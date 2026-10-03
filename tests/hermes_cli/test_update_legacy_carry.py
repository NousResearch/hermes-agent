"""After a failed PM takeover the retry must not record a stripped ledger.

Regression test for #126886: the takeover fails before PM commits runtime
facts, so the retry still faces a pre-PM install. Every update sync must
carry the old venv's extras (``legacy_selection``); otherwise the retry syncs
an empty set and commits it as the permanent feature ledger.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

from hermes_cli import venv_sync


def _legacy_checkout(tmp_path: Path, *anchors: str) -> Path:
    root = tmp_path / "co"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "uv.lock").write_text("lock-v1\n")
    site = root / "venv" / "lib" / "python3.11" / "site-packages"
    for anchor in anchors:
        (site / anchor).mkdir(parents=True)
    return root


def _home_token(tmp_path: Path):
    from hermes_constants import set_hermes_home_override

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return home, set_hermes_home_override(str(home))


def _capture_sync(monkeypatch):
    import pm

    calls = []

    def sync(extras=None, *, explicit, project_root, evict_incompatible_plugins):
        calls.append(extras)

    monkeypatch.setattr(pm, "venv_is_current", lambda *, project_root: False)
    monkeypatch.setattr(pm, "sync_venv", sync)
    return calls


CARRIED = ("fal_client", "telegram", "mcp")
EXPECTED = ["all", "computer-use", "fal", "mcp", "telegram"]


def test_startup_sync_carries_legacy_extras_without_facts(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override

    home, token = _home_token(tmp_path)
    try:
        root = _legacy_checkout(tmp_path, *CARRIED)
        calls = _capture_sync(monkeypatch)
        assert venv_sync.sync(root)["state"] == "synced"
        assert calls == [EXPECTED]
    finally:
        reset_hermes_home_override(token)


def test_startup_sync_defers_to_recorded_facts(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override
    from pm.environments import runtime_facts_path

    home, token = _home_token(tmp_path)
    try:
        root = _legacy_checkout(tmp_path, *CARRIED)
        facts = runtime_facts_path(root)
        facts.parent.mkdir(parents=True, exist_ok=True)
        facts.write_text(json.dumps(
            {"schema": 1, "packages": {"venv": {"stamp": "s", "extras": ["all", "fal"]}}}),
            encoding="utf-8")
        calls = _capture_sync(monkeypatch)
        assert venv_sync.sync(root)["state"] == "synced"
        assert calls == [None]
    finally:
        reset_hermes_home_override(token)


def test_update_prepare_carries_legacy_extras_without_facts(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override
    from hermes_cli import update_completion
    import pm.client
    import pm.environments
    from pm import receipt

    home, token = _home_token(tmp_path)
    try:
        root = _legacy_checkout(tmp_path, *CARRIED)
        calls = _capture_sync(monkeypatch)
        monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
        monkeypatch.setattr(venv_sync, "refuse_foreign_owned_venv", lambda r: None)
        monkeypatch.setattr(venv_sync, "arm_completion", lambda r: root / ".x")
        monkeypatch.setattr(venv_sync, "collect_superseded_generations", lambda r: None)
        monkeypatch.setattr(receipt, "worker_context", lambda cid: contextlib.nullcontext())
        monkeypatch.setattr(receipt, "last_for_update", lambda cid: None)
        monkeypatch.setattr(pm.environments, "project_python", lambda r: Path("/nonexistent"))
        monkeypatch.setattr(pm.environments, "activation_environment", lambda r: {})
        monkeypatch.setattr(subprocess, "call", lambda *a, **k: 0)
        request = {"source": str(root), "receipt": {"update_id": "u1"},
                   "bytecode_cache": str(tmp_path / "bc")}
        update_completion._prepare(request, tmp_path / "req.json", tmp_path / "res.json")
        assert calls == [EXPECTED]
    finally:
        reset_hermes_home_override(token)
