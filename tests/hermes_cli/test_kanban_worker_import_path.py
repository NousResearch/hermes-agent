"""Tests: a routed-profile kanban worker keeps the interpreter's import path.

A pm store interpreter has no editable install; ``hermes_cli`` is importable only
through the launcher-exported ``PYTHONPATH=<repo>:<venv site-packages>``. When the
multiplexed gateway dispatches for another profile, ``_default_spawn`` builds the
worker env with the secret scrub, which strips Hermes-owned PYTHONPATH entries —
so ``<store python> -m hermes_cli.main`` died with "No module named 'hermes_cli'"
on every gateway-dispatched worker while CLI dispatch (no scrub) worked.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _make_task(kb):
    return kb.Task(
        id="t_imp", title="import path", body=None, assignee="w", status="running",
        priority=0, created_by="test", created_at=1, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock="lock", claim_expires=None,
        tenant=None, current_run_id=1,
    )


def _spawn_routed(monkeypatch, tmp_path, hermes_argv) -> dict:
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("PYTHONPATH", REPO_ROOT)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: list(hermes_argv))
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    kbd._default_spawn(_make_task(kb), str(workspace))
    return captured["env"]


def test_module_form_worker_keeps_repo_on_pythonpath(monkeypatch, tmp_path):
    from hermes_cli import kanban_db_dispatch as kbd

    env = _spawn_routed(monkeypatch, tmp_path, kbd._module_hermes_argv())
    assert REPO_ROOT in env.get("PYTHONPATH", "").split(os.pathsep)


def test_foreign_launcher_does_not_inherit_hermes_pythonpath(monkeypatch, tmp_path):
    """A ``$HERMES_BIN``/PATH launcher may be another interpreter version; the
    Hermes-owned entries must stay stripped for it."""
    env = _spawn_routed(monkeypatch, tmp_path, ["/opt/other/bin/hermes"])
    assert REPO_ROOT not in env.get("PYTHONPATH", "").split(os.pathsep)
