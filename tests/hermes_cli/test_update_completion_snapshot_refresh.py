"""_prepare refreshes the committed snapshot when the venv sync is a no-op (#122425)."""
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pm
import pm.client
import pm.environments
import pm.receipt
import pm.workspace
from hermes_cli import source_stamp, update_completion, venv_sync
from pm.package import InstallError


def _stub_prepare(monkeypatch, tmp_path, calls, *, refresh_side_effect=None):
    root = tmp_path / "checkout"
    root.mkdir()
    update_id = "d" * 32
    request = {"source": str(root), "receipt": {"update_id": update_id},
               "bytecode_cache": str(tmp_path / "bytecode")}
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(pm, "venv_is_current", lambda *, project_root=None: True)
    monkeypatch.setattr(
        pm, "sync_venv",
        lambda *, explicit, project_root, evict_incompatible_plugins: calls.setdefault("synced", True),
    )
    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr(venv_sync, "refuse_foreign_owned_venv", lambda project_root: None)
    monkeypatch.setattr(venv_sync, "arm_completion", lambda project_root: None)
    monkeypatch.setattr(venv_sync, "collect_superseded_generations", lambda project_root: None)
    monkeypatch.setattr(pm.receipt, "worker_context", lambda update_id: nullcontext())
    monkeypatch.setattr(pm.receipt, "last_for_update", lambda update_id: {"update_id": update_id})
    monkeypatch.setattr(pm.environments, "project_python", lambda project_root: Path(sys.executable))
    monkeypatch.setattr(
        pm.environments, "activation_environment", lambda project_root: dict(os.environ))
    monkeypatch.setattr(
        source_stamp, "write_source_stamp",
        lambda project_root: calls.setdefault("stamped", []).append(Path(project_root)),
    )

    def fake_refresh(project_root):
        calls.setdefault("refreshed", []).append(Path(project_root))
        if refresh_side_effect is not None:
            refresh_side_effect()

    monkeypatch.setattr(pm.workspace, "sync_sources", fake_refresh)

    def fake_call(command, *, cwd, env):
        assert Path(cwd) == root
        result_path.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        return 0

    monkeypatch.setattr(update_completion.subprocess, "call", fake_call)
    return request, request_path, result_path


def test_prepare_refreshes_snapshot_when_sync_is_noop(tmp_path, monkeypatch):
    """The was_current/sync_sources block owns the headline fix: without it,
    a current venv leaves the committed snapshot on the old code, and this
    test fails because no refresh is recorded."""
    calls = {}
    request, request_path, result_path = _stub_prepare(monkeypatch, tmp_path, calls)

    assert update_completion._prepare(request, request_path, result_path) == 0

    root = Path(request["source"])
    assert calls.get("synced") is True
    assert calls.get("stamped") == [root]
    assert calls.get("refreshed") == [root]
    assert request["pm_receipt"] == {"update_id": request["receipt"]["update_id"]}


def test_prepare_contains_refresh_failure(tmp_path, monkeypatch):
    """A refresh InstallError must not report a good sync as a failed update."""
    def boom():
        raise InstallError("venv", "no dependency environment is committed for this install",
                           "run `hermes pm install`")

    calls = {}
    request, request_path, result_path = _stub_prepare(
        monkeypatch, tmp_path, calls, refresh_side_effect=boom)

    assert update_completion._prepare(request, request_path, result_path) == 0

    assert calls.get("synced") is True
    assert calls.get("stamped") == [Path(request["source"])]
    assert calls.get("refreshed") == [Path(request["source"])]
