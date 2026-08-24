"""Fresh-process boundary tests for post-update gateway restart work.

A broad ``sys.modules`` purge split the updater into old and new module objects
inside one interpreter. The restart phase now transfers its typed payload and
receipt state to a fresh Python process instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan


def test_restart_handoff_builds_a_fresh_interpreter_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        payload_path = Path(command[-1])
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["payload_path"] = payload_path
        captured["payload"] = json.loads(payload_path.read_text(encoding="utf-8"))
        status_path = Path(captured["payload"]["status_path"])
        captured["status_path"] = status_path
        status_path.write_text(
            json.dumps(
                {
                    "worker_completed": True,
                    "exit_code": 0,
                    "receipt_handoff_complete": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        update_cmd,
        "_m",
        lambda: SimpleNamespace(PROJECT_ROOT=tmp_path),
    )

    plan = UpdatePlan(
        install_method="git",
        profiles=["default"],
        runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=123)],
    )
    update_cmd._run_post_update_restart_in_fresh_process(
        gateway_mode=False,
        node_failures=["desktop"],
        _pre_update_plan=plan,
        _windows_gateway_resume={"resume_needed": True, "profiles": {}},
    )

    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1] == "-c"
    assert "_post_update_restart_worker" in command[2]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["payload"]["pre_update_plan"]["runtimes"][0]["pid"] == 123
    assert captured["payload"]["windows_gateway_resume"]["resume_needed"] is True
    assert not captured["payload_path"].exists(), "handoff payload must be removed"
    assert not captured["status_path"].exists(), "worker status must be removed"


def test_fresh_interpreter_has_one_coherent_module_identity(tmp_path):
    """Negative witness for the old split-module failure."""
    result_path = tmp_path / "identity.json"
    worker = """
import json, os, sys
import hermes_cli.main as main
import hermes_cli._startup_fast as startup_fast
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'pid': os.getpid(),
    'same_module': main._startup_fast is startup_fast,
    'cached_module': sys.modules.get('hermes_cli._startup_fast') is startup_fast,
}), encoding='utf-8')
"""
    completed = subprocess.run(
        [sys.executable, "-c", worker, str(result_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert completed.returncode == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["pid"] != os.getpid()
    assert result["same_module"] is True
    assert result["cached_module"] is True


def test_worker_rehydrates_plan_and_transferred_receipt(monkeypatch, tmp_path):
    import hermes_cli.update_receipt as update_receipt

    observed = {}
    finalized = []

    def fake_restart(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(update_cmd, "_run_post_update_restart", fake_restart)
    def fake_finalize(exit_code=None, stop_reason=""):
        finalized.append((exit_code, stop_reason))
        update_receipt._current = None

    monkeypatch.setattr(update_receipt, "finalize_pending_update_receipt", fake_finalize)
    update_receipt._current = None

    plan = UpdatePlan(
        install_method="git",
        profiles=["default"],
        runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=321)],
    )
    payload = {
        "gateway_mode": True,
        "node_failures": ["npm"],
        "pre_update_plan": plan.to_dict(),
        "windows_gateway_resume": {"resume_needed": True, "profiles": {}},
        "receipt_data": update_receipt.UpdateReceipt().data,
        "status_path": str(tmp_path / "worker-status.json"),
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    assert update_cmd._post_update_restart_worker(str(payload_path)) == 0
    assert observed["gateway_mode"] is True
    assert observed["node_failures"] == ["npm"]
    assert isinstance(observed["_pre_update_plan"], UpdatePlan)
    assert isinstance(observed["_pre_update_plan"].runtimes[0], RuntimeRecord)
    assert observed["_pre_update_plan"].runtimes[0].pid == 321
    assert observed["_windows_gateway_resume"]["resume_needed"] is True
    assert finalized == [(0, "fresh post-update restart worker")]
    worker_status = json.loads(
        (tmp_path / "worker-status.json").read_text(encoding="utf-8")
    )
    assert worker_status == {
        "worker_completed": True,
        "exit_code": 0,
        "receipt_handoff_complete": True,
    }
    update_receipt._current = None


def test_nonzero_worker_exit_propagates_and_clears_parent_receipt(
    monkeypatch, tmp_path
):
    import hermes_cli.update_receipt as update_receipt

    payload_paths = []
    marker_writes = []
    finalized = []

    def fake_run(command, **_kwargs):
        payload_paths.append(Path(command[-1]))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        update_cmd,
        "_m",
        lambda: SimpleNamespace(PROJECT_ROOT=tmp_path),
    )
    monkeypatch.setattr(
        update_cmd,
        "_write_gateway_update_exit_code",
        lambda success: marker_writes.append(success),
    )

    def fake_finalize(exit_code=None, stop_reason=""):
        finalized.append((exit_code, stop_reason))
        update_receipt._current = None

    monkeypatch.setattr(update_receipt, "finalize_pending_update_receipt", fake_finalize)
    update_receipt._current = update_receipt.UpdateReceipt()

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._run_post_update_restart_in_fresh_process(
            gateway_mode=True,
            node_failures=[],
            _pre_update_plan=None,
            _windows_gateway_resume=None,
        )

    assert exc_info.value.code == 7
    assert update_receipt._current is None
    assert marker_writes == [False]
    assert finalized == [
        (7, "fresh post-update restart worker did not acknowledge handoff")
    ]
    assert payload_paths and not payload_paths[0].exists()
    assert not Path(f"{payload_paths[0]}.status").exists()
