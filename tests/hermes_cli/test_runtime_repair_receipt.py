"""Runtime repair diagnostics must reach persisted update receipts (#111497)."""

import json
from types import SimpleNamespace

import pytest

from hermes_cli import managed_uv as uv
from hermes_cli import update_receipt as receipts


@pytest.mark.parametrize("status", ["safe", "repaired", "failed", "skipped", "not-applicable"])
@pytest.mark.parametrize("entry", ["update", "bootstrap"])
def test_runtime_result_reaches_persisted_receipt(tmp_path, monkeypatch, status, entry):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "_current", None)
    result = uv.RuntimeRepairResult(status, "repair diagnostic", "3.50.4", "3.53.1")
    monkeypatch.setattr(uv, "repair_vulnerable_runtime", lambda _: result)
    if entry == "update":
        monkeypatch.setattr(uv, "resolve_uv", lambda: "uv")
        monkeypatch.setattr(uv, "_uv_self_update_is_fresh", lambda: True)
        invoke = uv.update_managed_uv
    else:
        paths = iter([None, "uv"])
        monkeypatch.setattr(uv, "resolve_uv", lambda: next(paths))
        monkeypatch.setattr(uv, "_install_uv", lambda _: None)
        monkeypatch.setattr(uv, "_uv_version", lambda _: "test")
        invoke = uv.ensure_uv
    observed = []
    receipts.begin_update_receipt()
    invoke(repair_observer=observed.append)
    path = receipts.finalize_update_receipt("partial")
    data = json.loads(path.read_text())
    step, = data["steps"]
    assert step["name"] == "sqlite_runtime_repair"
    assert step["ok"] == (status in {"safe", "repaired"})
    assert all(value in step["detail"] for value in (
        result.status, result.detail, result.sqlite_before, result.sqlite_after))
    assert bool(data["skips"]) == (status in {"skipped", "not-applicable"})
    assert observed == [result]
    # The same repair hook is also used outside an update, without an active receipt.
    uv._run_runtime_repair("uv", observed.append)
    assert receipts._current is None
    assert observed == [result, result]


@pytest.mark.parametrize("stage, reason", [
    ("create", "candidate venv creation failed (rc=1): permission denied"),
    ("lock", "candidate dependency sync refused: uv.lock is missing"),
    ("sync", "candidate dependency sync failed (rc=1)"),
    ("smoke", "candidate venv smoke failed: missing module"),
])
def test_stage_rejection_preserves_reason_and_live_environment(tmp_path, monkeypatch, stage, reason):
    root = tmp_path / "checkout"
    live = root / "venv"
    live.mkdir(parents=True)
    sentinel = live / "sentinel"
    sentinel.write_text("unchanged")
    generation = root / ".hermes-runtime" / "python" / "generation"
    generation.mkdir(parents=True)
    if stage != "lock":
        (root / "uv.lock").write_text("lock")
    current = SimpleNamespace(wal_reset_vulnerable=True, sqlite_version_string="3.50.4")
    fixed = SimpleNamespace(sqlite_version_string="3.53.1")
    monkeypatch.setattr(uv, "probe_sqlite_runtime", lambda _: current)
    monkeypatch.setattr(uv, "_install_safe_python_generation", lambda *a, **kw: (
        generation, generation / "python", fixed))

    def run(argv, **kwargs):
        if argv[1] == "venv":
            from pathlib import Path
            Path(argv[2]).mkdir(parents=True)
        failed = (argv[1] == "venv" and stage == "create") or (argv[1] == "sync" and stage == "sync")
        return SimpleNamespace(returncode=int(failed), stderr="permission denied", stdout="")

    monkeypatch.setattr(uv.subprocess, "run", run)
    monkeypatch.setattr(uv, "_smoke_candidate_venv", lambda _: (False, "missing module", None))
    result = uv._repair_under_lock("uv", root=root, live=live, live_python=live / "python",
                                   runtime_root=root / ".hermes-runtime")
    assert result.status == "failed"
    assert result.detail == reason
    assert sentinel.read_text() == "unchanged"
    assert not generation.exists()
    assert not list((root / ".hermes-runtime").glob("venv-candidate-*"))
