"""Known-completed references recover without retiring live sibling work."""

import json
import subprocess
import sys
import threading
from contextlib import contextmanager

import pytest

from hermes_cli import execution_budget as budget


def _permit(home, key):
    permit, refusal = budget.try_acquire_execution_permit(profile_home=home, execution_id=key, surface="test")
    assert refusal is None
    return permit


def test_settled_release_recovers_automatically_after_transient_lock_failure(tmp_path, monkeypatch):
    from agent.execution_admission import release_conversation_permit

    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 2\n", encoding="utf-8")
    parent = _permit(tmp_path, "parent")
    child, refusal = budget.borrow_execution_permit(parent.vouch())
    assert refusal is None
    original = budget.registry._FileLock
    attempted = threading.Event()
    recovered = threading.Event()

    @contextmanager
    def fail_once(path, **kwargs):
        if not attempted.is_set():
            attempted.set()
            raise TimeoutError("synthetic release contention")
        with original(path, **kwargs):
            yield
        recovered.set()

    monkeypatch.setattr(budget.registry, "_FileLock", fail_once)
    try:
        release_conversation_permit(parent)
        assert attempted.is_set()
        assert recovered.wait(5), "settled release was never retried"
        # The retry owns the file lock; snapshot waits for its committed retirement.
        _, holders = budget.execution_budget_snapshot(profile_home=tmp_path)
        assert [entry["lease_id"] for entry in holders] == [child.lease.lease_id]
    finally:
        monkeypatch.setattr(budget.registry, "_FileLock", original)
        parent.release()
        child.release()


@pytest.mark.parametrize("receipt_encoding", ["utf-8", "utf-8-sig"])
def test_one_line_recovery_reconciles_only_durable_exact_completed_reference(tmp_path, monkeypatch, receipt_encoding):
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 2\n", encoding="utf-8")
    completed = _permit(tmp_path, "completed")
    live = _permit(tmp_path, "still-running")
    original = budget.registry._FileLock
    monkeypatch.setattr(budget.registry, "_FileLock", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("locked")))
    try:
        try:
            completed.release()
        except TimeoutError:
            pass
        receipt = completed.lease.state_path.parent / "execution_retirements" / f"{completed.lease.lease_id}.json"
        receipt.write_text(json.dumps(completed.identity), encoding=receipt_encoding)
        monkeypatch.setattr(budget.registry, "_FileLock", original)
        command = (
            "from hermes_cli.execution_budget import reconcile_execution_permits; "
            f"print(reconcile_execution_permits(profile_home={str(tmp_path)!r}))")
        result = subprocess.run([sys.executable, "-c", command], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1"
        rows = json.loads(completed.lease.state_path.read_text(encoding="utf-8-sig"))["entries"]
        assert [row["lease_id"] for row in rows] == [live.lease.lease_id]
        completed.release()
        assert budget.execution_budget_snapshot(profile_home=tmp_path)[1][0]["lease_id"] == live.lease.lease_id
    finally:
        monkeypatch.setattr(budget.registry, "_FileLock", original)
        completed.release()
        live.release()
