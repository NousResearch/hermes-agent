"""Nested command boundaries and copied contexts cannot close an outer receipt."""
import contextvars

from hermes_cli import update_receipt as receipts


def test_command_scope_retains_outer_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "_code_identity", lambda **kw: {})
    receipts.begin_update_receipt()
    outer = receipts.current_correlation_id()
    try:
        with receipts.update_receipt_scope():
            receipts.begin_update_receipt()
            receipts.finalize_update_receipt("success")
            assert receipts.finalize_pending_update_receipt(0) is None
        assert receipts.current_correlation_id() == outer
        assert receipts._current.get().data["outcome"] == "running"
    finally:
        receipts.finalize_update_receipt("success")


def test_copied_context_finalize_does_not_mutate_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "_code_identity", lambda **kw: {})
    receipts.begin_update_receipt()
    try:
        contextvars.copy_context().run(receipts.finalize_pending_update_receipt, 1, "child failed")
        parent = receipts._current.get().data
        assert parent["outcome"] == "running"
        assert "exit_code" not in parent
        assert "stop_reason" not in parent
    finally:
        receipts.finalize_update_receipt("success")


def test_failure_detail_survives_the_sys_exit_boundary(tmp_path, monkeypatch):
    """#132089: a mid-update sys.exit(1) reaches the command boundary as
    stop_reason "sys.exit(1)"; the reason recorded before the exit must land in
    the same receipt, or scheduled runs (stdout discarded) are undiagnosable."""
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "_code_identity", lambda **kw: {})
    receipts.begin_update_receipt()
    receipts.record_failure_detail("resolve main source channel: boom")
    path = receipts.finalize_pending_update_receipt(1, "sys.exit(1)")
    assert path is not None
    data = json.loads(path.read_text())
    assert data["outcome"] == "failed"
    assert data["stop_reason"] == "sys.exit(1)"
    assert data["failure_detail"] == "resolve main source channel: boom"


def test_failure_detail_is_noop_without_open_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipts.record_failure_detail("no receipt open")  # must not raise
