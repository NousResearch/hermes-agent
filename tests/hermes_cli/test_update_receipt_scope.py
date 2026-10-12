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
