"""Worker receipts cross the process seam by identity, never latest.json."""
import contextvars

import pytest

from pm import receipt


def test_worker_receipt_is_attributed_only_to_its_invoking_update(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def worker():
        with receipt.worker_context("update-a"):
            token = receipt.begin("sync")
            receipt.record_step("dependency-sync", False, "candidate refused")
            receipt.finalize("failed", 1, token=token)
            return receipt.last_completed()

    data = contextvars.Context().run(worker)
    assert data["update_id"] == "update-a"
    assert data["outcome"] == "failed"
    receipt.accept_worker_receipt(data, "update-a")
    assert receipt.last_for_update("update-a", consume=True) == data
    assert receipt.last_for_update("update-b") is None
    with pytest.raises(ValueError, match="correlation"):
        receipt.accept_worker_receipt(data, "update-b")
