"""Acknowledgement compacts only consumed outcomes; epochs reject replay after compaction."""
import pytest

from gateway.platforms.api_server_workspace_models import StartRequest
from gateway.platforms.api_server_workspace_store import WorkspaceConflict, WorkspaceKeyExpired, WorkspaceStore
from tools.process_registry_capture import ProcessByteCapture


def test_epoch_compaction_runs_continuously_without_reexecuting_old_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(WorkspaceStore, "OPERATIONS_PER_EPOCH", 2)
    monkeypatch.setattr(WorkspaceStore, "MAX_UNACKNOWLEDGED", 4)
    store = WorkspaceStore(tmp_path / "operations.db")
    body = StartRequest(session_id="owned", command="build")
    retained_key = store.admission_epoch + ":retained"
    retained, _ = store.reserve("owner", retained_key, "process", body)
    retained.status, retained.termination_confirmed = "completed", True
    capture = ProcessByteCapture(100)
    capture.append("stdout", b"original\xff")
    store.capture("owner", retained, capture)
    first_ack_key = ""
    try:
        for index in range(20):
            key = store.admission_epoch + f":{index}"
            receipt, created = store.reserve("owner", key, "process", body)
            assert created
            receipt.status, receipt.termination_confirmed = "completed", True
            store.save("owner", receipt)
            assert store.acknowledge("owner", key, receipt.operation_id).acknowledged
            if not first_ack_key:
                first_ack_key = key
        with pytest.raises(WorkspaceKeyExpired):
            store.reserve("owner", first_ack_key, "process", body)
        replay, created = store.reserve("owner", retained_key, "process", body)
        assert not created and replay.operation_id == retained.operation_id
        assert store.output("owner", retained.operation_id)["stdout"] == b"original\xff"
        assert store.connection.execute("SELECT COUNT(*) FROM operations").fetchone()[0] <= 3
        for index in range(3):
            store.reserve("owner", store.admission_epoch + f":unacknowledged-{index}", "process", body)
        with pytest.raises(WorkspaceConflict, match="Unacknowledged"):
            store.reserve("owner", store.admission_epoch + ":overflow", "process", body)
        assert store.output("owner", retained.operation_id)["stdout"] == b"original\xff"
    finally:
        store.close()
