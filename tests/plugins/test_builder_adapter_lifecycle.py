from pathlib import Path

from hermes_cli import kanban_db
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.native import NativeKanbanBackend
from tests.plugins.test_builder_adapter_schema import (
    FakeKanban,
    make_adapter,
    request_payload,
)


def test_lifecycle_maps_native_states_deterministically(tmp_path):
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    created = adapter.dispatch("principal", payload)
    assert created["status"] == "ACCEPTED"
    kanban.status = "running"
    running = adapter.get_status("principal", payload["dispatch_id"], payload["cycle_id"])
    assert running["status"] == "RUNNING"
    assert running["terminal"] is False


def test_cancel_fails_unknown_when_termination_unproven(tmp_path):
    kanban = FakeKanban()
    kanban.cancelled = False
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    adapter.dispatch("principal", payload)
    result = adapter.cancel(
        "principal", payload["dispatch_id"], payload["cycle_id"], "TIMEOUT"
    )
    assert result["status"] == "BLOCKED"
    assert result["side_effects_state"] == "UNKNOWN"
    assert result["errors"][0]["code"] == "CANCELLATION_UNCONFIRMED"


def test_cross_principal_status_is_denied(tmp_path):
    adapter = make_adapter(tmp_path)
    payload = request_payload(tmp_path)
    adapter.dispatch("principal", payload)
    try:
        adapter.get_status("attacker", payload["dispatch_id"], payload["cycle_id"])
    except AdapterError as error:
        assert error.code == "AUTHORIZATION_FAILED"
    else:
        raise AssertionError("cross-principal status access succeeded")


def test_real_accepted_dispatch_persists_task_and_releases_lease_last(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    board = "stage1-lifecycle"
    backend = NativeKanbanBackend(board=board)
    adapter = make_adapter(tmp_path, kanban=backend)
    payload = request_payload(tmp_path)

    accepted = adapter.dispatch("principal", payload)
    assert accepted["status"] == "ACCEPTED"
    assert accepted["kanban_task_id"]
    record = adapter.store.get(payload["dispatch_id"])
    assert record["task_id"] == accepted["kanban_task_id"]
    assert record["phase"] == "TASK_CREATED"
    with kanban_db.connect_closing(board=board) as conn:
        task = kanban_db.get_task(conn, accepted["kanban_task_id"])
        assert task is not None
        assert task.status == "ready"
    queued = adapter.get_status(
        "principal", payload["dispatch_id"], payload["cycle_id"]
    )
    assert queued["status"] == "QUEUED"
    assert queued["status"] != "UNKNOWN"

    class Completion:
        def __init__(self, *_args):
            pass

        def complete(self, *_args):
            return {"git": {"resulting_sha": "a" * 40}}

    monkeypatch.setattr(
        "plugins.builder_adapter.adapter.CompletionAttestor", Completion
    )
    with kanban_db.connect_closing(board=board) as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done',worker_pid=NULL,"
                "claim_lock=NULL WHERE id=?",
                (accepted["kanban_task_id"],),
            )
            conn.execute(
                "INSERT INTO governed_worker_lifecycle("
                "task_id,run_id,worker_pid,process_group,start_identity,"
                "completion_lease,state,terminal_callback_at,"
                "cleanup_confirmed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    accepted["kanban_task_id"],
                    1,
                    999999,
                    999999,
                    "darwin-bsdinfo-v1:1:1",
                    "lease-proof",
                    "terminated",
                    1.0,
                    2.0,
                ),
            )

    completed = adapter.get_status(
        "principal", payload["dispatch_id"], payload["cycle_id"]
    )
    assert completed["status"] == "SUCCEEDED"
    assert adapter.store.get(payload["dispatch_id"])["phase"] == "COMPLETED"
    with kanban_db.connect_closing(board=board) as conn:
        lifecycle = conn.execute(
            "SELECT state,completion_lease FROM governed_worker_lifecycle "
            "WHERE task_id=?",
            (accepted["kanban_task_id"],),
        ).fetchone()
    assert tuple(lifecycle) == ("attested", "")
    assert backend.release_completion_lease(accepted["kanban_task_id"]) is True
