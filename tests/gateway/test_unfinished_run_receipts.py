"""Offline recovery can enumerate all unfinished sessions without a client ledger."""

from contextlib import closing
from gateway.platforms.api_server_run_idempotency import (
    RunIdempotencyStore,
    TERMINAL_STATUSES,
)


def test_unfinished_receipts_survive_reopen_and_preserve_scope_and_outcomes(tmp_path):
    path = str(tmp_path / "runs.db")
    with closing(RunIdempotencyStore(path)) as store:
        for status in (
            "queued",
            "running",
            "waiting_for_approval",
            "stopping",
            *sorted(TERMINAL_STATUSES),
        ):
            store.reserve(
                "owner",
                status,
                status,
                status,
                {"session_id": "session-" + status, "status": status},
            )
        store.reserve(
            "foreign",
            "foreign",
            "foreign",
            "foreign",
            {"session_id": "other", "status": "running"},
        )
    with closing(RunIdempotencyStore(path)) as store:
        records = store.unfinished_runs("owner")
        assert {r["run_id"] for r in records} == {
            "queued",
            "running",
            "waiting_for_approval",
            "stopping",
        }
        assert {r["status"]["session_id"] for r in records} == {
            "session-queued",
            "session-running",
            "session-waiting_for_approval",
            "session-stopping",
        }
        assert store.unfinished_runs("missing") == []
        for status in TERMINAL_STATUSES:
            assert store.status_for_run("owner", status)["status"]["status"] == status
        store.update_status(
            "running", {"session_id": "session-running", "status": "interrupted"}
        )
        assert "running" not in {r["run_id"] for r in store.unfinished_runs("owner")}
        assert (
            store.status_for_run("foreign", "foreign")["status"]["status"] == "running"
        )
