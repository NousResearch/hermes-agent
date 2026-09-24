"""Delivery obligations keep their recovery contract on PostgreSQL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from gateway import delivery_ledger as ledger


def _record(obligation_id: str = "ob-1", *, profile: str | None = None) -> None:
    ledger.record_obligation(
        obligation_id=obligation_id,
        session_key="agent:main:telegram:dm:chat-1",
        platform="telegram",
        chat_id="chat-1",
        thread_id="topic-1",
        content="PostgreSQL delivery",
        adapter_profile=profile,
    )


def _row(database, obligation_id: str = "ob-1"):
    with database._read_ctx() as connection:
        return connection.execute(
            """SELECT state, attempts, content, last_error, adapter_profile,
                      owner_token, lease_expires_at, updated_at
               FROM delivery_obligations WHERE obligation_id = ?""",
            (obligation_id,),
        ).fetchone()


def test_postgres_state_machine_persists_without_local_sqlite(
    postgres_db, postgres_home
):
    _record()
    row = _row(postgres_db)
    assert (row["state"], row["attempts"], row["content"], row["adapter_profile"]) == (
        "pending",
        0,
        "PostgreSQL delivery",
        "default",
    )

    ledger.mark_attempting("ob-1")
    assert _row(postgres_db)["state"] == "attempting"
    ledger.mark_failed("ob-1", "send_path_degraded")
    assert _row(postgres_db)["last_error"] == "send_path_degraded"

    claimed = ledger.sweep_failed_for_runtime("telegram")
    assert len(claimed) == 1
    assert claimed[0]["needs_marker"] is True
    assert claimed[0]["attempts"] == 1
    assert ledger.release_runtime_claim("ob-1", "send_path_degraded") is True
    released = _row(postgres_db)
    assert (released["state"], released["attempts"], released["last_error"]) == (
        "failed",
        0,
        "send_path_degraded",
    )
    assert len(ledger.sweep_failed_for_runtime("telegram")) == 1
    ledger.mark_delivered("ob-1")
    assert _row(postgres_db)["state"] == "delivered"
    assert not (postgres_home / "state.db").exists()


def test_live_foreign_lease_blocks_recovery_until_expiry(postgres_db):
    _record()
    now = time.time()
    postgres_db._execute_write(
        lambda connection: connection.execute(
            """UPDATE delivery_obligations
           SET owner_token = ?, lease_expires_at = ? WHERE obligation_id = ?""",
            ("other-container", now + 60, "ob-1"),
        )
    )

    assert ledger.sweep_recoverable(now=now) == []
    claimed = ledger.sweep_recoverable(now=now + 61)
    assert len(claimed) == 1
    assert claimed[0]["needs_marker"] is False
    assert ledger.sweep_recoverable(now=now + 62) == []


def test_flood_deadline_and_profile_scope_are_preserved(postgres_db):
    _record(profile="reviewer")
    ledger.mark_failed("ob-1", "flood_control:30")
    row = _row(postgres_db)
    due = ledger.flood_not_before(row["updated_at"], "flood_control:30")

    assert (
        ledger.sweep_failed_for_runtime("telegram", now=due - 1, profile="default")
        == []
    )
    assert (
        ledger.sweep_failed_for_runtime("telegram", now=due - 1, profile="reviewer")
        == []
    )
    pending = ledger.pending_retries(now=due - 1)
    assert [(item["platform"], item["profile"]) for item in pending] == [
        ("telegram", "reviewer")
    ]

    claimed = ledger.sweep_failed_for_runtime("telegram", now=due, profile="reviewer")
    assert len(claimed) == 1
    assert claimed[0]["marker"] == ledger.FLOOD_MARKER


def test_postgres_prunes_only_old_terminal_rows(postgres_db):
    _record("delivered")
    ledger.mark_delivered("delivered")
    _record("pending")
    cutoff = time.time() - ledger._RETENTION_SECONDS - 1
    postgres_db._execute_write(
        lambda connection: connection.execute(
            "UPDATE delivery_obligations SET updated_at = ?",
            (cutoff,),
        )
    )

    _record("trigger-prune")

    assert _row(postgres_db, "delivered") is None
    assert _row(postgres_db, "pending")["state"] == "pending"


def test_two_processes_cannot_claim_one_expired_obligation(postgres_db, postgres_home):
    _record()
    postgres_db._execute_write(
        lambda connection: connection.execute(
            """UPDATE delivery_obligations
           SET owner_token = ?, lease_expires_at = ? WHERE obligation_id = ?""",
            ("stopped-container", time.time() - 1, "ob-1"),
        )
    )
    script = (
        "import json; "
        "from gateway.delivery_ledger import sweep_recoverable; "
        "print(json.dumps(sweep_recoverable(deliverable_platforms={'telegram'})))"
    )
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(postgres_home)
    first = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    second = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    first_stdout, first_stderr = first.communicate(timeout=30)
    second_stdout, second_stderr = second.communicate(timeout=30)
    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second_stderr

    claims = [json.loads(first_stdout), json.loads(second_stdout)]
    assert sorted(len(result) for result in claims) == [0, 1]
    assert _row(postgres_db)["attempts"] == 1


def test_writable_open_upgrades_version_one_delivery_schema(postgres_home):
    from hermes_state import SessionDB
    from hermes_state_postgres_schema import POSTGRES_SCHEMA_VERSION

    with SessionDB(postgres_home / "state.db") as database:

        def restore_version_one(connection):
            connection.execute("DROP TABLE delivery_obligations")
            connection.execute("UPDATE postgres_schema_version SET version = 1")

        database._execute_write(restore_version_one)

    with SessionDB(postgres_home / "state.db") as upgraded:
        with upgraded._read_ctx() as connection:
            assert (
                connection.execute(
                    "SELECT version FROM postgres_schema_version"
                ).fetchone()[0]
                == POSTGRES_SCHEMA_VERSION
            )
            assert (
                connection.execute(
                    "SELECT to_regclass('delivery_obligations')"
                ).fetchone()[0]
                == "delivery_obligations"
            )
