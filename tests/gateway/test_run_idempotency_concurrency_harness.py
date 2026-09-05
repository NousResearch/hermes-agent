"""Hermetic concurrency harness for the durable run reservation surface.

This is a separate test-owned surface from the cross-node JSON receipt harness.
It composes the canonical fake gateway process/connector fixture with the real
``RunIdempotencyStore`` from the immutable candidate under test.  The fake
connector is never connected, all databases live under pytest's temporary
directory, and the only process termination is a disposable child paused
inside its own SQLite transaction.

The assertions are journal-mode agnostic: this harness proves transaction and
recovery semantics, not WAL enablement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import queue
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.relay.stub_connector import StubConnector


_SCOPE = "harness:receipt-sqlite:scope"
_KEY = "harness:receipt-sqlite:same-key"


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _fake_delivery(
    connector: StubConnector, payload: dict[str, Any]
) -> dict[str, Any]:
    """Return a canonical in-memory connector ACK for a synthetic delivery."""

    connector.next_send_result = {
        "success": True,
        "message_id": payload["message_id"],
    }
    return asyncio.run(
        connector.send_outbound({"op": "send", "content": payload["message"]})
    )


def _reserve_worker(
    db_path: str,
    barrier: Any,
    result_queue: Any,
    worker_index: int,
    payload: dict[str, Any],
    fingerprint: str,
) -> None:
    """Race one independent production store connection against its sibling."""

    store = RunIdempotencyStore(db_path)
    try:
        barrier.wait(timeout=10)
        outcome, record = store.reserve(
            _SCOPE,
            _KEY,
            fingerprint,
            f"run-{payload['message_id']}",
            {
                "status": "queued",
                "message_id": payload["message_id"],
                "payload_digest": fingerprint,
                "fake_delivery_ack": True,
            },
        )
        result_queue.put(
            {
                "worker": worker_index,
                "outcome": outcome,
                "run_id": record["run_id"],
                "status": record["status"],
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "worker": worker_index,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        store.close()


def _run_reservation_race(
    db_path: Path,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run two separate workers and return their bounded result records."""

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(payloads))
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_reserve_worker,
            args=(
                str(db_path),
                barrier,
                result_queue,
                index,
                payload,
                _payload_digest(payload),
            ),
        )
        for index, payload in enumerate(payloads)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=15)
        lingering = [worker for worker in workers if worker.is_alive()]
        if lingering:
            for worker in lingering:
                worker.terminate()
            for worker in lingering:
                worker.join(timeout=5)
            pytest.fail("reservation workers did not finish within the bounded timeout")
        assert [worker.exitcode for worker in workers] == [0, 0]

        results = []
        for _ in workers:
            try:
                results.append(result_queue.get(timeout=5))
            except queue.Empty as exc:
                raise AssertionError("reservation worker returned no result") from exc
        return sorted(results, key=lambda result: result["worker"])
    finally:
        result_queue.close()
        result_queue.join_thread()


def test_simultaneous_different_deliveries_admit_only_one_reservation(
    tmp_path: Path,
) -> None:
    """Two ACKed identities cannot both reserve one idempotency key."""

    db_path = tmp_path / "same-key-race.db"
    bootstrap = RunIdempotencyStore(str(db_path))
    bootstrap.close()

    fake_connector = StubConnector(
        CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform="discord",
            label="Hermes fake gateway",
            max_message_length=2000,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=True,
            markdown_dialect="discord",
            len_unit="chars",
        )
    )
    payloads = [
        {"message_id": "fake-message-a", "message": "payload alpha"},
        {"message_id": "fake-message-b", "message": "payload beta"},
    ]
    assert [_fake_delivery(fake_connector, payload)["success"] for payload in payloads] == [
        True,
        True,
    ]
    results = _run_reservation_race(db_path, payloads)

    assert [result["outcome"] for result in results].count("created") == 1
    assert [result["outcome"] for result in results].count("conflict") == 1

    creator = next(result for result in results if result["outcome"] == "created")
    conflicted = next(result for result in results if result["outcome"] == "conflict")
    assert creator["run_id"] in {"run-fake-message-a", "run-fake-message-b"}
    assert conflicted["run_id"] == creator["run_id"]
    assert conflicted["status"]["message_id"] == creator["status"]["message_id"]

    verifier = RunIdempotencyStore(str(db_path))
    try:
        winner_payload = next(
            payload
            for payload in payloads
            if f"run-{payload['message_id']}" == creator["run_id"]
        )
        loser_payload = next(
            payload
            for payload in payloads
            if f"run-{payload['message_id']}" != creator["run_id"]
        )
        assert verifier.lookup(
            _SCOPE, _KEY, _payload_digest(winner_payload)
        )[0] == "reused"
        assert verifier.lookup(
            _SCOPE, _KEY, _payload_digest(loser_payload)
        )[0] == "conflict"
    finally:
        verifier.close()


def test_reservation_transaction_rollback_recovers_for_retry(tmp_path: Path) -> None:
    """An INSERT failure rolls back the reservation, allowing a clean retry."""

    db_path = tmp_path / "rollback.db"
    store = RunIdempotencyStore(str(db_path))
    scope = "harness:receipt-sqlite:rollback"
    key = "rollback-key"
    fingerprint = "rollback-fingerprint"
    try:
        store._conn.execute(
            """CREATE TEMP TRIGGER abort_harness_reservation
               AFTER INSERT ON run_idempotency
               BEGIN
                   SELECT RAISE(ABORT, 'synthetic reservation rollback');
               END"""
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="synthetic reservation rollback"):
            store.reserve(
                scope,
                key,
                fingerprint,
                "run-aborted",
                {
                    "status": "queued",
                    "message_id": "fake-rollback-message",
                    "fake_delivery_ack": True,
                },
            )

        assert not store._conn.in_transaction
        store._conn.execute("DROP TRIGGER abort_harness_reservation")
        store._conn.commit()
        assert store.lookup(scope, key, fingerprint) == ("missing", None)

        outcome, record = store.reserve(
            scope,
            key,
            fingerprint,
            "run-recovered",
            {
                "status": "queued",
                "message_id": "fake-retry-message",
                "fake_delivery_ack": True,
            },
        )
        assert outcome == "created"
        assert record["run_id"] == "run-recovered"
        assert store.lookup(scope, key, fingerprint)[0] == "reused"
    finally:
        store.close()


def test_process_kill_before_reservation_commit_recovers(tmp_path: Path) -> None:
    """A killed disposable worker cannot leave an uncommitted reservation."""

    db_path = tmp_path / "killed-reservation.db"
    ready_path = tmp_path / "commit-paused"
    repo_root = Path(__file__).resolve().parents[2]
    isolated_home = tmp_path / "child-home"
    isolated_home.mkdir()
    child_script = textwrap.dedent(
        """
        import pathlib
        import sys
        import time

        from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

        database = sys.argv[1]
        ready = pathlib.Path(sys.argv[2])
        store = RunIdempotencyStore(database)

        def pause_before_commit(statement):
            if statement.strip().upper().startswith("COMMIT"):
                ready.touch()
                while True:
                    time.sleep(0.05)

        store._conn.set_trace_callback(pause_before_commit)
        store.reserve(
            "harness:receipt-sqlite:kill",
            "kill-key",
            "kill-fingerprint",
            "run-killed",
            {
                "status": "queued",
                "message_id": "fake-kill-message",
                "fake_delivery_ack": True,
            },
        )
        """
    )
    child_env = {
        "HOME": str(isolated_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(repo_root),
        "PYTHONUTF8": "1",
        "TZ": "UTC",
    }
    child = subprocess.Popen(
        [sys.executable, "-c", child_script, str(db_path), str(ready_path)],
        cwd=repo_root,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_path.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.exists(), (
            "disposable worker did not pause before COMMIT "
            f"(returncode={child.returncode})"
        )
        child.kill()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stderr is not None:
            child.stderr.close()

    recovered = RunIdempotencyStore(str(db_path))
    try:
        outcome, record = recovered.reserve(
            "harness:receipt-sqlite:kill",
            "kill-key",
            "kill-fingerprint",
            "run-after-kill",
            {
                "status": "queued",
                "message_id": "fake-recovery-message",
                "fake_delivery_ack": True,
            },
        )
        assert outcome == "created"
        assert record["run_id"] == "run-after-kill"
        assert recovered.lookup(
            "harness:receipt-sqlite:kill", "kill-key", "kill-fingerprint"
        )[0] == "reused"
    finally:
        recovered.close()
