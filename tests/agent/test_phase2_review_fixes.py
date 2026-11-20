"""Regression tests for the greptile review findings on the authority store."""

from __future__ import annotations

import hashlib
import os
import threading
import time

import pytest

from agent import phase2_ledger as ledger_module
from agent.phase2_authority import (
    Phase2AuthorityStore,
    ResultRejection,
    validate_sealed_envelope,
)
from agent.phase2_sqlite import _DB_LOCK, _open_phase2_sqlite


def _node(node_id: str = "n-tool") -> dict:
    return {
        "node_id": node_id,
        "objective": f"execute {node_id}",
        "idempotency_key": hashlib.sha256(node_id.encode()).hexdigest(),
        "execution_surface": "local_tool",
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": "2030-01-01T00:00:00+00:00",
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "input_hash": "c" * 64,
    }


def _plan(*nodes: dict) -> dict:
    return {
        "contract_version": 2,
        "graph_id": "g-authority",
        "nodes": list(nodes or (_node(),)),
    }


def _granted(store: Phase2AuthorityStore, ttl_s: float = 300) -> dict:
    store.seal_plan(_plan(), policy_hash="d" * 64)
    return store.grant_node(
        "g-authority", "n-tool", holder="hermes:test", ttl_s=ttl_s, attempt_id="at-1"
    )


def test_unexpandable_home_root_reports_a_field_error(tmp_path):
    """A root like ~missing-user must fail validation, not raise RuntimeError."""

    store = Phase2AuthorityStore(tmp_path / "authority.db")
    envelope = dict(_granted(store))
    envelope["roots"] = ["~no-such-user-qa7/work"]

    assert "roots" in validate_sealed_envelope(envelope)


def test_complete_node_reads_the_clock_after_waiting_for_the_lock(tmp_path):
    """A worker queued behind the writer lock must not accept an expired lease."""

    store = Phase2AuthorityStore(tmp_path / "authority.db")
    envelope = _granted(store, ttl_s=1)
    outcome: dict = {}

    def worker() -> None:
        try:
            store.complete_node(envelope, result_hash="a" * 64)
            outcome["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - the assertion inspects it
            outcome["error"] = exc

    _DB_LOCK.acquire()
    try:
        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(1.6)
    finally:
        _DB_LOCK.release()
    thread.join(10)

    assert not outcome.get("ok")
    assert isinstance(outcome.get("error"), ResultRejection)


def test_closing_one_connection_keeps_the_pinned_descriptor_open(tmp_path):
    """Closing a reader must not release locks a live sibling connection holds."""

    db = tmp_path / "authority.db"
    first = _open_phase2_sqlite(db)
    second = _open_phase2_sqlite(db)
    fd = first._phase2_fd
    assert fd is not None

    first.close()
    os.fstat(fd)  # still open: closing here would drop the sibling's POSIX locks

    second.close()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_authority_events_node_lookup_index_exists(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    _granted(store)
    conn = store._connect()
    try:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    finally:
        conn.close()

    assert "authority_events_node_lookup" in names


def test_connect_uses_the_shared_journal_setup(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        ledger_module,
        "apply_wal_with_fallback",
        lambda conn, **kwargs: calls.append(kwargs) or "wal",
    )

    store = Phase2AuthorityStore(tmp_path / "authority.db")
    _granted(store)

    assert calls, "the store must open through apply_wal_with_fallback"
