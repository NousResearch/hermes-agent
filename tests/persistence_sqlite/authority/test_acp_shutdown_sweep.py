"""The gateway-stop ACP sweep reads one receipt per ACP session, not one per local session.

``settle_gateway_runtime`` ends every viewer-less ACP session at gateway stop (#118216).
``end_idle_acp_sessions`` used to find them by listing every local creation receipt and calling
``local_receipt`` on each one, so the stop parsed one receipt per local session ever created (CLI,
TUI, desktop, kanban, cron...) to find the few whose policy says acp. On a store with thousands of
local sessions that is seconds of parses inline on the gateway's event loop inside the stop
watchdog's budget.

The candidate ids now come from the indexed ``sessions`` table, and each candidate's own policy
still decides. The first case fails with the source change reverted (every receipt is read); the
second pins the peer-refresh relabel.
"""

from __future__ import annotations

from typing import Any

import pytest

from persistence.core.identity import SessionId
from persistence.sqlite.authority import _local
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved, seed


class _Authority:
    """The three things the sweep reads off a session authority."""

    def __init__(self, store: Any, epoch: int = 1) -> None:
        self.persistence = store
        self.epoch = epoch
        self.sessions: dict[str, Any] = {}


def _acp_store(store: Any, labels: dict[str, tuple[str, str | None]] | None = None) -> list[SessionId]:
    """2 ACP rows, 1 peer-refreshed ACP row, 20 CLI local sessions: 23 receipts, 3 ACP."""
    steps = [reserved(make_binding(f"acp{n}", request_id=f"acp{n}", source="acp")) for n in range(2)]
    steps += [reserved(make_binding("refreshed", request_id="refreshed", source="acp"))]
    steps += [reserved(make_binding(f"cli{n}", request_id=f"cli{n}", source="cli")) for n in range(20)]
    rows = labels if labels is not None else {
        "acp0": ("acp", "acp"), "acp1": ("acp", "acp"), "refreshed": ("local", "acp"),
        **{f"cli{n}": ("local", "cli") for n in range(20)}}

    def surface(db: Any) -> None:
        for sid, (source, created) in rows.items():
            db.store.holder()._execute_write(
                lambda conn, sid=sid, source=source, created=created: conn.execute(
                    "UPDATE sessions SET source=?, created_source=? WHERE id=?", (source, created, sid)))

    seed(store, *steps, surface)
    return [SessionId("acp0"), SessionId("acp1"), SessionId("refreshed")]


def _count_receipt_reads(monkeypatch) -> list[str]:
    read: list[str] = []
    real = _local.local_receipt

    def watching(holder, session_id):
        read.append(str(session_id))
        return real(holder, session_id)

    monkeypatch.setattr(_local, "local_receipt", watching)
    return read


def test_the_sweep_reads_only_the_acp_receipts(store, monkeypatch):
    from gateway.session_acp_lifecycle import end_idle_acp_sessions

    candidates = _acp_store(store)
    read = _count_receipt_reads(monkeypatch)
    ended = end_idle_acp_sessions(_Authority(store))

    assert set(read) == {str(sid) for sid in candidates}, read
    assert ended == 3, ended


def test_the_candidate_ids_include_a_peer_refreshed_acp_row(store):
    candidates = _acp_store(store)
    assert set(store.bindings.acp_session_ids()) == set(candidates)
    assert len(store.bindings.local_session_ids()) == 23


def test_an_acp_row_that_lost_the_label_is_not_a_candidate(store):
    """The candidate set is a lookup index: a row not marked ACP is not swept, and its receipt is never read."""
    labels = {"acp0": ("acp", "acp"), "acp1": ("local", "cli"), "refreshed": ("local", "acp"),
              **{f"cli{n}": ("local", "cli") for n in range(20)}}
    _acp_store(store, labels)
    assert set(store.bindings.acp_session_ids()) == {SessionId("acp0"), SessionId("refreshed")}
