"""Tier-2 LIVE-integration tests for the Telegram boot-redelivery guard (scope B).

These exercise the REAL SessionStore + SessionDB (a temp state.db), not mocks —
the anti-proxy gates the review demanded (AC-8 companion rows, AC-11 side-effect,
the B-1 update_id/message_id conflation mutation). An all-mock suite cannot catch
B-1: it would pass whether or not the two id spaces line up. These do.

SPEC: ~/.hermes/plans/2026-07-01_telegram-redelivery-guard-SPEC.md
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from gateway.telegram_redelivery import (
    RedeliverySuppressionCounter,
    decide_redelivery,
    in_redelivery_scope,
)


def _make_store(tmp_path, with_db=True):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state import SessionDB

    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    if with_db:
        store._db = SessionDB(db_path=tmp_path / "state.db")
    else:
        store._db = None
    store._loaded = True
    return store


def _answerable_via_store(store):
    return lambda sid, mid: store.has_platform_message_id_answerable(sid, str(mid))


# ── AC-11 (live anti-proxy): the two id spaces line up end-to-end ───────────


def test_live_suppress_when_message_id_present(tmp_path):
    """A re-delivered message whose message_id is really in the transcript, and
    whose update_id is within scope, SUPPRESSES — proving update_id-scope and
    message_id-answerability line up against the real store."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    # The agent answered this message: it's in the transcript by message_id 500.
    store._db.append_message(
        session_id="s1", role="user", content="what time is it",
        platform_message_id="500",
    )
    # Re-delivery: update_id=1000 (<= HWM 1000 → in scope), message_id=500.
    in_scope = in_redelivery_scope(1000, 1000, seconds_since_boot=5)
    suppress = decide_redelivery(
        in_scope=in_scope, session_id="s1", message_id="500", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress is True


def test_live_b1_mutation_feeding_update_id_misses(tmp_path):
    """THE B-1 MUTATION: if the guard fed the UPDATE_ID (1000) to the
    answerability lookup instead of the MESSAGE_ID (500), the lookup would miss
    (the transcript holds message_id 500, not 1000) → PROCESS → the guard is a
    silent no-op. This test proves feeding the wrong id fails to suppress, i.e.
    the id-space split is load-bearing and really tested."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    store._db.append_message(
        session_id="s1", role="user", content="q", platform_message_id="500",
    )
    in_scope = in_redelivery_scope(1000, 1000, seconds_since_boot=5)
    # MUTANT: pass update_id (1000) where message_id (500) belongs.
    suppress_wrong = decide_redelivery(
        in_scope=in_scope, session_id="s1", message_id="1000", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress_wrong is False, (
        "feeding update_id to the answerability lookup MUST miss (transcript "
        "holds message_id 500, not update_id 1000) — this is the B-1 no-op"
    )


def test_live_absent_message_processes(tmp_path):
    """A genuinely-new message (its message_id not in the transcript) PROCESSES
    even when in scope — real DB miss is answered=True, present=False → fail-open
    to PROCESS."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    in_scope = in_redelivery_scope(1000, 1000, seconds_since_boot=5)
    suppress = decide_redelivery(
        in_scope=in_scope, session_id="s1", message_id="777", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress is False


def test_live_no_db_fails_open(tmp_path):
    """No session DB → unanswerable → PROCESS (never suppress on uncertainty)."""
    store = _make_store(tmp_path, with_db=False)
    suppress = decide_redelivery(
        in_scope=True, session_id="s1", message_id="500", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress is False


# ── AC-11 side-effect: the observability path is exercised live (RC-C) ──────


def test_live_suppression_counter_and_log(caplog):
    counter = RedeliverySuppressionCounter()
    with caplog.at_level(logging.WARNING, logger="gateway.telegram_redelivery"):
        n1 = counter.record(update_id=1000, message_id=500)
        n2 = counter.record(update_id=1001, message_id=501)
    assert (n1, n2, counter.count) == (1, 2, 2)
    lines = [r.getMessage() for r in caplog.records if "PHASE=tg_redelivery_suppressed" in r.getMessage()]
    assert len(lines) == 2
    assert "update_id=1000" in lines[0] and "message_id=500" in lines[0]
    # content-free: no message body ever logged
    assert "what time is it" not in " ".join(lines)


# ── AC-8 (LIVE TIER): aggregate companion rows make each constituent answerable ─


class _AggEvent:
    """Minimal event carrying the buffer's accumulated constituent ids."""
    def __init__(self, message_id, constituent_ids):
        self.message_id = message_id
        self._constituent_message_ids = constituent_ids


class _Entry:
    def __init__(self, session_id):
        self.session_id = session_id


def _runner_with_store(store):
    """A bare object exposing just what the two guard methods touch, with the
    real _persist_telegram_aggregate_constituents bound to it."""
    from gateway.run import GatewayRunner

    class _Bare:
        pass
    bare = _Bare()
    bare.session_store = store
    # Bind the real unbound methods.
    bare._persist_telegram_aggregate_constituents = (
        GatewayRunner._persist_telegram_aggregate_constituents.__get__(bare, _Bare)
    )
    return bare


def test_live_aggregate_companion_rows_make_middle_constituent_answerable(tmp_path):
    """AC-8: persist a real aggregate of updates [N, N+1, N+2]; re-delivering the
    MIDDLE constituent N+1 must be answerable (present) → SUPPRESS. The turn row
    stamps only N; the companion observed rows carry N+1, N+2."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    # The conversational turn row stamps the FIRST constituent (300).
    store._db.append_message(
        session_id="s1", role="user", content="hello world part one",
        platform_message_id="300",
    )
    bare = _runner_with_store(store)
    event = _AggEvent(message_id="300", constituent_ids=["300", "301", "302"])
    bare._persist_telegram_aggregate_constituents(event, _Entry("s1"))

    # Now every constituent — including the MIDDLE one — is answerable.
    for mid in ("300", "301", "302"):
        answered, present = store.has_platform_message_id_answerable("s1", mid)
        assert (answered, present) == (True, True), f"constituent {mid} not answerable"
    # And the guard suppresses a re-delivery of the middle constituent.
    suppress = decide_redelivery(
        in_scope=in_redelivery_scope(9999, 9999, seconds_since_boot=5),
        session_id="s1", message_id="301", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress is True


def test_live_aggregate_mutation_stamp_only_first_leaves_middle_unanswerable(tmp_path):
    """AC-12(iv) mutation: if the persist path stamped ONLY the first constituent
    (drop companion rows), the middle re-delivery would MISS → the aggregate fix
    would not gate. Prove that: with only the turn row, N+1 is absent → PROCESS."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    store._db.append_message(
        session_id="s1", role="user", content="hello", platform_message_id="300",
    )
    # NO companion rows written (the mutation).
    answered, present = store.has_platform_message_id_answerable("s1", "301")
    assert (answered, present) == (True, False), "middle constituent must be absent without companion rows"
    suppress = decide_redelivery(
        in_scope=in_redelivery_scope(9999, 9999, seconds_since_boot=5),
        session_id="s1", message_id="301", is_edited=False,
        answerable_fn=_answerable_via_store(store),
    )
    assert suppress is False, "without companion rows the middle constituent is re-answered (the bug)"


def test_live_aggregate_single_message_writes_no_companions(tmp_path):
    """A non-aggregated turn (one constituent) writes no companion rows."""
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="telegram")
    store._db.append_message(
        session_id="s1", role="user", content="solo", platform_message_id="300",
    )
    bare = _runner_with_store(store)
    event = _AggEvent(message_id="300", constituent_ids=["300"])
    bare._persist_telegram_aggregate_constituents(event, _Entry("s1"))
    # 301 was never a constituent → still absent.
    answered, present = store.has_platform_message_id_answerable("s1", "301")
    assert (answered, present) == (True, False)
