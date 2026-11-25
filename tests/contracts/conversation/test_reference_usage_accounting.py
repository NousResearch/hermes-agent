"""Usage accounting follows the stored aggregate and attribution boundaries."""

import pytest

import logging

from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting
from tests.contracts._reference import open_reference_backend
from tests.contracts._scopes import profile_scope
from datetime import UTC


def test_failed_attribution_preserves_aggregate(monkeypatch, caplog):
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).usage.queue_usage("s", UsageDelta(tokens=TokenCounts(input_tokens=1), api_call_count=1))
        original = backend.open(scope).usage._attribute

        def refuse(*args, **kwargs):
            raise ValueError("attribution refused")

        monkeypatch.setattr(backend.open(scope).usage, "_attribute", refuse)
        # The writer logs the failed apply and drops the delta; the staged counters never land.
        with caplog.at_level(logging.WARNING):
            backend.open(scope).usage.queue_usage("s", UsageDelta(tokens=TokenCounts(input_tokens=7), api_call_count=2, route=BillingRoute("new", "provider")))
        assert any("attribution refused" in record.getMessage() for record in caplog.records)
        usage = accounting(backend.open(scope), "s")
        assert usage.tokens.input_tokens == 1
        assert usage.api_call_count == 1
        assert (backend.open(scope).sessions.get("s")).model == "new"
        monkeypatch.setattr(backend.open(scope).usage, "_attribute", original)
        backend.open(scope).usage.queue_usage("s", UsageDelta(tokens=TokenCounts(input_tokens=7), api_call_count=2))
        usage = accounting(backend.open(scope), "s")
        assert usage.tokens.input_tokens == 8
        assert usage.api_call_count == 3


def test_auxiliary_usage_backfills_session_without_main_usage():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).usage.record_auxiliary_usage("s", "vision", model="aux", billing_provider="p", tokens=TokenCounts(input_tokens=2), api_call_count=1)
        session = backend.open(scope).sessions.get("s")
        assert session is not None
        assert session.source == "unknown"
        assert session.model is None
        usage = accounting(backend.open(scope), "s")
        assert usage.tokens == TokenCounts()
        assert usage.api_call_count == 0
        assert usage.estimated_cost_usd is None
        assert (backend.open(scope).usage.auxiliary_usage_by_task("s"))["vision"].tokens.input_tokens == 2


def test_equal_time_route_updates_keep_insertion_tiebreak(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1700000000.0)
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for model in ("a", "b", "a"):
            backend.open(scope).usage.queue_usage("s", UsageDelta(api_call_count=1, route=BillingRoute(model, "p")))
        route = backend.open(scope).usage.recent_route("s")
        assert route.route.model == "b"
        assert route.api_call_count == 1


def test_totals_follow_session_parent_archive_and_message_counters():
    from datetime import datetime, timezone
    from persistence.conversation.messages import NewMessage
    from persistence.conversation.sessions import NewSession
    from persistence.conversation.usage import UsageTotalsQuery

    with open_reference_backend() as backend:
        scope = profile_scope("default")
        for sid, parent, count in [("root", None, 13), ("child", "root", 37), ("archived", None, 29)]:
            backend.open(scope).sessions.create(NewSession(session_id=sid, source="cli", parent_session_id=parent))
            backend.open(scope).usage.queue_usage(sid, UsageDelta(tokens=TokenCounts(input_tokens=count)))
            backend.open(scope).messages.append(sid, NewMessage(role="user", content="hello", timestamp=datetime(2023, 11, 14, tzinfo=UTC)))
        backend.open(scope).sessions.set_archived("archived", True)
        assert backend.open(scope).usage.flush_usage() is True
        assert (backend.open(scope).usage.usage_totals(UsageTotalsQuery(min_message_count=1))).tokens == 13
        assert (backend.open(scope).usage.usage_totals(UsageTotalsQuery(min_message_count=1, include_archived=True))).tokens == 42
        assert (backend.open(scope).usage.usage_totals(UsageTotalsQuery(min_message_count=2, include_archived=True))).tokens == 0
