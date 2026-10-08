"""Backend-neutral token-usage transport contract."""

from __future__ import annotations

import threading

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from token_usage_transport import TokenUsageTransport


_ROUTE_FIELDS = ("model", "billing_provider")
_SUM_FIELDS = ("input_tokens", "api_call_count")
_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")


def _transport(persist, *, idle_seconds=lambda: 30.0, coalesce=None):
    return TokenUsageTransport(
        persist,
        sum_fields=_SUM_FIELDS,
        cost_fields=_COST_FIELDS,
        route_fields=_ROUTE_FIELDS,
        idle_seconds=idle_seconds,
        coalesce=coalesce,
    )


def test_transport_coalesces_only_adjacent_equal_incremental_routes():
    applied = []
    transport = _transport(lambda session_id, **kwargs: applied.append((session_id, kwargs)))
    try:
        transport.apply_batch([
            ("a", {"input_tokens": 2, "api_call_count": 1, "model": "m1", "billing_provider": "p"}),
            ("a", {"input_tokens": 3, "api_call_count": 1, "model": "m1", "billing_provider": "p"}),
            ("a", {"input_tokens": 50, "api_call_count": 7, "absolute": True}),
            ("a", {"input_tokens": 4, "api_call_count": 1, "model": "m1", "billing_provider": "p"}),
        ])
    finally:
        transport.close()

    assert [(session_id, values["input_tokens"]) for session_id, values in applied] == [
        ("a", 5), ("a", 50), ("a", 4),
    ]


def test_transport_dead_writer_flush_claims_inflight_batch_before_apply():
    applied = threading.Event()
    release = threading.Event()
    calls = []

    def persist(session_id, **kwargs):
        calls.append((session_id, kwargs))
        applied.set()
        assert release.wait(timeout=5)

    transport = _transport(persist)
    try:
        transport.queue.append(("s", {"input_tokens": 1, "api_call_count": 1}))
        first = threading.Thread(target=lambda: transport.flush())
        first.start()
        assert applied.wait(timeout=5)
        assert transport.flush(timeout=0.05) is False
        release.set()
        first.join(timeout=5)
        assert transport.flush()
    finally:
        transport.close()

    assert calls == [("s", {"input_tokens": 1, "api_call_count": 1})]


def test_transport_logs_and_discards_only_failed_delta(caplog):
    calls = []

    def persist(session_id, **kwargs):
        calls.append(session_id)
        if session_id == "bad":
            raise RuntimeError("storage unavailable")

    transport = _transport(persist)
    try:
        with caplog.at_level("WARNING", logger="hermes_state"):
            transport.apply_batch([
                ("bad", {"input_tokens": 1, "model": "m"}),
                ("good", {"input_tokens": 2, "model": "other"}),
            ])
    finally:
        transport.close()

    assert calls == ["bad", "good"]
    failures = [record for record in caplog.records if "apply failed (session=bad)" in record.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], RuntimeError)


def test_transport_coalesce_failure_applies_raw_ordered_deltas_with_traceback(caplog):
    applied = []
    batch = [
        ("first", {"input_tokens": 2, "api_call_count": 1, "model": "m", "billing_provider": "p", "source": "turn"}),
        ("first", {"input_tokens": 3, "api_call_count": 1, "model": "m", "billing_provider": "p", "source": "replay"}),
        ("last", {"input_tokens": 5, "api_call_count": 1, "model": "other", "billing_provider": "q", "source": "turn"}),
    ]

    def broken_coalesce(deltas):
        assert deltas == batch
        raise RuntimeError("coalescer unavailable")

    transport = _transport(
        lambda session_id, **kwargs: applied.append((session_id, kwargs)),
        coalesce=broken_coalesce,
    )
    try:
        with caplog.at_level("WARNING", logger="hermes_state"):
            transport.apply_batch(batch)
    finally:
        transport.close()

    assert applied == batch
    failures = [record for record in caplog.records if "coalesce failed, applying raw batch" in record.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], RuntimeError)


def test_transport_post_close_persists_synchronously():
    calls = []
    transport = _transport(lambda session_id, **kwargs: calls.append((session_id, kwargs)))
    transport.close()
    transport.queue_delta("late", {"input_tokens": 2, "api_call_count": 1})
    assert calls == [("late", {"input_tokens": 2, "api_call_count": 1})]


def test_transport_writer_inherits_queued_profile_home(tmp_path):
    profile_home = tmp_path / "profiles" / "writer"
    profile_home.mkdir(parents=True)
    applied = []

    def persist(session_id, **kwargs):
        applied.append((session_id, kwargs, get_hermes_home(), threading.get_ident()))

    transport = _transport(persist)
    caller_thread = threading.get_ident()
    token = set_hermes_home_override(profile_home)
    try:
        transport.queue_delta("profile-session", {"input_tokens": 1})
        assert transport.flush()
    finally:
        transport.close()
        reset_hermes_home_override(token)

    assert [call[:3] for call in applied] == [
        ("profile-session", {"input_tokens": 1}, profile_home),
    ]
    assert applied[0][3] != caller_thread
