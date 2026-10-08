"""Acceptance tests — own-PR #132080 hardening (relay sync-generator close).

Red-team contract frozen from the design doc 验收场景 (6 scenarios / 13
predicates). Black-box only: every test drives the public ``relay_llm.stream``
harness shape anchored at the HEAD baseline
(``test_explicit_stream_close_surfaces_provider_close_failure``, :713) and
asserts observable close-path behaviour:

* S1 guard fidelity — the provider close runs inside an active
  ``relay_runtime.managed_callback_guard`` interval, with the request-captured
  contextvars visible.
* S2 retry unit — every bounded-retry attempt re-enters the guard with the
  request context visible (``run_callback(close)``); bare close count == 0.
* S3 in-bounds path — close inside the total budget returns cleanly, runs the
  generator ``finally`` in the request context, raises nothing to the caller,
  and emits no abandon warning.
* S4 timeout leg — a provider close held busy past the total budget is
  abandoned: ``close()`` returns WITHOUT re-raising the underlying busy error
  and >= 1 WARNING containing "abandon" lands on logger ``agent.relay_llm``.
* S5 constants — ``relay_llm._GENERATOR_BUSY_RETRY_DELAY`` and
  ``relay_llm._GENERATOR_BUSY_RETRY_TIMEOUT`` are monkeypatch-effective at
  call time; patched tiny bounds trigger the abandon path within 1s.

Contract literals (design doc): the two module-level constants, the "abandon"
warning wording, logger name ``agent.relay_llm``, close-must-not-re-raise on
timeout, guard enter/exit ordering. No implementation symbols beyond these are
asserted.

Construction note (auto-fix 留痕, see state.md 变更日志): the first cut held a
sibling worker thread blocked inside ``next(stream)`` across ``close()``; the
relay loop serving that live consumer keeps running, so the pre-existing
``loop.close()`` constraint (not the busy-window contract under test) failed
the test on base and head alike. The repaired construction uses a
deterministic fake raw stream whose ``close()`` reports the busy window —
no consumer thread stays inside ``next()`` — which is the issue's actual
shape (the interrupted caller unwinds before close) and the same mechanism
the committed regression tests pin.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import time

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime  # noqa: E402
from tests.agent.test_relay_llm import relay_turn  # noqa: E402,F401

# Request-level contextvar probe: set to REQUEST_VALUE while the managed stream
# is created, then re-pointed on the caller thread. A close that runs inside the
# captured request context (run_callback semantics) reads REQUEST_VALUE in the
# provider's close; a bare close on the caller/loop context cannot.
REQUEST_PROBE = contextvars.ContextVar("relay_acceptance_request_probe")
REQUEST_VALUE = "request-ctx-132080"
CALLER_VALUE = "caller-not-request"

ABANDON_WARNING_LOG_NAME = "agent.relay_llm"
_CHUNK = {"choices": [{"index": 0, "delta": {"content": "done"}}]}


def _make_guard_recorder():
    """Wrap ``relay_runtime.managed_callback_guard`` to trace enter/exit depth."""
    original = relay_runtime.managed_callback_guard
    state = {"depth": 0}
    events = []

    @contextlib.contextmanager
    def recording_guard(*args, **kwargs):
        state["depth"] += 1
        events.append(("guard-enter", state["depth"]))
        try:
            with original(*args, **kwargs):
                yield
        finally:
            events.append(("guard-exit", state["depth"]))
            state["depth"] -= 1

    return recording_guard, events, state


class _FakeCloseStream:
    """Deterministic raw provider stream: yields one chunk then exhausts.

    ``close()`` reports the attempt through ``recorder`` first, then raises the
    busy ``ValueError`` for the first ``busy_times`` attempts before succeeding —
    the deterministic stand-in for a generator still executing on a worker
    thread, without parking a consumer inside ``next()``.
    """

    def __init__(self, recorder, busy_times):
        self._chunks = iter([_CHUNK])
        self._recorder = recorder
        self._busy_times = busy_times

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self._recorder()
        if self._busy_times > 0:
            self._busy_times -= 1
            raise ValueError("generator already executing")


def _create_stream_in_request_context(request, provider_factory, **kwargs):
    """Create the managed stream while the probe holds REQUEST_VALUE, then
    re-point the caller thread so bare-context reads differ."""
    REQUEST_PROBE.set(REQUEST_VALUE)
    request_context = contextvars.copy_context()
    REQUEST_PROBE.set(CALLER_VALUE)
    holder = {}

    def create():
        holder["stream"] = relay_llm.stream(request, provider_factory, **kwargs)

    request_context.run(create)
    return holder["stream"]


def test_managed_close_runs_provider_close_inside_guard_and_request_context(
    relay_turn, monkeypatch
):
    """S1.P1 + S1.P2: the provider close of a managed sync generator executes
    inside an active managed_callback_guard interval — each close attempt
    strictly between a guard-enter and its guard-exit — with the request-
    captured contextvars visible, although the caller thread's own context
    does NOT hold the request values (a bare close cannot fake the trace)."""
    del relay_turn
    recording_guard, rec_events, rec_state = _make_guard_recorder()
    close_calls = []

    def record_close_attempt():
        close_calls.append((rec_state["depth"], REQUEST_PROBE.get(CALLER_VALUE)))

    def provider(_request):
        return _FakeCloseStream(record_close_attempt, busy_times=1)

    # Spy before consuming: the managed machinery drives the provider close
    # from the same guarded callback path as the pulls, so the recording
    # window must cover the whole managed lifecycle.
    monkeypatch.setattr(relay_runtime, "managed_callback_guard", recording_guard)

    stream = _create_stream_in_request_context(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": ""},
        metadata={"api_mode": "custom", "api_request_id": "request-s1-guard-fidelity"},
    )

    assert next(stream) == _CHUNK  # exhaustion drives the provider close path
    stream.close()  # must not raise on the happy seam

    # S1.P1: every close attempt ran strictly inside an active guard interval.
    assert len(close_calls) >= 2, close_calls  # busy once, then success
    assert all(depth >= 1 for depth, _probe in close_calls), close_calls
    guard_enters = [event for event in rec_events if event[0] == "guard-enter"]
    guard_exits = [event for event in rec_events if event[0] == "guard-exit"]
    assert len(guard_enters) == len(guard_exits)
    assert rec_state["depth"] == 0
    # S1.P2: request-level contextvars read inside close equal the request values.
    assert all(probe == REQUEST_VALUE for _depth, probe in close_calls), close_calls


def test_retry_unit_is_run_callback_close_not_bare_close(relay_turn, monkeypatch):
    """S2.P1: every bounded-retry attempt is the wrapped ``run_callback(close)``
    unit — each attempt is covered by its own guard entry with the request
    context visible — and no bare close ever happens (bare close attempts == 0:
    every recorded attempt is in-guard and in-request-context)."""
    del relay_turn
    recording_guard, rec_events, rec_state = _make_guard_recorder()
    close_calls = []

    def record_close_attempt():
        close_calls.append((rec_state["depth"], REQUEST_PROBE.get(CALLER_VALUE)))

    def provider(_request):
        return _FakeCloseStream(record_close_attempt, busy_times=3)

    # Small poll interval so the busy window spans several quick retries; the
    # total budget stays at its 2.0 default (in-bounds close).
    monkeypatch.setattr(relay_llm, "_GENERATOR_BUSY_RETRY_DELAY", 0.005)
    monkeypatch.setattr(relay_runtime, "managed_callback_guard", recording_guard)

    stream = _create_stream_in_request_context(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": ""},
        metadata={"api_mode": "custom", "api_request_id": "request-s2-retry-unit"},
    )

    assert next(stream) == _CHUNK
    stream.close()

    # attempts == 4: three busy polls + one success. The retry unit is the
    # whole run_callback(close): guard re-entered per attempt, never bare.
    assert len(close_calls) == 4, close_calls
    guard_enters = [event for event in rec_events if event[0] == "guard-enter"]
    guard_exits = [event for event in rec_events if event[0] == "guard-exit"]
    assert len(guard_enters) == len(guard_exits)  # enter/exit balance
    assert len(guard_enters) >= len(close_calls)  # >= one guard entry per attempt
    assert all(
        depth >= 1 and probe == REQUEST_VALUE for depth, probe in close_calls
    ), close_calls  # a bare close would read depth 0 / CALLER_VALUE
    assert rec_state["depth"] == 0


def test_in_bounds_close_returns_cleanly_without_abandon_warning(
    relay_turn, caplog
):
    """S3.P1: a close that completes within the total budget returns to the
    caller without raising, runs the generator ``finally`` in the request
    context, and emits no abandon warning."""
    del relay_turn
    entered_probe = []

    def provider(_request):
        try:
            yield _CHUNK
        finally:
            entered_probe.append(REQUEST_PROBE.get())

    stream = _create_stream_in_request_context(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": ""},
        metadata={"api_mode": "custom", "api_request_id": "request-s3-in-bounds"},
    )

    assert next(stream) == _CHUNK

    with caplog.at_level(logging.WARNING):
        stream.close()  # close_returned == true: a raise here fails the test

    assert entered_probe == [REQUEST_VALUE]  # finally ran in request context
    assert not any(
        record.levelname == "WARNING" and "abandon" in record.getMessage()
        for record in caplog.records
    )


def test_busy_close_past_timeout_abandons_with_warning_without_reraise(
    relay_turn, monkeypatch, caplog
):
    """S4.P1/P2/P3: a provider close held busy past the patched total budget is
    abandoned with a warning — the caller-facing path returns the pulled chunk
    and ``close()`` returns without re-raising the busy error, all within the
    wall-clock bound."""
    del relay_turn
    close_calls = []

    def record_close_attempt():
        close_calls.append(1)

    def provider(_request):
        return _FakeCloseStream(record_close_attempt, busy_times=10**9)  # permanently busy

    monkeypatch.setattr(relay_llm, "_GENERATOR_BUSY_RETRY_TIMEOUT", 0.1)
    monkeypatch.setattr(relay_llm, "_GENERATOR_BUSY_RETRY_DELAY", 0.01)

    stream = _create_stream_in_request_context(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": ""},
        metadata={"api_mode": "custom", "api_request_id": "request-s4-timeout-abandon"},
    )

    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        # The exhaustion-driven close polls past the 0.1s bound, is abandoned
        # with a warning, and must NOT surface the busy ValueError to the
        # consumer (issue #132048 symptom) ...
        assert next(stream) == _CHUNK  # S4.P1 leg 1: the managed pull survives
        stream.close()  # S4.P1 leg 2: must not re-raise
    elapsed = time.monotonic() - start

    abandon_warnings = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "abandon" in record.getMessage()
    ]
    assert len(close_calls) >= 2  # bounded polls really happened
    assert len(abandon_warnings) >= 1  # S4.P2: warning_records >= 1
    assert any(
        record.name == ABANDON_WARNING_LOG_NAME for record in abandon_warnings
    )
    assert elapsed < 5.0  # S4.P3


def test_busy_timeout_constants_are_monkeypatchable_and_effective(
    relay_turn, monkeypatch, caplog
):
    """S5.P1: with both constants patched to tiny values the abandon path fires
    within the patched budget (elapsed < 1.0s) and emits the warning — the
    constants are effective at call time, not baked in elsewhere."""
    del relay_turn
    close_calls = []

    def record_close_attempt():
        close_calls.append(1)

    def provider(_request):
        return _FakeCloseStream(record_close_attempt, busy_times=10**9)  # permanently busy

    monkeypatch.setattr(relay_llm, "_GENERATOR_BUSY_RETRY_TIMEOUT", 0.08)
    monkeypatch.setattr(relay_llm, "_GENERATOR_BUSY_RETRY_DELAY", 0.005)

    stream = _create_stream_in_request_context(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        finalizer=lambda: {"content": ""},
        metadata={"api_mode": "custom", "api_request_id": "request-s5-constants"},
    )

    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        assert next(stream) == _CHUNK  # abandon fires inside the pull cycle
        stream.close()  # must not raise
    elapsed = time.monotonic() - start

    abandon_warnings = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "abandon" in record.getMessage()
    ]
    assert len(close_calls) >= 2  # patched delay yielded multiple polls
    assert elapsed < 1.0  # S5.P1: patched bounds bound the wall clock
    assert len(abandon_warnings) >= 1  # warning_emitted == true


_ACCEPTANCE_CASE_IDS = (
    "test_managed_close_runs_provider_close_inside_guard_and_request_context",
    "test_retry_unit_is_run_callback_close_not_bare_close",
    "test_in_bounds_close_returns_cleanly_without_abandon_warning",
    "test_busy_close_past_timeout_abandons_with_warning_without_reraise",
    "test_busy_timeout_constants_are_monkeypatchable_and_effective",
)


def test_acceptance_case_ids_are_collectable():
    """S6.P2: the guard-fidelity and timeout-abandon acceptance cases are all
    present under their canonical ids in this module's collected namespace."""
    module = sys.modules[__name__]
    for case_id in _ACCEPTANCE_CASE_IDS:
        case = getattr(module, case_id, None)
        assert callable(case), f"acceptance case missing from collection: {case_id}"
        assert case_id.startswith("test_")

# S6.P1 (whole-file regression run, no new failures vs the baseline
# environmental failure ``test_stream_uses_rewritten_request_and_post_intercept_chunks``)
# is a real-process predicate: it is verified by running the canonical command
# from context.md over tests/agent/test_relay_llm.py plus this file —
# HERMES_PYTHON=<venv python> bash scripts/run_tests.sh <both files> -q —
# not by an in-file assertion.
