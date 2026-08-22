"""Parity tests for the single run_conversation finalization path."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from agent import conversation_loop, turn_finalizer


_COUNTER_NAMES = (
    "session_input_tokens",
    "session_output_tokens",
    "session_total_tokens",
    "session_estimated_cost_usd",
)


def _agent(session_id: str = "session-test", *, reason: str = "unknown") -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        reason=reason,
        session_input_tokens=101,
        session_output_tokens=23,
        session_total_tokens=124,
        session_estimated_cost_usd=0.0123,
    )


def _totals(agent: SimpleNamespace) -> tuple[object, ...]:
    return tuple(getattr(agent, name) for name in _COUNTER_NAMES)


@pytest.mark.parametrize(
    ("nature", "reason", "result"),
    [
        (
            "normal",
            "text_response(finish_reason=stop)",
            {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True},
        ),
        (
            "partial",
            "partial_stream_recovery",
            {
                "final_response": "partial",
                "messages": [],
                "api_calls": 2,
                "completed": False,
                "partial": True,
            },
        ),
        (
            "interrupted",
            "interrupted_by_user",
            {
                "final_response": None,
                "messages": [],
                "api_calls": 1,
                "completed": False,
                "interrupted": True,
            },
        ),
        (
            "budget_exhausted",
            "budget_exhausted",
            {
                "final_response": "budget summary",
                "messages": [],
                "api_calls": 4,
                "completed": False,
                "partial": True,
            },
        ),
        (
            "error",
            "all_retries_exhausted_no_response",
            {
                "final_response": "failed",
                "messages": [],
                "api_calls": 3,
                "completed": False,
                "failed": True,
                "error": "failed",
            },
        ),
    ],
)
def test_public_turn_result_and_totals_are_unchanged_and_finalized_once(
    monkeypatch: pytest.MonkeyPatch,
    nature: str,
    reason: str,
    result: dict[str, object],
) -> None:
    agent = _agent()
    before = _totals(agent)
    calls: list[tuple[str, str, BaseException | None]] = []

    def fake_impl(_agent: SimpleNamespace, *_args, **_kwargs) -> dict[str, object]:
        turn_finalizer._record_turn_exit_reason(reason)
        return result

    def capture(_agent: SimpleNamespace, observed_reason: str, error: BaseException | None) -> None:
        calls.append((_agent.session_id, observed_reason, error))

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    monkeypatch.setattr(turn_finalizer, "_log_turn_finalization", capture)

    observed = conversation_loop.run_conversation(agent, f"case:{nature}")

    assert observed is result
    assert observed == result
    assert _totals(agent) == before
    assert calls == [(agent.session_id, reason, None)]
    assert turn_finalizer._current_turn_exit_reason() == "unknown"


def test_exception_is_finalized_once_reraised_by_identity_and_context_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent()
    before = _totals(agent)
    sentinel = RuntimeError("sentinel")
    calls: list[tuple[str, str, BaseException | None]] = []

    def fake_impl(_agent: SimpleNamespace, *_args, **_kwargs) -> dict[str, object]:
        raise sentinel

    def capture(_agent: SimpleNamespace, reason: str, error: BaseException | None) -> None:
        calls.append((_agent.session_id, reason, error))

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    monkeypatch.setattr(turn_finalizer, "_log_turn_finalization", capture)

    with pytest.raises(RuntimeError) as raised:
        conversation_loop.run_conversation(agent, "exception")

    assert raised.value is sentinel
    assert _totals(agent) == before
    assert calls == [(agent.session_id, "exception(RuntimeError)", sentinel)]
    assert turn_finalizer._current_turn_exit_reason() == "unknown"


def test_contextvar_keeps_concurrent_turn_reasons_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2)
    lock = Lock()
    calls: list[tuple[str, str]] = []

    def fake_impl(agent: SimpleNamespace, *_args, **_kwargs) -> dict[str, object]:
        turn_finalizer._record_turn_exit_reason(agent.reason)
        barrier.wait(timeout=5)
        return {"final_response": agent.reason, "completed": True}

    def capture(agent: SimpleNamespace, reason: str, _error: BaseException | None) -> None:
        with lock:
            calls.append((agent.session_id, reason))

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    monkeypatch.setattr(turn_finalizer, "_log_turn_finalization", capture)

    agents = [_agent("session-a", reason="reason-a"), _agent("session-b", reason="reason-b")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: conversation_loop.run_conversation(item, "go"), agents))

    assert results == [
        {"final_response": "reason-a", "completed": True},
        {"final_response": "reason-b", "completed": True},
    ]
    assert sorted(calls) == [("session-a", "reason-a"), ("session-b", "reason-b")]
    assert turn_finalizer._current_turn_exit_reason() == "unknown"


def test_finalization_log_redacts_dynamic_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agent import redact

    sensitive = "dynamic-sensitive-value"
    calls: list[tuple[str, bool]] = []

    def fake_redact(value: str, *, force: bool = False) -> str:
        calls.append((value, force))
        return value.replace(sensitive, "[REDACTED]")

    monkeypatch.setattr(redact, "redact_sensitive_text", fake_redact)
    with caplog.at_level("INFO", logger="agent.conversation_loop"):
        turn_finalizer._log_turn_finalization(
            _agent(), f"local_processing_error({sensitive})", None
        )

    assert sensitive not in caplog.text
    assert "local_processing_error([REDACTED])" in caplog.text
    assert all(force for _value, force in calls)


def test_public_signature_matches_private_implementation() -> None:
    assert inspect.signature(conversation_loop.run_conversation) == inspect.signature(
        conversation_loop._run_conversation_impl
    )
