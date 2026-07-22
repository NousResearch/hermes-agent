"""Regression tests for opt-in compression wall-clock deadlines."""

import contextvars
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import (
    AuxiliaryWallClockTimeout,
    _call_fallback_candidate_sync,
    _aux_interrupt_protected,
    _get_task_wall_clock_timeout,
    call_llm,
)
from agent.context_compressor import (
    ContextCompressor,
    SummaryWallClockTimeout,
    _call_llm_with_wall_clock,
)


def _ok_response(content: str = "summary"):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="main/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def test_wall_clock_timeout_disabled_values_preserve_existing_behavior():
    for raw in (None, 0, "0", ""):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"wall_clock_timeout": raw},
        ):
            assert _get_task_wall_clock_timeout("compression") is None


def test_default_config_disables_compression_wall_clock_timeout():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["auxiliary"]["compression"]["wall_clock_timeout"] == 0


def test_wall_clock_timeout_above_effective_floor_is_preserved():
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"timeout": 120, "wall_clock_timeout": 600},
    ):
        assert _get_task_wall_clock_timeout("compression") == 600.0


def test_wall_clock_timeout_clamps_to_effective_compression_floor(caplog):
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"timeout": 120, "wall_clock_timeout": 30},
    ):
        assert _get_task_wall_clock_timeout("compression") == 300.0
    assert "clamped" in caplog.text.lower()


def test_disabled_deadline_uses_existing_inline_call(compressor):
    with (
        patch("agent.context_compressor._get_task_wall_clock_timeout", return_value=None),
        patch("agent.context_compressor.call_llm", return_value=_ok_response("inline")) as call,
        patch("agent.context_compressor._call_llm_with_wall_clock") as guarded,
    ):
        result = compressor._generate_summary([{"role": "user", "content": "hello"}])

    assert "inline" in result
    call.assert_called_once()
    guarded.assert_not_called()


def test_worker_propagates_context_and_interrupt_protection():
    marker = contextvars.ContextVar("compression_deadline_test", default="missing")
    marker.set("parent")
    observed = {}

    def fake_call_llm(**_kwargs):
        observed["context"] = marker.get()
        observed["protected"] = _aux_interrupt_protected()
        observed["deadline"] = _kwargs["wall_clock_deadline"]
        return _ok_response()

    deadline = time.monotonic() + 1.0
    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        response = _call_llm_with_wall_clock({}, deadline)

    assert response.choices[0].message.content == "summary"
    assert observed == {
        "context": "parent",
        "protected": True,
        "deadline": deadline,
    }


def test_worker_maps_auxiliary_deadline_timeout_before_summary_classification():
    with patch(
        "agent.context_compressor.call_llm",
        side_effect=AuxiliaryWallClockTimeout("deadline expired"),
    ):
        with pytest.raises(SummaryWallClockTimeout, match="deadline expired"):
            _call_llm_with_wall_clock({}, time.monotonic() + 1.0)


def test_worker_timeout_is_bounded_and_late_result_is_discarded():
    release = threading.Event()
    finished = threading.Event()

    def slow_call_llm(**_kwargs):
        release.wait(timeout=1.0)
        finished.set()
        return _ok_response("late")

    started = time.monotonic()
    with patch("agent.context_compressor.call_llm", side_effect=slow_call_llm):
        with pytest.raises(SummaryWallClockTimeout, match="timeout"):
            _call_llm_with_wall_clock({}, started + 0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    release.set()
    assert finished.wait(timeout=1.0)


def test_wall_clock_timeout_is_explicitly_classified_and_cooled_down(compressor):
    with (
        patch("agent.context_compressor._get_task_wall_clock_timeout", return_value=1.0),
        patch(
            "agent.context_compressor._call_llm_with_wall_clock",
            side_effect=SummaryWallClockTimeout("compression wall-clock timeout exceeded"),
        ) as guarded,
    ):
        assert compressor._generate_summary([{"role": "user", "content": "hello"}]) is None
        assert compressor._last_summary_network_failure is True
        assert compressor._consecutive_timeout_failures == 1
        assert compressor._summary_failure_cooldown_until > time.monotonic()
        # The cooldown must prevent an immediate second worker.
        assert compressor._generate_summary([{"role": "user", "content": "again"}]) is None

    guarded.assert_called_once()


def test_one_deadline_spans_summary_model_and_main_model_fallback(compressor):
    compressor.summary_model = "aux/model"
    deadlines = []

    class UnavailableError(Exception):
        status_code = 503

    def guarded_call(_call_kwargs, deadline):
        deadlines.append(deadline)
        if len(deadlines) == 1:
            raise UnavailableError("auxiliary model unavailable")
        raise SummaryWallClockTimeout("compression wall-clock timeout exceeded")

    with (
        patch("agent.context_compressor._get_task_wall_clock_timeout", return_value=600.0),
        patch(
            "agent.context_compressor._call_llm_with_wall_clock",
            side_effect=guarded_call,
        ),
    ):
        result = compressor._generate_summary([{"role": "user", "content": "hello"}])

    assert result is None
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert compressor._last_summary_network_failure is True


def test_wall_clock_timeout_preserves_messages_under_default_abort_policy(compressor):
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message {index}"}
        for index in range(10)
    ]

    with (
        patch("agent.context_compressor._get_task_wall_clock_timeout", return_value=1.0),
        patch(
            "agent.context_compressor._call_llm_with_wall_clock",
            side_effect=SummaryWallClockTimeout("compression wall-clock timeout exceeded"),
        ),
    ):
        result = compressor.compress(messages)

    assert result == messages
    assert compressor._last_compress_aborted is True
    assert compressor._last_summary_fallback_used is False
    assert compressor._last_summary_dropped_count == 0


def test_deadline_exhausted_on_recursive_entry_is_caught(compressor):
    compressor.summary_model = "aux/model"
    exhausted = time.monotonic() - 1.0

    with patch(
        "agent.context_compressor._get_task_wall_clock_timeout",
        return_value=MagicMock(),
    ):
        result = compressor._generate_summary(
            [{"role": "user", "content": "hello"}],
            _wall_clock_deadline=exhausted,
        )

    assert result is None
    assert compressor._last_summary_network_failure is True


def _routing_patches(primary_client, *, provider="auto"):
    return (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(provider, "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary_client, "primary-model"),
        ),
        patch(
            "agent.auxiliary_client._validate_llm_response",
            side_effect=lambda response, _task, **_kwargs: response,
        ),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={}),
    )


def _controlled_clock(*values):
    times = iter(values)
    last = values[-1]
    return lambda: next(times, last)


def test_real_call_llm_stops_transient_retry_after_deadline():
    primary = MagicMock()
    primary.base_url = "https://primary.example/v1"
    primary.chat.completions.create.side_effect = ConnectionError(
        "connection reset by peer"
    )
    p1, p2, p3, p4 = _routing_patches(primary)

    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "agent.auxiliary_client._is_transient_transport_error",
            side_effect=lambda exc: isinstance(exc, ConnectionError),
        ),
        patch("agent.auxiliary_client._is_timeout_error", return_value=False),
        patch("agent.auxiliary_client._transient_retry_count", return_value=1),
        patch("agent.auxiliary_client.time.sleep"),
        patch(
            "agent.auxiliary_client.time.monotonic",
            side_effect=_controlled_clock(10.0, 12.0),
        ),
    ):
        with pytest.raises(AuxiliaryWallClockTimeout):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                timeout=30.0,
                wall_clock_deadline=11.0,
            )

    assert primary.chat.completions.create.call_count == 1
    assert primary.chat.completions.create.call_args.kwargs["timeout"] == pytest.approx(
        1.0
    )


def test_real_call_llm_stops_fallback_request_after_deadline():
    primary = MagicMock()
    primary.base_url = "https://primary.example/v1"
    primary.chat.completions.create.side_effect = ConnectionError("connection lost")
    fallback = MagicMock()
    fallback.base_url = "https://fallback.example/v1"
    p1, p2, p3, p4 = _routing_patches(primary)

    with (
        p1,
        p2,
        p3,
        p4,
        patch(
            "agent.auxiliary_client._is_transient_transport_error",
            side_effect=lambda exc: isinstance(exc, ConnectionError),
        ),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch(
            "agent.auxiliary_client._is_connection_error",
            side_effect=lambda exc: isinstance(exc, ConnectionError),
        ),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(fallback, "fallback-model", "fallback"),
        ),
        patch(
            "agent.auxiliary_client.time.monotonic",
            side_effect=_controlled_clock(10.0, 12.0),
        ),
    ):
        with pytest.raises(AuxiliaryWallClockTimeout):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                timeout=30.0,
                wall_clock_deadline=11.0,
            )

    assert primary.chat.completions.create.call_count == 1
    fallback.chat.completions.create.assert_not_called()


def test_real_call_llm_stops_pool_retry_after_deadline():
    class RateLimitError(Exception):
        pass

    primary = MagicMock()
    primary.base_url = "https://primary.example/v1"
    primary.chat.completions.create.side_effect = RateLimitError("rate limited")
    p1, p2, p3, p4 = _routing_patches(primary, provider="openai-codex")

    with (
        p1,
        p2,
        p3,
        p4,
        patch("agent.auxiliary_client._is_transient_transport_error", return_value=False),
        patch(
            "agent.auxiliary_client._is_rate_limit_error",
            side_effect=lambda exc: isinstance(exc, RateLimitError),
        ),
        patch("agent.auxiliary_client._is_payment_error", return_value=False),
        patch("agent.auxiliary_client._is_auth_error", return_value=False),
        patch(
            "agent.auxiliary_client._recoverable_pool_provider",
            return_value="openai-codex",
        ),
        patch(
            "agent.auxiliary_client.time.monotonic",
            side_effect=_controlled_clock(10.0, 12.0),
        ),
    ):
        with pytest.raises(AuxiliaryWallClockTimeout):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                timeout=30.0,
                wall_clock_deadline=11.0,
            )

    assert primary.chat.completions.create.call_count == 1


def test_fallback_auth_refresh_cannot_start_after_deadline():
    class AuthError(Exception):
        pass

    fallback = MagicMock()
    fallback.base_url = "https://fallback.example/v1"
    fallback.chat.completions.create.side_effect = AuthError("expired token")
    refreshed = MagicMock()
    refreshed.base_url = "https://refreshed.example/v1"

    with (
        patch("agent.auxiliary_client._fallback_entry_timeout", return_value=None),
        patch(
            "agent.auxiliary_client._validate_llm_response",
            side_effect=lambda response, _task, **_kwargs: response,
        ),
        patch(
            "agent.auxiliary_client._is_auth_error",
            side_effect=lambda exc: isinstance(exc, AuthError),
        ),
        patch(
            "agent.auxiliary_client._auth_refresh_provider_for_route",
            return_value="custom",
        ),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(refreshed, "refreshed-model"),
        ),
        patch(
            "agent.auxiliary_client.time.monotonic",
            side_effect=_controlled_clock(10.0, 12.0),
        ),
    ):
        with pytest.raises(AuxiliaryWallClockTimeout):
            _call_fallback_candidate_sync(
                fallback,
                "fallback-model",
                "fallback_chain[0](custom)",
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                temperature=0.3,
                max_tokens=500,
                tools=None,
                effective_timeout=30.0,
                effective_extra_body={},
                reasoning_config=None,
                wall_clock_deadline=11.0,
            )

    assert fallback.chat.completions.create.call_count == 1
    refreshed.chat.completions.create.assert_not_called()
