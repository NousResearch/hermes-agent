"""A nearly full request with no preflight compression path left is logged before it is sent.

Compression is the only thing that shrinks a turn before the provider sees it. When it cannot
run (disabled, blocked for insufficient progress, deferred to real usage, in failure cooldown,
out of attempts), a request at 90%+ of the context window goes out as is and usually fails with a
context-length error. The gate logs why, so the failure that follows is explainable from the log.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from agent import turn_preflight_gate as gate

LOGGER = "agent.conversation_loop"


class _Compressor:
    def __init__(self, *, context_length=100_000, cooldown=None, defer=False):
        self.context_length = context_length
        self.threshold_tokens = int(context_length * 0.8)
        self._cooldown = cooldown
        self._defer = defer

    def get_active_compression_failure_cooldown(self):
        return self._cooldown

    def should_defer_preflight_to_real_usage(self, _tokens):
        return self._defer


def _warn(caplog, *, pressure, compressor=None, enabled=True, blocked=False,
          attempts=0, max_attempts=3, defer=False):
    compressor = compressor or _Compressor()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        gate._warn_if_context_exhaustion_unrecoverable(
            SimpleNamespace(compression_enabled=enabled),
            SimpleNamespace(_preflight_compression_blocked=blocked),
            compressor=compressor, request_pressure_tokens=pressure,
            compression_attempts=attempts, max_compression_attempts=max_attempts,
            defer_preflight=lambda _t: defer,
        )
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"enabled": False}, "compression disabled"),
        ({"blocked": True}, "insufficient progress"),
        ({"defer": True}, "deferred to provider usage"),
        ({"compressor": _Compressor(cooldown={"until": 1})}, "failure cooldown"),
        ({"attempts": 3, "max_attempts": 3}, "no compression attempts left"),
    ],
)
def test_warns_with_the_reason_compression_cannot_run(caplog, kwargs, reason):
    messages = _warn(caplog, pressure=95_000, **kwargs)

    assert len(messages) == 1
    assert "95%" in messages[0]
    assert reason in messages[0]


def test_silent_when_compression_can_still_run(caplog):
    assert _warn(caplog, pressure=95_000) == []


def test_silent_below_ninety_percent_of_the_window(caplog):
    assert _warn(caplog, pressure=89_000, enabled=False) == []


@pytest.mark.parametrize("pressure", [None, "n/a"])
def test_silent_on_unusable_pressure(caplog, pressure):
    assert _warn(caplog, pressure=pressure, enabled=False) == []


def test_silent_without_a_known_window(caplog):
    assert _warn(caplog, pressure=95_000, enabled=False, compressor=_Compressor(context_length=0)) == []


def test_run_preflight_gate_logs_before_handing_off(caplog, monkeypatch):
    import agent.conversation_loop as loop

    monkeypatch.setattr(loop, "_ollama_context_limit_error", lambda _agent, _tokens: None)
    handed_off = []

    def _fake_compression(agent, v, **kwargs):
        handed_off.append(kwargs["defer_preflight"](kwargs["request_pressure_tokens"]))
        return v

    monkeypatch.setattr(gate, "run_preflight_compression", _fake_compression)
    agent = SimpleNamespace(context_compressor=_Compressor(), compression_enabled=False,
                            _request_pressure_anchored=True)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        gate.run_preflight_gate(
            agent, request_pressure_tokens=96_000, _moa_prepared_request=None,
            pending_moa_prepared_request=None, messages=[], system_message=None, user_message="hi",
            active_system_prompt=None, conversation_history=[], api_call_count=1,
            compression_attempts=0, max_compression_attempts=3, effective_task_id="t",
            final_response=None, failed=False, _turn_exit_reason=None,
            _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
            _provider_overflow_recovery_pending=False, _last_preflight_pressure=None,
        )

    warnings = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any("compression disabled" in m for m in warnings)
    # An anchored figure is never deferred; the gate hands the same deferral to compression.
    assert handed_off == [False]
