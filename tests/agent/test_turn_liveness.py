"""Unit coverage for agent/turn_liveness.py config resolution (#95548/#95663).

AGENTS.md rejects new non-secret ``HERMES_*`` env knobs: the watchdog's
behavioral settings live in ``agent.turn_liveness`` in config.yaml, and the
resolver must validate them — a typo must never crash durable-turn startup,
and NaN/Inf must never silently disable the timeout or freeze the watcher
thread.
"""

from __future__ import annotations

import logging

from agent.turn_liveness import (
    DEFAULT_TURN_LIVENESS_POLL_S,
    DEFAULT_TURN_LIVENESS_TIMEOUT_S,
    resolve_turn_liveness_settings,
)


def test_defaults_when_config_missing_or_unset():
    # No config at all.
    assert resolve_turn_liveness_settings(None) == (
        DEFAULT_TURN_LIVENESS_TIMEOUT_S,
        DEFAULT_TURN_LIVENESS_POLL_S,
    )
    assert resolve_turn_liveness_settings({}) == (
        DEFAULT_TURN_LIVENESS_TIMEOUT_S,
        DEFAULT_TURN_LIVENESS_POLL_S,
    )
    # Section present but keys absent.
    assert resolve_turn_liveness_settings(
        {"agent": {"turn_liveness": {}}}
    ) == (DEFAULT_TURN_LIVENESS_TIMEOUT_S, DEFAULT_TURN_LIVENESS_POLL_S)


def test_precedence_explicit_values_win_over_defaults():
    timeout, poll = resolve_turn_liveness_settings(
        {"agent": {"turn_liveness": {"timeout_s": 30, "poll_s": 5}}}
    )
    assert timeout == 30.0
    assert poll == 5.0


def test_precedence_numeric_strings_accepted():
    # YAML may hand back strings; numeric coercion is part of the contract.
    timeout, poll = resolve_turn_liveness_settings(
        {"agent": {"turn_liveness": {"timeout_s": "45", "poll_s": "2.5"}}}
    )
    assert timeout == 45.0
    assert poll == 2.5


def test_timeout_zero_is_documented_opt_out():
    timeout, poll = resolve_turn_liveness_settings(
        {"agent": {"turn_liveness": {"timeout_s": 0, "poll_s": 15}}}
    )
    assert timeout is None
    assert poll == 15.0
    # Negative is the same documented "disabled" path.
    assert resolve_turn_liveness_settings(
        {"agent": {"turn_liveness": {"timeout_s": -1}}}
    )[0] is None


def test_typo_does_not_crash_and_falls_back_to_default(caplog):
    # The old raw float() env parsing raised ValueError into durable-turn
    # startup on a typo. The resolver must warn + default instead.
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        timeout, poll = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"timeout_s": "oops", "poll_s": "typo"}}}
        )
    assert timeout == DEFAULT_TURN_LIVENESS_TIMEOUT_S
    assert poll == DEFAULT_TURN_LIVENESS_POLL_S
    assert len(caplog.records) == 2
    assert all("agent.turn_liveness" in r.getMessage() for r in caplog.records)


def test_nan_timeout_falls_back_to_default_not_disabled(caplog):
    # float("nan") > 0 is False, so the old code silently disabled the
    # watchdog on NaN. The resolver must reject it and keep the default.
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        timeout, poll = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"timeout_s": float("nan")}}}
        )
    assert timeout == DEFAULT_TURN_LIVENESS_TIMEOUT_S  # NOT None
    assert poll == DEFAULT_TURN_LIVENESS_POLL_S
    assert len(caplog.records) == 1


def test_inf_poll_falls_back_to_default_not_frozen_watcher(caplog):
    # float("inf") poll made Event.wait(inf) hang the watcher thread
    # forever. The resolver must reject it.
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        timeout, poll = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"poll_s": float("inf")}}}
        )
    assert timeout == DEFAULT_TURN_LIVENESS_TIMEOUT_S
    assert poll == DEFAULT_TURN_LIVENESS_POLL_S
    assert len(caplog.records) == 1


def test_inf_timeout_falls_back_to_default(caplog):
    # inf timeout would never fire (idle < inf always true) — silent
    # disablement. Rejected.
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        timeout, _ = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"timeout_s": float("inf")}}}
        )
    assert timeout == DEFAULT_TURN_LIVENESS_TIMEOUT_S
    assert len(caplog.records) == 1


def test_non_positive_poll_falls_back_to_default(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        _, poll = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"poll_s": 0}}}
        )
    assert poll == DEFAULT_TURN_LIVENESS_POLL_S
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        _, poll = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": {"poll_s": -3}}}
        )
    assert poll == DEFAULT_TURN_LIVENESS_POLL_S


def test_malformed_sections_fall_back_to_defaults(caplog):
    # Non-dict `agent` or non-dict `turn_liveness` must not raise.
    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        result = resolve_turn_liveness_settings({"agent": "nonsense"})
    assert result == (DEFAULT_TURN_LIVENESS_TIMEOUT_S, DEFAULT_TURN_LIVENESS_POLL_S)

    with caplog.at_level(logging.WARNING, logger="agent.turn_liveness"):
        result = resolve_turn_liveness_settings(
            {"agent": {"turn_liveness": "nonsense"}}
        )
    assert result == (DEFAULT_TURN_LIVENESS_TIMEOUT_S, DEFAULT_TURN_LIVENESS_POLL_S)
    # The malformed section itself is surfaced as a warning.
    assert len(caplog.records) >= 1
