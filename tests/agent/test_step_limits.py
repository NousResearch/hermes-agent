"""KAN-319: Step limits normalization and regression test suite.

Tests:
1. Normalization functions: normalize_max_steps, normalize_max_iterations, agent_max_steps.
2. Turn finalizer budget fallback when max_steps is None, 0, or "5".
3. AIAgent and init_agent accept max_steps keyword argument.
4. _build_child_agent parameter compatibility (does not require max_steps, forwards to child).
5. finalize_turn accepts step_count as an optional parameter.
"""

import inspect
import logging
import sys
from types import SimpleNamespace

import pytest

from agent.step_limits import (
    agent_max_steps,
    normalize_max_iterations,
    normalize_max_steps,
)
from agent.turn_finalizer import _resolve_budget_fallback, finalize_turn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (0, None),
        (-1, None),
        (-10, None),
        (True, None),
        (False, None),
        ("", None),
        ("   ", None),
        ("abc", None),
        ([], None),
        ({}, None),
        (float("nan"), None),
        (5, 5),
        ("5", 5),
        (" 5 ", 5),
        ("7", 7),
        (3.0, 3),
    ],
)
def test_normalize_max_steps(raw, expected):
    assert normalize_max_steps(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, sys.maxsize),
        (0, sys.maxsize),
        (-5, sys.maxsize),
        ("x", sys.maxsize),
        (True, sys.maxsize),
        (False, sys.maxsize),
        (12, 12),
        ("40", 40),
        (" 250 ", 250),
    ],
)
def test_normalize_max_iterations(raw, expected):
    assert normalize_max_iterations(raw) == expected


def test_normalize_max_iterations_custom_default():
    assert normalize_max_iterations(None, default=250) == 250
    assert normalize_max_iterations(0, default=250) == 250
    assert normalize_max_iterations("invalid", default=250) == 250
    assert normalize_max_iterations(50, default=250) == 50


def test_agent_max_steps_tolerates_missing_or_none_attribute():
    assert agent_max_steps(object()) is None
    assert agent_max_steps(SimpleNamespace()) is None
    assert agent_max_steps(SimpleNamespace(max_steps=None)) is None
    assert agent_max_steps(SimpleNamespace(max_steps=0)) is None
    assert agent_max_steps(SimpleNamespace(max_steps="5")) == 5
    assert agent_max_steps(SimpleNamespace(max_steps=10)) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. Finalizer budget fallback with max_steps: None, 0, "5"
# ─────────────────────────────────────────────────────────────────────────────

_MISSING = object()


def _make_mock_agent(max_steps=_MISSING, *, max_iterations=60, remaining=10):
    calls = []

    def _handle_max_iterations(messages, api_call_count):
        calls.append(api_call_count)
        return "summary_response"

    agent = SimpleNamespace(
        max_iterations=max_iterations,
        iteration_budget=SimpleNamespace(remaining=remaining),
        quiet_mode=True,
        _handle_max_iterations=_handle_max_iterations,
        _emit_diagnostic_status=lambda *args, **kwargs: None,
        _safe_print=lambda *args, **kwargs: None,
    )
    if max_steps is not _MISSING:
        agent.max_steps = max_steps
    return agent, calls


def _call_resolve_budget_fallback(agent, *, step_count=0, api_call_count=1):
    return _resolve_budget_fallback(
        agent,
        final_response=None,
        api_call_count=api_call_count,
        step_count=step_count,
        interrupted=False,
        failed=False,
        messages=[],
        _turn_exit_reason="unknown",
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
        logger=logging.getLogger("test"),
    )


@pytest.mark.parametrize("max_steps", [None, 0, -1, "0", "invalid", True, False, _MISSING])
def test_finalizer_when_max_steps_is_none_or_zero(max_steps):
    """When max_steps is None or 0 (or unparsable), no step cap is enforced and turn does not fail."""
    agent, calls = _make_mock_agent(max_steps)
    final_response, reason, preserved, interrupted = _call_resolve_budget_fallback(
        agent, step_count=100, api_call_count=5
    )
    assert final_response is None
    assert reason == "unknown"
    assert calls == []
    assert preserved is False
    assert interrupted is False


def test_finalizer_when_max_steps_is_string_five_not_reached():
    """When max_steps is '5' and step_count < 5, loop continues normally."""
    agent, calls = _make_mock_agent("5")
    final_response, reason, _preserved, _interrupted = _call_resolve_budget_fallback(
        agent, step_count=4, api_call_count=4
    )
    assert final_response is None
    assert reason == "unknown"
    assert calls == []


def test_finalizer_when_max_steps_is_string_five_reached():
    """When max_steps is '5' and step_count >= 5, budget fallback is triggered."""
    agent, calls = _make_mock_agent("5")
    final_response, reason, _preserved, _interrupted = _call_resolve_budget_fallback(
        agent, step_count=5, api_call_count=5
    )
    assert final_response == "summary_response"
    assert calls == [5]
    # Classification prefix must match downstream expectations
    assert reason.startswith("max_iterations_reached(")


def test_finalize_turn_signature_step_count_is_optional():
    """finalize_turn must have step_count as an optional parameter with default 0."""
    sig = inspect.signature(finalize_turn)
    assert "step_count" in sig.parameters
    assert sig.parameters["step_count"].default == 0


def test_finalize_turn_legacy_call_without_step_count():
    """Calling _resolve_budget_fallback without step_count kwarg uses default 0 without error."""
    agent, calls = _make_mock_agent(None)
    final_response, reason, _, _ = _resolve_budget_fallback(
        agent,
        final_response=None,
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=[],
        _turn_exit_reason="unknown",
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
        logger=logging.getLogger("test"),
    )
    assert final_response is None
    assert reason == "unknown"
    assert calls == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. AIAgent and _build_child_agent signature & parameter acceptance
# ─────────────────────────────────────────────────────────────────────────────

def test_aiagent_accepts_max_steps_kwarg():
    """AIAgent.__init__ and agent_init.init_agent must accept max_steps."""
    from run_agent import AIAgent
    from agent.agent_init import init_agent

    aiagent_params = inspect.signature(AIAgent.__init__).parameters
    assert "max_steps" in aiagent_params
    assert aiagent_params["max_steps"].default is None

    init_agent_params = inspect.signature(init_agent).parameters
    assert "max_steps" in init_agent_params
    assert init_agent_params["max_steps"].default is None


def test_build_child_agent_does_not_require_max_steps():
    """_build_child_agent in delegate_tool must not require max_steps parameter."""
    from tools.delegate_tool import _build_child_agent

    params = inspect.signature(_build_child_agent).parameters
    assert "max_steps" not in params or params["max_steps"].default is not inspect.Parameter.empty
