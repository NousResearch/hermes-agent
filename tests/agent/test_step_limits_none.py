"""KAN-319: ``max_steps`` / ``max_iterations`` set to ``None`` (or junk) must never crash.

Regression for ``TypeError: '>=' not supported between instances of 'int' and
'NoneType'`` raised by ``turn_finalizer._resolve_budget_fallback`` when
``agent.max_steps`` existed but was ``None`` (``getattr``'s default does not
apply to an attribute that is present).
"""

import inspect
import logging
import sys
from types import SimpleNamespace

import pytest

from agent.step_limits import agent_max_steps, normalize_max_iterations, normalize_max_steps
from agent.turn_finalizer import _resolve_budget_fallback, finalize_turn


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None), (0, None), (-3, None), (True, None), (False, None),
        ("", None), ("  ", None), ("abc", None), ([], None), (float("nan"), None),
        (5, 5), ("7", 7), (" 9 ", 9), (3.0, 3),
    ],
)
def test_normalize_max_steps(raw, expected):
    assert normalize_max_steps(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, sys.maxsize), (0, sys.maxsize), ("x", sys.maxsize), (True, sys.maxsize), (12, 12), ("40", 40)],
)
def test_normalize_max_iterations(raw, expected):
    assert normalize_max_iterations(raw) == expected


def test_normalize_max_iterations_custom_default():
    assert normalize_max_iterations(None, default=250) == 250


def test_agent_max_steps_tolerates_missing_attribute():
    assert agent_max_steps(object()) is None
    assert agent_max_steps(SimpleNamespace(max_steps=None)) is None
    assert agent_max_steps(SimpleNamespace(max_steps="4")) == 4


def _agent(max_steps, *, max_iterations=60, remaining=10):
    calls = []

    def _handle_max_iterations(messages, api_call_count):
        calls.append(api_call_count)
        return "summary"

    agent = SimpleNamespace(
        max_iterations=max_iterations,
        iteration_budget=SimpleNamespace(remaining=remaining),
        quiet_mode=True,
        _handle_max_iterations=_handle_max_iterations,
        _emit_diagnostic_status=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
    )
    if max_steps is not _MISSING:
        agent.max_steps = max_steps
    return agent, calls


_MISSING = object()


def _resolve(agent, *, step_count=3, api_call_count=3, **kw):
    return _resolve_budget_fallback(
        agent, final_response=None, api_call_count=api_call_count, step_count=step_count,
        interrupted=False, failed=False, messages=[], _turn_exit_reason="unknown",
        _pending_verification_response=None, _pending_verification_response_previewed=False,
        logger=logging.getLogger("test"), **kw,
    )


@pytest.mark.parametrize("max_steps", [None, _MISSING, 0, -1, "abc", True])
def test_no_step_cap_does_not_crash_or_trigger_fallback(max_steps):
    agent, calls = _agent(max_steps)
    final_response, reason, preserved, interrupted = _resolve(agent)
    assert final_response is None
    assert reason == "unknown"
    assert calls == []
    assert preserved is False and interrupted is False


@pytest.mark.parametrize("max_steps", [3, "3", 2])
def test_reached_step_cap_requests_summary_with_compatible_reason(max_steps):
    agent, calls = _agent(max_steps)
    final_response, reason, _, _ = _resolve(agent, step_count=3)
    assert final_response == "summary"
    assert calls == [3]
    # Downstream classifiers key on this prefix (kanban recovery, explainers, API routes).
    assert reason.startswith("max_iterations_reached(")


def test_step_count_below_cap_is_not_exhausted():
    agent, calls = _agent(10)
    final_response, reason, _, _ = _resolve(agent, step_count=3)
    assert (final_response, reason, calls) == (None, "unknown", [])


def test_step_count_defaults_for_legacy_callers():
    """Callers that predate ``step_count`` (tests, plugins) must keep working."""
    agent, calls = _agent(None)
    final_response, reason, _, _ = _resolve_budget_fallback(
        agent, final_response=None, api_call_count=3, interrupted=False, failed=False, messages=[],
        _turn_exit_reason="unknown", _pending_verification_response=None,
        _pending_verification_response_previewed=False, logger=logging.getLogger("test"),
    )
    assert (final_response, reason, calls) == (None, "unknown", [])


def test_finalize_turn_step_count_is_optional():
    param = inspect.signature(finalize_turn).parameters["step_count"]
    assert param.default == 0


def test_aiagent_accepts_max_steps_kwarg():
    """delegate_tool passes ``max_steps=`` to AIAgent; it must be a real parameter."""
    from run_agent import AIAgent
    from agent.agent_init import init_agent

    assert inspect.signature(AIAgent.__init__).parameters["max_steps"].default is None
    assert inspect.signature(init_agent).parameters["max_steps"].default is None


def test_build_child_agent_does_not_require_max_steps():
    """The production caller never passed ``max_steps``; it must not be a required param."""
    from tools.delegate_tool import _build_child_agent

    param = inspect.signature(_build_child_agent).parameters.get("max_steps")
    assert param is None or param.default is not inspect.Parameter.empty
