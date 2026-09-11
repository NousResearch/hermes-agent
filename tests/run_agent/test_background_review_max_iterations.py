"""Regression tests for the configurable background-review iteration budget.

The background-review fork in ``agent/background_review.py`` historically hardcoded
``_REVIEW_MAX_ITERATIONS = 16``. Operators had no way to lower the worst-case cost of
the review (the fork is auto-spawned on most turns past the nudge interval) or raise
it for unusually heavy reviews without patching source. This file pins the
``_review_max_iterations`` resolver and its wiring into ``_run_review_fork`` so the
behaviour stays tunable.

Mirrors ``tests/run_agent/test_background_review_input_budget.py`` (the same resolver
pattern for the input-token budget introduced alongside it).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ===================================================================
# Group 1: resolver edge cases
# ===================================================================


@pytest.mark.parametrize(
    ("config_value", "expected"),
    [
        ({}, 16),
        ({"max_iterations": 32}, 32),
        ({"max_iterations": 1}, 1),  # tight cost floor
        ({"max_iterations": 256}, 256),  # unbounded-feel; global max_iterations still applies
        ({"max_iterations": "8"}, 8),  # numeric string coerced
        ({"max_iterations": 0}, 16),  # 0 falls back to default — cannot silently disable
        ({"max_iterations": -5}, 16),  # negative falls back to default
        ({"max_iterations": "not-a-number"}, 16),  # garbage falls back to default
        ({"max_iterations": None}, 16),  # explicit null falls back to default
    ],
)
def test_review_max_iterations_resolution(config_value, expected):
    """Config parsing: default, override, garbage fallback, no clamp.

    Operator intent passes through when it's a positive int; everything else falls back
    to the documented default so a typo cannot hand ``AIAgent`` an invalid
    ``max_iterations`` (which the constructor would either reject or accept as
    ``sys.maxsize``).
    """
    from agent.background_review import _review_max_iterations

    assert _review_max_iterations(config_value) == expected


def test_review_max_iterations_resolves_via_full_config_path():
    """When the resolver is called via the real ``load_config_readonly`` path,
    a deeply-nested ``auxiliary.background_review.max_iterations`` is honoured."""
    from agent.background_review import _review_max_iterations

    cfg = {"auxiliary": {"background_review": {"max_iterations": 24}}}
    # ``load_config_readonly`` is imported lazily inside ``_background_review_task_config``;
    # patch at the source module so the lazy import picks up the fake.
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert _review_max_iterations() == 24


def test_review_max_iterations_resolves_via_full_config_path_default():
    """When the config key is absent, the resolver returns the module default even when
    called via the load_config_readonly path (no operator key, no surprise)."""
    from agent.background_review import _review_max_iterations, _REVIEW_MAX_ITERATIONS

    cfg = {"auxiliary": {"background_review": {"enabled": True, "max_input_tokens": 600000}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert _review_max_iterations() == _REVIEW_MAX_ITERATIONS == 16


# ===================================================================
# Group 2: wiring into _run_review_fork
# ===================================================================


def _make_agent():
    """Minimal AIAgent (mirrors ``test_background_review_input_budget.py::_make_loop_agent``
    but stripped — we don't run a tool loop here, just observe what ``max_iterations`` is
    passed to the fork factory)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def test_run_review_fork_uses_configured_max_iterations(monkeypatch):
    """End-to-end: the value passed to ``build_cache_parity_fork`` is whatever
    ``_review_max_iterations`` resolves from the active config."""
    from agent import background_review

    captured_kwargs: dict = {}

    def _fake_build_cache_parity_fork(agent, task_cfg, *, max_iterations):
        captured_kwargs["max_iterations"] = max_iterations
        # Return a fork agent with no-op run_conversation so we can observe the kwarg
        # without standing up a real AIAgent (the call site would otherwise blow up on
        # attributes the fork uses post-construction).
        fork = SimpleNamespace(
            release_clients=lambda: None,
            run_conversation=lambda **kw: {"completed": True, "messages": [], "final_response": ""},
        )
        return fork, {}, False

    monkeypatch.setattr(
        background_review, "build_cache_parity_fork", _fake_build_cache_parity_fork,
    )
    # Bypass the tool whitelist / thread machinery so we can observe the kwarg only.
    monkeypatch.setattr(background_review, "_track_review_fork", lambda *a, **kw: None)
    # ``set_thread_tool_whitelist`` and ``clear_thread_tool_whitelist`` are imported lazily
    # inside _run_review_fork from ``hermes_cli.plugins``; patch at the source module.
    monkeypatch.setattr("hermes_cli.plugins.set_thread_tool_whitelist", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.clear_thread_tool_whitelist", lambda: None)
    monkeypatch.setattr(background_review, "_review_tool_whitelist", lambda *a, **kw: ([], set()))

    agent = _make_agent()
    task_cfg = {"max_iterations": 4}

    background_review._run_review_fork(
        agent, messages_snapshot=[], prompt="review me",
        task_cfg=task_cfg, review_run=None,
        st=background_review._ReviewForkState(),
    )

    assert captured_kwargs["max_iterations"] == 4, (
        f"fork must receive the configured 4 iterations, got {captured_kwargs['max_iterations']!r}"
    )


def test_run_review_fork_uses_default_when_unconfigured(monkeypatch):
    """End-to-end: missing ``max_iterations`` key falls through to the module default
    (regression guard against re-hardcoding the constant in the future)."""
    from agent import background_review

    captured_kwargs: dict = {}

    def _fake_build_cache_parity_fork(agent, task_cfg, *, max_iterations):
        captured_kwargs["max_iterations"] = max_iterations
        fork = SimpleNamespace(
            release_clients=lambda: None,
            run_conversation=lambda **kw: {"completed": True, "messages": [], "final_response": ""},
        )
        return fork, {}, False

    monkeypatch.setattr(
        background_review, "build_cache_parity_fork", _fake_build_cache_parity_fork,
    )
    monkeypatch.setattr(background_review, "_track_review_fork", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.set_thread_tool_whitelist", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.clear_thread_tool_whitelist", lambda: None)
    monkeypatch.setattr(background_review, "_review_tool_whitelist", lambda *a, **kw: ([], set()))

    agent = _make_agent()
    background_review._run_review_fork(
        agent, messages_snapshot=[], prompt="review me",
        task_cfg={}, review_run=None,
        st=background_review._ReviewForkState(),
    )

    assert captured_kwargs["max_iterations"] == background_review._REVIEW_MAX_ITERATIONS == 16