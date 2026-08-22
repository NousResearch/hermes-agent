"""Regression tests for restoring a rate-limited primary during a long turn."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from agent import conversation_loop
from agent.conversation_loop import _restore_primary_runtime_for_iteration


def test_restores_primary_prompt_identity_when_runtime_recovers():
    restore = Mock(return_value=True)
    agent = SimpleNamespace(
        _fallback_activated=True,
        _cached_system_prompt="Model: primary\nProvider: primary-provider",
        _restore_primary_runtime=restore,
    )

    result = _restore_primary_runtime_for_iteration(agent, "Model: fallback")

    assert result == agent._cached_system_prompt
    restore.assert_called_once_with()


def test_keeps_fallback_prompt_when_cooldown_has_not_expired():
    restore = Mock(return_value=False)
    agent = SimpleNamespace(
        _fallback_activated=True,
        _cached_system_prompt="Model: primary",
        _restore_primary_runtime=restore,
    )

    result = _restore_primary_runtime_for_iteration(agent, "Model: fallback")

    assert result == "Model: fallback"
    restore.assert_called_once_with()


def test_skips_restore_probe_for_primary_iterations():
    restore = Mock(return_value=True)
    agent = SimpleNamespace(
        _fallback_activated=False,
        _cached_system_prompt="Model: primary",
        _restore_primary_runtime=restore,
    )

    result = _restore_primary_runtime_for_iteration(agent, "Model: primary")

    assert result == "Model: primary"
    restore.assert_not_called()


def test_outer_iteration_probes_before_building_the_next_request():
    tree = ast.parse(inspect.getsource(conversation_loop.run_conversation))
    outer_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and any(
            isinstance(name, ast.Name) and name.id == "api_call_count"
            for name in ast.walk(node.test)
        )
    )

    restore_lines = [
        node.lineno
        for node in ast.walk(outer_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_restore_primary_runtime_for_iteration"
    ]
    redirect_lines = [
        node.lineno
        for node in ast.walk(outer_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_drain_pending_redirect"
    ]

    assert restore_lines, "the outer loop must re-check primary restoration"
    assert redirect_lines, "expected the normal outer-loop request setup"
    assert min(restore_lines) < min(redirect_lines)
