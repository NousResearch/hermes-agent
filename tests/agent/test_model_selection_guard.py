"""Common managed-route guard: real-constructor validation (design §5, §12).

Pure — asserts the guard rejects any mismatch between a receipted decision
and what was actually constructed, and that the disabled-fallback kwargs are
exactly the receipted route (never merged with a parent/profile default).
"""
from __future__ import annotations

import pytest

from agent.model_selection_types import RoutingBlocked


def _decision():
    return {
        "policy_id": "p1", "policy_revision": 1,
        "requirements": {"role": "builder", "execution_kind": "kanban", "execution_id": "t_1",
                          "attempt_id": "1", "slot_id": "", "task_class": "cross-component",
                          "quality": "deep", "reasoning": "high"},
        "selected": {"route_id": "a", "route_revision": 1, "provider": "openai", "model": "gpt-x",
                     "endpoint": "https://api.openai.com/v1", "maker": "openai"},
        "rejections": {}, "alternates": [], "selection_timestamp": 100,
    }


def test_matching_actual_route_passes():
    from agent.model_selection_guard import validate_actual_route

    validate_actual_route(_decision(), actual_provider="openai", actual_model="gpt-x",
                           actual_endpoint="https://api.openai.com/v1", actual_reasoning="high")


def test_mismatched_provider_or_model_blocks_before_send():
    from agent.model_selection_guard import validate_actual_route

    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        validate_actual_route(_decision(), actual_provider="anthropic", actual_model="gpt-x")
    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        validate_actual_route(_decision(), actual_provider="openai", actual_model="other-model")


def test_mismatched_reasoning_blocks():
    from agent.model_selection_guard import validate_actual_route

    with pytest.raises(RoutingBlocked, match="reasoning_unsupported"):
        validate_actual_route(_decision(), actual_provider="openai", actual_model="gpt-x",
                               actual_reasoning="low")


def test_managed_child_kwargs_is_exact_route_no_inherited_fallback():
    from agent.model_selection_guard import managed_child_kwargs

    kwargs = managed_child_kwargs(_decision())
    assert kwargs == {"provider": "openai", "model": "gpt-x",
                       "endpoint": "https://api.openai.com/v1", "reasoning_effort": "high"}
