"""Unit tests for the identity-derived memory-recall policy (#40170).

These test the pure decision function directly — no mocked conditionals — so
they assert the actual operator/customer boundary that production uses.
"""

from __future__ import annotations

import pytest

from agent.memory_recall_policy import (
    is_local_operator_surface,
    operator_scoped_recall,
    suppress_memory_recall,
)


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "acp", "local", "", None])
def test_local_surfaces_are_operator_scoped(platform):
    """The operator's own local surfaces always allow auto-recall, regardless
    of chat_type or the (irrelevant) operator flag."""
    assert is_local_operator_surface(platform) is True
    assert operator_scoped_recall(
        platform=platform, chat_type="dm", requester_is_operator=False
    ) is True
    assert suppress_memory_recall(
        platform=platform, chat_type=None, requester_is_operator=False
    ) is False


@pytest.mark.parametrize("platform", ["telegram", "discord", "slack", "whatsapp", "signal", "matrix"])
def test_gateway_dm_from_operator_is_scoped(platform):
    """A 1:1 gateway DM from a confirmed operator injects recall."""
    assert operator_scoped_recall(
        platform=platform, chat_type="dm", requester_is_operator=True
    ) is True
    assert suppress_memory_recall(
        platform=platform, chat_type="dm", requester_is_operator=True
    ) is False


@pytest.mark.parametrize("platform", ["telegram", "discord", "slack", "whatsapp", "signal", "matrix"])
def test_gateway_dm_from_customer_is_suppressed(platform):
    """A 1:1 gateway DM from a non-operator (customer) suppresses recall."""
    assert operator_scoped_recall(
        platform=platform, chat_type="dm", requester_is_operator=False
    ) is False
    assert suppress_memory_recall(
        platform=platform, chat_type="dm", requester_is_operator=False
    ) is True


@pytest.mark.parametrize("chat_type", ["group", "channel", "thread", "forum", "supergroup"])
def test_gateway_group_always_suppressed(chat_type):
    """Shared multi-user contexts are never operator-private — recall is
    suppressed even if the current sender happens to be the operator."""
    assert operator_scoped_recall(
        platform="telegram", chat_type=chat_type, requester_is_operator=True
    ) is False
    assert suppress_memory_recall(
        platform="telegram", chat_type=chat_type, requester_is_operator=True
    ) is True


def test_unknown_gateway_platform_is_gated_by_identity_not_name():
    """A platform outside any hardcoded list is still covered: a customer DM on
    a novel plugin platform suppresses; the operator on it does not."""
    assert suppress_memory_recall(
        platform="some_new_plugin_platform", chat_type="dm", requester_is_operator=False
    ) is True
    assert suppress_memory_recall(
        platform="some_new_plugin_platform", chat_type="dm", requester_is_operator=True
    ) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
