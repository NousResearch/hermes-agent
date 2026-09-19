"""Behavior contract for the transport-neutral inbound mention gate."""

import pytest

from gateway.platforms.inbound_mention import (
    DEFAULT_REQUIRE_MENTION,
    InboundMentionFacts,
    resolve_inbound_mention_decision,
)


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        ("telegram", False),
        ("whatsapp", False),
        ("dingtalk", False),
        ("discord", True),
        ("slack", True),
        ("matrix", True),
        ("mattermost", True),
        ("feishu", True),
        ("signal", False),
    ],
)
def test_adapter_defaults_are_explicit(adapter, expected):
    assert DEFAULT_REQUIRE_MENTION[adapter] is expected


@pytest.mark.parametrize(
    "facts, expected",
    [
        (InboundMentionFacts(is_dm=True), True),
        (InboundMentionFacts(is_dm=False, is_mentioned=True), True),
        (InboundMentionFacts(is_dm=False, matches_custom_pattern=True), True),
        (InboundMentionFacts(is_dm=False, command_addresses_bot=True), True),
        (InboundMentionFacts(is_dm=False, is_free_response_scope=True), True),
        (InboundMentionFacts(is_dm=False, is_participating_thread=True), True),
        (InboundMentionFacts(is_dm=False), False),
    ],
)
def test_required_mention_acceptance_rungs(facts, expected):
    assert resolve_inbound_mention_decision(facts, require_mention=True) is expected


def test_disabled_requirement_accepts_unmentioned_group_message():
    assert resolve_inbound_mention_decision(InboundMentionFacts(is_dm=False), require_mention=False)
