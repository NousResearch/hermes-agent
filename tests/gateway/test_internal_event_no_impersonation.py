"""Synthetic internal events must never be attributed to the human user.

Regression guard for the whole bug class: the gateway synthesises inbound
``MessageEvent``s that no human typed (boot auto-resume re-prompts, queued
continuations, background-process completion notices).  These already carry
``MessageEvent.internal=True`` so they bypass authorization.  Two user-visible
surfaces still treated them as human-authored:

1. The shared-multi-user sender prefix stamped ``"[<display name>] "`` onto
   them, so a group session saw a ghost message apparently sent by a
   participant who never spoke.  Worse, prefixing an EMPTY synthetic event
   makes it non-empty, which suppresses the blank-text recovery-note
   substitution downstream.
2. The inbound log line recorded ``user=<display name>``, sending incident
   forensics chasing a phantom "user sent an empty message".
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _shared_group_source(**overrides) -> SessionSource:
    kwargs = dict(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    kwargs.update(overrides)
    return SessionSource(**kwargs)


def _shared_group_config() -> GatewayConfig:
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
        group_sessions_per_user=False,
    )


@pytest.mark.asyncio
async def test_internal_event_is_not_prefixed_with_a_user_name():
    """A synthetic continuation must not be stamped with a participant's name."""
    runner = _make_runner(_shared_group_config())
    source = _shared_group_source()
    event = MessageEvent(text="continue", source=source, internal=True)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "continue"
    assert "[Alice]" not in result


@pytest.mark.asyncio
async def test_empty_internal_event_stays_empty():
    """An EMPTY synthetic event must stay empty.

    Prefixing turns "" into "[Alice] ", which is both an impersonation and a
    silent behavior change: downstream blank-text handling (the reason-aware
    recovery system-note substitution) only fires on empty text.
    """
    runner = _make_runner(_shared_group_config())
    source = _shared_group_source()
    event = MessageEvent(text="", source=source, internal=True)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == ""


@pytest.mark.asyncio
async def test_slack_internal_event_gets_no_author_mention():
    """Sibling call path: the Slack author-mention variant of the same prefix."""
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.SLACK: PlatformConfig(enabled=True, token="fake")},
        )
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_name="team-channel",
        chat_type="group",
        user_id="U123",
        user_name="Alice",
        thread_id="171.000",
    )
    event = MessageEvent(text="continue", source=source, internal=True)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "continue"
    assert "U123" not in result


@pytest.mark.asyncio
async def test_human_event_still_gets_the_sender_prefix():
    """Control: the feature itself is preserved for genuine human messages."""
    runner = _make_runner(_shared_group_config())
    source = _shared_group_source()
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice] hello"


def test_internal_event_log_identity_is_system_not_the_user():
    """The inbound log line must not name a human for a synthetic event."""
    from gateway.run import describe_inbound_event_author

    source = _shared_group_source(user_id="7788")
    internal_event = MessageEvent(text="", source=source, internal=True)
    human_event = MessageEvent(text="hello", source=source)

    assert describe_inbound_event_author(internal_event, source) == "<system:internal>"
    # Control: a real human message still logs the human's identity.
    assert describe_inbound_event_author(human_event, source) == "Alice"


def test_log_identity_falls_back_to_user_id_then_unknown():
    """Behavior contract for the non-internal fallback ladder."""
    from gateway.run import describe_inbound_event_author

    id_only = _shared_group_source(user_name=None, user_id="7788")
    assert describe_inbound_event_author(
        MessageEvent(text="hi", source=id_only), id_only
    ) == "7788"

    anonymous = _shared_group_source(user_name=None, user_id=None)
    assert describe_inbound_event_author(
        MessageEvent(text="hi", source=anonymous), anonymous
    ) == "unknown"
