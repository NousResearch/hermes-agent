"""Tests for reply-to pointer injection in _prepare_inbound_message_text.

The `[Replying to: "..."]` prefix is a *disambiguation pointer*, not
deduplication. It must always be injected when the user explicitly replies
to a prior message — even when the quoted text already exists somewhere
in the conversation history. History can contain the same or similar text
multiple times, and without an explicit pointer the agent has to guess
which prior message the user is referencing.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


@pytest.mark.asyncio
async def test_reply_prefix_injected_when_text_absent_from_history():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Japan is great for culture, food, and efficiency.",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[{"role": "user", "content": "unrelated"}],
    )

    assert result is not None
    assert result.startswith(
        '[Replying to: "Japan is great for culture, food, and efficiency."]'
    )
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_reply_prefix_still_injected_when_text_in_history():
    """Regression test: the pointer must survive even when the quoted text
    already appears in history. Previously a `found_in_history` guard
    silently dropped the prefix, leaving the agent to guess which prior
    message the user was referencing."""
    runner = _make_runner()
    source = _source()
    quoted = "Japan is great for culture, food, and efficiency."
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text=quoted,
    )

    history = [
        {"role": "user", "content": "I'm thinking of going to Japan or Italy."},
        {
            "role": "assistant",
            "content": (
                f"{quoted} Italy is better if you prefer a relaxed pace."
            ),
        },
        {"role": "user", "content": "How long should I stay?"},
        {"role": "assistant", "content": "For Japan, 10-14 days is ideal."},
    ]

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=history,
    )

    assert result is not None
    assert result.startswith(f'[Replying to: "{quoted}"]')
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_reply_prefix_own_message_variant():
    """Replying to the bot's own message uses the dedicated wording so the
    agent knows the quote is its own prior answer, not another user's."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="yes, send the same",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Draft A: ready to send to alice@example.com",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[{"role": "user", "content": "unrelated"}],
    )

    assert result is not None
    assert result.startswith(
        '[Replying to your previous message: "Draft A: ready to send to alice@example.com"]'
    )
    assert result.endswith("yes, send the same")


@pytest.mark.asyncio
async def test_busy_steer_text_carries_reply_prefix():
    """Regression test (#101866): with ``busy_input_mode: steer`` a mid-turn
    follow-up is injected via ``running_agent.steer()`` and never passes
    through ``_prepare_inbound_message_text`` — the reply-to prefix must be
    applied in ``_prepare_busy_steer_text`` itself, or the agent only sees
    the bare text and has to guess which prior message is referenced."""
    runner = _make_runner()
    event = MessageEvent(
        text="yes, send the same",
        source=_source(),
        reply_to_message_id="42",
        reply_to_text="Draft A: ready to send to alice@example.com",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_busy_steer_text(event)

    assert result == (
        '[Replying to your previous message: '
        '"Draft A: ready to send to alice@example.com"]\n\n'
        "yes, send the same"
    )


@pytest.mark.asyncio
async def test_busy_steer_text_without_reply_unchanged():
    """A plain mid-turn steer without reply context must be forwarded
    verbatim — no prefix, no formatting change."""
    runner = _make_runner()
    event = MessageEvent(
        text="also add a footnote",
        source=_source(),
    )

    result = await runner._prepare_busy_steer_text(event)

    assert result == "also add a footnote"


@pytest.mark.asyncio
async def test_busy_steer_empty_text_stays_empty():
    """An empty payload must keep falling back to queue semantics even when
    reply metadata is present — the prefix never turns empty text into a
    steerable one."""
    runner = _make_runner()
    event = MessageEvent(
        text="",
        source=_source(),
        reply_to_message_id="42",
        reply_to_text="Draft A: ready to send to alice@example.com",
    )

    result = await runner._prepare_busy_steer_text(event)

    assert result == ""


