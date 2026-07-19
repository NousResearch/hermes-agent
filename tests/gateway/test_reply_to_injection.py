"""Tests for reply-to pointer injection in _prepare_inbound_message_text.

The `[Replying to: "..."]` prefix is a *disambiguation pointer*, not
deduplication. It must always be injected when the user explicitly replies
to a prior message — even when the quoted text already exists somewhere
in the conversation history. History can contain the same or similar text
multiple times, and without an explicit pointer the agent has to guess
which prior message the user is referencing.
"""
import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.config import load_config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


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


async def _prepare_reply_with_limit(
    monkeypatch,
    tmp_path,
    configured_value,
    *,
    text_length: int = 12_000,
) -> str:
    """Exercise config.yaml -> gateway loader -> reply slicing end to end."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"gateway:\n  reply_snippet_max_length: {configured_value!r}\n",
        encoding="utf-8",
    )
    source = _source()
    result = await _make_runner()._prepare_inbound_message_text(
        event=MessageEvent(
            text="follow-up",
            source=source,
            reply_to_message_id="42",
            reply_to_text="x" * text_length,
        ),
        source=source,
        history=[],
    )
    assert result is not None
    return result


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
async def test_own_message_reply_prefix_marks_assistant_message():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this one",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Use the direct train.",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to your previous message: "Use the direct train."]')
    assert result.endswith("this one")


@pytest.mark.asyncio
async def test_no_prefix_without_reply_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_no_prefix_when_reply_to_text_is_empty():
    """reply_to_message_id alone without text (e.g. a reply to a media-only
    message) should not produce an empty `[Replying to: ""]` prefix."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="hi",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hi"


@pytest.mark.asyncio
async def test_reply_snippet_truncated_to_500_chars():
    runner = _make_runner()
    source = _source()
    long_text = "x" * 800
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text=long_text,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to: "' + "x" * 500 + '"]')
    assert "x" * 501 not in result


def test_default_reply_snippet_limit_comes_from_canonical_config(tmp_path):
    """The default is part of merged config, not only a call-site fallback."""
    token = set_hermes_home_override(tmp_path)
    try:
        assert load_config()["gateway"]["reply_snippet_max_length"] == 500
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_reply_snippet_respects_configured_max_length(monkeypatch, tmp_path):
    """A real config.yaml override propagates through the gateway loader."""
    result = await _prepare_reply_with_limit(
        monkeypatch,
        tmp_path,
        200,
        text_length=800,
    )
    assert result.startswith('[Replying to: "' + "x" * 200 + '"]')
    assert "x" * 201 not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_value",
    [0, -1, 10_001, True, 1.5, "0", "-1", "10001", "not-an-integer"],
)
async def test_invalid_reply_snippet_limits_fall_back_to_500(
    monkeypatch,
    tmp_path,
    configured_value,
):
    result = await _prepare_reply_with_limit(
        monkeypatch,
        tmp_path,
        configured_value,
    )
    assert result.startswith('[Replying to: "' + "x" * 500 + '"]')
    assert "x" * 501 not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_value", "expected_length"),
    [(1, 1), (10_000, 10_000), ("200", 200)],
)
async def test_reply_snippet_limit_accepts_boundaries(
    monkeypatch,
    tmp_path,
    configured_value,
    expected_length,
):
    result = await _prepare_reply_with_limit(
        monkeypatch,
        tmp_path,
        configured_value,
    )
    assert result.startswith('[Replying to: "' + "x" * expected_length + '"]')
    assert "x" * (expected_length + 1) not in result


@pytest.mark.asyncio
async def test_reply_snippet_limit_expands_env_var(monkeypatch, tmp_path):
    """Runtime config expansion reaches the reply-context budget."""
    monkeypatch.setenv("REPLY_SNIPPET_LIMIT", "200")
    result = await _prepare_reply_with_limit(
        monkeypatch,
        tmp_path,
        "${REPLY_SNIPPET_LIMIT}",
        text_length=800,
    )

    assert result.startswith('[Replying to: "' + "x" * 200 + '"]')
    assert "x" * 201 not in result
