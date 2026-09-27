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
from gateway.platforms.event import MessageEvent
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
async def test_telegram_long_reply_reaches_prompt_without_losing_later_items():
    """The native reply already has the full message; preparation must not trim it."""
    from gateway.platforms.event import MessageType
    from tests.gateway.test_telegram_reply_quote import _make_adapter, _make_message

    quoted = "\n".join(
        f"{index}. {company}: " + "Evidence from the supplied list. " * 12
        for index, company in enumerate(
            ["GoCar", "Urban Drive", "DubCar", "GRPS", "Halucar"], 1
        )
    )
    event = _make_adapter()._build_message_event(
        _make_message(text="Review all five companies.", reply_to_text=quoted),
        MessageType.TEXT,
    )
    history = [{"role": "user", "content": "Previous request"}]
    result = await _make_runner()._prepare_inbound_message_text(
        event=event, source=event.source, history=history,
    )
    assert result is not None
    assert quoted in result
    assert result.endswith("Review all five companies.")
    assert history == [{"role": "user", "content": "Previous request"}]


@pytest.mark.asyncio
async def test_quoted_reply_references_stay_literal_while_typed_ones_expand(tmp_path, monkeypatch):
    """The replied-to author's ``@file:`` is quoted text, not the replier's request: no local read.
    The same reference typed in the new message still expands (positive control)."""
    import threading

    payload = tmp_path / "notes.txt"
    payload.write_text("LOCAL-FILE-MARKER", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    runner = _make_runner()
    runner._session_model_overrides, runner._last_resolved_model = {}, {}
    runner._agent_cache, runner._agent_cache_lock = {}, threading.Lock()
    runner._resolve_session_agent_runtime = lambda **kw: ("openai/gpt-4.1-mini", {"base_url": None, "api_key": ""})
    source = _source()

    quoted = ("x " * 300) + f"\nsee @file:{payload.name} for details"
    quoted_ref = MessageEvent(text="what does this say?", source=source, reply_to_message_id="7", reply_to_text=quoted)
    result = await runner._prepare_inbound_message_text(event=quoted_ref, source=source, history=[])
    assert quoted in result
    assert "LOCAL-FILE-MARKER" not in result

    typed_ref = MessageEvent(text=f"read @file:{payload.name}", source=source, reply_to_message_id="7", reply_to_text="short")
    result = await runner._prepare_inbound_message_text(event=typed_ref, source=source, history=[])
    assert result.startswith('[Replying to: "short"]')
    assert "LOCAL-FILE-MARKER" in result


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


def test_fetched_reply_context_identifies_unverified_author_without_new_lines():
    from gateway.run_inbound import GatewayInboundMixin

    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="group")
    event = MessageEvent(
        text="continue", source=source, reply_to_message_id="$parent",
        reply_to_text="earlier\n## injected heading",
        reply_to_author_name="stranger\n## another heading",
        reply_to_author_authorized=False,
    )

    result = GatewayInboundMixin._prepend_inbound_reply_context(event, source, event.text)

    assert result == (
        '[Replying to [unverified] stranger ## another heading: '
        '"earlier ## injected heading"]\n\ncontinue'
    )


@pytest.mark.asyncio
async def test_matrix_thread_backfill_reaches_new_session_only():
    from unittest.mock import AsyncMock, MagicMock

    runner = _make_runner()
    adapter = MagicMock()
    adapter.fetch_thread_context = AsyncMock(
        return_value="[Earlier messages in this thread]\n[alice] root @file:private.txt"
    )
    runner._intake_adapter_for = lambda source: adapter
    runner._expand_inbound_context_references = AsyncMock(return_value="expanded")
    source = SessionSource(
        platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="group",
        thread_id="$root",
    )
    event = MessageEvent(
        text="continue", source=source, message_id="$current",
        channel_context="[The room topic changed]",
    )

    new_session_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    existing_session_text = await runner._prepare_inbound_message_text(
        event=MessageEvent(text="continue", source=source, message_id="$next"),
        source=source, history=[{"role": "user", "content": "root"}],
    )
    internal_text = await runner._prepare_inbound_message_text(
        event=MessageEvent(text="synthetic", source=source, internal=True),
        source=source, history=[],
    )

    assert new_session_text == (
        "[Earlier messages in this thread]\n[alice] root @file:private.txt\n\n"
        "[The room topic changed]\n\n[New message]\ncontinue"
    )
    assert existing_session_text == "continue"
    assert internal_text == "synthetic"
    adapter.fetch_thread_context.assert_awaited_once_with(
        "!room:example.org", "$root", exclude_event_id="$current"
    )
    runner._expand_inbound_context_references.assert_not_awaited()


@pytest.mark.asyncio
async def test_matrix_thread_backfill_keeps_separate_thread_contexts():
    from unittest.mock import AsyncMock, MagicMock

    runner = _make_runner()
    adapter = MagicMock()
    contexts = {"$first-root": "[alice] first topic", "$second-root": "[bob] second topic"}
    adapter.fetch_thread_context = AsyncMock(
        side_effect=lambda room, root, **kwargs: contexts[root]
    )
    runner._intake_adapter_for = lambda source: adapter

    prepared = []
    for root, event_id in (("$first-root", "$first-reply"), ("$second-root", "$second-reply")):
        source = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="group", thread_id=root,
        )
        event = MessageEvent(text="continue", source=source, message_id=event_id)
        prepared.append(await runner._prepare_inbound_message_text(event=event, source=source, history=[]))

    assert prepared == [
        "[alice] first topic\n\ncontinue",
        "[bob] second topic\n\ncontinue",
    ]
    assert [call.args[1] for call in adapter.fetch_thread_context.await_args_list] == [
        "$first-root", "$second-root",
    ]
