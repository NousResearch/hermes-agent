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


@pytest.mark.asyncio
async def test_deferred_expansion_keeps_reply_quote_literal_and_expands_typed_body():
    """The composed deferred path must isolate the user's body from the reply pointer."""
    import asyncio

    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    runner = _make_runner()
    source = _source()
    calls = []

    async def expand(_source, _session_key, message, *, turn_route, warning_sender):
        calls.append((message, turn_route, warning_sender))
        return message + "\n\nEXPANDED-CONTEXT"

    runner._expand_inbound_context_references = expand  # type: ignore[method-assign]
    quoted = "Please inspect @file:quoted.txt; it belongs to the other author."

    quoted_event = MessageEvent(
        text="what does this say?", source=source, reply_to_message_id="7", reply_to_text=quoted,
    )
    quoted_staged = await runner._prepare_inbound_message_text(
        event=quoted_event, source=source, history=[], defer_context_references=True,
    )
    assert quoted_staged is not None
    quoted_ref = getattr(quoted_event, "_gateway_context_reference_message")
    quoted_ctx = TurnContext(
        source=source, session_key="session", message=quoted_staged,
        persist_user_message=quoted_staged, context_reference_message=quoted_ref,
    )
    assert await asyncio.to_thread(
        TurnRunner(runner, quoted_ctx)._prepare_context_references_for_realized_route,
        {"model": "selected", "runtime": {}},
    )
    assert calls == []
    assert isinstance(quoted_ctx.message, str)
    assert isinstance(quoted_ctx.persist_user_message, str)
    assert quoted in quoted_ctx.message
    assert "EXPANDED-CONTEXT" not in quoted_ctx.message
    assert "EXPANDED-CONTEXT" not in quoted_ctx.persist_user_message

    typed_event = MessageEvent(
        text="read @file:notes.txt", source=source, reply_to_message_id="8", reply_to_text="short",
    )
    typed_staged = await runner._prepare_inbound_message_text(
        event=typed_event, source=source, history=[], defer_context_references=True,
    )
    assert typed_staged is not None
    typed_ref = getattr(typed_event, "_gateway_context_reference_message")
    typed_ctx = TurnContext(
        source=source, session_key="session", message=typed_staged,
        persist_user_message=typed_staged, context_reference_message=typed_ref,
    )
    assert await asyncio.to_thread(
        TurnRunner(runner, typed_ctx)._prepare_context_references_for_realized_route,
        {"model": "selected", "runtime": {}},
    )
    assert [call[0] for call in calls] == ["read @file:notes.txt"]
    assert isinstance(typed_ctx.message, str)
    assert isinstance(typed_ctx.persist_user_message, str)
    assert typed_ctx.message.startswith('[Replying to: "short"]')
    assert typed_ctx.message.endswith("read @file:notes.txt\n\nEXPANDED-CONTEXT")
    assert typed_ctx.persist_user_message.endswith("read @file:notes.txt\n\nEXPANDED-CONTEXT")




@pytest.mark.asyncio
async def test_run_agent_inner_forwards_authored_body_into_turn_context():
    """The local branch must hand the authored body to the real TurnContext, not drop it."""
    runner = object.__new__(GatewayRunner)
    runner._get_proxy_url = lambda: None
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="review-chat")
    disp = runner._RunAgentDisplay(
        user_config={}, platform_key="telegram", needs_progress_queue=False,
        resolve_display_setting=lambda *args: False,
    )
    runner._run_agent_display_settings = lambda source: disp
    body = "Please summarize this."
    decorated = "[Replying to earlier discussion]\n\n" + body
    observed = []

    class StopBeforeWorker(Exception):
        pass

    def capture(ctx, *args):
        observed.append(ctx)
        raise StopBeforeWorker

    runner._run_agent_bind_turn_wiring = capture
    with pytest.raises(StopBeforeWorker):
        await runner._run_agent_inner(
            decorated, "", [], source, "physical", session_key="review",
            context_reference_message=body,
        )
    assert len(observed) == 1
    assert observed[0].message == decorated
    assert observed[0].context_reference_message == body
