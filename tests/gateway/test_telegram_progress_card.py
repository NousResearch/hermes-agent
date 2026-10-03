"""Telegram plan-aware progress card for long-running turns (issue #124600).

RED-first: on unfixed main ``gateway.telegram_progress_card`` does not exist,
so every test here fails at import/collection time.
"""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.telegram_progress_card import (
    EDIT_FLOOR_SECONDS,
    FALLBACK_REFRESH_SECONDS,
    TelegramProgressCard,
    clean_commentary_segment,
    render_progress_card,
)


def test_edit_floor_and_fallback_defaults():
    """The 60s edit floor and 300s fallback are documented module constants."""
    assert EDIT_FLOOR_SECONDS == 60.0
    assert FALLBACK_REFRESH_SECONDS == 300.0


def test_clean_keeps_model_commentary():
    text = "I'll inspect the repo first, then run the tests."
    assert clean_commentary_segment(text) == text


def test_clean_drops_tool_telemetry_lines():
    seg = "🔍 web_search...\nI'll inspect the repo first."
    cleaned = clean_commentary_segment(seg)
    assert "web_search" not in cleaned
    assert "I'll inspect the repo first." in cleaned


def test_clean_drops_thinking_scratch_lines():
    seg = "💬 considering which tool to call next\nChecking the failing test now."
    cleaned = clean_commentary_segment(seg)
    assert "considering which tool" not in cleaned
    assert "Checking the failing test now." in cleaned


def test_clean_redacts_internal_paths():
    seg = (
        "The failure is in /home/operator/.config/hermes/state.db according to the log."
    )
    cleaned = clean_commentary_segment(seg)
    assert "/home/operator/.config/hermes/state.db" not in cleaned
    assert "state.db" in cleaned
    assert "The failure is in" in cleaned


def test_clean_silent_without_prose():
    assert clean_commentary_segment("🔍 web_search...") == ""
    assert clean_commentary_segment("/home/operator/.config/hermes/state.db") == ""
    assert clean_commentary_segment("   \n  ") == ""


def test_clean_keeps_urls_intact():
    """The path-redaction rule must not eat URL path segments (#124976)."""
    for text in (
        "Fetching https://docs.example.com/api/v2/reference now",
        "See https://example.com/docs/page for details.",
    ):
        cleaned = clean_commentary_segment(text)
        assert "https://" in cleaned
        assert cleaned == text


def test_clean_keeps_markdown_plan_lines():
    """A bulleted/heading plan is prose: the leading-symbol filter must not drop it,
    because ``observe`` pins the first surviving segment as the card's plan (#124976)."""
    cleaned = clean_commentary_segment("- Read the config\n- patch the handler")
    assert cleaned == "Read the config\npatch the handler"
    assert clean_commentary_segment("# Read the config") == "Read the config"
    # Emoji-prefixed status lines are still dropped.
    assert clean_commentary_segment("🔧 Fixing the handler") == ""


def test_silent_with_no_commentary():
    card = TelegramProgressCard()
    assert card.render(now=1000.0) is None
    assert render_progress_card(plan=None, now=None, elapsed_seconds=300.0) is None


def test_first_multiline_segment_pins_plan():
    card = TelegramProgressCard()
    assert (
        card.observe("Step 1: reproduce\nStep 2: fix\nStep 3: verify", now=1000.0)
        is True
    )
    text = card.render(now=1060.0)
    assert text is not None
    assert "Plan:" in text
    assert "Step 1: reproduce" in text
    assert "Now:" not in text


def test_later_segments_render_as_now():
    card = TelegramProgressCard()
    card.observe("Step 1: reproduce\nStep 2: fix", now=1000.0)
    card.observe("Reproduced locally, editing the guard now.", now=1010.0)
    text = card.render(now=1070.0)
    assert "Plan:" in text
    assert "Now: Reproduced locally, editing the guard now." in text


def test_first_send_happens_on_commentary_not_on_tick():
    """The card is sendable as soon as the first clean segment arrives."""
    card = TelegramProgressCard()
    assert card.should_send(now=1000.0) is False
    card.observe("Step 1: reproduce\nStep 2: fix", now=1000.0)
    assert card.should_send(now=1000.0) is True


def test_edit_floor_holds_rapid_updates():
    card = TelegramProgressCard()
    card.observe("Step 1: reproduce\nStep 2: fix", now=1000.0)
    card.mark_sent("msg-1", now=1000.0)
    card.observe("Still reproducing.", now=1010.0)
    assert card.due_for_refresh(now=1010.0) is False
    assert card.due_for_refresh(now=1060.0) is True


def test_fallback_refreshes_header_during_silence():
    """No new segments for 300s: the elapsed header still refreshes."""
    card = TelegramProgressCard()
    card.observe("Step 1: reproduce\nStep 2: fix", now=1000.0)
    card.mark_sent("msg-1", now=1000.0)
    before = card.render(now=1001.0)
    assert card.due_for_refresh(now=1299.0) is False
    assert card.due_for_refresh(now=1300.0) is True
    after = card.render(now=1300.0)
    assert after is not None and after != before


def _make_card_runner(monkeypatch, adapter, emits=(True, False)):
    """Bare GatewayRunner wired only for the Telegram progress-card path."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setenv("HERMES_TELEGRAM_PROGRESS_POLL", "0.01")
    monkeypatch.setenv("HERMES_TELEGRAM_PROGRESS_EDIT_FLOOR", "0.05")
    monkeypatch.setenv("HERMES_TELEGRAM_PROGRESS_FALLBACK", "0.2")
    runner = object.__new__(GatewayRunner)
    runner._draining = runner._restart_requested = False
    runner._sessions = {}
    runner._delivery_adapter_for = lambda source: adapter
    states = list(emits)

    def _emit(*a, **k):
        return states.pop(0) if states else False

    runner._should_emit_long_running_notification = _emit
    source = SimpleNamespace(
        chat_id="c1",
        platform=Platform.TELEGRAM,
        thread_id=None,
        scope_id=None,
        user_id=None,
    )
    ctx = TurnContext(source=source, session_key="sess")
    ctx.agent_holder[0] = MagicMock()
    ctx.telegram_progress_queue = queue.Queue()
    ctx._cleanup_progress = True
    ctx._status_thread_metadata = None
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    return runner, disp, ctx


@pytest.mark.asyncio
async def test_card_sends_on_first_commentary_and_tracks_cleanup(monkeypatch):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    runner, disp, ctx = _make_card_runner(monkeypatch, adapter)
    ctx.telegram_progress_queue.put_nowait("Step 1: reproduce\nStep 2: fix")

    await asyncio.wait_for(
        runner._run_agent_notify_long_running(disp, ctx, [None]),
        timeout=10,
    )

    adapter.send.assert_awaited_once()
    sent_text = adapter.send.await_args.args[1]
    assert "Plan:" in sent_text
    assert "card-1" in ctx._cleanup_msg_ids


@pytest.mark.asyncio
async def test_card_stays_silent_without_commentary(monkeypatch):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    runner, disp, ctx = _make_card_runner(
        monkeypatch, adapter, emits=(True, True, False)
    )

    await asyncio.wait_for(
        runner._run_agent_notify_long_running(disp, ctx, [None]),
        timeout=10,
    )

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert ctx._cleanup_msg_ids == []


@pytest.mark.asyncio
async def test_card_edits_single_card_in_place(monkeypatch):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="card-1")
    )
    runner, disp, ctx = _make_card_runner(
        monkeypatch,
        adapter,
        emits=(True, True, True, True, False),
    )
    ctx.telegram_progress_queue.put_nowait("Step 1: reproduce\nStep 2: fix")

    async def _run():
        task = asyncio.create_task(
            runner._run_agent_notify_long_running(disp, ctx, [None])
        )
        await asyncio.sleep(0.05)
        ctx.telegram_progress_queue.put_nowait("Reproduced locally, editing now.")
        await asyncio.wait_for(task, timeout=10)

    await _run()

    adapter.send.assert_awaited_once()
    assert adapter.edit_message.await_count >= 1
    for call in adapter.edit_message.await_args_list:
        assert call.args[1] == "card-1"
    assert "Now:" in adapter.edit_message.await_args.args[2]
