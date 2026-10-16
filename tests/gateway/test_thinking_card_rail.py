"""Thinking card (per-point rows): narration flows into one per-turn card, never thread replies.

Behavior contracts under test:
- REV-1: no reasoning content (think-tagged blocks) reaches the card feed; tag-only bursts
  never render.
- The interim-narration sink consumes bursts when the card is on and stays inert when off.
- Each narration burst becomes its OWN task_update row; rows fall back to one editable text
  message on native failure; an egress decline is terminal (no laundering).
- The anti-dup delta contract: a row (with its details) is appended EXACTLY ONCE -- Slack
  concatenates details when a row is re-sent across frames (live-proven on the probe card).
  Reopening a sealed stream re-seeds the full projection into the fresh stream.
- Tool and thinking streams are separate slots of the same thread.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent.history_commentary import visible_commentary
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.run_turn_runner_thinking import (
    BURST_SEPARATOR,
    MAX_FEED_CHARS,
    _ThinkingCardState,
    _TRUNCATION_ROW,
    sanitize_thinking_burst,
)
from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


# ── REV-1 sink guard ─────────────────────────────────────────────────────────

def test_sanitize_strips_reasoning_blocks():
    text = "Checking the config now.\n<reasoning>SECRET-TOKEN-XYZ</reasoning>\nDone with that."
    cleaned = sanitize_thinking_burst(text)
    assert cleaned is not None
    assert "SECRET-TOKEN-XYZ" not in cleaned
    assert "Checking the config now." in cleaned


def test_sanitize_drops_tag_only_burst():
    assert sanitize_thinking_burst("<thinking>hidden reasoning only</thinking>") is None
    assert sanitize_thinking_burst("   ") is None
    assert sanitize_thinking_burst(None) is None


def test_sanitize_matches_live_commentary_contract():
    # Same think-strip + redaction as the interim lane's own visible_commentary.
    text = "Prose before.\n<reasoning>inner</reasoning>Prose after."
    assert sanitize_thinking_burst(text) == (visible_commentary(text).strip() or None)


def test_sanitize_redacts_secrets():
    text = "The key is sk-live-abcdef1234567890abcd and it works."
    cleaned = sanitize_thinking_burst(text) or ""
    assert "sk-live-abcdef1234567890abcd" not in cleaned


# ── feed assembly ────────────────────────────────────────────────────────────

def test_feed_preserves_order_and_separator():
    st = _ThinkingCardState(adapter=None)
    st.append("first burst")
    st.append("second burst")
    assert st.feed_text == "first burst" + BURST_SEPARATOR + "second burst"


def test_feed_cap_is_visible_truncation():
    st = _ThinkingCardState(adapter=None)
    st.append("x" * (MAX_FEED_CHARS + 500))
    text = st.feed_text
    assert len(text) <= MAX_FEED_CHARS + 200
    assert text.startswith("x" * 100)  # chronological head kept
    assert "truncated" in text  # never silent (acceptance criterion 2)
    # The marker is its own row; further bursts are dropped, never silently mixed in.
    assert st.feed_parts[-1] == _TRUNCATION_ROW
    assert st.append("later burst") is False
    assert "later burst" not in st.feed_text


# ── interim-narration sink ───────────────────────────────────────────────────

def _route_ctx(cards_on: bool):
    q = queue.Queue()
    ctx = SimpleNamespace(_native_slack_thinking_cards=cards_on, thinking_queue=q)
    return ctx, q


def test_route_interim_narration_on_and_off():
    ctx_on, q_on = _route_ctx(True)
    runner = SimpleNamespace(_ctx=ctx_on)
    assert TurnRunner._route_interim_narration(runner, "narration burst") is True
    assert q_on.get_nowait() == {"type": "thinking", "text": "narration burst"}

    ctx_off, q_off = _route_ctx(False)
    assert TurnRunner._route_interim_narration(SimpleNamespace(_ctx=ctx_off), "x") is False
    assert q_off.empty()


def test_route_skips_already_streamed_but_still_consumes():
    # already_streamed text belongs to the answer stream; the sink must not double-render it,
    # but must keep the interim lane from sending a reply.
    ctx, q = _route_ctx(True)
    runner = SimpleNamespace(_ctx=ctx)
    assert TurnRunner._route_interim_narration(runner, "answer text", already_streamed=True) is True
    assert q.empty()


# ── drain: cumulative feed, fallbacks ────────────────────────────────────────

def _drain_ctx(q, adapter):
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="D1"),
        thinking_queue=q,
        _native_slack_thinking_cards=True,
        _progress_reply_to="100.1",
        _progress_metadata={"thread_id": "100.1"},
        _cleanup_progress=False,
        _run_still_current=lambda: not q.empty(),
        agent_holder=[None],
    )
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
    return TurnRunner(runner, ctx)


def _thinking_adapter(**kwargs):
    adapter = SimpleNamespace(
        name="slack",
        send_thinking_card_progress=AsyncMock(
            return_value=kwargs.get("card_result", SendResult(success=True, message_id="900.1"))),
        stop_thinking_card_progress=AsyncMock(return_value=None),
        send=AsyncMock(return_value=kwargs.get("send_result", SendResult(success=True, message_id="900.2"))),
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id="900.2")),
    )
    return adapter


@pytest.mark.asyncio
async def test_drain_publishes_cumulative_feed_and_stops():
    adapter = _thinking_adapter()
    q = queue.Queue()
    q.put({"type": "thinking", "text": "first burst"})
    q.put({"type": "thinking", "text": "second burst"})
    await _drain_ctx(q, adapter).send_thinking_card_messages()

    assert adapter.send_thinking_card_progress.await_count == 2
    first = adapter.send_thinking_card_progress.await_args_list[0].kwargs["bursts"]
    second = adapter.send_thinking_card_progress.await_args_list[1].kwargs["bursts"]
    assert first == ["first burst"]
    assert second == ["first burst", "second burst"]
    adapter.stop_thinking_card_progress.assert_awaited_once()
    adapter.send.assert_not_awaited()  # no thread replies (Option A)


@pytest.mark.asyncio
async def test_drain_applies_reasoning_guard():
    adapter = _thinking_adapter()
    q = queue.Queue()
    q.put({"type": "thinking", "text": "Visible work.\n<reasoning>LEAK-ME</reasoning>"})
    q.put({"type": "thinking", "text": "<thinking>all hidden</thinking>"})
    await _drain_ctx(q, adapter).send_thinking_card_messages()

    assert adapter.send_thinking_card_progress.await_count == 1
    bursts = adapter.send_thinking_card_progress.await_args.kwargs["bursts"]
    joined = BURST_SEPARATOR.join(bursts)
    assert "LEAK-ME" not in joined
    assert "Visible work." in joined
    # tag-only burst never rendered
    assert "all hidden" not in joined


@pytest.mark.asyncio
async def test_drain_falls_back_to_one_editable_message():
    adapter = _thinking_adapter(card_result=SendResult(success=False, error="boom"))
    q = queue.Queue()
    q.put({"type": "thinking", "text": "first burst"})
    q.put({"type": "thinking", "text": "second burst"})
    await _drain_ctx(q, adapter).send_thinking_card_messages()

    # Both publishes failed natively -> ONE editable text message, updated in place.
    assert adapter.send.await_count == 1
    edited = [c for c in adapter.edit_message.await_args_list]
    assert len(edited) == 1
    assert edited[0].kwargs["content"] == "first burst" + BURST_SEPARATOR + "second burst"


@pytest.mark.asyncio
async def test_drain_egress_decline_is_terminal(monkeypatch):
    import gateway.relay.egress as egress

    monkeypatch.setattr(egress, "declined_send", lambda result: True)
    adapter = _thinking_adapter()
    q = queue.Queue()
    q.put({"type": "thinking", "text": "first burst"})
    q.put({"type": "thinking", "text": "second burst"})
    await _drain_ctx(q, adapter).send_thinking_card_messages()

    # A refusal must not be laundered into plain-text delivery (P5(b) semantics).
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()


# ── adapter wire shape (per-point rows) ─────────────────────────────────────

def _slack_adapter(monkeypatch):
    adapter = SlackAdapter(PlatformConfig(extra={"reply_in_thread": False}))
    client = SimpleNamespace(api_call=AsyncMock(side_effect=lambda *a, **kw: {"ts": "900.1"}))
    monkeypatch.setattr(adapter, "_get_client", Mock(return_value=client))
    adapter._app = object()
    return adapter, client


@pytest.mark.asyncio
async def test_thinking_card_wire_shape(monkeypatch):
    adapter, client = _slack_adapter(monkeypatch)
    md = {"thread_ts": "100.1"}
    await adapter.send_thinking_card_progress("D1", ["feed one"], reply_to=None, metadata=md)
    await adapter.send_thinking_card_progress("D1", ["feed one", "feed two"], reply_to=None, metadata=md)
    await adapter.stop_thinking_card_progress("D1", reply_to=None, metadata=md)

    calls = [c.args[0] for c in client.api_call.await_args_list]
    assert calls[0] == "chat.startStream"
    assert calls[-1] == "chat.stopStream"
    start = client.api_call.await_args_list[0].kwargs["json"]
    # Plan mode: the title survives the turn seal (live-verified, probe P); timeline folds it away.
    assert start["task_display_mode"] == "plan"
    assert start["thread_ts"] == "100.1"

    # Anti-dup delta contract: frame 1 = row 0 only, frame 2 = row 1 only.
    # A burst's details appear EXACTLY ONCE across frames -- Slack concatenates
    # details when a row is re-sent (live-proven duplication).
    appends = [c.kwargs["json"] for c in client.api_call.await_args_list if c.args[0] == "chat.appendStream"]
    assert len(appends) == 3  # two arrival frames, then the seal's completion flip
    for body in appends:
        assert "markdown_text" not in body
        assert "chunks" in body
    arrival = [chunk for body in appends[:2] for chunk in body["chunks"] if chunk["type"] == "task_update"]
    flip = [chunk for chunk in appends[2]["chunks"] if chunk["type"] == "task_update"]
    assert [r["id"] for r in arrival] == ["thinking-0", "thinking-1"]
    assert [r["details"] for r in arrival] == ["feed one", "feed two"]
    assert all(r["status"] == "in_progress" for r in arrival)  # icon stays in-progress until seal
    assert arrival[0]["title"] == "feed one"  # point label = paragraph head, 256-capped
    # Seal flips rows complete via DETAILS-LESS re-sends (Row B keeps the text;
    # a with-details re-send CONCATENATES it -- Row A, status probe 3).
    assert [r["id"] for r in flip] == ["thinking-0", "thinking-1"]
    assert all(r["status"] == "complete" for r in flip)
    assert all("details" not in r for r in flip)
    # Arrival frames carry the plan title; the seal flip is rows-only.
    for body in appends[:2]:
        assert body["chunks"][0] == {"type": "plan_update", "title": body["chunks"][0]["title"]}
    assert all(c["type"] == "task_update" for c in appends[2]["chunks"])
    assert appends[0]["chunks"][0]["title"] != "Hermes is working"
    assert not adapter._native_task_card_streams


@pytest.mark.asyncio
async def test_card_title_uses_bot_display_name(monkeypatch):
    # fhu's final ask: the card title is '[bot name] is thinking' -- whichever bot
    # is actually serving the turn, resolved via the shared display-name path.
    adapter, client = _slack_adapter(monkeypatch)
    md = {"thread_ts": "100.1"}
    adapter._bot_user_id = "U-BOT"
    monkeypatch.setattr(adapter, "_resolve_user_name", AsyncMock(return_value="Ji Youyan"))
    await adapter.send_thinking_card_progress("D1", ["burst"], reply_to=None, metadata=md)
    appends = [c.kwargs["json"] for c in client.api_call.await_args_list if c.args[0] == "chat.appendStream"]
    assert appends[0]["chunks"][0]["title"] == "Ji Youyan is thinking"

    # Fallback chain never breaks a send: unresolvable name -> auth handle -> "Hermes".
    adapter2, client2 = _slack_adapter(monkeypatch)
    monkeypatch.setattr(adapter2, "_resolve_user_name", AsyncMock(return_value="U-BOT"))
    adapter2._bot_user_id = "U-BOT"
    await adapter2.send_thinking_card_progress("D1", ["burst"], reply_to=None, metadata=md)
    appends2 = [c.kwargs["json"] for c in client2.api_call.await_args_list if c.args[0] == "chat.appendStream"]
    assert appends2[0]["chunks"][0]["title"] == "Hermes is thinking"


def test_burst_label_is_paragraph_head_capped_at_256():
    # Approved mockup rule: label = the first <title cap> characters of the PARAGRAPH
    # (not just its first line); cut labels get an ellipsis; details keep the whole burst.
    multiline = "first line\nsecond line continues"
    assert SlackAdapter._thinking_burst_chunk(multiline, 0)["title"] == multiline
    long_burst = "x" * 300
    chunk = SlackAdapter._thinking_burst_chunk(long_burst, 0)
    assert chunk["title"] == "x" * 255 + "…"
    assert len(chunk["title"]) == 256
    assert chunk["details"] == long_burst


@pytest.mark.asyncio
async def test_thinking_card_reopen_reseeds_full_projection(monkeypatch):
    # A stream sealed server-side (~5m20s) reopens as a FRESH card: the reopen frame
    # must carry every row again (the fresh stream has no concat history), while the
    # normal delta frames never re-send one.
    adapter, client = _slack_adapter(monkeypatch)
    md = {"thread_ts": "100.1"}

    class _Sealed(Exception):
        response = {"error": "message_not_in_streaming_state"}

    appends_seen = {"n": 0}

    async def api_call(method, json=None, **kw):
        if method == "chat.appendStream":
            appends_seen["n"] += 1
            if appends_seen["n"] == 2:
                raise _Sealed()
        return {"ts": "900.2"}

    client.api_call = AsyncMock(side_effect=api_call)
    await adapter.send_thinking_card_progress("D1", ["b one"], reply_to=None, metadata=md)
    await adapter.send_thinking_card_progress("D1", ["b one", "b two"], reply_to=None, metadata=md)

    calls = [c.args[0] for c in client.api_call.await_args_list]
    assert calls.count("chat.startStream") == 2  # reopened exactly once
    reseed = client.api_call.await_args_list[-1].kwargs["json"]["chunks"]
    rows = [c for c in reseed if c["type"] == "task_update"]
    assert [r["details"] for r in rows] == ["b one", "b two"]  # fresh stream seeded fully
    assert all(r["status"] == "in_progress" for r in rows)  # turn is still running


@pytest.mark.asyncio
async def test_tool_and_thinking_streams_are_separate_slots(monkeypatch):
    adapter, client = _slack_adapter(monkeypatch)
    md = {"thread_ts": "100.1"}
    await adapter.send_native_task_card_progress(
        "D1", tasks=[{"id": "t1", "title": "terminal", "status": "in_progress"}],
        reply_to=None, metadata=md)
    await adapter.send_thinking_card_progress("D1", ["thinking feed"], reply_to=None, metadata=md)

    slots = sorted(key[3] for key in adapter._native_task_card_streams)
    assert slots == ["thinking", "tools"]
    # Stopping one card does not stop the other.
    await adapter.stop_thinking_card_progress("D1", reply_to=None, metadata=md)
    assert [key[3] for key in adapter._native_task_card_streams] == ["tools"]


def test_thinking_cards_enabled_config():
    assert SlackAdapter(PlatformConfig(extra={"thinking_cards": True})).thinking_cards_enabled() is True
    assert SlackAdapter(PlatformConfig(extra={"thinkingCards": "yes"})).thinking_cards_enabled() is True
    assert SlackAdapter(PlatformConfig(extra={"thinking_cards": False})).thinking_cards_enabled() is False
    assert SlackAdapter(PlatformConfig(extra={})).thinking_cards_enabled() is False
