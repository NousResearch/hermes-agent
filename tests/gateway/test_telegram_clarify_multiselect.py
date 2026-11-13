"""Telegram multi-select clarify: toggle buttons, Send, and the typed fallback.

The clarify primitive already accepts a comma separated reply for ``multi_select`` entries; these
tests pin the inline keyboard path that previously resolved on the first tap.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_clarify_binding import ClarifyPrompt


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _clear():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _prompt(adapter, clarify_id, session_key="sk-multi", message_id=555):
    msg = MagicMock()
    msg.chat_id = 12345
    msg.message_id = message_id
    msg.text = "Pick"
    msg.chat.type = "private"
    adapter._clarify_state[clarify_id] = ClarifyPrompt.from_sent(session_key, msg)
    return msg


async def _tap(adapter, msg, data):
    query = AsyncMock()
    query.data = data
    query.message = msg
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())
    return query


class TestMultiSelectRender:
    def setup_method(self):
        _clear()

    @pytest.mark.asyncio
    async def test_multi_prompt_renders_hint_and_keeps_other(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = MagicMock()
        msg.message_id = 100
        adapter._bot.send_message = AsyncMock(return_value=msg)
        cm.register("cidM", "sk-m", "Pick", ["red", "green"], multi_select=True)

        result = await adapter.send_clarify("12345", "Pick", ["red", "green"], "cidM", "sk-m")

        assert result.success is True
        text = adapter._bot.send_message.call_args[1]["text"]
        assert "Select one or more" in text
        assert "2. green" in text
        assert "cidM" in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_single_select_prompt_has_no_hint(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = MagicMock()
        msg.message_id = 101
        adapter._bot.send_message = AsyncMock(return_value=msg)
        cm.register("cidS", "sk-s", "Pick", ["red", "green"], multi_select=False)

        await adapter.send_clarify("12345", "Pick", ["red", "green"], "cidS", "sk-s")

        assert "Select one or more" not in adapter._bot.send_message.call_args[1]["text"]


class TestMultiSelectCallbacks:
    def setup_method(self):
        _clear()

    @pytest.mark.asyncio
    async def test_toggle_then_send_resolves_with_both_labels(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidM")
        cm.register("cidM", "sk-multi", "Pick", ["red", "green", "blue"], multi_select=True)

        await _tap(adapter, msg, "cl:cidM:t0")
        await _tap(adapter, msg, "cl:cidM:t2")
        assert adapter._clarify_multi["cidM"] == {0, 2}
        with cm._lock:
            entry = cm._entries["cidM"]
        assert not entry.event.is_set()

        await _tap(adapter, msg, "cl:cidM:go")

        with cm._lock:
            entry = cm._entries["cidM"]
        # The tool decodes a JSON array into one answer per pick; "red, blue" would read as one
        # typed answer (#154195).
        from tools.clarify_tool import _clean_answer
        assert _clean_answer(entry.response, True) == ["red", "blue"]
        assert entry.event.is_set()
        assert "cidM" not in adapter._clarify_state
        assert "cidM" not in adapter._clarify_multi

    @pytest.mark.asyncio
    async def test_send_strips_the_recommended_label(self):
        from tools import clarify_gateway as cm
        from tools.clarify_tool import _clean_answer, mark_recommended
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidR")
        cm.register("cidR", "sk-multi", "Pick", mark_recommended(["red", "green"]), multi_select=True)

        await _tap(adapter, msg, "cl:cidR:t0")
        await _tap(adapter, msg, "cl:cidR:t1")
        await _tap(adapter, msg, "cl:cidR:go")

        with cm._lock:
            assert _clean_answer(cm._entries["cidR"].response, True) == ["red", "green"]

    @pytest.mark.asyncio
    async def test_refused_toggle_edit_keeps_the_visible_selection(self):
        """A pick Telegram refused to show must not be submitted by Send."""
        from telegram.error import BadRequest
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidF")
        cm.register("cidF", "sk-multi", "Pick", ["red", "green", "blue"], multi_select=True)

        await _tap(adapter, msg, "cl:cidF:t0")
        refused = AsyncMock(side_effect=BadRequest("Bad Request: message can't be edited"))
        query = AsyncMock()
        query.data, query.message = "cl:cidF:t2", msg
        query.from_user = MagicMock(id="777", first_name="Tester")
        query.edit_message_reply_markup = refused
        update = MagicMock(callback_query=query)
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())
        query.answer.assert_called_once()
        await _tap(adapter, msg, "cl:cidF:go")

        from tools.clarify_tool import _clean_answer
        with cm._lock:
            assert _clean_answer(cm._entries["cidF"].response, True) == ["red"]

    @pytest.mark.asyncio
    async def test_tap_on_an_expired_card_retires_it(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidX")
        cm.register("cidX", "sk-multi", "Pick", ["red", "green"], multi_select=True)
        cm.resolve_gateway_clarify("cidX", "typed answer")  # answered elsewhere (typed reply / timeout)

        query = await _tap(adapter, msg, "cl:cidX:t0")

        assert "cidX" not in adapter._clarify_state
        assert query.edit_message_text.call_args.kwargs["reply_markup"] is None
        assert "expired" in query.edit_message_text.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_session_reset_during_toggle_edit_retires_prompt(self):
        """A reset while Telegram is editing must not report a stale checkbox selection."""
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidRace")
        cm.register("cidRace", "sk-multi", "Pick", ["red", "green"], multi_select=True)

        async def reset_while_editing(**_kwargs):
            assert cm.clear_session("sk-multi") == 1

        query = AsyncMock()
        query.data, query.message = "cl:cidRace:t0", msg
        query.from_user = MagicMock(id="777", first_name="Tester")
        query.edit_message_reply_markup = AsyncMock(side_effect=reset_while_editing)
        update = MagicMock(callback_query=query)
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        assert "cidRace" not in adapter._clarify_state
        assert "cidRace" not in adapter._clarify_multi
        assert query.edit_message_text.call_args.kwargs["reply_markup"] is None
        assert "expired" in query.edit_message_text.call_args.kwargs["text"]
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_off_before_send(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidT")
        cm.register("cidT", "sk-multi", "Pick", ["red", "green"], multi_select=True)

        await _tap(adapter, msg, "cl:cidT:t1")
        await _tap(adapter, msg, "cl:cidT:t1")
        assert adapter._clarify_multi["cidT"] == set()
        with cm._lock:
            assert not cm._entries["cidT"].event.is_set()

    @pytest.mark.asyncio
    async def test_send_without_selection_keeps_prompt_armed(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidE")
        cm.register("cidE", "sk-multi", "Pick", ["red", "green"], multi_select=True)

        query = await _tap(adapter, msg, "cl:cidE:go")

        with cm._lock:
            entry = cm._entries["cidE"]
        assert entry.response is None
        assert not entry.event.is_set()
        assert "cidE" in adapter._clarify_state
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_legacy_numeric_tap_still_resolves_single_choice(self):
        from tools import clarify_gateway as cm
        adapter = _make_adapter()
        msg = _prompt(adapter, "cidL", "sk-single")
        cm.register("cidL", "sk-single", "Pick", ["red", "green"], multi_select=False)

        await _tap(adapter, msg, "cl:cidL:1")

        with cm._lock:
            entry = cm._entries["cidL"]
        assert entry.response == "green"
        assert entry.event.is_set()


class _Card:
    """Holds the markup Telegram shows; the first edit is lost (sent, timed out, never applied)."""

    def __init__(self, shown):
        self.shown, self.calls = shown, 0

    async def edit(self, reply_markup=None, **_kwargs):
        from telegram.error import BadRequest, TimedOut
        self.calls += 1
        markup = reply_markup
        if self.calls == 1:
            raise TimedOut("Timed out")
        if markup == self.shown:
            raise BadRequest("Bad Request: message is not modified: specified new message content and "
                             "reply markup are exactly the same as a current content and reply markup")
        self.shown = markup


@pytest.mark.asyncio
@pytest.mark.parametrize("pre_ticked, expected", [(False, None), (True, ["red"])])
async def test_retap_after_lost_toggle_edit_adopts_what_the_card_shows(pre_ticked, expected):
    """A re-tap after a lost edit gets "not modified": the card shows the requested ticks, so Send
    must submit exactly those (lost tick: nothing; lost untick: still red)."""
    from plugins.platforms.telegram import adapter as tg
    with patch.object(tg, "InlineKeyboardButton", lambda text, callback_data: (text, callback_data)), \
            patch.object(tg, "InlineKeyboardMarkup", lambda rows: tuple(map(tuple, rows))):
        await _retap_after_lost_edit(pre_ticked, expected)


async def _retap_after_lost_edit(pre_ticked, expected):
    from plugins.platforms.telegram.telegram_clarify_multi import multi_keyboard
    from tools import clarify_gateway as cm
    _clear()
    adapter = _make_adapter()
    msg = _prompt(adapter, "cidLost")
    cm.register("cidLost", "sk-multi", "Pick", ["red", "green"], multi_select=True)
    if pre_ticked:
        await _tap(adapter, msg, "cl:cidLost:t0")
    card = _Card(multi_keyboard(adapter, "cidLost", 2, {0} if pre_ticked else set()))

    for _ in range(2):  # the lost edit, then the user's corrective re-tap
        query = AsyncMock(data="cl:cidLost:t0", message=msg, from_user=MagicMock(id="777", first_name="T"))
        query.edit_message_reply_markup.side_effect = card.edit
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    assert "Selected" in query.answer.call_args.kwargs["text"]
    await _tap(adapter, msg, "cl:cidLost:go")

    from tools.clarify_tool import _clean_answer
    with cm._lock:
        response = cm._entries["cidLost"].response
    assert (None if response is None else _clean_answer(response, True)) == expected
