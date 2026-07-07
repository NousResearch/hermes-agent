"""Integration tests for Telegram guest mode reply flow (Bot API 10.0).

This covers the text-only "Branch 1" foundation: a stub fires unconditionally
on send_typing, the reply is buffered, and on_processing_complete edits the
stub with the final text via editMessageText(inline_message_id, ...).

Media delivery (deliver_<token> Branch 2/3 dispatch, the media-button OPC
path) is out of scope here and lands in a follow-up PR.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Telegram library mock
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> TelegramAdapter:
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {"guest_mode": True}
    adapter = TelegramAdapter(cfg)
    adapter._bot = MagicMock()
    adapter._bot.do_api_request = AsyncMock(return_value={"inline_message_id": "imi_abc"})
    return adapter


def _register_guest_chat(adapter: TelegramAdapter, chat_id="42") -> None:
    """Pre-populate state as if branch-1 processing started."""
    adapter._pending_guest_queries[chat_id] = "gqid_test"
    adapter._guest_only_chats.add(chat_id)
    adapter._guest_inline_message_ids[chat_id] = False  # slot open


# ---------------------------------------------------------------------------
# Branch 1 — stub fires unconditionally on send_typing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_stub_fires_on_send_typing():
    """Stub fires on send_typing for any query — no content classification."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)

    with patch.object(adapter, "_guest_fire_text_stub", new_callable=AsyncMock) as mock_stub:
        await adapter.send_typing("42")
        mock_stub.assert_awaited_once_with("42")


@pytest.mark.asyncio
async def test_branch1_stub_fires_for_media_keyword_query():
    """No classification suppression — stub fires even for 'download this video'."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)

    with patch.object(adapter, "_guest_fire_text_stub", new_callable=AsyncMock) as mock_stub:
        await adapter.send_typing("42")
        mock_stub.assert_awaited_once_with("42")


# ---------------------------------------------------------------------------
# Branch 1 OPC — text result: editMessageText on imi
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_opc_text_result_edits_stub():
    """OPC text result: editMessageText(inline_message_id, final_text)."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_reply_buffer["42"] = "Here is your answer."
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"
    outcome = ProcessingOutcome.SUCCESS

    await adapter.on_processing_complete(event, outcome)

    calls = adapter._bot.do_api_request.await_args_list
    methods = [c.args[0] for c in calls]
    assert "editMessageText" in methods

    edit_call = next(c for c in calls if c.args[0] == "editMessageText")
    kw = edit_call.kwargs["api_kwargs"]
    assert kw["inline_message_id"] == "imi_abc"
    assert "Here is your answer." in kw["text"]


# ---------------------------------------------------------------------------
# Caller authorization gate (fail-closed) — _handle_guest_message_update
#
# The human caller for a guest update is in raw guest_bot_caller_user, not
# from_user. Unauthorized callers are denied before any state registration or
# API call; empty allowlist / missing caller => deny.
# ---------------------------------------------------------------------------

import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402


def _make_guest_update(caller_id="999", chat_id="42", text="@bot hi",
                       gqid="gq1", update_id=1,
                       caller_key="guest_bot_caller_user"):
    raw = {"guest_query_id": gqid, "chat": {"id": int(chat_id), "type": "supergroup"}, "text": text}
    if caller_id is not None:
        raw[caller_key] = {"id": int(caller_id), "username": "someone"}
    update = MagicMock()
    update.api_kwargs = {"guest_message": raw}
    update.update_id = update_id
    return update, raw


def _fake_guest_msg(chat_id="42", text="@bot hi", chat_type="supergroup"):
    msg = MagicMock()
    msg.text = text
    msg.caption = None
    msg.chat.id = int(chat_id)
    msg.chat.type = chat_type
    msg.from_user = None  # guest messages carry the caller in guest_bot_caller_user
    return msg


@pytest.mark.asyncio
async def test_guest_caller_unauthorized_is_denied():
    """A caller not in any allowlist is denied before state/API — reads the id
    from guest_bot_caller_user (from_user is None here)."""
    adapter = _make_adapter()
    update, _ = _make_guest_update(caller_id="999")
    msg = _fake_guest_msg()

    with patch.object(_tg_adapter_mod.Message, "de_json", return_value=msg), \
         patch.object(adapter, "_is_callback_user_authorized", return_value=False) as mock_auth, \
         patch.object(adapter, "_should_process_message") as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    assert mock_auth.call_args.args[0] == "999"  # caller id from the raw field
    assert adapter._pending_guest_queries == {}
    adapter._bot.do_api_request.assert_not_called()
    mock_should.assert_not_called()


@pytest.mark.asyncio
async def test_guest_caller_authorized_passes_gate():
    """An authorized caller passes the gate and reaches normal processing."""
    adapter = _make_adapter()
    update, _ = _make_guest_update(caller_id="123304346")
    msg = _fake_guest_msg()

    with patch.object(_tg_adapter_mod.Message, "de_json", return_value=msg), \
         patch.object(adapter, "_is_callback_user_authorized", return_value=True) as mock_auth, \
         patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    mock_should.assert_called_once()  # gate passed → reached processing


@pytest.mark.asyncio
async def test_guest_missing_caller_field_denies_fail_closed():
    """No guest_bot_caller_user and no from_user => empty caller => real
    _is_callback_user_authorized denies (no runner, no env allowlist)."""
    adapter = _make_adapter()
    update, _ = _make_guest_update(caller_id=None)  # no caller field at all
    msg = _fake_guest_msg()

    with patch.object(_tg_adapter_mod.Message, "de_json", return_value=msg):
        await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.do_api_request.assert_not_called()


@pytest.mark.asyncio
async def test_guest_allowlisted_caller_via_env_passes(monkeypatch):
    """End-to-end through the real gate: env allowlist authorizes the caller."""
    adapter = _make_adapter()
    update, _ = _make_guest_update(caller_id="999")
    msg = _fake_guest_msg()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    with patch.object(_tg_adapter_mod.Message, "de_json", return_value=msg), \
         patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_should.assert_called_once()  # real gate allowed the env-allowlisted caller


# ---------------------------------------------------------------------------
# Session isolation per guest caller — different callers in the same chat must
# not share a session (context bleed). Post-gate the caller is authorized, so
# stamping the real caller id makes guest sessions key like group sessions.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guest_session_isolated_per_caller(monkeypatch):
    from types import SimpleNamespace
    from gateway.session import SessionSource, build_session_key
    from gateway.config import Platform

    adapter = _make_adapter()
    adapter._bot.username = "testbot"
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    update, _ = _make_guest_update(caller_id="999", chat_id="42")
    msg = _fake_guest_msg(chat_id="42")

    src = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id=None)
    ev = SimpleNamespace(source=src, text="hi", channel_prompt=None)
    captured = {}

    with patch.object(_tg_adapter_mod.Message, "de_json", return_value=msg), \
         patch.object(adapter, "_build_message_event", return_value=ev), \
         patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event", side_effect=lambda e: captured.update(event=e)):
        await adapter._handle_guest_message_update(update, MagicMock())

    # The real guest caller id is stamped onto the source...
    assert captured["event"].source.user_id == "999"
    # ...so the session key isolates per caller within the chat.
    key = build_session_key(captured["event"].source, group_sessions_per_user=True)
    assert key.endswith(":42:999")

    # Two different callers in the same chat get distinct sessions (no bleed).
    k_a = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id="999"),
        group_sessions_per_user=True,
    )
    k_b = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group", user_id="888"),
        group_sessions_per_user=True,
    )
    assert k_a != k_b
