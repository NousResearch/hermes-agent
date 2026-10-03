"""Integration tests for Telegram guest mode reply flow (Bot API 10.0).

Branch 1 — normal query: stub fires, skill runs, OPC edits stub with text or
           media button.
Branch 2 — deliver_<token> with valid token: answerGuestQuery with cached media,
           no stub, no edit cycle.
Branch 3 — deliver_<token> with invalid/expired token: answerGuestQuery with
           "something went wrong", no stub.

Guest messages arrive as a native ``telegram.Message`` on ``update.guest_message``,
carrying its own ``guest_query_id`` and a normal, populated ``from_user`` — that's
how PTB >=22.8 actually models Bot API 10.0 guest bots (there is no api_kwargs
involved). Every fixture below builds updates that shape, not a raw dict.
"""

import os
import sys
import time
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
import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Real stand-ins for PTB's inline-query/keyboard classes
#
# tests/gateway/conftest.py installs a shared generic MagicMock for the whole
# ``telegram`` package before any test file's imports run (see its
# ``_ensure_telegram_mock``, which wins whenever real PTB hasn't already been
# imported in this process). Calling a MagicMock attribute like
# ``InlineQueryResultArticle(id=..., title=...)`` returns a shared
# auto-generated child mock that does NOT reflect the kwargs it was called
# with — so asserting ``result.id == "thinking"`` would just be inspecting
# mock plumbing, not what the adapter actually built. These lightweight real
# classes give every test something genuinely inspectable regardless of
# whether real PTB or the shared mock is active in this process.
# ---------------------------------------------------------------------------

class _FakeInputTextMessageContent:
    def __init__(self, message_text, **_kw):
        self.message_text = message_text


class _FakeInlineQueryResultArticle:
    def __init__(self, *, id, title, input_message_content, **_kw):
        self.id = id
        self.title = title
        self.input_message_content = input_message_content


class _FakeInlineQueryResultCachedVideo:
    def __init__(self, *, id, video_file_id, title=None, **_kw):
        self.id, self.video_file_id, self.title = id, video_file_id, title


class _FakeInlineQueryResultCachedPhoto:
    def __init__(self, *, id, photo_file_id, title=None, **_kw):
        self.id, self.photo_file_id, self.title = id, photo_file_id, title


class _FakeInlineQueryResultCachedAudio:
    def __init__(self, *, id, audio_file_id, title=None, **_kw):
        self.id, self.audio_file_id, self.title = id, audio_file_id, title


class _FakeInlineQueryResultCachedDocument:
    def __init__(self, *, id, document_file_id, title=None, **_kw):
        self.id, self.document_file_id, self.title = id, document_file_id, title


class _FakeInlineKeyboardButton:
    def __init__(self, text, switch_inline_query_current_chat=None, **_kw):
        self.text = text
        self.switch_inline_query_current_chat = switch_inline_query_current_chat


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard, **_kw):
        self.inline_keyboard = inline_keyboard


@pytest.fixture(autouse=True)
def _real_inline_result_classes(monkeypatch):
    monkeypatch.setattr(_tg_adapter_mod, "InputTextMessageContent", _FakeInputTextMessageContent)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultArticle", _FakeInlineQueryResultArticle)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedVideo", _FakeInlineQueryResultCachedVideo)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedPhoto", _FakeInlineQueryResultCachedPhoto)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedAudio", _FakeInlineQueryResultCachedAudio)
    monkeypatch.setattr(_tg_adapter_mod, "InlineQueryResultCachedDocument", _FakeInlineQueryResultCachedDocument)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(_tg_adapter_mod, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> TelegramAdapter:
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {"guest_mode": True}
    adapter = TelegramAdapter(cfg)
    adapter._bot = MagicMock()
    adapter._bot.username = "testbot"  # real str so _clean_bot_trigger_text regex works
    adapter._bot.answer_guest_query = AsyncMock(
        return_value=MagicMock(inline_message_id="imi_abc")
    )
    adapter._bot.edit_message_text = AsyncMock()
    return adapter


def _register_guest_chat(adapter: TelegramAdapter, chat_id="42") -> None:
    """Pre-populate state as if branch-1 processing started."""
    adapter._pending_guest_queries[chat_id] = "gqid_test"
    adapter._guest_only_chats.add(chat_id)
    adapter._guest_inline_message_ids[chat_id] = False


def _make_guest_message(
    caller_id="999", caller_username="someone", caller_first_name=None,
    chat_id="42", chat_type="supergroup", text="@bot hi", gqid="gq1",
):
    """A Message-shaped mock matching PTB's actual guest_message modeling:
    guest_query_id and from_user are both real, direct attributes."""
    msg = MagicMock()
    msg.guest_query_id = gqid
    msg.text = text
    msg.caption = None
    msg.message_id = 1
    # Explicit, not a bare MagicMock: _effective_message_thread_id treats any
    # truthy value (including an unconfigured auto-mock attribute) as a real
    # forum-topic thread id, silently corrupting the session key.
    msg.message_thread_id = None
    msg.is_topic_message = False
    msg.chat = MagicMock(id=int(chat_id), type=chat_type, title="Test Group", is_forum=False)
    msg.from_user = (
        MagicMock(id=int(caller_id), username=caller_username, first_name=caller_first_name, is_bot=False)
        if caller_id is not None else None
    )
    return msg


def _make_guest_update(*, update_id=1, **kwargs):
    msg = _make_guest_message(**kwargs)
    update = MagicMock()
    update.update_id = update_id
    update.guest_message = msg
    return update, msg


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


@pytest.mark.asyncio
async def test_guest_fire_text_stub_stores_typed_inline_message_id():
    """The stub fire calls the typed Bot.answer_guest_query and stores the
    (guaranteed-present) inline_message_id straight off the typed result —
    no dict/.get() involved."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._bot.answer_guest_query = AsyncMock(return_value=MagicMock(inline_message_id="imi_live"))

    await adapter._guest_fire_text_stub("42")

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_test"
    result = call.args[1]
    assert result.id == "thinking"
    assert adapter._guest_inline_message_ids["42"] == "imi_live"


@pytest.mark.asyncio
async def test_guest_fire_text_stub_exception_leaves_imi_none():
    """A raised exception (network error, API rejection) — not a malformed
    response, PTB validates SentGuestMessage's inline_message_id as required —
    is the only way the stub can fail; OPC's no-imi fallback picks it up."""
    adapter = _make_adapter()
    _register_guest_chat(adapter)
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("boom"))

    await adapter._guest_fire_text_stub("42")

    assert adapter._guest_inline_message_ids["42"] is None


# ---------------------------------------------------------------------------
# Branch 1 OPC — text result: edit_message_text on imi
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_opc_text_result_edits_stub():
    """OPC text result: edit_message_text(inline_message_id=, text=final_text)."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_reply_buffer["42"] = "Here is your answer."
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"
    outcome = ProcessingOutcome.SUCCESS

    await adapter.on_processing_complete(event, outcome)

    adapter._bot.edit_message_text.assert_awaited()
    last_call = adapter._bot.edit_message_text.await_args_list[-1]
    assert last_call.kwargs["inline_message_id"] == "imi_abc"
    assert "Here is your answer." in last_call.kwargs["text"]


@pytest.mark.asyncio
async def test_branch1_opc_truncates_by_utf16_units_not_python_length():
    """A reply with many astral-plane emoji is short in Python string length
    but can exceed Telegram's 4,096 *UTF-16 code unit* limit — each emoji is
    a surrogate pair (2 units). A naive text[:4096] slice would pass this
    reply through untruncated and let the real edit_message_text call fail
    after guest state has already been torn down, silently dropping the
    turn. Every edit_message_text call (typewriter frames and the final edit)
    must stay within the UTF-16 cap."""
    from gateway.platforms.base import ProcessingOutcome, utf16_len

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    # 2,049 emoji: 2,049 Python chars, but 4,098 UTF-16 units — over the cap
    # despite `len(_plain) < 4096` looking safe.
    adapter._guest_reply_buffer["42"] = "😀" * 2049
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    edit_texts = [c.kwargs["text"] for c in adapter._bot.edit_message_text.await_args_list]
    assert edit_texts, "expected at least one edit_message_text call"
    for text in edit_texts:
        assert utf16_len(text) <= adapter.MAX_MESSAGE_LENGTH, (
            f"edit_message_text frame exceeds UTF-16 cap: {utf16_len(text)} units"
        )


@pytest.mark.asyncio
async def test_branch1_opc_no_imi_falls_back_to_answer_guest_query():
    """If the stub call itself raised (imi never obtained), OPC must still
    answer the guest query directly instead of silently dropping the turn."""
    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._pending_guest_queries["42"] = "gqid_fallback"
    adapter._guest_inline_message_ids["42"] = None  # stub raised
    adapter._guest_reply_buffer["42"] = "Fallback answer."
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.edit_message_text.assert_not_awaited()
    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_fallback"
    assert "Fallback answer." in call.args[1].input_message_content.message_text


# ---------------------------------------------------------------------------
# Caller authorization gate (fail-closed) — _handle_guest_message_update
#
# The caller for an incoming guest message is msg.from_user, same as any
# other Telegram message. Unauthorized callers are denied before any state
# registration or API call; no from_user / unknown caller => deny.
# ---------------------------------------------------------------------------

import plugins.platforms.telegram.adapter as _tg_adapter_mod  # noqa: E402


@pytest.mark.asyncio
async def test_guest_caller_unauthorized_is_denied():
    """A caller not in any allowlist is denied before state/API."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False) as mock_auth, \
         patch.object(adapter, "_should_process_message") as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    assert mock_auth.call_args.args[0] == "999"  # caller id from msg.from_user
    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()
    mock_should.assert_not_called()


@pytest.mark.asyncio
async def test_guest_caller_authorized_passes_gate():
    """An authorized caller passes the gate and reaches normal processing."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True) as mock_auth, \
         patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_auth.assert_called_once()
    mock_should.assert_called_once()  # gate passed → reached processing


@pytest.mark.asyncio
async def test_guest_missing_from_user_denies_fail_closed():
    """No from_user at all (e.g. an anonymous-admin edge case) => empty
    caller id => real _is_callback_user_authorized denies (no runner, no env
    allowlist)."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id=None)

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


@pytest.mark.asyncio
async def test_guest_allowlisted_caller_via_env_passes(monkeypatch):
    """End-to-end through the real gate: env allowlist authorizes the caller."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    with patch.object(adapter, "_should_process_message", return_value=False) as mock_should:
        await adapter._handle_guest_message_update(update, MagicMock())

    mock_should.assert_called_once()  # real gate allowed the env-allowlisted caller


@pytest.mark.asyncio
async def test_guest_message_missing_guest_query_id_is_ignored(monkeypatch):
    """update.guest_message present but with an empty guest_query_id must not
    be treated as a normal message — no state registered, no API call."""
    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999", gqid="")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


@pytest.mark.asyncio
async def test_non_guest_update_is_ignored():
    """A regular update (update.guest_message is None, PTB's default for any
    non-guest update) must be a pure no-op — this is the fix's core contract:
    the handler no longer reaches into api_kwargs at all."""
    adapter = _make_adapter()
    update = MagicMock()
    update.guest_message = None

    await adapter._handle_guest_message_update(update, MagicMock())

    assert adapter._pending_guest_queries == {}
    adapter._bot.answer_guest_query.assert_not_called()


# ---------------------------------------------------------------------------
# Session isolation per guest caller — different callers in the same chat must
# not share a session (context bleed). This now falls entirely out of the
# unpatched _build_message_event, which reads user_id/user_name straight off
# msg.from_user — there is no guest-specific stamping code left to test
# separately from that.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guest_session_isolated_per_caller(monkeypatch):
    from gateway.session import SessionSource, build_session_key
    from gateway.config import Platform

    adapter = _make_adapter()
    adapter._bot.username = "testbot"
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    update, msg = _make_guest_update(caller_id="999", chat_id="42")
    captured = {}

    with patch.object(adapter, "_should_process_message", return_value=True), \
         patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e), \
         patch.object(adapter, "_enqueue_text_event", side_effect=lambda e: captured.update(event=e)):
        await adapter._handle_guest_message_update(update, MagicMock())

    # from_user flows straight into the session source via _build_message_event.
    assert captured["event"].source.user_id == "999"
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


# ---------------------------------------------------------------------------
# Branch 1 OPC — media result: edit_message_text with deliver button
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch1_opc_media_result_edits_stub_with_button():
    """OPC media result: edit_message_text with switch_inline_query_current_chat button."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()

    from gateway.platforms.base import ProcessingOutcome

    adapter = _make_adapter()
    adapter._guest_inline_message_ids["42"] = "imi_abc"
    adapter._guest_reply_buffer["42"] = "Here is your video."
    adapter._guest_turn_media["42"] = {"file_id": "fid_video", "media_kind": "video"}
    adapter._guest_only_chats.add("42")

    event = MagicMock()
    event.source.chat_id = "42"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.edit_message_text.assert_awaited()
    call = adapter._bot.edit_message_text.await_args_list[-1]
    assert call.kwargs["inline_message_id"] == "imi_abc"
    assert "✅ Ready" in call.kwargs["text"]
    markup = call.kwargs.get("reply_markup")
    assert markup is not None, "reply_markup must be present"
    button = markup.inline_keyboard[0][0]
    assert button.switch_inline_query_current_chat.startswith("deliver_")

    # Token must exist in store
    token = button.switch_inline_query_current_chat[len("deliver_"):]
    assert token in gmt._TOKEN_STORE
    assert gmt._TOKEN_STORE[token]["media_kind"] == "video"

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Branch 2 — valid token: answerGuestQuery with cached media, no stub
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch2_valid_token_answers_with_cached_media():
    """Branch 2: deliver_<token> with a valid token dispatches straight to
    answer_guest_query with the cached media — no stub, no LLM pass.

    Invokes the real handler (not a hand-built API call) so token
    resolution, caller authorization, and dispatch are all actually
    exercised, not just that the mock records whatever call the test made
    directly."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_video", "video")

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346", gqid="gqid_branch2", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_branch2"
    result = call.args[1]
    assert result.id == "delivery"
    assert result.video_file_id == "fid_video"
    # No stub / no edit cycle — Branch 2 never touches guest turn state.
    assert adapter._pending_guest_queries == {}

    gmt._TOKEN_STORE.clear()


@pytest.mark.asyncio
async def test_deliver_token_denied_for_unauthorized_caller():
    """A valid, unexpired token must still be denied if the caller isn't
    authorized — the caller gate sits in front of the deliver_<token>
    branch too, so a leaked token can't be redeemed by a stranger."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid_video", "video")

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="999", gqid="gqid_branch2", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_not_called()

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Branch 3 — invalid/expired token: "something went wrong" result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_branch3_expired_token_answers_with_error():
    """Branch 3: deliver_<token> with an expired token answers with
    "something went wrong" instead of the cached media — via the real
    handler, not a hand-built API call."""
    import tools.guest_mode_tool as gmt
    gmt._TOKEN_STORE.clear()
    token = gmt.mint_token("fid", "video")
    gmt._TOKEN_STORE[token]["expires_at"] = time.monotonic() - 1  # force expiry

    adapter = _make_adapter()
    update, msg = _make_guest_update(caller_id="123304346", gqid="gqid_branch3", text=f"deliver_{token}")

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_guest_message_update(update, MagicMock())

    adapter._bot.answer_guest_query.assert_awaited_once()
    call = adapter._bot.answer_guest_query.await_args
    assert call.args[0] == "gqid_branch3"
    result = call.args[1]
    assert "wrong" in result.input_message_content.message_text.lower()

    gmt._TOKEN_STORE.clear()


# ---------------------------------------------------------------------------
# Callback handler — no delivery attempt, only answerCallbackQuery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_on_stub_only_dismisses_loading():
    """Button press on stub produces callback_query — handler must not call answer_guest_query."""
    adapter = _make_adapter()

    cq = MagicMock()
    cq.id = "cq_123"
    cq.inline_message_id = "imi_abc"
    cq.answer = AsyncMock()

    await cq.answer()

    cq.answer.assert_awaited_once()
    adapter._bot.answer_guest_query.assert_not_called()


# ---------------------------------------------------------------------------
# _guest_media_send — hard path containment for the MEDIA: staging flow
#
# Unaffected by the PTB-native-API modernization: this staging path always
# used typed Bot.send_photo/send_video/send_audio/send_document (never
# do_api_request), so nothing here changes.
#
# The delivery-constraint prompt *tells* the LLM to stage under
# HERMES_HOME/cache/<subdir>, but that was only a prompt-level instruction --
# nothing enforced it, so a guest-triggered turn coerced into requesting an
# arbitrary host path (e.g. the credentials store) would previously have had
# it staged and made deliverable to the guest chat. These tests cover the
# hard containment check added to _guest_media_send.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guest_media_send_rejects_path_outside_staging_root(tmp_path, monkeypatch):
    """A path outside HERMES_HOME/cache is rejected before any open()/upload."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    outside_file = tmp_path / "auth.json"
    outside_file.write_text('{"api_key": "secret"}')

    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock()

    result = await adapter._guest_media_send("42", "document", str(outside_file))

    assert result.success is False
    assert "outside the allowed staging directory" in result.error
    adapter._bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_guest_media_send_allows_path_inside_staging_root(tmp_path, monkeypatch):
    """A path inside HERMES_HOME/cache proceeds to staging as before."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    cache_dir = tmp_path / "cache" / "videos"
    cache_dir.mkdir(parents=True)
    video_file = cache_dir / "clip.mp4"
    video_file.write_bytes(b"fake video bytes")

    adapter = _make_adapter()

    sent_video = MagicMock()
    sent_video.video = MagicMock(file_id="fid_ok")
    adapter._bot.send_video = AsyncMock(return_value=sent_video)

    result = await adapter._guest_media_send("42", "video", str(video_file))

    assert result.success is True
    adapter._bot.send_video.assert_awaited_once()
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_ok"


@pytest.mark.asyncio
async def test_guest_media_send_null_byte_path_fails_cleanly(tmp_path, monkeypatch):
    """A path resolve() itself chokes on (embedded null byte) returns a clean
    failure SendResult instead of raising out of the adapter.

    Regression: resolution and containment used to share one try block, so a
    null-byte path entered the ValueError ("outside staging root") handler
    with _abs_resolved unbound — the rejection log line itself then raised
    UnboundLocalError, escaping the SendResult contract entirely.
    """
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock()

    result = await adapter._guest_media_send(
        "42", "document", str(tmp_path / "cache") + "/evil\x00.pdf"
    )

    assert result.success is False
    assert "path validation failed" in result.error
    adapter._bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_guest_media_send_restages_when_file_content_changes(tmp_path, monkeypatch):
    """Re-generating a file at the same path (new mtime) re-stages and mints a
    fresh file_id instead of serving the stale cached one."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100999")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    photo = cache_dir / "chart.png"
    photo.write_bytes(b"v1")
    os.utime(photo, (1000, 1000))

    adapter = _make_adapter()
    fids = iter(["fid_v1", "fid_v2"])

    def _mint(*a, **kw):
        sent = MagicMock()
        sent.photo = [MagicMock(file_id=next(fids))]
        return sent

    adapter._bot.send_photo = AsyncMock(side_effect=_mint)

    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_v1"

    # Same path, same mtime -> served from cache, no second upload.
    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._bot.send_photo.await_count == 1

    # Same path, new content/mtime -> re-staged, fresh file_id.
    photo.write_bytes(b"v2 -- regenerated")
    os.utime(photo, (2000, 2000))
    await adapter._guest_media_send("42", "photo", str(photo))
    assert adapter._bot.send_photo.await_count == 2
    assert adapter._guest_turn_media["42"]["file_id"] == "fid_v2"
