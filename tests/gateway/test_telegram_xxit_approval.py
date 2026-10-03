"""Tests for Telegram XXIT inline keyboard approval buttons (Phase 1)."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


# ===========================================================================
# send_xxit_approval — inline keyboard buttons
# ===========================================================================

class TestTelegramXXITApproval:
    """Test the send_xxit_approval method sends XXIT InlineKeyboard buttons."""

    @pytest.mark.asyncio
    async def test_sends_inline_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_xxit_approval(
            chat_id="12345",
            title="XXIT Task 승인 요청",
            question="배포 파이프라인을 실행하시겠습니까?",
            session_key="agent:main:telegram:group:12345:99",
            xxit_id="A1B2C3D4E5F6",
        )

        assert result.success is True
        assert result.message_id == "42"

        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "XXIT Task 승인 요청" in kwargs["text"]
        assert "배포 파이프라인을 실행하시겠습니까?" in kwargs["text"]
        assert kwargs["reply_markup"] is not None  # InlineKeyboardMarkup

    @pytest.mark.asyncio
    async def test_callback_data_format(self, monkeypatch):
        """callback_data가 xa:a:, xa:r:, xa:h: 접두사로 64바이트 제한을 준수하는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        captured_buttons = []

        def fake_button(text, callback_data=None, url=None):
            captured_buttons.append((text, callback_data, url))
            return MagicMock()

        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton", fake_button
        )

        captured_rows = []

        def fake_markup(rows):
            captured_rows.append(rows)
            return MagicMock()

        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", fake_markup
        )

        await adapter.send_xxit_approval(
            chat_id="12345",
            title="T",
            question="Q",
            session_key="s",
            xxit_id="A1B2C3D4E5F6",
            clickup_url="https://clickup.com/task/12345",
        )

        # callback_data가 있는 버튼들 수집
        callback_buttons = [(t, cd) for (t, cd, u) in captured_buttons if cd is not None]
        url_buttons = [(t, u) for (t, cd, u) in captured_buttons if u is not None]

        # callback_data 버튼: 승인, 반려, 보류 (3개)
        assert len(callback_buttons) == 3, f"Expected 3 callback buttons, got {len(callback_buttons)}"
        for (text, cb_data) in callback_buttons:
            assert cb_data.startswith("xa:"), f"Expected xa: prefix, got {cb_data}"
            assert len(cb_data.encode("utf-8")) <= 64, f"callback_data exceeds 64 bytes: {cb_data} ({len(cb_data.encode('utf-8'))} bytes)"

        # 승인 버튼
        approve = [cd for (t, cd) in callback_buttons if "승인" in t][0]
        assert approve == "xa:a:A1B2C3D4E5F6"

        # 반려 버튼
        reject = [cd for (t, cd) in callback_buttons if "반려" in t][0]
        assert reject == "xa:r:A1B2C3D4E5F6"

        # 보류 버튼
        hold = [cd for (t, cd) in callback_buttons if "보류" in t][0]
        assert hold == "xa:h:A1B2C3D4E5F6"

        # URL 버튼 (상세보기)
        assert len(url_buttons) == 1
        url_text, url_val = url_buttons[0]
        assert "ClickUp" in url_text
        assert url_val == "https://clickup.com/task/12345"

    @pytest.mark.asyncio
    async def test_callback_data_64byte_limit_max_id(self):
        """최대 12자 아이디로도 64바이트 제한을 준수하는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        captured_buttons = []

        def fake_button(text, callback_data=None, url=None):
            captured_buttons.append((text, callback_data, url))
            return MagicMock()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton", fake_button
        )

        await adapter.send_xxit_approval(
            chat_id="12345",
            title="T", question="Q", session_key="s",
            xxit_id="123456789012",  # 정확히 12자
        )
        monkeypatch.undo()

        for (text, cb_data, _) in captured_buttons:
            if cb_data is not None:
                size = len(cb_data.encode("utf-8"))
                assert size <= 64, f"callback_data {cb_data} is {size} bytes (>64)"
                # 예: xa:a:123456789012 = 3+1+1+12 = 17바이트

    @pytest.mark.asyncio
    async def test_no_clickup_url_omits_detail_button(self, monkeypatch):
        """clickup_url이 없으면 상세보기 버튼이 생략되는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        captured_buttons = []

        def fake_button(text, callback_data=None, url=None):
            captured_buttons.append((text, callback_data, url))
            return MagicMock()

        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton", fake_button
        )

        await adapter.send_xxit_approval(
            chat_id="12345", title="T", question="Q", session_key="s",
            xxit_id="ABC", clickup_url=None,
        )

        url_buttons = [(t, u) for (t, cd, u) in captured_buttons if u is not None]
        assert len(url_buttons) == 0, "clickup_url이 None이면 URL 버튼이 없어야 함"

    @pytest.mark.asyncio
    async def test_session_key_stored_on_send(self):
        """send_xxit_approval 호출 시 _xxit_approval_state에 session_key가 저장되는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        await adapter.send_xxit_approval(
            chat_id="12345", title="T", question="Q", session_key="s:key:1",
            xxit_id="XXIT-001", clickup_url=None,
        )

        assert "XXIT-001" in adapter._xxit_approval_state
        assert adapter._xxit_approval_state["XXIT-001"] == "s:key:1"

    @pytest.mark.asyncio
    async def test_xxit_id_unique_per_request(self):
        """서로 다른 xxit_id로 두 번 보내면 각각 다른 state entry가 생기는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        await adapter.send_xxit_approval(
            chat_id="12345", title="T1", question="Q1", session_key="s1",
            xxit_id="ID-001", clickup_url=None,
        )
        await adapter.send_xxit_approval(
            chat_id="12345", title="T2", question="Q2", session_key="s2",
            xxit_id="ID-002", clickup_url=None,
        )

        assert adapter._xxit_approval_state["ID-001"] == "s1"
        assert adapter._xxit_approval_state["ID-002"] == "s2"
        assert len(adapter._xxit_approval_state) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
    @pytest.mark.parametrize("duplicate_session", ["s1", "s2"])
    async def test_concurrent_duplicate_send_preserves_owner_and_allows_retry(
        self, outcome, duplicate_session,
    ):
        adapter = _make_adapter()
        sending = asyncio.Event()
        release = asyncio.Event()

        async def send_message(**kwargs):
            sending.set()
            await release.wait()
            if outcome == "failure":
                raise RuntimeError("Send failed")
            return SimpleNamespace(message_id=1)

        adapter._bot.send_message.side_effect = send_message
        first = asyncio.create_task(adapter.send_xxit_approval(
            "12345", "T", "Q", "s1", "X1"))
        try:
            await asyncio.wait_for(sending.wait(), timeout=5)
            duplicate = await asyncio.wait_for(adapter.send_xxit_approval(
                "12345", "T", "Q", duplicate_session, "X1"), timeout=5)
            assert not duplicate.success
            assert "duplicate" in duplicate.error.lower()
            adapter._bot.send_message.assert_awaited_once()
            assert adapter._xxit_approval_state["X1"] == "s1"

            if outcome == "cancelled":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            else:
                release.set()
                result = await asyncio.wait_for(first, timeout=5)
                assert result.success is (outcome == "success")

            if outcome == "success":
                assert adapter._xxit_approval_state["X1"] == "s1"
            else:
                assert "X1" not in adapter._xxit_approval_state
                adapter._bot.send_message.side_effect = None
                adapter._bot.send_message.return_value = SimpleNamespace(message_id=2)
                retry = await adapter.send_xxit_approval("12345", "T", "Q", "s2", "X1")
                assert retry.success
                assert adapter._xxit_approval_state["X1"] == "s2"
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)


# _handle_callback_query — XXIT approval button clicks
# ===========================================================================

class TestTelegramXXITCallbackDispatch:
    """Test _handle_callback_query routing for xa: prefixes."""

    @pytest.mark.asyncio
    async def test_xa_prefix_routed_to_xxit_handler(self):
        """callback_data가 xa:로 시작하면 _handle_xxit_approval_callback으로 dispatch되는지 확인."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        # 발신 메시지가 있다고 가정 (callback_ctx 용)
        mock_msg = MagicMock()
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
        mock_msg.message_id = 99
        mock_msg.text = "XXIT 승인 요청"
        mock_chat = MagicMock()
        mock_chat.type = "private"
        mock_msg.chat = mock_chat

        adapter._xxit_approval_state["XXIT-001"] = "agent:main:telegram:group:12345:99"

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="777", first_name="Choi"),
            data="xa:a:XXIT-001",
            message=mock_msg,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        # 핸들러를 직접 호출하는 대신 _handle_callback_query를 통해 라우팅 확인
        with patch.object(adapter, "_handle_xxit_approval_callback", wraps=adapter._handle_xxit_approval_callback) as mock_handler:
            await adapter._handle_callback_query(update, context)
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_xa_callback_stale_handled(self):
        """이미 처리된(xxit_state에 없는) 콜백은 'already resolved' 응답 후 무시."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        mock_msg = MagicMock()
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
        mock_msg.message_id = 99
        mock_msg.text = "XXIT 승인 요청"
        mock_chat = MagicMock()
        mock_chat.type = "private"
        mock_msg.chat = mock_chat

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="777", first_name="Choi"),
            data="xa:a:NOEXIST",
            message=mock_msg,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        # _callback_authorized 통과 가정
        with patch.object(adapter, "_callback_authorized", return_value=True):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        called_text = query.answer.call_args[1].get("text", "")
        assert "already been resolved" in called_text or "resolved" in called_text.lower()

    @pytest.mark.asyncio
    async def test_xa_callback_auth_denied(self):
        """인증 실패 시 콜백이 silently 무시됨 (UNAUTHORIZED 응답 후 return)."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        mock_msg = MagicMock()
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
        mock_msg.message_id = 99
        mock_msg.text = "XXIT 승인 요청"
        mock_chat = MagicMock()
        mock_chat.type = "private"
        mock_msg.chat = mock_chat

        adapter._xxit_approval_state["XXIT-001"] = "agent:main:telegram:group:12345:99"

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="EVIL", first_name="BadActor"),
            data="xa:a:XXIT-001",
            message=mock_msg,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        # Preserve the real auth gate's denial response; stub only its policy decision.
        with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
            await adapter._handle_callback_query(update, context)

        assert adapter._xxit_approval_state["XXIT-001"] == "agent:main:telegram:group:12345:99"
        query.answer.assert_called_once()
        called_text = query.answer.call_args[1].get("text", "")
        assert "not authorized" in called_text.lower() or "UNAUTHORIZED" in called_text or len(called_text) > 0

    @pytest.mark.asyncio
    async def test_xa_callback_invalid_action_ignored(self):
        """xa:<invalid>:<id> 형식 → 조용히 무시."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        mock_msg = MagicMock()
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
        mock_msg.message_id = 99
        mock_msg.text = "XXIT 승인 요청"
        mock_chat = MagicMock()
        mock_chat.type = "private"
        mock_msg.chat = mock_chat

        adapter._xxit_approval_state["XXIT-001"] = "agent:main:telegram:group:12345:99"

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="777", first_name="Choi"),
            data="xa:z:XXIT-001",  # z는 유효하지 않은 action
            message=mock_msg,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        with patch.object(adapter, "_callback_authorized", return_value=True):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "Invalid action" in query.answer.call_args[1].get("text", "")

    @pytest.mark.asyncio
    async def test_xa_callback_invalid_format_ignored(self):
        """xa:foo (파싱 불가) → 조용히 무시."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

        mock_msg = MagicMock()
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
        mock_msg.message_id = 99
        mock_msg.text = "XXIT 승인 요청"
        mock_chat = MagicMock()
        mock_chat.type = "private"
        mock_msg.chat = mock_chat

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="777", first_name="Choi"),
            data="xa:onlyonepart",  # : 없음 → len(parts) != 3
            message=mock_msg,
            answer=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace()

        await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "Invalid XXIT approval data" in query.answer.call_args[1].get("text", "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("xxit_id,valid", [("x" * 59, True), ("x" * 60, False),
                                              ("한" * 19, True), ("한" * 20, False), ("", False)])
    async def test_callback_data_64byte_limit_test_samples(self, xxit_id, valid):
        adapter = _make_adapter()
        adapter._bot.send_message.return_value = SimpleNamespace(message_id=1)
        result = await adapter.send_xxit_approval(
            "12345", "<Task> & review", "Proceed?\nNext line", "s", xxit_id)
        assert result.success is valid
        if valid:
            sent = adapter._bot.send_message.call_args.kwargs
            assert sent["text"] == "<b>&lt;Task&gt; &amp; review</b>\n\nProceed?\nNext line"
            for row in sent["reply_markup"].inline_keyboard:
                for button in row:
                    assert len(button.callback_data.encode("utf-8")) <= 64
        else:
            adapter._bot.send_message.assert_not_awaited()
            assert not adapter._xxit_approval_state


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["a", "r", "h"])
@pytest.mark.parametrize("failure", [None, "missing_origin", "no_handler", "wrong_session", "cancelled"])
async def test_xxit_admission_routes_once_or_retains_retry(tmp_path, monkeypatch, action, failure):
    """Real store, identity, and adapter admission; fake only Telegram transport and the LLM."""
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import GatewayConfig
    from gateway.session import SessionSource, SessionStore

    adapter = _make_adapter()
    runner = GatewayAuthorizationMixin()
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter.gateway_runner = runner
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    adapter.set_message_handler(AsyncMock())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="group",
                           user_id="chief", thread_id="99")
    entry = runner.session_store.get_or_create_session(source)
    key = entry.session_key
    adapter._active_sessions[key] = asyncio.Event()
    # Keep the real admission path on its busy queue, with no model call or network.
    adapter._session_tasks[key] = asyncio.current_task()
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=777, first_name="Choi <owner>"),
        data=f"xa:{action}:X1",
        message=SimpleNamespace(chat_id=888, chat=SimpleNamespace(type="private"),
                                message_thread_id=None, message_id=4),
        answer=AsyncMock(), edit_message_text=AsyncMock())
    adapter._xxit_approval_state["X1"] = key
    if failure == "missing_origin":
        entry.origin = None
    elif failure == "no_handler":
        adapter._message_handler = None
    elif failure == "wrong_session":
        entry.origin.thread_id = "100"
    try:
        if failure == "cancelled":
            admitting = asyncio.Event()
            async def pause_admission(event):
                admitting.set()
                await asyncio.Event().wait()

            # Cancel inside real admit_internal_event, before the adapter accepts it.
            with monkeypatch.context() as m:
                m.setattr(adapter, "handle_message", pause_admission)
                task = asyncio.create_task(adapter._handle_callback_query(
                    SimpleNamespace(callback_query=query), None))
                try:
                    await asyncio.wait_for(admitting.wait(), timeout=5)
                    assert adapter._xxit_approval_state["X1"] == key
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert adapter._xxit_approval_state["X1"] == key
            assert not adapter._pending_messages
            query.answer.assert_not_awaited()
            query.edit_message_text.assert_not_awaited()
            # Retry through real admission and the normal one-shot checks below.
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        if failure and failure != "cancelled":
            assert adapter._xxit_approval_state["X1"] == key
            assert not adapter._pending_messages
            assert "다시 시도" in query.answer.call_args.kwargs["text"]
            query.edit_message_text.assert_not_awaited()
        else:
            assert "X1" not in adapter._xxit_approval_state
            event = adapter._pending_messages[key]
            assert event._gateway_accepted is True
            assert event.source.chat_id == "12345" and event.source.thread_id == "99"
            assert event.source.user_id == "chief"
            assert event.metadata["action"] == action
            assert event.metadata["user_id"] == "777"
            assert event.metadata["gateway_session_key"] == key
            assert event.allow_gateway_control is False
            assert query.edit_message_text.call_args.kwargs["reply_markup"] is None
            assert "Choi &lt;owner&gt;" in query.edit_message_text.call_args.kwargs["text"]
            before = event.text
            await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
            assert adapter._pending_messages[key].text == before
            query.edit_message_text.assert_awaited_once()
            assert "already been resolved" in query.answer.call_args.kwargs["text"]
    finally:
        adapter._active_sessions.clear()
        adapter._session_tasks.clear()
        for db in runner.session_store._db_handles.values():
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("callback_outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("finish_first", ["send", "callback"])
async def test_xxit_send_and_callback_keep_exclusive_id_ownership(
    tmp_path, monkeypatch, send_outcome, callback_outcome, finish_first,
):
    """A claim cannot free an unfinished send's ID or resurrect its failed state."""
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import GatewayConfig
    from gateway.session import SessionSource, SessionStore

    adapter = _make_adapter()
    runner = GatewayAuthorizationMixin()
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter.gateway_runner = runner
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    adapter.set_message_handler(AsyncMock())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="group",
                           user_id="chief", thread_id="99")
    key = runner.session_store.get_or_create_session(source).session_key
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=777, first_name="Owner"), data="xa:a:X1",
        message=SimpleNamespace(chat_id=888, chat=SimpleNamespace(type="private"),
                                message_thread_id=None, message_id=4),
        answer=AsyncMock(), edit_message_text=AsyncMock())
    started = {name: asyncio.Event() for name in ("send", "callback")}
    release = {name: asyncio.Event() for name in started}
    handle_message = adapter.handle_message
    admitted = []

    async def send_message(**kwargs):
        # Only the original send blocks; duplicate bugs must fail assertions, not hang.
        if not started["send"].is_set():
            started["send"].set()
            await release["send"].wait()
            if send_outcome == "failure":
                raise RuntimeError("Send failed")
        return SimpleNamespace(message_id=1)

    async def pause_admission(event):
        started["callback"].set()
        await release["callback"].wait()
        if callback_outcome == "failure":
            raise RuntimeError("Admission failed")
        await handle_message(event)
        admitted.append(event)

    async def assert_duplicate_refused():
        for duplicate_key in (key, "other-session"):
            duplicate = await adapter.send_xxit_approval("12345", "T", "Q", duplicate_key, "X1")
            assert not duplicate.success
            assert "duplicate" in duplicate.error.lower()
        adapter._bot.send_message.assert_awaited_once()

    async def finish(name):
        outcome = send_outcome if name == "send" else callback_outcome
        if outcome == "cancelled":
            tasks[name].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[name]
        else:
            release[name].set()
            result = await asyncio.wait_for(tasks[name], timeout=5)
            if name == "send":
                assert result.success is (outcome == "success")

    adapter._bot.send_message.side_effect = send_message
    monkeypatch.setattr(adapter, "handle_message", pause_admission)
    tasks = {"send": asyncio.create_task(adapter.send_xxit_approval("12345", "T", "Q", key, "X1"))}
    try:
        await asyncio.wait_for(started["send"].wait(), timeout=5)
        tasks["callback"] = asyncio.create_task(adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), None))
        await asyncio.wait_for(started["callback"].wait(), timeout=5)
        await assert_duplicate_refused()
        # A second tap cannot enter admission while the first tap owns the claim.
        duplicate_query = SimpleNamespace(**{**vars(query), "answer": AsyncMock()})
        await asyncio.wait_for(adapter._handle_callback_query(
            SimpleNamespace(callback_query=duplicate_query), None), timeout=5)
        assert "already been resolved" in duplicate_query.answer.call_args.kwargs["text"]

        await finish(finish_first)
        await assert_duplicate_refused()
        await finish("callback" if finish_first == "send" else "send")
        assert len(admitted) == (1 if callback_outcome == "success" else 0)
        monkeypatch.setattr(adapter, "handle_message", handle_message)
        if send_outcome == "success" and callback_outcome != "success":
            assert adapter._xxit_approval_state["X1"] == key
            await assert_duplicate_refused()
            await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        assert "X1" not in adapter._xxit_approval_state
        if key in adapter._pending_messages:
            event = adapter._pending_messages[key]
            assert event._gateway_accepted is True
            assert event.metadata["gateway_session_key"] == key
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        assert "already been resolved" in query.answer.call_args.kwargs["text"]
        retry = await adapter.send_xxit_approval("12345", "T", "Q", "other-session", "X1")
        assert retry.success
        assert adapter._xxit_approval_state["X1"] == "other-session"
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        adapter._active_sessions.clear()
        adapter._session_tasks.clear()
        for db in runner.session_store._db_handles.values():
            db.close()
