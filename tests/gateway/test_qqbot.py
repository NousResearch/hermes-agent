"""Tests for the QQ Bot platform adapter."""

import asyncio
import os
from unittest import mock

import httpx
import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra):
    """Build a PlatformConfig(enabled=True, extra=extra) for testing."""
    return PlatformConfig(enabled=True, extra=extra)


# ---------------------------------------------------------------------------
# check_qq_requirements
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# QQAdapter.__init__
# ---------------------------------------------------------------------------

class TestQQAdapterInit:
    def _make(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"QQ_APP_ID": "env_id", "QQ_CLIENT_SECRET": "env_sec"}, clear=False):
            adapter = self._make()
            assert adapter._app_id == "env_id"
            assert adapter._client_secret == "env_sec"


    def test_dm_policy_default(self):
        adapter = self._make(app_id="a", client_secret="b")
        assert adapter._dm_policy == "pairing"



    def test_allow_from_parsing_string(self):
        adapter = self._make(app_id="a", client_secret="b", allow_from="x, y , z")
        assert adapter._allow_from == ["x", "y", "z"]




# ---------------------------------------------------------------------------
# _coerce_list
# ---------------------------------------------------------------------------

class TestCoerceList:
    def _fn(self, value):
        from plugins.platforms.qqbot.utils import coerce_list as _coerce_list
        return _coerce_list(value)

    def test_none(self):
        assert self._fn(None) == []

    def test_string(self):
        assert self._fn("a, b ,c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _is_voice_content_type
# ---------------------------------------------------------------------------

class TestIsVoiceContentType:
    def _fn(self, content_type, filename):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter._is_voice_content_type(content_type, filename)


    def test_voice_extension_fallback_when_content_type_empty(self):
        """content_type='' with audio extension → True (extension fallback)."""
        assert self._fn("", "file.silk") is True




# ---------------------------------------------------------------------------
# Voice attachment SSRF protection
# ---------------------------------------------------------------------------

class TestVoiceAttachmentSSRFProtection:
    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_connect_uses_redirect_guard_hook(self):
        from plugins.platforms.qqbot.adapter import QQAdapter

        client = mock.AsyncMock()
        api = mock.Mock()
        api.setup = mock.Mock()
        api.get_gateway_url = mock.AsyncMock(side_effect=RuntimeError("stop after client creation"))
        with mock.patch(
            "plugins.platforms.qqbot.adapter.platform_http_client", return_value=client,
        ) as http_client, mock.patch("qqbot_agent_sdk.QQApiClient", return_value=api):
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            connected = asyncio.run(adapter.connect())

        assert connected is False
        http_client.assert_called_once_with()
        api.setup.assert_called_once_with(client)

    def test_connect_failure_logs_the_error_without_a_traceback(self, caplog):
        import logging

        from plugins.platforms.qqbot.adapter import QQAdapter

        client = mock.AsyncMock()
        api = mock.Mock()
        api.setup = mock.Mock()
        api.get_gateway_url = mock.AsyncMock(
            side_effect=RuntimeError(
                "QQ Bot token response missing access_token: "
                "{'code': 100016, 'message': 'invalid appid or secret'}"
            )
        )
        with mock.patch(
            "plugins.platforms.qqbot.adapter.platform_http_client", return_value=client,
        ), mock.patch("qqbot_agent_sdk.QQApiClient", return_value=api), caplog.at_level(
            logging.ERROR, logger="plugins.platforms.qqbot.adapter",
        ):
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            connected = asyncio.run(adapter.connect())

        assert connected is False
        assert "invalid appid or secret" in caplog.text
        assert "Traceback" not in caplog.text

    @pytest.mark.asyncio
    async def test_connect_stays_unconfirmed_until_sdk_ready(self):
        from plugins.platforms.qqbot.adapter import QQAdapter

        client = mock.AsyncMock()
        api = mock.Mock()
        api.setup = mock.Mock()
        api.get_gateway_url = mock.AsyncMock(return_value="wss://example.test/websocket")

        class _WS:
            def __init__(self, callbacks, log_tag=None):
                self.callbacks = callbacks

            def start(self, gateway_url, main_loop):
                self.started = (gateway_url, main_loop)

        with mock.patch(
            "plugins.platforms.qqbot.adapter.platform_http_client", return_value=client,
        ), mock.patch("qqbot_agent_sdk.QQApiClient", return_value=api), mock.patch(
            "plugins.platforms.qqbot.sdk_bridge.scoped_websocket_class", return_value=_WS,
        ):
            adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
            connected = await adapter.connect()

        assert connected is True
        assert adapter._sdk_connected is False
        assert adapter.is_connected is False
        assert adapter.send_path_degraded is True
        adapter._sdk_callbacks().on_connected()
        assert adapter._sdk_connected is True
        assert adapter.is_connected is True
        assert adapter.send_path_degraded is False

    def test_open_failure_before_listen_is_retryable_fatal(self):
        from plugins.platforms.qqbot.adapter import QQAdapter

        adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
        adapter._running = True
        adapter._qq_ws = object()
        adapter._qq_api = mock.Mock()
        adapter._sdk_connected = True
        callbacks = adapter._sdk_callbacks()
        callbacks.on_disconnected()
        callbacks.on_fatal_error(
            "qq_gateway_open_failed",
            "QQ gateway socket exited before the listen loop started",
        )

        assert adapter._sdk_connected is False
        assert adapter.is_connected is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is True
        assert adapter.fatal_error_code == "qq_gateway_open_failed"

    def test_media_download_headers_carry_hermes_user_agent(self):
        from plugins.platforms.qqbot.adapter import QQAdapter

        adapter = QQAdapter(_make_config(app_id="a", client_secret="b"))
        adapter._qq_api = mock.Mock(access_token="tok")
        headers = adapter._qq_media_headers()

        assert headers["Authorization"] == "QQBot tok"
        assert "Hermes/" in headers["User-Agent"]
        assert headers["User-Agent"].startswith("QQBotAdapter/")


# ---------------------------------------------------------------------------
# Voice attachment temp-file cleanup
# ---------------------------------------------------------------------------

class TestVoiceAttachmentTempCleanup:
    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))

    def _setup_download_mocks(self, adapter, content=b"RIFFmock-wav-audio-data"):
        response = mock.Mock()
        response.content = content
        response.headers = {"content-type": "audio/wav"}
        response.raise_for_status = mock.Mock()

        adapter._http_client = mock.AsyncMock()
        adapter._http_client.get = mock.AsyncMock(return_value=response)

    def test_temp_wav_cleaned_up_on_stt_failure(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        self._setup_download_mocks(adapter)
        seen = {}

        async def _raise_transport_error(path):
            seen["wav_path"] = path
            raise httpx.TransportError("boom")

        with mock.patch("tools.url_safety.is_safe_url", return_value=True):
            adapter._call_stt = mock.AsyncMock(side_effect=_raise_transport_error)
            transcript = asyncio.run(
                adapter._stt_voice_attachment(
                    "https://cdn.qq.com/voice.silk",
                    "audio/silk",
                    "voice.silk",
                    voice_wav_url="https://cdn.qq.com/voice.wav",
                )
            )

        assert transcript is None
        assert "wav_path" in seen
        assert not os.path.exists(seen["wav_path"])


# ---------------------------------------------------------------------------
# _strip_at_mention
# ---------------------------------------------------------------------------

class TestStripAtMention:
    def _fn(self, content):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter._strip_at_mention(content)

    def test_removes_mention(self):
        result = self._fn("@BotUser hello there")
        assert result == "hello there"


# ---------------------------------------------------------------------------
# _is_dm_allowed
# ---------------------------------------------------------------------------

class TestDmAllowed:
    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_open_policy_with_opt_in(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="open")
        assert adapter._is_dm_allowed("any_user") is True
        assert adapter._is_dm_intake_allowed("any_user") is True


    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", dm_policy="allowlist", allow_from="user1,user2")
        assert adapter._is_dm_allowed("user1") is True


# ---------------------------------------------------------------------------
# _is_group_allowed
# ---------------------------------------------------------------------------

class TestGroupAllowed:
    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))


    def test_allowlist_match(self):
        adapter = self._make_adapter(app_id="a", client_secret="b", group_policy="allowlist", group_allow_from="grp1")
        assert adapter._is_group_allowed("grp1", "user1") is True


    def test_pairing_default_blocks_groups(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        assert adapter._group_policy == "pairing"
        assert adapter._is_group_allowed("grp1", "user1") is False


# ---------------------------------------------------------------------------
# _resolve_stt_config
# ---------------------------------------------------------------------------

class TestResolveSTTConfig:
    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))

    def test_no_config(self):
        adapter = self._make_adapter(app_id="a", client_secret="b")
        with mock.patch.dict(os.environ, {}, clear=True):
            assert adapter._resolve_stt_config() is None

    def test_call_stt_posts_with_configured_timeout(self, tmp_path):
        """The configured ``stt.timeout`` reaches the STT HTTP request; default 60s, not the
        old fixed 30s (#112939). Drives ``_call_stt`` so a regression at the call-site is caught."""
        wav = tmp_path / "v.wav"
        wav.write_bytes(b"RIFF")
        posted = []

        class _FakeClient:
            async def post(self, url, **kwargs):
                posted.append(kwargs)
                return httpx.Response(200, json={"text": "hi"}, request=httpx.Request("POST", url))

        with mock.patch.dict(os.environ, {}, clear=True):
            for stt, expected in (({"apiKey": "k", "provider": "zai"}, 60.0),
                                  ({"apiKey": "k", "provider": "zai", "timeout": "95"}, 95.0)):
                adapter = self._make_adapter(app_id="a", client_secret="b", stt=stt)
                adapter._http_client = _FakeClient()
                assert asyncio.run(adapter._call_stt(str(wav))) == "hi"
                assert posted[-1]["timeout"] == expected


# ---------------------------------------------------------------------------
# _detect_message_type
# ---------------------------------------------------------------------------
# _wait_for_reconnection / send reconnection wait
# ---------------------------------------------------------------------------


class TestSdkTextSend:
    """Outbound text goes through QQApiClient.send_text, including the markdown flag."""

    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b", **extra))

    @pytest.mark.asyncio
    async def test_markdown_flag_reaches_sdk(self):
        adapter = self._make_adapter(markdown_support=True)
        adapter._qq_api = mock.Mock()
        adapter._qq_api.send_text = mock.AsyncMock(return_value={"id": "m1"})

        result = await adapter._send_text_to("c2c", "openid", "hello")

        assert result.success
        adapter._qq_api.send_text.assert_awaited_once()
        assert adapter._qq_api.send_text.await_args.kwargs["markdown"] is True


class TestWaitForReconnection:
    """Test that send() waits for reconnection instead of silently dropping."""

    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(**extra))

    @pytest.mark.asyncio
    async def test_send_waits_and_succeeds_on_reconnect(self):
        """send() should wait for reconnection and then deliver the message."""
        adapter = self._make_adapter(app_id="a", client_secret="b")
        # Initially disconnected
        adapter._running = False
        adapter._qq_api = mock.Mock()
        adapter._qq_api.send_text = mock.AsyncMock(return_value={"id": "msg_123"})
        adapter._RECONNECT_POLL_INTERVAL = 0.1
        adapter._RECONNECT_WAIT_SECONDS = 5.0

        # Schedule reconnection after a short delay
        async def reconnect_after_delay():
            await asyncio.sleep(0.2)
            adapter._running = True
            adapter._qq_ws = object()
            adapter._sdk_connected = True

        asyncio.get_event_loop().create_task(reconnect_after_delay())

        result = await adapter.send("test_openid", "Hello, world!")
        assert result.success
        assert result.message_id == "msg_123"


# ---------------------------------------------------------------------------
# Regression for #78183: httpx timeout empty-string defeats _is_timeout_error
# ---------------------------------------------------------------------------

class TestQQTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, qqbot's send
    paths must preserve the exception type name so the base-layer timeout guard
    (_is_timeout_error) can still recognise it and suppress the duplicate-
    delivery plain-text fallback."""

    def _make_adapter(self, **extra):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b", **extra))

    @pytest.mark.asyncio
    async def test_send_chunk_preserves_read_timeout_type(self, monkeypatch):
        from gateway.platforms.base import BasePlatformAdapter
        import plugins.platforms.qqbot.adapter as qq_adapter_mod

        # Skip the real 1s+2s retry backoff.
        monkeypatch.setattr(qq_adapter_mod.asyncio, "sleep", mock.AsyncMock())
        adapter = self._make_adapter()

        async def _boom(*args, **kwargs):
            raise httpx.ReadTimeout("")

        adapter._send_c2c_text = _boom

        result = await adapter._send_chunk("test_openid", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )



# ---------------------------------------------------------------------------
# Inline keyboards — approval + update-prompt flows
# ---------------------------------------------------------------------------

class TestApprovalButtonData:
    def test_parse_allow_once(self):
        from plugins.platforms.qqbot.keyboards import parse_approval_button_data
        result = parse_approval_button_data("approve:agent:main:qqbot:c2c:UID:allow-once")
        assert result == ("agent:main:qqbot:c2c:UID", "allow-once")


    def test_parse_empty_returns_none(self):
        from plugins.platforms.qqbot.keyboards import parse_approval_button_data
        assert parse_approval_button_data("") is None
        assert parse_approval_button_data(None) is None  # type: ignore[arg-type]


class TestUpdatePromptButtonData:
    def test_parse_yes(self):
        from plugins.platforms.qqbot.keyboards import parse_update_prompt_button_data
        assert parse_update_prompt_button_data("update_prompt:y") == "y"


class TestBuildApprovalKeyboard:

    def test_button_data_embeds_session_key(self):
        from plugins.platforms.qqbot.keyboards import build_approval_keyboard
        kb = build_approval_keyboard("agent:main:qqbot:c2c:UID")
        datas = [b.action.data for b in kb.content.rows[0].buttons]
        assert datas[0] == "approve:agent:main:qqbot:c2c:UID:allow-once"
        assert datas[1] == "approve:agent:main:qqbot:c2c:UID:allow-always"
        assert datas[2] == "approve:agent:main:qqbot:c2c:UID:deny"




class TestBuildApprovalText:


    def test_truncates_long_commands(self):
        from plugins.platforms.qqbot.keyboards import (
            ApprovalRequest, build_approval_text,
        )
        long = "x" * 1000
        req = ApprovalRequest(
            session_key="s", title="t", command_preview=long, cwd="/x",
        )
        text = build_approval_text(req)
        # Preview is truncated to 300 chars; 1000 "x"s would still push the
        # body past 300, but the inline preview specifically must be capped.
        # 2 backtick fences; the content line in between is separate.
        xs_in_preview = sum(line.count("x") for line in text.split("\n") if line and "```" not in line)
        assert xs_in_preview <= 301  # 300 xs + one-off tolerance


class TestInteractionEventParsing:
    def test_parse_c2c_interaction(self):
        from plugins.platforms.qqbot.keyboards import parse_interaction_event
        raw = {
            "id": "interaction-42",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {
                    "button_data": "approve:sess:allow-once",
                    "button_id": "allow",
                },
            },
        }
        ev = parse_interaction_event(raw)
        assert ev.id == "interaction-42"
        assert ev.scene == "c2c"
        assert ev.chat_type == 2
        assert ev.user_openid == "user-1"
        assert ev.button_data == "approve:sess:allow-once"
        assert ev.button_id == "allow"
        assert ev.operator_openid == "user-1"


class TestAdapterInteractionDispatch:
    """End-to-end verification of _on_interaction including ACK + callback."""

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_callback_invoked_with_parsed_event(self):
        adapter = self._make_adapter()

        # Stub ACK so we don't require a live http_client.
        ack_calls = []

        async def fake_ack(interaction_id, code=0):
            ack_calls.append((interaction_id, code))

        adapter._acknowledge_interaction = fake_ack  # type: ignore[assignment]

        received = []

        async def cb(event):
            received.append(event)

        adapter.set_interaction_callback(cb)
        await adapter._on_interaction({
            "id": "i-1",
            "chat_type": 2,
            "user_openid": "user-1",
            "data": {
                "type": 11,
                "resolved": {"button_data": "approve:agent:main:qqbot:c2c:u:deny", "button_id": "deny"},
            },
        })

        assert len(ack_calls) == 1
        assert ack_calls[0][0] == "i-1"
        assert len(received) == 1
        assert received[0].button_data == "approve:agent:main:qqbot:c2c:u:deny"
        assert received[0].scene == "c2c"


# ---------------------------------------------------------------------------
# Quoted-message handling (message_type=103 → msg_elements)
# ---------------------------------------------------------------------------

class TestProcessQuotedContext:
    """Verify the quoted-message pipeline: text + voice STT + images + files."""

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_non_quote_message_returns_empty(self):
        adapter = self._make_adapter()
        d = {"message_type": 0, "content": "hi"}
        out = await adapter._process_quoted_context(d)
        assert out == {"quote_block": "", "image_urls": [], "image_media_types": []}


    @pytest.mark.asyncio
    async def test_quote_with_voice_attachment_runs_stt(self):
        adapter = self._make_adapter()

        # Capture what attachments are passed into _process_attachments.
        captured = []

        async def fake_process(atts):
            captured.append(atts)
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": ["[Voice] hello from the quoted audio"],
                "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [{
                "content": "",
                "attachments": [
                    {"content_type": "audio/silk",
                     "url": "https://qq-cdn/x.silk",
                     "filename": "rec.silk"}
                ],
            }],
        }
        out = await adapter._process_quoted_context(d)

        # The quoted voice attachment must actually flow through STT.
        assert captured and len(captured[0]) == 1
        assert captured[0][0]["content_type"] == "audio/silk"
        assert "[Quoted message]:" in out["quote_block"]
        assert "hello from the quoted audio" in out["quote_block"]


    @pytest.mark.asyncio
    async def test_multiple_elements_concatenated(self):
        adapter = self._make_adapter()

        async def fake_process(atts):
            assert len(atts) == 2
            return {
                "image_urls": [], "image_media_types": [],
                "voice_transcripts": [], "attachment_info": "",
            }

        adapter._process_attachments = fake_process  # type: ignore[assignment]

        d = {
            "message_type": 103,
            "msg_elements": [
                {"content": "first", "attachments": [{"content_type": "image/png", "url": "a"}]},
                {"content": "second", "attachments": [{"content_type": "image/png", "url": "b"}]},
            ],
        }
        out = await adapter._process_quoted_context(d)
        assert "first" in out["quote_block"]
        assert "second" in out["quote_block"]




# ---------------------------------------------------------------------------
# Gateway-contract approval UX — send_exec_approval + default dispatcher
# ---------------------------------------------------------------------------

class TestDefaultInteractionDispatch:
    """Verify the adapter's default INTERACTION_CREATE router."""

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    def test_default_callback_installed_on_init(self):
        """Fresh adapter has a working default interaction callback."""
        adapter = self._make_adapter()
        assert adapter._interaction_callback is not None
        assert adapter._interaction_callback == adapter._default_interaction_dispatch


    @pytest.mark.asyncio
    async def test_approval_click_once_maps_to_once(self):
        """'allow-once' button → resolve_gateway_approval(session, 'once')."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        # Patch the *module-level* function that _default_interaction_dispatch
        # imports lazily.
        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from plugins.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i",
                "chat_type": 2,
                "user_openid": "u-42",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:dm:u-42:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:main:qqbot:dm:u-42", "once", False)]


    @pytest.mark.asyncio
    async def test_approval_click_rejects_unauthorized_operator(self):
        adapter = self._make_adapter()
        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from plugins.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 1,
                "group_openid": "g-1",
                "group_member_openid": "attacker",
                "data": {"resolved": {"button_data": "approve:agent:main:qqbot:group:g-1:owner:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == []

    @pytest.mark.asyncio
    async def test_update_prompt_click_writes_response_file(self, tmp_path, monkeypatch):
        """update_prompt:y click writes 'y' to ~/.hermes/.update_response."""
        adapter = self._make_adapter()
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: hermes_home,
        )

        from plugins.platforms.qqbot.keyboards import parse_interaction_event
        event = parse_interaction_event({
            "id": "i", "chat_type": 2, "user_openid": "u-1",
            "data": {"resolved": {"button_data": "update_prompt:y"}},
        })
        await adapter._default_interaction_dispatch(event)

        response = hermes_home / ".update_response"
        assert response.exists()
        assert response.read_text() == "y"


class TestProfileNamespaceApprovalAuthz:
    """Named-profile (multiplex) session keys must authorize like ``main``.

    ``build_session_key`` namespaces named-profile keys as ``agent:<profile>:...``
    while the default profile keeps the legacy ``agent:main`` prefix
    (``gateway/session.py::_session_key_namespace``). The interaction authz
    parser used to require the literal ``main`` in the namespace slot, so
    every approval button click in a named profile was rejected as
    unauthorized and the pending approval timed out (fail-closed block).
    """

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @staticmethod
    def _parse(key):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter._parse_gateway_session_key(key)

    def test_parse_accepts_named_profile_namespace(self):
        parsed = self._parse("agent:coder:qqbot:dm:u-1")
        assert parsed is not None
        assert parsed["platform"] == "qqbot"
        assert parsed["chat_type"] == "dm"
        assert parsed["chat_id"] == "u-1"

    def test_parse_accepts_main_namespace_with_user_id(self):
        parsed = self._parse("agent:main:qqbot:group:g-1:owner")
        assert parsed is not None
        assert parsed["platform"] == "qqbot"
        assert parsed["chat_type"] == "group"
        assert parsed["chat_id"] == "g-1"
        assert parsed["user_id"] == "owner"

    def test_parse_still_rejects_non_agent_and_malformed_keys(self):
        assert self._parse("session:main:qqbot:c2c:u-1") is None
        assert self._parse("agent::qqbot:c2c:u-1") is None
        assert self._parse("agent:main") is None
        assert self._parse("") is None

    @pytest.mark.asyncio
    async def test_c2c_click_on_named_profile_key_resolves(self):
        """Approval click carrying a named-profile c2c key resolves (was rejected)."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from plugins.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i",
                "chat_type": 2,
                "user_openid": "u-42",
                "data": {"resolved": {"button_data": "approve:agent:coder:qqbot:c2c:u-42:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:coder:qqbot:c2c:u-42", "once", False)]

    @pytest.mark.asyncio
    async def test_group_click_on_named_profile_key_authorizes_session_owner(self):
        """Group approval click under a named profile authorizes the session owner."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from plugins.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 1,
                "group_openid": "g-1",
                "group_member_openid": "owner",
                "data": {"resolved": {"button_data": "approve:agent:coder:qqbot:group:g-1:owner:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == [("agent:coder:qqbot:group:g-1:owner", "once", False)]

    @pytest.mark.asyncio
    async def test_named_profile_key_still_rejects_wrong_operator(self):
        """The namespace relaxation must not weaken the operator check."""
        adapter = self._make_adapter()

        resolve_calls = []

        def fake_resolve(session_key, choice, resolve_all=False):
            resolve_calls.append((session_key, choice, resolve_all))
            return 1

        import tools.approval
        orig = tools.approval.resolve_gateway_approval
        tools.approval.resolve_gateway_approval = fake_resolve
        try:
            from plugins.platforms.qqbot.keyboards import parse_interaction_event
            event = parse_interaction_event({
                "id": "i", "chat_type": 1,
                "group_openid": "g-1",
                "group_member_openid": "attacker",
                "data": {"resolved": {"button_data": "approve:agent:coder:qqbot:group:g-1:owner:allow-once"}},
            })
            await adapter._default_interaction_dispatch(event)
        finally:
            tools.approval.resolve_gateway_approval = orig

        assert resolve_calls == []




class TestSendUpdatePrompt:
    """Verify the cross-adapter send_update_prompt signature + behaviour."""

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_delegates_to_send_with_keyboard(self):
        adapter = self._make_adapter()

        captured = {}

        async def fake_swk(chat_id, content, keyboard, reply_to=None):
            from gateway.platforms.base import SendResult
            captured["chat_id"] = chat_id
            captured["content"] = content
            captured["keyboard"] = keyboard
            captured["reply_to"] = reply_to
            return SendResult(success=True, message_id="mid")

        adapter.send_with_keyboard = fake_swk  # type: ignore[assignment]
        adapter._last_msg_id["u1"] = "prev-msg"

        result = await adapter.send_update_prompt(
            chat_id="u1", prompt="Continue with update?",
            default="y", session_key="ignored", metadata={"x": 1},
        )
        assert result.success
        assert "Continue with update?" in captured["content"]
        assert captured["reply_to"] == "prev-msg"
        # Keyboard has the Yes/No buttons.
        dd = captured["keyboard"].to_dict()
        datas = [b["action"]["data"] for b in dd["content"]["rows"][0]["buttons"]]
        assert datas == ["update_prompt:y", "update_prompt:n"]

# ---------------------------------------------------------------------------
# _process_attachments: video/file path exposure
# ---------------------------------------------------------------------------

class TestProcessAttachmentsPathExposure:
    """Verify that video and file attachments include the cached local path."""

    def _make_adapter(self):
        from plugins.platforms.qqbot.adapter import QQAdapter
        return QQAdapter(_make_config(app_id="a", client_secret="b"))

    @pytest.mark.asyncio
    async def test_video_attachment_includes_path(self):
        adapter = self._make_adapter()

        # Mock _download_and_cache to return a known path
        async def fake_download(url, ct, original_name=""):
            return "/tmp/cache/video_abc123.mp4"

        adapter._download_and_cache = fake_download  # type: ignore[assignment]

        attachments = [
            {
                "content_type": "video/mp4",
                "url": "https://multimedia.nt.qq.com.cn/download/video123",
                "filename": "my_video.mp4",
            }
        ]
        result = await adapter._process_attachments(attachments)

        assert result["image_urls"] == []
        assert result["voice_transcripts"] == []
        info = result["attachment_info"]
        assert "my_video.mp4" in info
        assert "/tmp/cache/video_abc123.mp4" in info




