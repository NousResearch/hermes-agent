"""Tests for XMPP platform adapter.

These tests encode the contract for the XMPP gateway adapter and were
written before the implementation. Running pytest before M2 lands will
produce predictable failures (no Platform.XMPP, no XmppAdapter, no
wiring) — that's the TDD signal.

See:
- ROADMAP.md (project root) — M1 lists each test by name
- docs/adr/0001-xmpp-adapter-architecture.md — design contract
- gateway/platforms/ADDING_A_PLATFORM.md — upstream integration checklist
"""
import asyncio
import inspect
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Skip the entire module if slixmpp isn't installed — the adapter requires it,
# but the test file itself shouldn't break suite collection in environments
# without the optional [xmpp] extra.
slixmpp = pytest.importorskip("slixmpp")

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _Jid:
    """Minimal slixmpp.JID surrogate for stanza stubs."""
    def __init__(self, full):
        self.full = full
        self.bare = full.split("/", 1)[0] if "/" in full else full
        self.resource = full.split("/", 1)[1] if "/" in full else ""

    def __str__(self):
        return self.full


class _StanzaStub(dict):
    """Duck-typed slixmpp Message stanza supporting __getitem__ and get_from()."""
    def __init__(self, *, stanza_type, from_jid, body, stanza_id="incoming-1"):
        super().__init__(type=stanza_type, body=body, id=stanza_id)
        self["from"] = _Jid(from_jid)

    def get_from(self):
        return self["from"]


def _make_xmpp_adapter(monkeypatch, jid="hermes@example.org", password="pw", **extra):
    """Construct an XmppAdapter with sensible test defaults."""
    monkeypatch.setenv("XMPP_ALLOWED_USERS", extra.pop("allowed_users", ""))
    monkeypatch.setenv("XMPP_ALLOW_ALL_USERS", extra.pop("allow_all", ""))
    from gateway.platforms.xmpp import XmppAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "jid": jid,
        "password": password,
        **extra,
    }
    return XmppAdapter(config)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestXmppConfigLoading:
    def test_apply_env_overrides_loads_xmpp(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.setenv("XMPP_PASSWORD", "secret")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.XMPP in config.platforms
        xc = config.platforms[Platform.XMPP]
        assert xc.enabled is True
        assert xc.extra["jid"] == "hermes@example.org"
        assert xc.extra["password"] == "secret"

    def test_xmpp_requires_both_jid_and_password(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        # XMPP_PASSWORD intentionally unset
        monkeypatch.delenv("XMPP_PASSWORD", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.XMPP not in config.platforms

    def test_xmpp_optional_host_and_port(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.setenv("XMPP_PASSWORD", "secret")
        monkeypatch.setenv("XMPP_HOST", "xmpp.example.org")
        monkeypatch.setenv("XMPP_PORT", "5223")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        xc = config.platforms[Platform.XMPP]
        assert xc.extra["host"] == "xmpp.example.org"
        assert xc.extra["port"] == 5223


# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------

class TestXmppAdapterInit:
    def test_init_parses_basic_config(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter.jid == "hermes@example.org"
        assert adapter.platform == Platform.XMPP

    def test_init_parses_allowlist(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, allowed_users="alice@example.org,bob@example.org"
        )
        assert "alice@example.org" in adapter.allowed_users
        assert "bob@example.org" in adapter.allowed_users
        assert len(adapter.allowed_users) == 2

    def test_init_does_not_construct_slixmpp_client(self, monkeypatch):
        # Per ADR-0001: connect() instantiates the client; __init__ must not.
        # Keeps unit tests cheap and makes the connect lifecycle observable.
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter.client is None

    def test_init_parses_muc_rooms_with_optional_nick(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch,
            muc_rooms="room1@conference.example.org,room2@conference.example.org/customnick",
        )
        rooms_by_jid = {r.room: r for r in adapter.muc_rooms}
        assert "room1@conference.example.org" in rooms_by_jid
        assert "room2@conference.example.org" in rooms_by_jid
        assert rooms_by_jid["room2@conference.example.org"].nick == "customnick"


# ---------------------------------------------------------------------------
# Inbound message dispatch
# ---------------------------------------------------------------------------

class TestXmppMessageDispatch:
    """Verify that _on_message converts a stanza into the correct
    MessageEvent and dispatches via BasePlatformAdapter.handle_message.

    We spy on ``adapter.handle_message`` directly (not via
    ``set_message_handler``) because the base's handle_message runs the
    full agent pipeline — invoking it with an AsyncMock handler triggers
    media extraction and other downstream work that isn't relevant to
    these unit tests.
    """

    @pytest.mark.asyncio
    async def test_dm_message_dispatches_to_handler(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="alice@example.org/laptop", body="hello"
        )
        await adapter._on_message(stanza)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello"
        assert event.source.platform == Platform.XMPP
        assert event.source.chat_type == "dm"
        # Bare JID per ADR-0001: resources don't fracture sessions.
        assert event.source.chat_id == "alice@example.org"
        assert event.source.user_id == "alice@example.org"

    @pytest.mark.asyncio
    async def test_groupchat_message_dispatches_with_muc_metadata(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch,
            muc_rooms="hermes-room@conference.example.org",
            allowed_users="alice@example.org",
        )
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="groupchat",
            from_jid="hermes-room@conference.example.org/alice",
            body="howdy",
        )
        await adapter._on_message(stanza)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "hermes-room@conference.example.org"
        # MUC nick is the from-resource and goes into user_name.
        assert event.source.user_name == "alice"

    @pytest.mark.asyncio
    async def test_self_message_filtered(self, monkeypatch):
        # Adapter must drop messages whose bare JID matches its own
        # (slixmpp will echo our own MUC messages back to _on_message).
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="hermes@example.org/laptop", body="echo"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_sender_filtered(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="mallory@evil.example/x", body="hi"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allow_all_users_short_circuits_allowlist(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allow_all="1")
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="anyone@elsewhere.example/x", body="hi"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_headline_message_dropped(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="headline", from_jid="alice@example.org/laptop", body="MOTD"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_stanza_dropped(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()

        stanza = _StanzaStub(
            stanza_type="error", from_jid="alice@example.org/laptop", body=""
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Outbound: send / send_typing / send_image_file
# ---------------------------------------------------------------------------

class TestXmppSend:
    @pytest.mark.asyncio
    async def test_send_text_emits_chat_stanza(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.client = MagicMock()
        sent = []

        def _capture(**kw):
            sent.append(kw)
            out = MagicMock()
            out.__getitem__ = lambda self, key: "stanza-1" if key == "id" else ""
            return out

        adapter.client.send_message = _capture

        result = await adapter.send("alice@example.org", "hello")
        assert result.success is True
        assert result.message_id  # populated from stanza id
        assert sent[0]["mto"] == "alice@example.org"
        assert sent[0]["mbody"] == "hello"
        assert sent[0]["mtype"] == "chat"

    @pytest.mark.asyncio
    async def test_send_to_known_muc_uses_groupchat_type(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, muc_rooms="hermes-room@conference.example.org"
        )
        adapter.client = MagicMock()
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw) or MagicMock()

        await adapter.send("hermes-room@conference.example.org", "group hello")
        assert sent[0]["mtype"] == "groupchat"

    @pytest.mark.asyncio
    async def test_send_typing_emits_chat_state_composing(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.client = MagicMock()
        adapter._registered_plugins.add("xep_0085")
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw)

        await adapter.send_typing("alice@example.org")
        assert any(kw.get("mchat_state") == "composing" for kw in sent)

    @pytest.mark.asyncio
    async def test_send_typing_noop_when_xep0085_unavailable(self, monkeypatch):
        # Regression: if XEP-0085 plugin failed to register, send_typing must
        # silently skip rather than calling send_message with an unknown
        # mchat_state kwarg (which raises TypeError in slixmpp).
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.client = MagicMock()
        # Note: xep_0085 NOT added to _registered_plugins
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw)

        await adapter.send_typing("alice@example.org")
        await adapter.stop_typing("alice@example.org")
        assert sent == []

    @pytest.mark.asyncio
    async def test_send_image_file_uploads_via_xep_0363(self, tmp_path, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.client = MagicMock()
        adapter._registered_plugins.add("xep_0363")
        # XEP-0363 plugin returns a public URL the adapter then sends in the body.
        upload_mock = AsyncMock(return_value="https://files.example.org/abc.jpg")
        adapter.client.__getitem__ = MagicMock(return_value=MagicMock(upload_file=upload_mock))
        send_capture = MagicMock(return_value=MagicMock(__getitem__=lambda s, k: "img-1"))
        adapter.client.send_message = send_capture

        path = tmp_path / "x.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

        result = await adapter.send_image_file("alice@example.org", str(path), caption="cap")
        assert result.success is True
        upload_mock.assert_awaited_once()
        # content_type hint should be passed for known extensions (XEP-0363 §5).
        upload_kwargs = upload_mock.call_args.kwargs
        assert upload_kwargs.get("content_type") == "image/jpeg"
        sent_kw = send_capture.call_args.kwargs
        assert "https://files.example.org/abc.jpg" in (sent_kw.get("mbody") or "")


# ---------------------------------------------------------------------------
# get_chat_info
# ---------------------------------------------------------------------------

class TestXmppGetChatInfo:
    @pytest.mark.asyncio
    async def test_dm_chat_info(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        info = await adapter.get_chat_info("alice@example.org")
        assert info["chat_id"] == "alice@example.org"
        assert info["type"] == "dm"
        assert "name" in info

    @pytest.mark.asyncio
    async def test_muc_chat_info(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, muc_rooms="hermes-room@conference.example.org"
        )
        info = await adapter.get_chat_info("hermes-room@conference.example.org")
        assert info["type"] == "group"

    def test_is_muc_recognizes_common_subdomain_conventions(self, monkeypatch):
        # Various XMPP servers use different MUC subdomain conventions.
        # The heuristic should cover the common ones; explicit XMPP_MUC_ROOMS
        # config still takes precedence for any non-standard prefix.
        adapter = _make_xmpp_adapter(monkeypatch)
        for room in (
            "team@conference.example.org",
            "team@muc.example.org",
            "team@rooms.example.org",
            "team@chat.example.org",
            "team@groups.example.org",
        ):
            assert adapter._is_muc(room), f"expected MUC heuristic to match {room}"
        # 1:1 chats must not be misclassified as groupchat.
        assert adapter._is_muc("alice@example.org") is False


# ---------------------------------------------------------------------------
# Connect: TLS posture (ADR-0001 §5)
# ---------------------------------------------------------------------------

class TestXmppConnectTLSPosture:
    @pytest.mark.asyncio
    async def test_connect_forces_starttls(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        fake_client = MagicMock()
        # connect() now waits for the session to come up before returning, so
        # simulate session_start firing during client.connect().
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )

        with patch(
            "gateway.platforms.xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ) as ctor:
            ok = await adapter.connect()
            assert ok is True
            assert ctor.called
            # ADR-0001 §5: plaintext must be refused. Assert the knob slixmpp
            # actually honors (enable_plaintext / enable_starttls); the legacy
            # force_starttls attribute is inert in current slixmpp.
            assert fake_client.enable_plaintext is False
            assert fake_client.enable_starttls is True


# ---------------------------------------------------------------------------
# SessionSource roundtrip
# ---------------------------------------------------------------------------

class TestXmppSessionSourceRoundtrip:
    def test_dm_roundtrip(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        src = adapter.build_source(
            chat_id="alice@example.org",
            chat_type="dm",
            user_id="alice@example.org",
            user_name="Alice",
        )
        from gateway.session import SessionSource
        roundtripped = SessionSource.from_dict(src.to_dict())
        assert roundtripped.platform == Platform.XMPP
        assert roundtripped.chat_id == "alice@example.org"
        assert roundtripped.chat_type == "dm"
        assert roundtripped.user_id == "alice@example.org"


# ---------------------------------------------------------------------------
# 16-step integration wiring (ADDING_A_PLATFORM.md)
# ---------------------------------------------------------------------------

class TestXmppIntegrationWiring:
    """Verifies the global wiring required by ADDING_A_PLATFORM.md.

    These read source via inspect rather than executing the integration
    points, which keeps the tests fast and side-effect-free.
    """

    def test_platform_enum_has_xmpp(self):
        assert Platform.XMPP.value == "xmpp"

    def test_authorization_maps_present(self):
        # Authorization (``_is_user_authorized`` / ``_get_unauthorized_dm_behavior``)
        # lives in the ``gateway.authz_mixin`` mixin that the gateway runner
        # composes in. XMPP must be registered in those maps so
        # XMPP_ALLOWED_USERS / XMPP_ALLOW_ALL_USERS gate inbound senders.
        from gateway import authz_mixin
        src = inspect.getsource(authz_mixin)
        assert "XMPP_ALLOWED_USERS" in src
        assert "XMPP_ALLOW_ALL_USERS" in src

    def test_adapter_factory_creates_xmpp(self):
        from gateway import run
        src = inspect.getsource(run)
        assert "Platform.XMPP" in src
        assert "from gateway.platforms.xmpp" in src

    def test_send_message_tool_routes_xmpp(self):
        from tools import send_message_tool as smt_module
        src = inspect.getsource(smt_module)
        assert '"xmpp"' in src or "'xmpp'" in src
        assert "Platform.XMPP" in src

    def test_cron_scheduler_routes_xmpp(self):
        from cron import scheduler
        src = inspect.getsource(scheduler)
        assert '"xmpp"' in src or "'xmpp'" in src

    def test_platform_hints_include_xmpp(self):
        from agent.prompt_builder import PLATFORM_HINTS
        assert "xmpp" in PLATFORM_HINTS

    def test_status_display_lists_xmpp(self):
        from hermes_cli import status
        src = inspect.getsource(status)
        assert "XMPP" in src or "xmpp" in src
        assert "XMPP_JID" in src

    def test_gateway_setup_wizard_includes_xmpp(self):
        from hermes_cli import gateway as gw
        src = inspect.getsource(gw)
        assert '"xmpp"' in src or "'xmpp'" in src


# ---------------------------------------------------------------------------
# Fatal-error handling / reconnect signalling (issue #28919)
# ---------------------------------------------------------------------------

class TestXmppFatalErrorHandling:
    """Regression tests for issue #28919.

    The original adapter reacted to slixmpp's per-mechanism ``failed_auth``
    (which fires once for *each* SASL mechanism the server rejects, even when a
    later mechanism succeeds) and so marked a perfectly healthy connection
    dead. It also marked an unexpected disconnect without ever notifying the
    gateway, so the reconnect watcher never ran and the bridge silently died
    inside an otherwise-healthy gateway. The adapter must now:

      * react to ``failed_all_auth`` (fires once, after *all* mechanisms fail),
        never ``failed_auth``;
      * call the fatal-error handler on any mid-life fatal so the gateway's
        reconnect watcher is driven;
      * distinguish a deliberate ``disconnect()`` from an unexpected drop;
      * not let a late ``disconnected`` clobber a precise auth failure.
    """

    @pytest.mark.asyncio
    async def test_connect_registers_failed_all_auth_not_failed_auth(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        events = []
        fake_client.add_event_handler = MagicMock(
            side_effect=lambda ev, handler: events.append(ev)
        )
        with patch(
            "gateway.platforms.xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            await adapter.connect()
        assert "failed_all_auth" in events
        assert "failed_auth" not in events

    @pytest.mark.asyncio
    async def test_connect_accepts_is_reconnect_kwarg(self, monkeypatch):
        # BasePlatformAdapter.connect(*, is_reconnect=False): the gateway's
        # reconnect watcher calls connect(is_reconnect=True).
        adapter = _make_xmpp_adapter(monkeypatch)
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        with patch(
            "gateway.platforms.xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            assert await adapter.connect(is_reconnect=True) is True

    @pytest.mark.asyncio
    async def test_session_start_marks_session_established(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter.client = MagicMock()
        adapter.client.get_roster = AsyncMock()
        adapter._session_ready = asyncio.Event()
        await adapter._on_session_start(None)
        assert adapter._session_established is True
        assert adapter._session_ready.is_set()

    @pytest.mark.asyncio
    async def test_failed_all_auth_sets_nonretryable_fatal_and_notifies(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        notified = []
        adapter.set_fatal_error_handler(lambda a: notified.append(a))
        await adapter._on_failed_all_auth(None)
        assert adapter.has_fatal_error
        assert adapter.fatal_error_code == "xmpp_auth_failed"
        assert adapter.fatal_error_retryable is False
        assert notified == [adapter]

    @pytest.mark.asyncio
    async def test_unexpected_disconnect_escalates_retryable_and_notifies(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        notified = []
        adapter.set_fatal_error_handler(lambda a: notified.append(a))
        adapter._session_established = True  # a live session was up
        await adapter._on_disconnected(None)
        assert adapter.has_fatal_error
        assert adapter.fatal_error_code == "xmpp_connection_lost"
        assert adapter.fatal_error_retryable is True
        assert notified == [adapter]
        assert adapter._session_established is False

    @pytest.mark.asyncio
    async def test_deliberate_disconnect_does_not_escalate(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        notified = []
        adapter.set_fatal_error_handler(lambda a: notified.append(a))
        adapter._closing = True  # disconnect() sets this before teardown
        await adapter._on_disconnected(None)
        assert not adapter.has_fatal_error
        assert notified == []

    @pytest.mark.asyncio
    async def test_disconnect_after_auth_failure_keeps_auth_fatal(self, monkeypatch):
        # failed_all_auth fires, then slixmpp tears the stream down and fires
        # 'disconnected'. The later disconnect must not overwrite the precise,
        # non-retryable auth error with a generic retryable connection-lost.
        adapter = _make_xmpp_adapter(monkeypatch)
        await adapter._on_failed_all_auth(None)
        await adapter._on_disconnected(None)
        assert adapter.fatal_error_code == "xmpp_auth_failed"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_disconnect_sets_closing_flag(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        await adapter.disconnect()
        assert adapter._closing is True

    @pytest.mark.asyncio
    async def test_connect_passes_configured_host_and_port(self, monkeypatch):
        # XMPP_HOST/XMPP_PORT must actually reach slixmpp.connect — a prior bug
        # passed connect(address=...) (rejected by current slixmpp) and the
        # fallback silently dropped the host.
        adapter = _make_xmpp_adapter(
            monkeypatch, host="xmpp.example.org", port=5223
        )
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        with patch(
            "gateway.platforms.xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            ok = await adapter.connect()
        assert ok is True
        fake_client.connect.assert_called_once_with(
            host="xmpp.example.org", port=5223
        )

    @pytest.mark.asyncio
    async def test_connect_timeout_escalates_retryable_fatal(self, monkeypatch):
        # If the session never establishes (unreachable server / wrong host),
        # connect() must report failure with a retryable fatal so the gateway
        # keeps the platform in its reconnect queue — not silently 'connected'.
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter._CONNECT_TIMEOUT_SECS = 0.05  # don't actually wait
        fake_client = MagicMock()
        fake_client.connect = MagicMock(return_value=None)  # never fires session_start
        with patch(
            "gateway.platforms.xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            ok = await adapter.connect()
        assert ok is False
        assert adapter.has_fatal_error
        assert adapter.fatal_error_code == "xmpp_connect_timeout"
        assert adapter.fatal_error_retryable is True

    @pytest.mark.asyncio
    async def test_media_methods_accept_base_keyword_names(self, monkeypatch, tmp_path):
        # The gateway calls these by the BasePlatformAdapter keyword names
        # (audio_path/video_path/file_path/image_path). A renamed positional
        # arg raises TypeError and the attachment silently never delivers.
        from unittest.mock import AsyncMock as _AsyncMock
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter._upload_and_send = _AsyncMock(return_value=MagicMock(success=True))
        f = tmp_path / "media.bin"
        f.write_bytes(b"data")
        await adapter.send_voice(chat_id="a@example.org", audio_path=str(f))
        await adapter.send_video(chat_id="a@example.org", video_path=str(f), caption="v")
        await adapter.send_document(chat_id="a@example.org", file_path=str(f), caption="d")
        await adapter.send_image_file(chat_id="a@example.org", image_path=str(f))
        assert adapter._upload_and_send.await_count == 4
