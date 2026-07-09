"""Tests for the XMPP platform adapter plugin.

The adapter ships as a bundled platform plugin under
``plugins/platforms/xmpp/`` and registers through the platform registry —
see gateway/platforms/ADDING_A_PLATFORM.md (Plugin Path). These tests cover:

1. adapter init / config parsing (JID, allowlists, MUC rooms)
2. inbound stanza → MessageEvent dispatch (DM, MUC, self-echo, authz)
3. outbound send routing (chat vs groupchat), typing, XEP-0363 uploads
4. TLS posture on connect (plaintext refused)
5. fatal-error handling / reconnect signalling (issue #28919)
6. register() metadata + env-enablement + standalone-send hook shapes
"""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Skip the entire module if slixmpp isn't installed — the adapter requires it,
# but the test file itself shouldn't break suite collection in environments
# without the optional [xmpp] extra.
slixmpp = pytest.importorskip("slixmpp")

from gateway.config import Platform, PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/xmpp/adapter.py under plugin_adapter_xmpp so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_xmpp = load_plugin_adapter("xmpp")

XmppAdapter = _xmpp.XmppAdapter


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
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "jid": jid,
        "password": password,
        **extra,
    }
    return XmppAdapter(config)


class _CaptureCtx:
    """Capture the kwargs the plugin passes to ctx.register_platform()."""
    def __init__(self):
        self.kwargs = None

    def register_platform(self, **kwargs):
        self.kwargs = kwargs


# ---------------------------------------------------------------------------
# Env enablement (replaces the pre-plugin _apply_env_overrides integration)
# ---------------------------------------------------------------------------

class TestXmppEnvEnablement:
    def test_env_enablement_seeds_extra(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.setenv("XMPP_PASSWORD", "secret")
        for var in ("XMPP_HOST", "XMPP_PORT", "XMPP_MUC_ROOMS", "XMPP_MUC_NICK",
                    "XMPP_HOME_CHANNEL"):
            monkeypatch.delenv(var, raising=False)

        seed = _xmpp._env_enablement()
        assert seed is not None
        assert seed["jid"] == "hermes@example.org"
        assert seed["password"] == "secret"
        assert "host" not in seed
        assert "home_channel" not in seed

    def test_env_enablement_requires_both_jid_and_password(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.delenv("XMPP_PASSWORD", raising=False)
        assert _xmpp._env_enablement() is None

    def test_env_enablement_optional_host_port_and_home(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.setenv("XMPP_PASSWORD", "secret")
        monkeypatch.setenv("XMPP_HOST", "xmpp.example.org")
        monkeypatch.setenv("XMPP_PORT", "5223")
        monkeypatch.setenv("XMPP_HOME_CHANNEL", "me@example.org")
        monkeypatch.setenv("XMPP_HOME_CHANNEL_NAME", "Reports")

        seed = _xmpp._env_enablement()
        assert seed["host"] == "xmpp.example.org"
        assert seed["port"] == 5223
        # The special home_channel key becomes a HomeChannel on the
        # PlatformConfig via the core env-enablement hook.
        assert seed["home_channel"] == {"chat_id": "me@example.org", "name": "Reports"}

    def test_env_enablement_ignores_bad_port(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "hermes@example.org")
        monkeypatch.setenv("XMPP_PASSWORD", "secret")
        monkeypatch.setenv("XMPP_PORT", "not-a-port")
        seed = _xmpp._env_enablement()
        assert "port" not in seed


# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------

class TestXmppAdapterInit:
    def test_init_parses_basic_config(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter.jid == "hermes@example.org"
        assert adapter.platform == Platform("xmpp")

    def test_init_parses_allowlist(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, allowed_users="alice@example.org,bob@example.org"
        )
        assert "alice@example.org" in adapter.allowed_users
        assert "bob@example.org" in adapter.allowed_users
        assert len(adapter.allowed_users) == 2

    def test_init_does_not_construct_slixmpp_client(self, monkeypatch):
        # connect() instantiates the client; __init__ must not. Keeps unit
        # tests cheap and makes the connect lifecycle observable.
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
        assert event.source.platform == Platform("xmpp")
        assert event.source.chat_type == "dm"
        # Bare JID: resources don't fracture sessions.
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
# Connect: TLS posture
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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ) as ctor:
            ok = await adapter.connect()
            assert ok is True
            assert ctor.called
            # Plaintext must be refused. Assert the knob slixmpp actually
            # honors (enable_plaintext / enable_starttls); the legacy
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
        # Platform("xmpp") resolves via the bundled-plugin pseudo-member
        # (identity-stable through Platform._missing_).
        assert roundtripped.platform == Platform("xmpp")
        assert roundtripped.chat_id == "alice@example.org"
        assert roundtripped.chat_type == "dm"
        assert roundtripped.user_id == "alice@example.org"


# ---------------------------------------------------------------------------
# Plugin registration contract (replaces the pre-plugin core wiring tests)
# ---------------------------------------------------------------------------

class TestXmppRegistration:
    """The plugin hooks replace every core integration point the built-in
    path required (authz maps, cron delivery, send_message routing, prompt
    hints, setup wizard, status display) — assert each hook is wired.
    """

    def _registered_kwargs(self):
        ctx = _CaptureCtx()
        _xmpp.register(ctx)
        assert ctx.kwargs is not None
        return ctx.kwargs

    def test_register_platform_name_resolves_in_enum(self):
        kw = self._registered_kwargs()
        assert kw["name"] == "xmpp"
        # Bundled plugin dir makes Platform("xmpp") a valid, identity-stable
        # pseudo-member.
        assert Platform("xmpp") is Platform("xmpp")
        assert Platform("xmpp").value == "xmpp"

    def test_register_wires_authorization_envs(self):
        kw = self._registered_kwargs()
        assert kw["allowed_users_env"] == "XMPP_ALLOWED_USERS"
        assert kw["allow_all_env"] == "XMPP_ALLOW_ALL_USERS"

    def test_register_wires_cron_delivery(self):
        kw = self._registered_kwargs()
        assert kw["cron_deliver_env_var"] == "XMPP_HOME_CHANNEL"
        assert callable(kw["standalone_sender_fn"])
        assert asyncio.iscoroutinefunction(kw["standalone_sender_fn"])

    def test_register_wires_env_enablement_and_setup(self):
        kw = self._registered_kwargs()
        assert kw["env_enablement_fn"] is _xmpp._env_enablement
        assert callable(kw["setup_fn"])
        assert kw["required_env"] == ["XMPP_JID", "XMPP_PASSWORD"]

    def test_register_provides_platform_hint(self):
        kw = self._registered_kwargs()
        hint = kw["platform_hint"]
        assert "XMPP" in hint
        assert "XEP-0363" in hint

    def test_validate_config_requires_jid_and_password(self):
        kw = self._registered_kwargs()
        validate = kw["validate_config"]
        assert validate(PlatformConfig(extra={"jid": "a@b", "password": "x"})) is True
        assert validate(PlatformConfig(extra={"jid": "a@b"})) is False
        assert validate(PlatformConfig()) is False
        # is_connected shares the same "credentials present" definition.
        assert kw["is_connected"] is validate

    def test_adapter_factory_builds_adapter(self, monkeypatch):
        monkeypatch.setenv("XMPP_ALLOWED_USERS", "")
        monkeypatch.setenv("XMPP_ALLOW_ALL_USERS", "")
        kw = self._registered_kwargs()
        adapter = kw["adapter_factory"](
            PlatformConfig(extra={"jid": "hermes@example.org", "password": "pw"})
        )
        assert isinstance(adapter, XmppAdapter)
        assert adapter.jid == "hermes@example.org"

    @pytest.mark.asyncio
    async def test_standalone_send_success_shape(self, monkeypatch):
        ok = AsyncMock(return_value={"success": True, "message_id": "m1", "error": None})
        monkeypatch.setattr(_xmpp, "send_xmpp_message", ok)
        result = await _xmpp._standalone_send(
            PlatformConfig(), "alice@example.org", "hi"
        )
        assert result == {"success": True, "message_id": "m1"}

    @pytest.mark.asyncio
    async def test_standalone_send_error_shape(self, monkeypatch):
        fail = AsyncMock(return_value={"success": False, "error": "auth failed"})
        monkeypatch.setattr(_xmpp, "send_xmpp_message", fail)
        result = await _xmpp._standalone_send(
            PlatformConfig(), "alice@example.org", "hi"
        )
        assert result == {"error": "auth failed"}

    @pytest.mark.asyncio
    async def test_standalone_send_never_raises(self, monkeypatch):
        boom = AsyncMock(side_effect=RuntimeError("socket exploded"))
        monkeypatch.setattr(_xmpp, "send_xmpp_message", boom)
        result = await _xmpp._standalone_send(
            PlatformConfig(), "alice@example.org", "hi"
        )
        assert "error" in result and "socket exploded" in result["error"]


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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
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
        adapter = _make_xmpp_adapter(monkeypatch)
        adapter._upload_and_send = AsyncMock(return_value=MagicMock(success=True))
        f = tmp_path / "media.bin"
        f.write_bytes(b"data")
        await adapter.send_voice(chat_id="a@example.org", audio_path=str(f))
        await adapter.send_video(chat_id="a@example.org", video_path=str(f), caption="v")
        await adapter.send_document(chat_id="a@example.org", file_path=str(f), caption="d")
        await adapter.send_image_file(chat_id="a@example.org", image_path=str(f))
        assert adapter._upload_and_send.await_count == 4


# ---------------------------------------------------------------------------
# Message chunking / MAX_MESSAGE_LENGTH (delivery pipeline)
# ---------------------------------------------------------------------------

class TestXmppChunking:
    """The gateway's stream consumer clips at 4096 chars for adapters that
    don't define MAX_MESSAGE_LENGTH; the adapter must define it and its own
    send paths must split (never truncate) longer bodies.
    """

    def test_max_message_length_default(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter.MAX_MESSAGE_LENGTH == 10000

    def test_max_message_length_override_via_extra(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, max_message_length="2000")
        assert adapter.MAX_MESSAGE_LENGTH == 2000

    def test_max_message_length_override_via_env(self, monkeypatch):
        monkeypatch.setenv("XMPP_MAX_MESSAGE_LENGTH", "3000")
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter.MAX_MESSAGE_LENGTH == 3000

    def test_max_message_length_invalid_falls_back(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, max_message_length="not-a-number")
        assert adapter.MAX_MESSAGE_LENGTH == 10000

    def test_chunk_body_short_is_single_chunk(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter._chunk_body("hello") == ["hello"]
        assert adapter._chunk_body("") == [""]

    def test_chunk_body_splits_long_content(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, max_message_length="100")
        content = "word " * 100  # 500 chars
        chunks = adapter._chunk_body(content)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)
        # No content lost (chunk indicators like "(1/5)" may be added).
        joined = "".join(chunks)
        assert "word" in joined

    def test_fallback_split_prefers_boundaries(self, monkeypatch):
        chunks = XmppAdapter._fallback_split("aaa bbb\nccc ddd eee fff", 10)
        assert all(len(c) <= 10 for c in chunks)
        assert "".join(c.replace(" ", "").replace("\n", "") for c in chunks) == \
            "aaabbbcccdddeeefff"

    def test_fallback_split_handles_unbreakable_run(self, monkeypatch):
        chunks = XmppAdapter._fallback_split("x" * 25, 10)
        assert chunks == ["x" * 10, "x" * 10, "x" * 5]

    @pytest.mark.asyncio
    async def test_send_splits_long_plaintext_into_stanzas(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, max_message_length="100")
        adapter.client = MagicMock()
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw) or MagicMock()

        result = await adapter.send("alice@example.org", "word " * 100)
        assert result.success is True
        assert len(sent) > 1
        assert all(len(kw["mbody"]) <= 100 for kw in sent)


# ---------------------------------------------------------------------------
# MUC correctness (reflection filter, history replay, _known_dms)
# ---------------------------------------------------------------------------

class TestXmppMucCorrectness:
    @pytest.mark.asyncio
    async def test_muc_reflection_of_own_nick_filtered(self, monkeypatch):
        # In a MUC the from-JID is room@host/nick, so the bare-JID self-check
        # never matches; the bot's own reflected messages must be dropped by
        # nick or it replies to itself in an endless loop.
        adapter = _make_xmpp_adapter(
            monkeypatch, muc_rooms="room@conference.example.org"
        )
        adapter.handle_message = AsyncMock()
        stanza = _StanzaStub(
            stanza_type="groupchat",
            from_jid="room@conference.example.org/hermes",  # default nick
            body="echo of our own message",
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_muc_reflection_respects_per_room_nick(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, muc_rooms="room@conference.example.org/butler"
        )
        adapter.handle_message = AsyncMock()
        stanza = _StanzaStub(
            stanza_type="groupchat",
            from_jid="room@conference.example.org/butler",
            body="echo",
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_muc_history_replay_dropped(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, muc_rooms="room@conference.example.org"
        )
        adapter.handle_message = AsyncMock()
        stanza = _StanzaStub(
            stanza_type="groupchat",
            from_jid="room@conference.example.org/alice",
            body="old message from history",
        )
        stanza.get_plugin = MagicMock(return_value=MagicMock())  # delay stamp present
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_registers_message_handler_once(self, monkeypatch):
        # slixmpp dispatches MUC stanzas to both "message" and
        # "groupchat_message"; registering both would dispatch every group
        # message to the agent twice.
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
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            await adapter.connect()
        assert events.count("message") == 1
        assert "groupchat_message" not in events

    @pytest.mark.asyncio
    async def test_known_dm_beats_muc_domain_heuristic(self, monkeypatch):
        # Snikket-style servers put users on chat.example.com, which matches
        # the MUC subdomain heuristic. A JID we've seen DM us must never be
        # replied to with mtype=groupchat.
        adapter = _make_xmpp_adapter(
            monkeypatch, allowed_users="alice@chat.example.com"
        )
        adapter.handle_message = AsyncMock()
        assert adapter._is_muc("alice@chat.example.com") is True  # heuristic alone
        stanza = _StanzaStub(
            stanza_type="chat", from_jid="alice@chat.example.com/phone", body="hi"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_awaited_once()
        assert adapter._is_muc("alice@chat.example.com") is False  # learned DM


# ---------------------------------------------------------------------------
# Presence subscription (enables OMEMO device-list PEP push to peers)
# ---------------------------------------------------------------------------

class _PresenceStub:
    def __init__(self, from_jid):
        self._from = _Jid(from_jid)

    def get_from(self):
        return self._from


class TestXmppPresenceSubscription:
    @pytest.mark.asyncio
    async def test_subscribe_from_allowed_user_is_approved_mutually(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.client = MagicMock()
        sent = []
        adapter.client.send_presence = lambda **kw: sent.append(kw)
        await adapter._on_subscribe(_PresenceStub("alice@example.org/phone"))
        types = [kw.get("ptype") for kw in sent]
        assert "subscribed" in types  # approve their request
        assert "subscribe" in types   # subscribe back → mutual
        assert all(kw["pto"] == "alice@example.org" for kw in sent)

    @pytest.mark.asyncio
    async def test_subscribe_from_unauthorized_is_ignored(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.client = MagicMock()
        sent = []
        adapter.client.send_presence = lambda **kw: sent.append(kw)
        await adapter._on_subscribe(_PresenceStub("mallory@evil.example/x"))
        assert sent == []

    @pytest.mark.asyncio
    async def test_subscribe_allow_all_approves_anyone(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allow_all="1")
        adapter.client = MagicMock()
        sent = []
        adapter.client.send_presence = lambda **kw: sent.append(kw)
        await adapter._on_subscribe(_PresenceStub("anyone@elsewhere.example/x"))
        assert any(kw.get("ptype") == "subscribed" for kw in sent)

    @pytest.mark.asyncio
    async def test_session_start_requests_subscription_to_allowed_users(self, monkeypatch):
        adapter = _make_xmpp_adapter(
            monkeypatch, allowed_users="alice@example.org,bob@example.org"
        )
        adapter.client = MagicMock()
        adapter.client.get_roster = AsyncMock()
        adapter._session_ready = asyncio.Event()
        sent = []
        adapter.client.send_presence = lambda **kw: sent.append(kw)
        await adapter._on_session_start(None)
        subs = {kw["pto"] for kw in sent if kw.get("ptype") == "subscribe"}
        assert subs == {"alice@example.org", "bob@example.org"}

    @pytest.mark.asyncio
    async def test_connect_gates_auto_authorize_and_registers_handler(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, allowed_users="alice@example.org")
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        events = []
        fake_client.add_event_handler = MagicMock(
            side_effect=lambda ev, h: events.append(ev)
        )
        with patch(
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            await adapter.connect()
        assert "presence_subscribe" in events
        # Blanket auto-authorize would approve anyone; it must be off so only
        # allow-listed peers get into the roster.
        assert fake_client.roster.auto_authorize is False
        assert fake_client.roster.auto_subscribe is False


# ---------------------------------------------------------------------------
# OMEMO (XEP-0384)
# ---------------------------------------------------------------------------

omemo_required = pytest.mark.skipif(
    not _xmpp.SLIXMPP_OMEMO_AVAILABLE,
    reason="slixmpp-omemo not installed (ships with hermes-agent[xmpp])",
)


class TestXmppOmemoConfig:
    def test_omemo_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("XMPP_OMEMO_ENABLED", raising=False)
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter._omemo_enabled is True

    def test_omemo_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("XMPP_OMEMO_ENABLED", "false")
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter._omemo_enabled is False

    def test_omemo_disabled_via_extra(self, monkeypatch):
        adapter = _make_xmpp_adapter(monkeypatch, omemo_enabled="off")
        assert adapter._omemo_enabled is False

    def test_omemo_storage_path_default_and_override(self, monkeypatch):
        monkeypatch.delenv("XMPP_OMEMO_STORAGE_PATH", raising=False)
        adapter = _make_xmpp_adapter(monkeypatch)
        assert adapter._omemo_storage_path.endswith("xmpp_omemo.json")
        adapter2 = _make_xmpp_adapter(monkeypatch, omemo_storage_path="/tmp/keys.json")
        assert adapter2._omemo_storage_path == "/tmp/keys.json"


@omemo_required
class TestXmppOmemoLazyLoad:
    """OMEMO ships with the platform and auto-installs (Matrix-style): the
    base types are None at import if the package isn't present yet, then
    ``_ensure_omemo_loaded`` rebinds them and rebuilds the plugin classes
    after the lazy install — the ``ensure_and_bind`` idiom.
    """

    def test_ensure_omemo_loaded_rebuilds_after_reset(self, monkeypatch):
        # Simulate the state right after import on a box where the package
        # wasn't installed yet, then a successful lazy install rebinds.
        monkeypatch.setattr(_xmpp, "SLIXMPP_OMEMO_AVAILABLE", False)
        monkeypatch.setattr(_xmpp, "_XEP_0384Impl", None)
        monkeypatch.setattr(_xmpp, "_StorageImpl", None)

        import tools.lazy_deps as ld

        def fake_ensure_and_bind(feature, importer, target_globals, *, prompt=False):
            assert feature == "platform.xmpp"  # OMEMO folded into the platform group
            target_globals.update(importer())   # emulate real post-install rebind
            return True

        monkeypatch.setattr(ld, "ensure_and_bind", fake_ensure_and_bind)

        assert _xmpp._ensure_omemo_loaded() is True
        assert _xmpp.SLIXMPP_OMEMO_AVAILABLE is True
        assert _xmpp._XEP_0384Impl is not None
        assert _xmpp._StorageImpl is not None
        assert issubclass(_xmpp._XEP_0384Impl, _xmpp.XEP_0384)

    def test_ensure_omemo_loaded_failure_stays_tls_only(self, monkeypatch):
        monkeypatch.setattr(_xmpp, "SLIXMPP_OMEMO_AVAILABLE", False)
        monkeypatch.setattr(_xmpp, "_XEP_0384Impl", None)
        import tools.lazy_deps as ld
        monkeypatch.setattr(ld, "ensure_and_bind", lambda *a, **k: False)
        assert _xmpp._ensure_omemo_loaded() is False

    def test_ensure_omemo_loaded_idempotent_when_present(self, monkeypatch):
        # When already loaded it must short-circuit without touching lazy_deps.
        import tools.lazy_deps as ld

        def _boom(*a, **k):
            raise AssertionError("ensure_and_bind should not be called when loaded")

        monkeypatch.setattr(ld, "ensure_and_bind", _boom)
        assert _xmpp._ensure_omemo_loaded() is True

    def test_check_requirements_loads_omemo(self):
        # check_fn runs before adapter construction; it must leave OMEMO ready
        # so connect() sees SLIXMPP_OMEMO_AVAILABLE=True.
        assert _xmpp.check_xmpp_requirements() is True
        assert _xmpp.SLIXMPP_OMEMO_AVAILABLE is True
        assert _xmpp._XEP_0384Impl is not None


@omemo_required
class TestXmppOmemoStorage:
    @pytest.mark.asyncio
    async def test_storage_roundtrip_and_permissions(self, tmp_path):
        path = tmp_path / "omemo" / "keys.json"
        store = _xmpp._StorageImpl(path)
        await store._store("k", {"nested": [1, 2]})
        assert (await store._load("k")).from_just() == {"nested": [1, 2]}
        assert (path.stat().st_mode & 0o777) == 0o600  # holds private keys
        # Atomic write leaves no temp file behind.
        assert not path.with_name("keys.json.tmp").exists()

    @pytest.mark.asyncio
    async def test_corrupt_store_starts_fresh_loudly(self, tmp_path, caplog):
        # A corrupt key store means a new OMEMO identity — that must be
        # logged loudly, never silently swallowed.
        import logging as _logging
        path = tmp_path / "keys.json"
        path.write_text("{not valid json")
        with caplog.at_level(_logging.WARNING):
            store = _xmpp._StorageImpl(path)
        assert any("fresh identity" in r.message for r in caplog.records)
        loaded = await store._load("k")
        assert type(loaded).__name__ == "Nothing"
        # The store recovers: a subsequent write persists and reloads cleanly.
        await store._store("k", {"ok": True})
        store2 = _xmpp._StorageImpl(path)
        assert (await store2._load("k")).from_just() == {"ok": True}


@omemo_required
class TestXmppOmemoTransport:
    def _omemo_adapter(self, monkeypatch, **extra):
        adapter = _make_xmpp_adapter(monkeypatch, **extra)
        adapter.client = MagicMock()
        adapter._registered_plugins.update({"xep_0384", "xep_0085"})
        return adapter

    @pytest.mark.asyncio
    async def test_send_routes_dm_through_encryption(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch)
        adapter._send_encrypted = AsyncMock(
            return_value=_xmpp.SendResult(success=True, message_id="enc-1")
        )
        result = await adapter.send("alice@example.org", "secret")
        adapter._send_encrypted.assert_awaited_once_with("alice@example.org", "secret")
        assert result.message_id == "enc-1"

    @pytest.mark.asyncio
    async def test_send_muc_stays_plaintext(self, monkeypatch):
        adapter = self._omemo_adapter(
            monkeypatch, muc_rooms="room@conference.example.org"
        )
        adapter._send_encrypted = AsyncMock()
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw) or MagicMock()
        await adapter.send("room@conference.example.org", "group message")
        adapter._send_encrypted.assert_not_awaited()
        assert sent and sent[0]["mtype"] == "groupchat"

    @pytest.mark.asyncio
    async def test_encryption_failure_falls_back_to_plaintext(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch)
        adapter._send_encrypted = AsyncMock(side_effect=RuntimeError("no session"))
        sent = []
        adapter.client.send_message = lambda **kw: sent.append(kw) or MagicMock()
        result = await adapter.send("alice@example.org", "hello")
        assert result.success is True
        assert sent and sent[0]["mbody"] == "hello"

    @pytest.mark.asyncio
    async def test_send_encrypted_one_sends_encrypted_stanza(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch)
        encrypted_msg = MagicMock()
        encrypted_msg.__getitem__ = MagicMock(return_value="enc-id")
        xep_0384 = MagicMock()
        xep_0384.encrypt_message = AsyncMock(return_value=(encrypted_msg, {}))
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)
        adapter.client.make_message = MagicMock(return_value=MagicMock())

        result = await adapter._send_encrypted_one("alice@example.org", "secret")
        assert result.success is True
        xep_0384.encrypt_message.assert_awaited_once()
        encrypted_msg.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_encrypted_one_plaintext_when_no_devices(self, monkeypatch):
        # encrypt_message returns None when the recipient has no usable OMEMO
        # devices; the reply must still be delivered (TLS-only).
        adapter = self._omemo_adapter(monkeypatch)
        xep_0384 = MagicMock()
        xep_0384.encrypt_message = AsyncMock(return_value=(None, {}))
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)
        adapter.client.make_message = MagicMock(return_value=MagicMock())
        plain = MagicMock()
        plain.__getitem__ = MagicMock(return_value="plain-id")
        adapter.client.send_message = MagicMock(return_value=plain)

        result = await adapter._send_encrypted_one("alice@example.org", "hello")
        assert result.success is True
        adapter.client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_encrypted_chunks_long_bodies(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch, max_message_length="100")
        adapter._send_encrypted_one = AsyncMock(
            return_value=_xmpp.SendResult(success=True)
        )
        await adapter._send_encrypted("alice@example.org", "word " * 100)
        assert adapter._send_encrypted_one.await_count > 1

    @pytest.mark.asyncio
    async def test_inbound_encrypted_message_is_decrypted(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()

        decrypted = MagicMock()
        decrypted.get = MagicMock(
            side_effect=lambda k, d=None: "decrypted text" if k == "body" else d
        )
        device_info = MagicMock(device_id=42)
        xep_0384 = MagicMock()
        xep_0384.is_encrypted = MagicMock(return_value=True)
        xep_0384.decrypt_message = AsyncMock(return_value=(decrypted, device_info))
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)

        stanza = _StanzaStub(
            stanza_type="chat",
            from_jid="alice@example.org/phone",
            body="This message is OMEMO encrypted.",
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "decrypted text"
        assert event.raw_message is decrypted

    @pytest.mark.asyncio
    async def test_inbound_decrypt_failure_drops_message(self, monkeypatch):
        # A failed decrypt must NOT dispatch the ciphertext fallback body
        # ("This message is OMEMO encrypted.") to the agent as user text.
        adapter = self._omemo_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()
        xep_0384 = MagicMock()
        xep_0384.is_encrypted = MagicMock(return_value=True)
        xep_0384.decrypt_message = AsyncMock(side_effect=RuntimeError("no session"))
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)

        stanza = _StanzaStub(
            stanza_type="chat",
            from_jid="alice@example.org/phone",
            body="This message is OMEMO encrypted.",
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inbound_plaintext_passes_through_untouched(self, monkeypatch):
        adapter = self._omemo_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()
        xep_0384 = MagicMock()
        xep_0384.is_encrypted = MagicMock(return_value=False)
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="alice@example.org/phone", body="plain hi"
        )
        await adapter._on_message(stanza)
        adapter.handle_message.assert_awaited_once()
        assert adapter.handle_message.await_args.args[0].text == "plain hi"

    @pytest.mark.asyncio
    async def test_unauthorized_encrypted_dm_never_reaches_decryption(self, monkeypatch):
        # Authorization gates BEFORE decryption. Decrypting first would let
        # unauthorized senders force X3DH session builds, blind-trust key
        # store writes, and automatic key-exchange replies (adversarial
        # review finding).
        adapter = self._omemo_adapter(monkeypatch, allowed_users="alice@example.org")
        adapter.handle_message = AsyncMock()
        xep_0384 = MagicMock()
        xep_0384.is_encrypted = MagicMock(return_value=True)
        xep_0384.decrypt_message = AsyncMock()
        adapter.client.__getitem__ = MagicMock(return_value=xep_0384)

        stanza = _StanzaStub(
            stanza_type="chat", from_jid="mallory@evil.example/x", body="ciphertext"
        )
        await adapter._on_message(stanza)
        xep_0384.decrypt_message.assert_not_awaited()
        adapter.handle_message.assert_not_awaited()
        assert "mallory@evil.example" not in adapter._known_dms

    @pytest.mark.asyncio
    async def test_partial_encrypted_failure_does_not_downgrade(self, monkeypatch):
        # Once any encrypted chunk was delivered, a later chunk failure must
        # NOT resend the whole message as plaintext (duplication + silent
        # retroactive E2E downgrade).
        adapter = self._omemo_adapter(monkeypatch, max_message_length="100")
        calls = {"n": 0}

        async def _one(chat_id, chunk):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("session died")
            return _xmpp.SendResult(success=True)

        adapter._send_encrypted_one = _one
        sent_plain = []
        adapter.client.send_message = lambda **kw: sent_plain.append(kw) or MagicMock()

        result = await adapter.send("alice@example.org", "word " * 100)
        assert result.success is False
        assert sent_plain == []

    @pytest.mark.asyncio
    async def test_first_chunk_failure_still_falls_back_to_plaintext(self, monkeypatch):
        # Nothing delivered yet → plaintext fallback is safe and preferred
        # over dropping the reply.
        adapter = self._omemo_adapter(monkeypatch)
        adapter._send_encrypted_one = AsyncMock(side_effect=RuntimeError("no devices"))
        sent_plain = []
        adapter.client.send_message = lambda **kw: sent_plain.append(kw) or MagicMock()

        result = await adapter.send("alice@example.org", "hello")
        assert result.success is True
        assert sent_plain and sent_plain[0]["mbody"] == "hello"


@omemo_required
class TestXmppOmemoConnect:
    @pytest.mark.asyncio
    async def test_connect_registers_omemo_plugin(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMPP_OMEMO_ENABLED", raising=False)
        adapter = _make_xmpp_adapter(
            monkeypatch, omemo_storage_path=str(tmp_path / "keys.json")
        )
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        registered = []
        fake_client.register_plugin = MagicMock(
            side_effect=lambda name, *a, **k: registered.append(name)
        )
        with patch(
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ), patch("plugin_adapter_xmpp._register_slixmpp_plugin") as reg:
            ok = await adapter.connect()
        assert ok is True
        assert "xep_0384" in registered
        assert reg.called
        assert "xep_0384" in adapter._registered_plugins

    @pytest.mark.asyncio
    async def test_connect_skips_omemo_when_disabled(self, monkeypatch):
        monkeypatch.setenv("XMPP_OMEMO_ENABLED", "false")
        adapter = _make_xmpp_adapter(monkeypatch)
        fake_client = MagicMock()
        fake_client.connect = MagicMock(
            side_effect=lambda *a, **k: adapter._session_ready.set()
        )
        registered = []
        fake_client.register_plugin = MagicMock(
            side_effect=lambda name, *a, **k: registered.append(name)
        )
        with patch(
            "plugin_adapter_xmpp.slixmpp.ClientXMPP", return_value=fake_client
        ):
            ok = await adapter.connect()
        assert ok is True
        assert "xep_0384" not in registered
