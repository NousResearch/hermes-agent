"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
import pytest

from gateway.run import GatewayRunner
from gateway.platforms.base import BasePlatformAdapter


class _FakeAdapter:
    def __init__(self, token=None):
        self.token = token


class TestCredentialFingerprint:
    def test_none_without_token(self):
        assert GatewayRunner._adapter_credential_fingerprint(_FakeAdapter()) is None

    def test_stable_and_log_safe(self):
        a = _FakeAdapter(token="secret-bot-token")
        fp1 = GatewayRunner._adapter_credential_fingerprint(a)
        fp2 = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="secret-bot-token"))
        assert fp1 == fp2  # stable
        assert "secret-bot-token" not in (fp1 or "")  # never the raw token
        assert len(fp1) == 16

    def test_distinct_tokens_distinct_fp(self):
        a = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-A"))
        b = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-B"))
        assert a != b

    def test_reads_alt_attrs(self):
        class _AltAdapter:
            def __init__(self):
                self.bot_token = "alt-token"
        assert GatewayRunner._adapter_credential_fingerprint(_AltAdapter()) is not None


class TestProfileMessageHandler:
    @pytest.mark.asyncio
    async def test_stamps_profile_on_unstamped_source(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coder"

    @pytest.mark.asyncio
    async def test_does_not_override_existing_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = "writer"  # already stamped (e.g. by URL prefix)

        class _Evt:
            source = _Src()

        await handler(_Evt())
        assert seen["profile"] == "writer"


class TestPortBindingHardError:
    """A secondary profile enabling a port-binding platform aborts startup."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_raises(self, monkeypatch):
        from gateway.run import MultiplexConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        # reviewer profile config enables webhook (a port-binding platform)
        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        with pytest.raises(MultiplexConfigError) as ei:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert "webhook" in str(ei.value)
        assert "reviewer" in str(ei.value)

    @pytest.mark.asyncio
    async def test_secondary_non_binding_platform_ok(self, monkeypatch):
        """A non-port-binding platform (e.g. telegram) is NOT rejected."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        # _create_adapter returns None here (no real telegram token wiring), so
        # the loop simply connects nothing — the key assertion is NO raise.
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: None)

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0  # nothing connected, but no MultiplexConfigError

    def test_port_binding_set_covers_known_listeners(self):
        from gateway.run import _PORT_BINDING_PLATFORM_VALUES
        # Every adapter that binds a TCP port must be in the guard set.
        for p in ("webhook", "api_server", "msgraph_webhook", "feishu",
                  "wecom_callback", "bluebubbles", "sms"):
            assert p in _PORT_BINDING_PLATFORM_VALUES



class TestAdapterForSource:
    def test_default_adapter_without_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._active_profile_name = lambda: "default"
        runner.adapters = {"telegram": "default-tg"}
        runner._profile_adapters = {}

        class _Src:
            platform = "telegram"
            profile = None

        assert runner._adapter_for_source(_Src()) == "default-tg"

    def test_secondary_profile_adapter(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._active_profile_name = lambda: "default"
        runner.adapters = {"telegram": "default-tg"}
        runner._profile_adapters = {"secondary": {"telegram": "secondary-tg"}}

        class _Src:
            platform = "telegram"
            profile = "secondary"

        assert runner._adapter_for_source(_Src()) == "secondary-tg"

    def test_active_profile_uses_default_map(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._active_profile_name = lambda: "default"
        runner.adapters = {"telegram": "default-tg"}
        runner._profile_adapters = {"default": {"telegram": "profile-default-tg"}}

        class _Src:
            platform = "telegram"
            profile = "default"

        assert runner._adapter_for_source(_Src()) == "profile-default-tg"


class _TypingAdapter:
    """Fake adapter that records resume_typing_for_chat calls."""
    def __init__(self, name):
        self.name = name
        self.resumed = []

    def resume_typing_for_chat(self, chat_id):
        self.resumed.append(chat_id)


class TestSlashCommandProfileRouting:
    """Slash commands must route through _adapter_for_source, not adapters.get.

    Regression: /approve, /deny, /status, /model, /goal used
    self.adapters.get(source.platform) which ignores source.profile and
    can resume/inspect the wrong profile's adapter.
    """

    def _make_runner(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._active_profile_name = lambda: "default"
        default_tg = _TypingAdapter("default")
        secondary_tg = _TypingAdapter("secondary")
        runner.adapters = {"telegram": default_tg}
        runner._profile_adapters = {"secondary": {"telegram": secondary_tg}}
        return runner, default_tg, secondary_tg

    def _make_src(self):
        class _Src:
            platform = "telegram"
            chat_id = "chat-2"
            profile = "secondary"
            user_id = "u1"
            thread_id = None
        return _Src()

    def _make_event(self):
        src = self._make_src()
        class _Event:
            def __init__(self, s):
                self.source = s
            def get_command_args(self):
                return ""
        return _Event(src)

    @pytest.mark.asyncio
    async def test_approve_routes_to_secondary_adapter(self, monkeypatch):
        """ /approve must resume typing on the secondary adapter, not default."""
        runner, default_tg, secondary_tg = self._make_runner()
        runner._pending_approvals = {}

        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda *a, **kw: 1,  # one pending approval
        )
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval",
            lambda *a, **kw: True,
        )
        runner._session_key_for_source = lambda s: "secondary-session"

        result = await runner._handle_approve_command(self._make_event())

        assert secondary_tg.resumed == ["chat-2"]
        assert default_tg.resumed == []  # default adapter must NOT be touched

    @pytest.mark.asyncio
    async def test_deny_routes_to_secondary_adapter(self, monkeypatch):
        """ /deny must resume typing on the secondary adapter, not default."""
        runner, default_tg, secondary_tg = self._make_runner()
        runner._pending_approvals = {}

        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda *a, **kw: 1,
        )
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval",
            lambda *a, **kw: True,
        )
        runner._session_key_for_source = lambda s: "secondary-session"

        result = await runner._handle_deny_command(self._make_event())

        assert secondary_tg.resumed == ["chat-2"]
        assert default_tg.resumed == []


class _ConcreteAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for testing profile stamping."""
    async def connect(self): pass
    async def disconnect(self): pass
    async def send(self, *a, **kw): pass
    async def get_chat_info(self, *a, **kw): return {}


class TestProfileStampingBeforeSessionKey:
    """handle_message must stamp source.profile from adapter.profile_name
    BEFORE computing the session key, so multiplexed profiles get isolated
    sessions (agent:<profile>:…) instead of sharing agent:main:…."""

    def _make_adapter(self, profile_name="secondary"):
        from gateway.config import Platform, PlatformConfig
        adapter = _ConcreteAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
        adapter.profile_name = profile_name
        # Set a no-op handler so handle_message doesn't return early
        async def _noop(event):
            return None
        adapter._message_handler = _noop
        return adapter

    def _make_source(self, profile=None):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="u1",
            profile=profile,
        )

    def test_profile_stamped_from_adapter(self):
        import asyncio
        from gateway.platforms.base import MessageEvent

        adapter = self._make_adapter("secondary")
        source = self._make_source(profile=None)
        assert source.profile is None  # not yet stamped

        event = MessageEvent(source=source, text="/status")
        asyncio.run(adapter.handle_message(event))

        assert source.profile == "secondary"

    def test_existing_profile_not_overwritten(self):
        import asyncio
        from gateway.platforms.base import MessageEvent

        adapter = self._make_adapter("secondary")
        source = self._make_source(profile="already-set")

        event = MessageEvent(source=source, text="/status")
        asyncio.run(adapter.handle_message(event))

        assert source.profile == "already-set"  # not overwritten
