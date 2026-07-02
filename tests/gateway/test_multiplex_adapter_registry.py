"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
from pathlib import Path
import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import GatewayRunner


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

    def test_reads_config_token(self):
        class _Cfg:
            token = "config-token"

        class _ConfigAdapter:
            config = _Cfg()

        assert GatewayRunner._adapter_credential_fingerprint(_ConfigAdapter()) is not None


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
    async def test_overrides_existing_profile_from_secondary_adapter(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = "default"  # stale process/default profile on adapter event

        class _Evt:
            source = _Src()

        await handler(_Evt())
        assert seen["profile"] == "coder"

    @pytest.mark.asyncio
    async def test_runs_delegated_handler_under_profile_home(self, tmp_path, monkeypatch):
        from hermes_constants import get_hermes_home

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "coder"
        default_home.mkdir(parents=True)
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text("TELEGRAM_BOT_TOKEN=profile-token\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(default_home))

        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            seen["home"] = Path(get_hermes_home())
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder", profile_home)

        class _Src:
            profile = "default"

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen == {"profile": "coder", "home": profile_home}


class TestProfileOutboundAdapterSelection:
    @pytest.mark.asyncio
    async def test_streaming_turn_uses_profile_owned_adapter(self, monkeypatch):
        """Streaming delivery must not borrow the default same-platform adapter."""
        import gateway.run as gateway_run
        from gateway.config import GatewayConfig, Platform, StreamingConfig
        from gateway.session import SessionSource

        class _Adapter:
            SUPPORTS_MESSAGE_EDITING = True
            MAX_MESSAGE_LENGTH = 4096

            def __init__(self, name):
                self.name = name
                self._pending_messages = {}
                self.typing_calls = 0

            def message_len_fn(self, text):
                return len(text)

            def get_pending_message(self, _session_key):
                return None

            def has_pending_interrupt(self, _session_key):
                return False

            async def send_typing(self, *_args, **_kwargs):
                self.typing_calls += 1

            def register_post_delivery_callback(self, *_args, **_kwargs):
                pass

            def pause_typing_for_chat(self, *_args, **_kwargs):
                pass

        class _StreamConsumer:
            instances = []

            def __init__(self, *, adapter, **_kwargs):
                self.adapter = adapter
                self.deltas = []
                self.final_response_sent = False
                self.final_content_delivered = False
                self._done = False
                self.instances.append(self)

            async def run(self):
                while not self._done:
                    await asyncio.sleep(0.01)

            def on_delta(self, text):
                self.deltas.append(text)
                self.final_response_sent = True
                self.final_content_delivered = True

            def on_commentary(self, text):
                self.on_delta(text)

            def on_segment_break(self):
                pass

            def finish(self):
                self._done = True

        class _Agent:
            def __init__(self, **kwargs):
                self.session_id = kwargs["session_id"]
                self.model = kwargs["model"]
                self.tools = []
                self.context_compressor = SimpleNamespace(
                    last_prompt_tokens=12,
                    context_length=4096,
                )
                self.session_prompt_tokens = 12
                self.session_completion_tokens = 3
                self.stream_delta_callback = None
                self.interim_assistant_callback = None
                self.tool_progress_callback = None
                self.status_callback = None
                self.notice_callback = None
                self.notice_clear_callback = None
                self.event_callback = None
                self.background_review_callback = None
                self.clarify_callback = None
                self.gateway_approval_callback = None
                self.reasoning_config = None
                self.service_tier = None
                self.request_overrides = {}

            def run_conversation(self, user_message, conversation_history=None, **_kwargs):
                if self.stream_delta_callback:
                    self.stream_delta_callback("hello")
                return {
                    "final_response": "hello",
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": "hello"},
                    ],
                    "api_calls": 1,
                    "completed": True,
                    "history_offset": len(conversation_history or []),
                }

            def interrupt(self, *_args, **_kwargs):
                pass

        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _Agent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
        })
        monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _StreamConsumer)

        import hermes_cli.tools_config as tools_config
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args, **_kwargs: {"core"})

        default_adapter = _Adapter("default")
        profile_adapter = _Adapter("private-supervisor")
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: default_adapter}
        runner._profile_adapters = {"private-supervisor": {Platform.TELEGRAM: profile_adapter}}
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.streaming = StreamingConfig(enabled=True, transport="edit", edit_interval=0.01)
        runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
        runner.session_store = SimpleNamespace(_entries={})
        runner._session_db = None
        runner._agent_cache = {}
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._session_run_generation = {}
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._pending_skills_reload_notes = {}
        runner._prefill_messages = []
        runner._ephemeral_system_prompt = ""
        runner._reasoning_config = None
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._draining = False
        runner._resolve_session_agent_runtime = lambda **_kwargs: ("test-model", {"api_key": "token"})
        runner._resolve_session_reasoning_config = lambda **_kwargs: None
        runner._resolve_turn_agent_config = lambda message, model, runtime: {"model": model, "runtime": runtime}
        runner._load_service_tier = lambda: None
        runner._agent_config_signature = lambda *_args, **_kwargs: ("sig",)
        runner._extract_cache_busting_config = lambda _config: ()
        runner._thread_metadata_for_source = lambda *_args, **_kwargs: None

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="u1",
            profile="private-supervisor",
        )

        result = await runner._run_agent_inner(
            message="hi",
            context_prompt="",
            history=[],
            source=source,
            session_id="sid",
            session_key="agent:private-supervisor:telegram:dm:12345",
        )

        assert result["final_response"] == "hello"
        assert _StreamConsumer.instances
        assert _StreamConsumer.instances[0].adapter is profile_adapter


class TestSecondarySuppressedAdapters:
    """Secondary profiles skip listener/singleton adapters and keep startup alive."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_skipped(self, monkeypatch):
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
        created = []
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: created.append(p))

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0
        assert created == []

    @pytest.mark.asyncio
    async def test_secondary_signal_skipped(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.SIGNAL: PlatformConfig(enabled=True, extra={"http_url": "http://127.0.0.1:8080"}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        created = []
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: created.append(p))

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0
        assert created == []

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

    def test_suppressed_set_covers_known_listeners(self):
        from gateway.run import _SECONDARY_SUPPRESSED_PLATFORM_VALUES
        # Every adapter that binds a TCP port must be in the guard set.
        for p in ("webhook", "api_server", "msgraph_webhook", "feishu",
                  "wecom_callback", "bluebubbles", "sms"):
            assert p in _SECONDARY_SUPPRESSED_PLATFORM_VALUES
        assert "signal" in _SECONDARY_SUPPRESSED_PLATFORM_VALUES

