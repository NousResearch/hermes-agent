"""Tests for gateway reset notices and session config surfacing."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.run import GatewayRunner


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    return GatewayRunner.__new__(GatewayRunner)


def _patch_info(tmp_path, config_yaml, model, runtime):
    """Return a context-manager stack that patches _format_session_info deps."""
    cfg_path = tmp_path / "config.yaml"
    if config_yaml is not None:
        cfg_path.write_text(config_yaml)
    return (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.run._resolve_gateway_model", return_value=model),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
    )


class TestFormatSessionInfo:



    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 32768\n",
                                  "test-model",
                                  {"provider": "custom", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info


    def test_local_endpoint_shown(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: qwen3:8b\n  provider: custom\n  base_url: http://localhost:11434/v1\n  context_length: 8192\n",
            "qwen3:8b",
            {"provider": "custom", "base_url": "http://localhost:11434/v1", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "localhost:11434" in info
        assert "8K" in info

    def test_moa_preset_names_the_billed_aggregator(self, runner, tmp_path):
        """#112359: the preset name hides who pays; /model must name the acting aggregator."""
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: review\n  provider: moa\n",
                                  "review", {"provider": "moa", "base_url": "", "api_key": ""})
        moa_cfg = {"moa": {"presets": {"review": {
            "reference_models": [{"provider": "openai", "model": "gpt-5.5"}],
            "aggregator": {"provider": "nous", "model": "claude-opus-4.8"},
        }}}}
        with p1, p2, p3, patch("hermes_cli.config.load_config", return_value=moa_cfg):
            info = runner._format_session_info()
        assert "nous:claude-opus-4.8" in info

    def test_named_custom_provider_keeps_context_pin_without_model_base_url(
        self, runner, tmp_path
    ):
        """Session-reset banner must honor model.context_length for named custom providers.

        Repro: /status shows 262144 from config while the reset banner said
        ``131K tokens (detected)`` because empty model.base_url + runtime URL
        falsely cleared the pin and fell through to the Qwen family default.
        """
        model = "custom-local-agentw/Qwen-AgentWorld-35B-A3B-Q5_K_XL"
        config_yaml = (
            "model:\n"
            f"  default: {model}\n"
            "  provider: custom-local-agentw\n"
            "  context_length: 262144\n"
            "custom_providers:\n"
            "  - name: custom-local-agentw\n"
            "    base_url: http://127.0.0.1:8080/v1\n"
            "    models: {}\n"
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            config_yaml,
            model,
            {
                "provider": "custom-local-agentw",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "",
            },
        )
        with p1, p2, p3, patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[
                {
                    "name": "custom-local-agentw",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {},
                }
            ],
        ), patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *args, **kwargs: (
                kwargs.get("config_context_length")
                if kwargs.get("config_context_length")
                else 131072
            ),
        ):
            info = runner._format_session_info()
        assert "262K" in info
        assert "config" in info
        assert "131K" not in info


class TestResetNoticeSessionInfo:
    """#59003: the auto-reset banner must report the serving profile's config,
    not the multiplexer's base config."""

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    def _source(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", user_id="u1",
            profile="planner",
        )

    def _homes(self, tmp_path):
        base = tmp_path / "base"
        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n  default: base-model\n  provider: custom\n  context_length: 1000\n")
        profile.joinpath("config.yaml").write_text(
            "model:\n  default: profile-model\n  provider: anthropic\n  context_length: 2000\n")
        return base, profile

    def test_multiplex_uses_profile_config(self, runner, tmp_path):
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True)
        with patch("gateway.run._hermes_home", base), \
             patch.object(GatewayRunner, "_resolve_profile_home_for_source", return_value=profile), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert info.splitlines()[0] == "👤 **Profile:** `planner`"
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

    def test_profile_survives_model_resolution_failure(self, runner, tmp_path):
        from gateway.session_identity import RoutingIdentity

        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        source = self._source()
        source._identity = RoutingIdentity(
            transport_profile="receiver",
            runtime_profile="planner",
            authorization_home=tmp_path / "receiver",
            runtime_home=profile,
        )
        runner.config = SimpleNamespace(multiplex_profiles=True)
        with patch.object(GatewayRunner, "_format_session_info", side_effect=RuntimeError("broken model")):
            info = runner._reset_notice_session_info(source)
        assert info == "👤 **Profile:** `planner`"

    def test_standalone_ignores_stale_source_profile_without_identity(self, runner):
        runner.config = SimpleNamespace(multiplex_profiles=False)
        source = self._source()
        resolved = SimpleNamespace(
            model="test-model", provider="custom", base_url="", context_length=8192,
            context_source="config",
        )

        with patch.object(runner, "_profile_scope_for_source", return_value=nullcontext()), \
             patch.object(runner, "_active_profile_name", return_value="default"), \
             patch("gateway.run._resolve_gateway_model_context", return_value=resolved):
            info = runner._reset_notice_session_info(source)

        assert info.splitlines()[0] == "👤 **Profile:** `default`"
        assert "planner" not in info


def _write_profile_config(home: Path, model: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    home.joinpath("config.yaml").write_text(
        f"model:\n  default: {model}\n  provider: custom\n  context_length: 8192\n",
        encoding="utf-8",
    )


def test_canonical_runtime_profile_follows_default_routed_default(tmp_path, monkeypatch, runner):
    """The receiving bot stays ``default`` while the canonical runtime moves A → B → A."""
    from gateway.config import GatewayConfig, Platform
    from gateway.profile_routing import parse_profile_routes
    from gateway.session import SessionSource
    from gateway.session_identity import resolve_identity

    root = tmp_path / ".hermes"
    planner = root / "profiles" / "planner"
    _write_profile_config(root, "default-model")
    _write_profile_config(planner, "planner-model")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))

    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config._runtime_profile_homes = (("default", root), ("planner", planner))
    runner.config.profile_routes = parse_profile_routes([{
        "name": "planner-topic", "platform": "telegram", "profile": "planner", "chat_id": "B",
    }])
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {"planner": {}}

    sources = [
        SessionSource(platform=Platform.TELEGRAM, chat_id=chat, user_id="user", chat_type="dm")
        for chat in ("A", "B", "A-again")
    ]
    identities = [
        resolve_identity(source, runner=runner, transport_profile="default", primary_home=root)
        for source in sources
    ]
    notices = [runner._reset_notice_session_info(source) for source in sources]

    assert [(identity.transport_profile, identity.runtime_profile) for identity in identities] == [
        ("default", "default"), ("default", "planner"), ("default", "default"),
    ]
    assert [notice.splitlines()[0] for notice in notices] == [
        "👤 **Profile:** `default`", "👤 **Profile:** `planner`", "👤 **Profile:** `default`",
    ]
    assert "default-model" in notices[0] and "planner-model" in notices[1]
    assert "default-model" in notices[2]


@pytest.mark.parametrize("profile_name", ["default", "writer"])
def test_standalone_reset_notice_names_active_profile(profile_name, tmp_path, monkeypatch, runner):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource

    root = tmp_path / ".hermes"
    home = root if profile_name == "default" else root / "profiles" / profile_name
    _write_profile_config(home, f"{profile_name}-model")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner._primary_profile_name = profile_name
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user", chat_type="dm")

    with patch("gateway.run._hermes_home", home):
        info = runner._reset_notice_session_info(source)

    assert info.splitlines()[0] == f"👤 **Profile:** `{profile_name}`"
    assert f"{profile_name}-model" in info


def _reset_runner(source):
    from datetime import datetime

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.session import SessionEntry, build_session_key

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    session_key = build_session_key(source)
    entry = SessionEntry(
        session_key=session_key,
        session_id="fresh-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.reset_session.return_value = entry
    runner.session_store._entries = {session_key: entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncMock()
    runner._session_db._db = MagicMock()
    runner._session_db._db.is_telegram_topic_mode_enabled.return_value = False
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/reset", "/new Project Atlas"])
async def test_reset_replies_show_profile_for_both_aliases_and_titles(command):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", user_id="user", chat_type="dm",
    )
    runner = _reset_runner(source)

    resolved = SimpleNamespace(
        model="test-model", provider="custom", base_url="", context_length=8192,
        context_source="config",
    )
    with patch("gateway.run._resolve_gateway_model_context", return_value=resolved):
        reply = str(await runner._handle_reset_command(MessageEvent(text=command, source=source)))

    assert "👤 **Profile:** `" in reply
    assert reply.index("👤 **Profile:**") < reply.index("◆ Model:")
    if command.endswith("Project Atlas"):
        assert "New session started: Project Atlas" in reply
