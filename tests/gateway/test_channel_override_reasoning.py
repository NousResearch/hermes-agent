"""``reasoning_effort`` in ``platforms.<p>.channel_overrides`` (#67031).

Priority: session ``/reasoning`` > the chat's channel entry > per-model ``agent.reasoning_overrides``
> global ``agent.reasoning_effort``. The entry is the first match along chat -> thread -> parent, as
for the channel ``model``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

CHANNEL = "111"
THREAD = "222"
OTHER = "333"

HIGH = {"enabled": True, "effort": "high"}
LOW = {"enabled": True, "effort": "low"}
XHIGH = {"enabled": True, "effort": "xhigh"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-5\n"
        "agent:\n  reasoning_effort: low\n  reasoning_overrides:\n    claude-opus-4.5: xhigh\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    return hermes_home


def _runner(overrides: dict[str, ChannelOverride]):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner.config = GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides),
    })
    return runner


def _channel_source(chat_id: str = CHANNEL) -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="group", user_id="u1")


def _thread_source() -> SessionSource:
    # Discord keys an in-thread message on the thread's own id; the channel is the parent.
    return SessionSource(platform=Platform.DISCORD, chat_id=THREAD, chat_type="thread", thread_id=THREAD,
                         parent_chat_id=CHANNEL, user_id="u1")


class TestResolution:
    def test_channel_effort_beats_global(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        assert runner._resolve_session_reasoning_config(source=_channel_source()) == HIGH
        # Another channel keeps the profile's level.
        assert runner._resolve_session_reasoning_config(source=_channel_source(OTHER)) == LOW

    def test_channel_effort_beats_per_model_override(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        assert runner._resolve_session_reasoning_config(source=_channel_source(), model="claude-opus-4.5") == HIGH
        assert runner._resolve_session_reasoning_config(
            source=_channel_source(OTHER), model="claude-opus-4.5") == XHIGH

    def test_session_reasoning_beats_channel(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        source = _channel_source()
        runner._session_state(runner._session_key_for_source(source)).conversation.reasoning_override = XHIGH
        assert runner._resolve_session_reasoning_config(source=source) == XHIGH

    def test_thread_inherits_parent_channel_effort(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        assert runner._resolve_session_reasoning_config(source=_thread_source()) == HIGH

    def test_thread_entry_wins_over_parent(self, home):
        runner = _runner({
            CHANNEL: ChannelOverride(reasoning_effort="high"),
            THREAD: ChannelOverride(reasoning_effort="xhigh"),
        })
        assert runner._resolve_session_reasoning_config(source=_thread_source()) == XHIGH

    def test_thread_entry_without_effort_falls_to_profile_not_parent(self, home):
        # First matching entry decides, as for the channel model and allow_silence.
        runner = _runner({
            CHANNEL: ChannelOverride(reasoning_effort="high"),
            THREAD: ChannelOverride(model="claude-opus-4.5"),
        })
        assert runner._resolve_session_reasoning_config(source=_thread_source(), model="claude-opus-4.5") == XHIGH

    @pytest.mark.parametrize("value", ["none", False, "disabled"])
    def test_channel_can_disable_thinking(self, home, value):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort=value)})
        assert runner._resolve_session_reasoning_config(source=_channel_source()) == {"enabled": False}

    def test_unknown_value_warns_and_falls_through(self, home, caplog):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="hgih")})
        with caplog.at_level(logging.WARNING):
            assert runner._resolve_session_reasoning_config(source=_channel_source()) == LOW
        assert "hgih" in caplog.text

    def test_no_source_skips_the_channel_level(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        assert runner._resolve_session_reasoning_config() == LOW


class TestConfigParsing:
    def test_round_trip_keeps_false(self):
        ov = ChannelOverride.from_dict({"model": "m", "reasoning_effort": False})
        assert ov.reasoning_effort is False
        assert ov.to_dict() == {"model": "m", "reasoning_effort": False}
        assert ChannelOverride.from_dict({"model": "m"}).to_dict() == {"model": "m"}

    def test_loaded_from_config_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "platforms:\n  discord:\n    enabled: true\n    channel_overrides:\n"
            f"      \"{CHANNEL}\":\n        reasoning_effort: high\n",
            encoding="utf-8",
        )
        cfg = load_gateway_config()
        assert cfg.platforms[Platform.DISCORD].channel_overrides[CHANNEL].reasoning_effort == "high"


class TestReasoningCommand:
    @staticmethod
    def _event(text: str, chat_id: str = CHANNEL) -> MessageEvent:
        return MessageEvent(text=text, source=_channel_source(chat_id))

    @pytest.mark.asyncio
    async def test_status_names_the_channel_scope(self, home):
        from agent.i18n import t

        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        reply = await runner._handle_reasoning_command(self._event("/reasoning"))
        assert t("gateway.reasoning.scope_channel") in reply
        assert "high" in reply
        assert t("gateway.reasoning.scope_global") in await runner._handle_reasoning_command(
            self._event("/reasoning", chat_id=OTHER))

        await runner._handle_reasoning_command(self._event("/reasoning xhigh"))
        assert t("gateway.reasoning.scope_session") in await runner._handle_reasoning_command(
            self._event("/reasoning"))

    @pytest.mark.asyncio
    async def test_global_pick_in_a_channel_chat_keeps_the_chosen_level(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        source = _channel_source()
        await runner._handle_reasoning_command(self._event("/reasoning minimal --global"))
        # Saved globally, and this chat runs it although its channel entry says high.
        assert "reasoning_effort: minimal" in (home / "config.yaml").read_text(encoding="utf-8")
        assert runner._resolve_session_reasoning_config(source=source) == {"enabled": True, "effort": "minimal"}

    @pytest.mark.asyncio
    async def test_global_pick_elsewhere_still_clears_the_session_override(self, home):
        runner = _runner({CHANNEL: ChannelOverride(reasoning_effort="high")})
        source = _channel_source(OTHER)
        await runner._handle_reasoning_command(self._event("/reasoning minimal --global", chat_id=OTHER))
        assert runner._peek_session_state(runner._session_key_for_source(source)).conversation.reasoning_override is None
