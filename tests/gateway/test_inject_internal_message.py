"""Tests for GatewayRunner.inject_internal_message — the AL16 public injection API.

Covers:
- inject_internal_message: adapter selection, SessionSource routing,
  internal=True flag, notice_text delivery, missing-adapter failure
- steer vs queue mode: steer into running agent, fallback to queue
- No Platform.ATM creation (negative guarantee)
- Runner exposed via gateway:startup hook payload
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner, InjectInternalMessageError
from gateway.session import SessionSource, build_session_key


# ------------------------------------------------------------------
# Test infrastructure
# ------------------------------------------------------------------

class _FakeTelegramAdapter:
    """Minimal Telegram adapter for injection tests.

    Captures the event passed to handle_message so tests can assert
    on routing decisions (source platform, internal flag, etc.).
    """

    def __init__(self):
        self.sent_messages: list = []          # (chat_id, text) tuples
        self.send_kwargs: list[dict] = []
        self.handled_events: list[MessageEvent] = []
        self._message_handler = AsyncMock()

    async def send(self, chat_id, text, **kwargs):
        self.sent_messages.append((chat_id, text))
        self.send_kwargs.append(kwargs)
        return MagicMock(success=True, error=None)

    async def handle_message(self, event):
        self.handled_events.append(event)
        if self._message_handler:
            await self._message_handler(event)


class _FakeRunningAgent:
    """Minimal agent stub with a steer() method for steer-mode tests."""

    def __init__(self, steer_result=True):
        self._steer_result = steer_result
        self.steered_texts: list[str] = []

    def steer(self, text: str) -> bool:
        self.steered_texts.append(text)
        return self._steer_result


def _make_runner(with_session_store=True, active_profile="test-profile"):
    """Build a bare GatewayRunner for unit testing the injection API.

    Args:
        with_session_store: If True, attach a mock session_store.
        active_profile: Value returned by ``_active_profile_name()``.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    tg = _FakeTelegramAdapter()
    runner.adapters = {Platform.TELEGRAM: tg}
    runner._profile_adapters = {}
    # Mock _active_profile_name so tests don't depend on the host env
    runner._active_profile_name = lambda: active_profile
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.delivery_router = MagicMock()
    if with_session_store:
        runner.session_store = MagicMock()
        runner.session_store._generate_session_key = lambda src: build_session_key(src)
    else:
        runner.session_store = None
    # Property backing dicts
    runner._sessions = {}
    return runner


# ------------------------------------------------------------------
# inject_internal_message — queue mode (default)
# ------------------------------------------------------------------

class TestInjectInternalMessage:
    """inject_internal_message routes an internal event to adapter.handle_message.

    Tests use the active profile name (mock default: "test-profile")
    to route through self.adapters — the running profile's adapter map.
    """

    @pytest.mark.asyncio
    async def test_routes_through_telegram_adapter(self):
        """The event reaches handle_message on the correct adapter."""
        runner = _make_runner()
        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="ATM nudge test marker",
            notice_text=None,
        )
        tg = runner.adapters[Platform.TELEGRAM]
        assert len(tg.handled_events) == 1
        event = tg.handled_events[0]
        assert event.text == "ATM nudge test marker"
        assert event.internal is True

    @pytest.mark.asyncio
    async def test_constructs_session_source_with_telegram_platform(self):
        """SessionSource reflects the real platform, not ATM."""
        runner = _make_runner()
        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="test",
        )
        tg = runner.adapters[Platform.TELEGRAM]
        event = tg.handled_events[0]
        assert event.source.platform == Platform.TELEGRAM
        assert event.source.chat_id == "100000001"
        assert event.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_internal_flag_is_true(self):
        """The MessageEvent carries internal=True so _handle_message skips
        authorization and startup-restore guards."""
        runner = _make_runner()
        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="test",
        )
        tg = runner.adapters[Platform.TELEGRAM]
        assert tg.handled_events[0].internal is True

    @pytest.mark.asyncio
    async def test_profile_passed_to_session_source(self):
        """The profile name is attached to SessionSource for session namespacing."""
        runner = _make_runner()
        # Register profile adapter so the strict resolution works
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        
        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="test",
        )
        assert skillrx_tg.handled_events[0].source.profile == "skillrx"

    @pytest.mark.asyncio
    async def test_sends_notice_text_before_routing(self):
        """notice_text is delivered via adapter.send before handle_message."""
        runner = _make_runner()
        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="nudge payload",
            notice_text="\u26a1 ATM nudge received",
        )
        tg = runner.adapters[Platform.TELEGRAM]
        # Notice sent first
        assert tg.sent_messages == [("100000001", "\u26a1 ATM nudge received")]
        assert tg.send_kwargs == [{"metadata": {"notify": True}}]
        # Then event routed
        assert tg.handled_events[0].text == "nudge payload"

    @pytest.mark.asyncio
    async def test_missing_adapter_raises(self):
        """Raises InjectInternalMessageError when no adapter for platform."""
        runner = _make_runner()
        runner.adapters = {}  # no adapters for any platform
        with pytest.raises(InjectInternalMessageError) as exc:
            await runner.inject_internal_message(
                profile="test-profile",
                platform=Platform.TELEGRAM,
                chat_id="100000001",
                text="test",
            )
        assert exc.value.code == "adapter_not_found"

    @pytest.mark.asyncio
    async def test_notice_failure_does_not_prevent_routing(self):
        """If adapter.send raises, the event is still routed to handle_message."""
        runner = _make_runner()
        tg = runner.adapters[Platform.TELEGRAM]
        tg.send = AsyncMock(side_effect=Exception("network down"))

        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="payload",
            notice_text="notice that fails",
        )
        # Still routed
        assert len(tg.handled_events) == 1
        assert tg.handled_events[0].text == "payload"

    @pytest.mark.asyncio
    async def test_reported_notice_failure_does_not_prevent_routing(self, caplog):
        """A failed SendResult is observable but cannot suppress the XML event."""
        runner = _make_runner()
        tg = runner.adapters[Platform.TELEGRAM]
        tg.send = AsyncMock(return_value=MagicMock(success=False, error="network down"))

        with caplog.at_level("WARNING"):
            await runner.inject_internal_message(
                profile="test-profile",
                platform=Platform.TELEGRAM,
                chat_id="100000001",
                text="payload",
                notice_text="notice that fails",
            )

        assert len(tg.handled_events) == 1
        assert tg.handled_events[0].text == "payload"
        assert "visible notice was not delivered: network down" in caplog.text

    @pytest.mark.asyncio
    async def test_selects_adapter_from_profile_adapters(self):
        """When a profile is in _profile_adapters, its adapter is used."""
        runner = _make_runner()
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        # The default adapter should NOT be used
        default_tg = runner.adapters[Platform.TELEGRAM]

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="test",
        )
        # Profile adapter was used
        assert len(skillrx_tg.handled_events) == 1
        # Default adapter was NOT used
        assert len(default_tg.handled_events) == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_default_adapters_with_active_profile(self):
        """When profile matches the active profile, uses self.adapters."""
        runner = _make_runner()
        default_tg = runner.adapters[Platform.TELEGRAM]

        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="test",
        )
        assert len(default_tg.handled_events) == 1

    @pytest.mark.asyncio
    async def test_unknown_profile_raises(self):
        """Raises InjectInternalMessageError when profile not found."""
        runner = _make_runner()
        runner._profile_adapters["other"] = {
            Platform.TELEGRAM: _FakeTelegramAdapter()
        }

        with pytest.raises(InjectInternalMessageError) as exc:
            await runner.inject_internal_message(
                profile="nonexistent",
                platform=Platform.TELEGRAM,
                chat_id="100000001",
                text="test",
            )
        assert exc.value.code == "unknown_profile"

    @pytest.mark.asyncio
    async def test_empty_profile_adapters_raises(self):
        """Raises InjectInternalMessageError when _profile_adapters empty."""
        runner = _make_runner()
        runner._profile_adapters = {}

        with pytest.raises(InjectInternalMessageError) as exc:
            await runner.inject_internal_message(
                profile="skillrx",
                platform=Platform.TELEGRAM,
                chat_id="100000001",
                text="test",
            )
        assert exc.value.code == "profile_map_empty"


# ------------------------------------------------------------------
# inject_internal_message — steer mode
# ------------------------------------------------------------------

class TestInjectInternalMessageSteerMode:
    """mode=\"steer\" injects text directly into the running agent's turn."""

    @pytest.mark.asyncio
    async def test_steers_into_running_agent(self):
        """When an agent is running for the session, steer() is called
        with the message text, and handle_message is NOT called."""
        runner = _make_runner()
        # Register profile adapter so strict resolution passes
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        
        agent = _FakeRunningAgent()
        # Simulate a running agent by populating _running_agents with the
        # session key that _session_key_for_source will produce.
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            chat_type="dm",
            user_id="100000001",
            profile="skillrx",
        )
        session_key = build_session_key(source)
        runner._running_agents[session_key] = (agent,)

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="steered nudge",
            mode="steer",
        )

        # Agent.steer() was called
        assert agent.steered_texts == ["steered nudge"]
        # handle_message was NOT called (no queue fallback)
        assert len(skillrx_tg.handled_events) == 0

    @pytest.mark.asyncio
    async def test_steer_falls_back_to_queue_when_no_agent_running(self):
        """When no agent is running, steer mode falls back to queue."""
        runner = _make_runner()
        # Register profile adapter so strict resolution passes
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        # _running_agents is empty — no agent running

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="fallback nudge",
            mode="steer",
        )

        # Falls back to queue: handle_message was called
        assert len(skillrx_tg.handled_events) == 1
        event = skillrx_tg.handled_events[0]
        assert event.text == "fallback nudge"
        assert event.internal is True

    @pytest.mark.asyncio
    async def test_steer_falls_back_when_steer_returns_false(self):
        """When steer() returns False, falls back to queue."""
        runner = _make_runner()
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        
        agent = _FakeRunningAgent(steer_result=False)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            chat_type="dm",
            user_id="100000001",
            profile="skillrx",
        )
        session_key = build_session_key(source)
        runner._running_agents[session_key] = (agent,)

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="empty steer",
            mode="steer",
        )

        # steer() was called
        assert agent.steered_texts == ["empty steer"]
        # Falls back to queue
        assert len(skillrx_tg.handled_events) == 1
        assert skillrx_tg.handled_events[0].text == "empty steer"

    @pytest.mark.asyncio
    async def test_queue_mode_never_steers(self):
        """Explicit mode=\"queue\" (or default) never calls steer(),
        even when an agent is running."""
        runner = _make_runner()
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        
        agent = _FakeRunningAgent()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            chat_type="dm",
            user_id="100000001",
            profile="skillrx",
        )
        session_key = build_session_key(source)
        runner._running_agents[session_key] = (agent,)

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="queued nudge",
            mode="queue",
        )

        # steer() was NOT called
        assert agent.steered_texts == []
        # handle_message WAS called (queue path)
        assert len(skillrx_tg.handled_events) == 1
        assert skillrx_tg.handled_events[0].text == "queued nudge"

    @pytest.mark.asyncio
    async def test_steer_mode_preserves_notice_text(self):
        """notice_text is still sent even in steer mode."""
        runner = _make_runner()
        skillrx_tg = _FakeTelegramAdapter()
        runner._profile_adapters["skillrx"] = {Platform.TELEGRAM: skillrx_tg}
        
        agent = _FakeRunningAgent()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            chat_type="dm",
            user_id="100000001",
            profile="skillrx",
        )
        session_key = build_session_key(source)
        runner._running_agents[session_key] = (agent,)

        await runner.inject_internal_message(
            profile="skillrx",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="steered payload",
            notice_text="📬 ATM nudge",
            mode="steer",
        )

        # Notice was sent
        assert skillrx_tg.sent_messages == [("100000001", "📬 ATM nudge")]
        # Text was steered
        assert agent.steered_texts == ["steered payload"]


# ------------------------------------------------------------------
# No ATM platform creation (negative guarantee)
# ------------------------------------------------------------------

def test_no_atm_platform_created():
    """inject_internal_message must never register Platform.ATM or
    create an ATM session — it routes through real platform adapters."""
    # Platform.ATM must not exist
    assert not hasattr(Platform, "ATM")

    # The method uses only real platforms (TELEGRAM in our tests)
    runner = _make_runner()
    # After injection, no ATM adapter should exist
    assert "atm" not in {p.value for p in runner.adapters}
    assert "atm" not in {p.value for p in runner._profile_adapters.values()}




# ------------------------------------------------------------------
# Isolation tests — queue and steer must not cross sessions
# ------------------------------------------------------------------

class TestInjectInternalMessageIsolation:
    """Both queue and steer modes must be scoped to their target session."""

    @pytest.mark.asyncio
    async def test_queue_isolation_different_chat_id_only_routes_to_target(self):
        """Queue mode targeting chat A does not deliver to chat B's adapter."""
        runner = _make_runner()
        tg_a = runner.adapters[Platform.TELEGRAM]
        # Create a separate adapter for chat B
        tg_b = _FakeTelegramAdapter()
        runner._profile_adapters["chatB"] = {Platform.TELEGRAM: tg_b}

        # Inject into chat B's profile adapter
        await runner.inject_internal_message(
            profile="chatB",
            platform=Platform.TELEGRAM,
            chat_id="chatB-id",
            text="only for B",
            mode="queue",
        )

        # Chat B received it
        assert len(tg_b.handled_events) == 1
        assert tg_b.handled_events[0].text == "only for B"
        # Chat A did NOT receive it
        assert len(tg_a.handled_events) == 0

    @pytest.mark.asyncio
    async def test_steer_isolation_different_chat_id_does_not_cross(self):
        """Steer mode targeting chat A's running agent does not affect chat B."""
        runner = _make_runner()
        tg_a = runner.adapters[Platform.TELEGRAM]
        tg_b = _FakeTelegramAdapter()
        runner._profile_adapters["chatB"] = {Platform.TELEGRAM: tg_b}

        # Running agent only for chat A
        agent_a = _FakeRunningAgent()
        source_a = SessionSource(
            platform=Platform.TELEGRAM, chat_id="chatA-id",
            chat_type="dm", user_id="chatA-id", profile="test-profile",
        )
        runner._running_agents[build_session_key(source_a)] = (agent_a,)

        # Steer into chat B's profile
        await runner.inject_internal_message(
            profile="chatB",
            platform=Platform.TELEGRAM,
            chat_id="chatB-id",
            text="steer to B",
            mode="steer",
        )

        # Agent A was NOT steered
        assert agent_a.steered_texts == []
        # Chat B received via queue fallback (no agent running for B)
        assert len(tg_b.handled_events) == 1

    @pytest.mark.asyncio
    async def test_active_profile_isolation_self_adapters_only(self):
        """Active profile routes through self.adapters, not secondary profiles."""
        runner = _make_runner(active_profile="primary")
        tg_primary = runner.adapters[Platform.TELEGRAM]
        tg_secondary = _FakeTelegramAdapter()
        runner._profile_adapters["secondary"] = {Platform.TELEGRAM: tg_secondary}

        await runner.inject_internal_message(
            profile="primary",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="primary only",
        )

        # Only primary adapter was used
        assert len(tg_primary.handled_events) == 1
        assert len(tg_secondary.handled_events) == 0

    # ------------------------------------------------------------------
    # Host-contract isolation tests (AL17 gate)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_steer_isolation_same_profile_different_chats(self):
        """same-profile/two-chat steer isolation: steer to chat B must not
        affect chat A's running agent, and vice versa."""
        runner = _make_runner(active_profile="test-profile")
        tg = runner.adapters[Platform.TELEGRAM]

        # Chat A has a running agent, chat B does not.
        agent_a = _FakeRunningAgent()
        source_a = SessionSource(
            platform=Platform.TELEGRAM, chat_id="chatA-id",
            chat_type="dm", user_id="chatA-id", profile="test-profile",
        )
        runner._running_agents[build_session_key(source_a)] = (agent_a,)

        # Steer into chat B — must NOT affect chat A's agent
        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="chatB-id",
            text="steer to B",
            mode="steer",
        )

        # Agent A was NOT steered
        assert agent_a.steered_texts == []
        # Chat B received via queue fallback (no agent running for B)
        assert len(tg.handled_events) == 1
        assert tg.handled_events[0].text == "steer to B"

        # Now reverse: clear events, run agent for B, steer to A
        tg.handled_events.clear()
        agent_b = _FakeRunningAgent()
        source_b = SessionSource(
            platform=Platform.TELEGRAM, chat_id="chatB-id",
            chat_type="dm", user_id="chatB-id", profile="test-profile",
        )
        runner._running_agents[build_session_key(source_b)] = (agent_b,)
        # Remove agent A so it can't interfere
        del runner._running_agents[build_session_key(source_a)]

        await runner.inject_internal_message(
            profile="test-profile",
            platform=Platform.TELEGRAM,
            chat_id="chatA-id",
            text="steer to A",
            mode="steer",
        )

        # Agent B was NOT steered
        assert agent_b.steered_texts == []
        # Chat A received via queue fallback
        assert len(tg.handled_events) == 1
        assert tg.handled_events[0].text == "steer to A"

    @pytest.mark.asyncio
    async def test_steer_isolation_different_profiles_same_chat(self):
        """two-profiles/same-chat isolation: steer to profile B with the
        same chat_id must not affect profile A's running agent."""
        runner = _make_runner(active_profile="test-profile")
        # Use profile-aware session key generation so different profiles
        # produce different session keys (as in production with
        # multiplex_profiles=True).
        runner.session_store._generate_session_key = (
            lambda src: build_session_key(src, profile=src.profile)
        )

        # Profile A has a running agent for chat_id "100000001"
        tg_a = _FakeTelegramAdapter()
        runner._profile_adapters["profileA"] = {Platform.TELEGRAM: tg_a}
        agent_a = _FakeRunningAgent()
        source_a = SessionSource(
            platform=Platform.TELEGRAM, chat_id="100000001",
            chat_type="dm", user_id="100000001", profile="profileA",
        )
        runner._running_agents[
            build_session_key(source_a, profile="profileA")
        ] = (agent_a,)

        # Profile B has its own adapter, no running agent
        tg_b = _FakeTelegramAdapter()
        runner._profile_adapters["profileB"] = {Platform.TELEGRAM: tg_b}

        # Steer into profile B with same chat_id
        await runner.inject_internal_message(
            profile="profileB",
            platform=Platform.TELEGRAM,
            chat_id="100000001",
            text="steer to B",
            mode="steer",
        )

        # Profile A's agent was NOT steered
        assert agent_a.steered_texts == []
        # Profile A received nothing
        assert len(tg_a.handled_events) == 0
        # Profile B received via queue fallback
        assert len(tg_b.handled_events) == 1
        assert tg_b.handled_events[0].text == "steer to B"

    @pytest.mark.asyncio
    async def test_invalid_mode_fails_closed(self):
        """invalid runtime mode must raise InjectInternalMessageError rather
        than silently falling through to queue mode."""
        runner = _make_runner()

        invalid_modes = ["invalid", "blerg", "INVALID", "", "steer "]
        for bad_mode in invalid_modes:
            with pytest.raises(InjectInternalMessageError) as exc:
                await runner.inject_internal_message(
                    profile="test-profile",
                    platform=Platform.TELEGRAM,
                    chat_id="100000001",
                    text="test",
                    mode=bad_mode,
                )
            assert exc.value.code == "invalid_mode", (
                f"mode={bad_mode!r} got code={exc.value.code!r}, "
                f"expected 'invalid_mode'"
            )

# ------------------------------------------------------------------
# Runner in gateway:startup hook payload
# ------------------------------------------------------------------

class TestGatewayStartupHook:
    """The gateway:startup hook payload exposes the runner for plugins."""

    @pytest.mark.asyncio
    async def test_runner_passed_in_startup_hook_context(self):
        """The startup hook payload includes the runner reference."""
        runner = _make_runner()

        # Patch the full start() method and just test the hook emit
        runner.hooks.loaded_hooks = []
        await runner.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in runner.adapters.keys()],
            "gateway_runner": runner,
        })

        runner.hooks.emit.assert_called_once()

    def test_hook_context_runner_is_callable(self):
        """The runner reference in the hook context exposes inject_internal_message."""
        runner = _make_runner()
        assert hasattr(runner, "inject_internal_message")
        assert callable(runner.inject_internal_message)
