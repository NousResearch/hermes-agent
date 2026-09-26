"""Slash gating must come from the profile that serves the source, not the default one (#121705).

Under ``gateway.multiplex_profiles`` the runner's own ``GatewayConfig`` is the DEFAULT profile's.
Every slash-gating call site passed that config to ``policy_for_source``, so a platform configured
only in a secondary profile had no ``PlatformConfig`` to find: the lookup returned the disabled
(allow-everything) policy and every allowlisted user of that profile became ``unrestricted``, free
to run ``/update``, ``/sethome`` and the rest. A restart did not help, because the config it read
was never the right one.

The adapter that answers a source belongs to the profile that owns it, so its ``extra`` is the
authoritative gating config. These tests pin that the policy follows the adapter.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource

SECONDARY_GATING = {
    "allow_admin_from": ["admin-1"],
    "user_allowed_commands": ["status"],
}


def _plain(text: str) -> str:
    """Markdown emphasis varies per tier in the renderer; compare on the text."""
    return text.replace("*", "")


def _source(user_id: str = "user-9", chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id="c1",
        user_name=f"name-{user_id}",
        chat_type=chat_type,
    )


def _adapter_with_extra(extra: dict) -> MagicMock:
    """An adapter carrying a real PlatformConfig, the way a live profile adapter does."""
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    return adapter


def _runner(*, default_profile_platforms: dict, serving_adapter):
    """A runner whose own config is the DEFAULT profile's, serving a secondary profile's source."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms=default_profile_platforms)
    runner.adapters = {Platform.DISCORD: serving_adapter or MagicMock(send=AsyncMock())}
    # The multiplex registry is exercised by its own tests; what matters here is which config
    # the policy is read from once the serving adapter is known.
    runner._delivery_adapter_for = lambda _source: serving_adapter
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]),
                                   loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:discord:dm:c1", session_id="sess-1",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.DISCORD, chat_type="dm", total_tokens=0)
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *a, **k: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **k: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


class TestSecondaryProfileGating:
    @pytest.mark.asyncio
    async def test_whoami_reports_the_serving_profiles_tier(self):
        """The reported symptom: a gated non-admin came back as `unrestricted`."""
        runner = _runner(
            default_profile_platforms={},  # the platform lives only in the secondary profile
            serving_adapter=_adapter_with_extra(SECONDARY_GATING),
        )

        result = await runner._handle_message(
            MessageEvent(text="/whoami", source=_source(), message_id="m1"))

        assert "unrestricted" not in _plain(result), (
            "the secondary profile's gating was ignored and the user is unrestricted: " + result
        )
        assert "Tier: user" in _plain(result), result
        assert "/status" in result  # from the secondary profile's user_allowed_commands

    @pytest.mark.asyncio
    async def test_the_serving_profiles_admin_is_admin(self):
        """The admin list must come from the same profile, or nobody is an admin there."""
        runner = _runner(
            default_profile_platforms={},
            serving_adapter=_adapter_with_extra(SECONDARY_GATING),
        )

        result = await runner._handle_message(
            MessageEvent(text="/whoami", source=_source(user_id="admin-1"), message_id="m1"))

        assert "Tier: admin" in _plain(result), result

    def test_the_policy_is_read_from_the_serving_adapter(self):
        """Focused on the resolver: the default profile's config must not win."""
        runner = _runner(
            default_profile_platforms={},
            serving_adapter=_adapter_with_extra(SECONDARY_GATING),
        )

        policy = runner._slash_policy_for_source(_source())

        assert policy.enabled, "gating from the serving profile was dropped"
        assert "admin-1" in policy.admin_user_ids
        assert not policy.is_admin("user-9")

    def test_a_gated_default_profile_does_not_leak_into_an_ungated_one(self):
        """The mirror of the bug: the default profile's admin list must not gate a secondary
        profile that configured none."""
        runner = _runner(
            default_profile_platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True, token="***", extra={"allow_admin_from": ["someone-else"]}),
            },
            serving_adapter=_adapter_with_extra({}),
        )

        policy = runner._slash_policy_for_source(_source())

        assert not policy.enabled, "the default profile's gating leaked into another profile"

    def test_falls_back_to_the_runner_config_without_a_serving_adapter(self):
        """Single-profile behaviour is unchanged when no adapter answers the source."""
        runner = _runner(
            default_profile_platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True, token="***", extra=SECONDARY_GATING),
            },
            serving_adapter=None,
        )

        policy = runner._slash_policy_for_source(_source())

        assert policy.enabled
        assert "admin-1" in policy.admin_user_ids
