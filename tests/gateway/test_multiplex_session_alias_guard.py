"""Startup invariant: no session ID reachable from two profile namespaces.

The corruption precondition for a multiplexed gateway is one session ID
aliased across profiles — from the first inbound message the two namespaces
share that session's cache, transcript, and turn lease. These pin the guard
that refuses startup when the root routing index is already in that state.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.session import (
    MultiplexSessionCollisionError,
    SessionEntry,
    SessionStore,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _entry(session_key: str, session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
    )


def _store(tmp_path, entries, **config_kwargs) -> SessionStore:
    """A loaded store whose routing index is exactly *entries*."""
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(**config_kwargs),
        )
    store._loaded = True
    store._entries = {e.session_key: e for e in entries}
    return store


class TestAliasGuard:
    def test_two_namespaces_on_one_session_trips_the_guard(self, tmp_path):
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:111", "sess-shared"),
                _entry("agent:reviewer:telegram:222", "sess-shared"),
            ],
            multiplex_profiles=True,
        )

        with pytest.raises(MultiplexSessionCollisionError) as excinfo:
            store.assert_no_cross_profile_session_aliases()

        assert "1 cross-profile session collision(s)" in str(excinfo.value)

    def test_error_leaks_no_routing_keys_or_session_ids(self, tmp_path):
        """This message reaches operator logs and runtime status."""
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:private-chat-111", "sess-secret"),
                _entry("agent:reviewer:telegram:private-chat-222", "sess-secret"),
            ],
            multiplex_profiles=True,
        )

        with pytest.raises(MultiplexSessionCollisionError) as excinfo:
            store.assert_no_cross_profile_session_aliases()

        message = str(excinfo.value)
        assert "sess-secret" not in message
        assert "private-chat" not in message
        assert "reviewer" not in message

    def test_clean_index_passes(self, tmp_path):
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:111", "sess-default"),
                _entry("agent:reviewer:telegram:222", "sess-reviewer"),
                _entry("agent:ops:discord:333", "sess-ops"),
            ],
            multiplex_profiles=True,
        )

        store.assert_no_cross_profile_session_aliases()

    def test_aliases_within_one_profile_are_legitimate(self, tmp_path):
        """Several routing keys may point at one session inside a namespace."""
        store = _store(
            tmp_path,
            [
                _entry("agent:reviewer:telegram:111", "sess-reviewer"),
                _entry("agent:reviewer:telegram:111:thread:7", "sess-reviewer"),
            ],
            multiplex_profiles=True,
        )

        store.assert_no_cross_profile_session_aliases()

    def test_main_and_default_are_the_same_namespace(self, tmp_path):
        """`agent:main:` is the default profile's on-disk spelling."""
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:111", "sess-default"),
                _entry("agent:default:telegram:222", "sess-default"),
            ],
            multiplex_profiles=True,
        )

        store.assert_no_cross_profile_session_aliases()

    def test_unprofiled_keys_are_ignored(self, tmp_path):
        """Legacy keys carry no namespace and cannot prove a collision."""
        store = _store(
            tmp_path,
            [
                _entry("telegram:111", "sess-shared"),
                _entry("agent:reviewer:telegram:222", "sess-shared"),
            ],
            multiplex_profiles=True,
        )

        store.assert_no_cross_profile_session_aliases()

    def test_counts_each_colliding_session_once(self, tmp_path):
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:111", "sess-a"),
                _entry("agent:reviewer:telegram:222", "sess-a"),
                _entry("agent:ops:telegram:333", "sess-a"),
                _entry("agent:main:discord:444", "sess-b"),
                _entry("agent:reviewer:discord:555", "sess-b"),
            ],
            multiplex_profiles=True,
        )

        with pytest.raises(MultiplexSessionCollisionError, match="2 cross-profile"):
            store.assert_no_cross_profile_session_aliases()

    def test_multiplex_off_is_a_no_op(self, tmp_path):
        """One namespace cannot alias across two, so the check does not apply."""
        store = _store(
            tmp_path,
            [
                _entry("agent:main:telegram:111", "sess-shared"),
                _entry("agent:reviewer:telegram:222", "sess-shared"),
            ],
        )

        store.assert_no_cross_profile_session_aliases()

    def test_multiplex_off_does_not_load_the_index(self, tmp_path):
        """Cheap enough to sit on every startup path, multiplexed or not."""
        store = _store(tmp_path, [])
        calls = []
        store._ensure_loaded = lambda: calls.append(1)

        store.assert_no_cross_profile_session_aliases()

        assert calls == []

    def test_empty_index_passes(self, tmp_path):
        store = _store(tmp_path, [], multiplex_profiles=True)

        store.assert_no_cross_profile_session_aliases()


class TestStartupAbort:
    @pytest.mark.asyncio
    async def test_gateway_aborts_before_connecting_adapters(self, tmp_path):
        """A collision must be caught while no platform can deliver a message."""
        from gateway.run import GATEWAY_FATAL_CONFIG_EXIT_CODE, GatewayRunner

        home_token = set_hermes_home_override(tmp_path)
        try:
            runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
            collision_check = AsyncMock(
                side_effect=MultiplexSessionCollisionError(
                    "multiplex routing contains 1 cross-profile session collision(s)"
                )
            )
            runner._async_session_store = SimpleNamespace(
                _store=runner.session_store,
                assert_no_cross_profile_session_aliases=collision_check,
            )
            runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
            runner._start_loop_liveness_guards = Mock()
            runner._request_clean_exit = Mock()
            runner._create_adapter = Mock()

            with (
                patch("gateway.run.faulthandler.enable"),
                patch("gateway.status.write_runtime_status"),
                patch(
                    "agent.monitoring.gateway_health_export."
                    "start_gateway_health_export",
                    return_value=SimpleNamespace(enabled=False),
                ),
                patch(
                    "hermes_cli.security_advisories.detect_compromised",
                    return_value=[],
                ),
            ):
                result = await runner.start()

            assert result is True
            collision_check.assert_awaited_once()
            runner._create_adapter.assert_not_called()
            runner._request_clean_exit.assert_called_once()
            assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
        finally:
            reset_hermes_home_override(home_token)
