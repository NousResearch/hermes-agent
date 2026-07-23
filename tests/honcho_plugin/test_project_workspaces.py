"""Tests for per-channel project workspace routing (honcho-projects.json).

Covers plugins/memory/honcho/session.py routing: terminal-segment pattern
matching, per-workspace child managers with pinned clients, cross-instance
write routing via the session workspace stamp, unmapped-key passthrough,
and malformed-mapping-file tolerance.
"""

import json
import logging
import os

from unittest.mock import MagicMock, patch

from hermes_constants import get_hermes_home
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
)


def _write_project_map(payload) -> None:
    """Write $HERMES_HOME/honcho-projects.json (str payloads written raw)."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "honcho-projects.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _make_config() -> HonchoClientConfig:
    # write_frequency="turn" keeps tests synchronous (no async writer thread).
    return HonchoClientConfig(api_key="test-key", write_frequency="turn")


def _make_manager() -> HonchoSessionManager:
    return HonchoSessionManager(config=_make_config())


_MAP = {
    "projects": {
        "myproject": {
            "sessions": {
                "telegram-group--100123456789-1": "telegram-topic-one",
                "slack-group-C0EXAMPLE123": "slack",
            }
        },
        "otherproject": {
            "sessions": {
                "telegram-group--100123456789-1578": "telegram",
            }
        },
    }
}


# ---------------------------------------------------------------------------
# Pattern matching (_match_project_route)
# ---------------------------------------------------------------------------


class TestMatchProjectRoute:
    def test_exact_terminal_match(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        route = mgr._match_project_route("telegram:group:-100123456789:1578")
        assert route == ("otherproject", "telegram")

    def test_topic_id_prefix_does_not_collide(self):
        # Pattern "…-1" (topic 1) must NOT match "…-1578" (topic 1578): a
        # plain substring match would route topic 1578 into topic 1's project.
        _write_project_map(_MAP)
        mgr = _make_manager()
        assert mgr._match_project_route("telegram:group:-100123456789:1") == (
            "myproject",
            "telegram-topic-one",
        )
        assert mgr._match_project_route("telegram:group:-100123456789:1578") == (
            "otherproject",
            "telegram",
        )

    def test_pattern_followed_by_separator_matches(self):
        # Slack thread keys extend past the channel id with "-<thread>".
        _write_project_map(_MAP)
        mgr = _make_manager()
        route = mgr._match_project_route("slack:group:C0EXAMPLE123:thread-4567")
        assert route == ("myproject", "slack")

    def test_pattern_mid_key_without_separator_does_not_match(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        assert mgr._match_project_route("slack:group:C0EXAMPLE123999") is None

    def test_longest_pattern_wins(self):
        _write_project_map({
            "projects": {
                "broad": {"sessions": {"group--100123456789-1": "broad-name"}},
                "narrow": {"sessions": {"telegram-group--100123456789-1": "narrow-name"}},
            }
        })
        mgr = _make_manager()
        route = mgr._match_project_route("telegram:group:-100123456789:1")
        assert route == ("narrow", "narrow-name")

    def test_unmapped_key_returns_none(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        assert mgr._match_project_route("discord:999888777") is None

    def test_no_mapping_file_returns_none(self):
        mgr = _make_manager()
        assert mgr._match_project_route("telegram:group:-100123456789:1") is None

    def test_manager_without_config_does_not_route(self):
        _write_project_map(_MAP)
        mgr = HonchoSessionManager()
        assert mgr._match_project_route("telegram:group:-100123456789:1") is None

    def test_project_child_does_not_route_again(self):
        _write_project_map(_MAP)
        child = HonchoSessionManager(
            honcho=MagicMock(),
            config=_make_config(),
            project_workspace="myproject",
        )
        assert child._match_project_route("telegram:group:-100123456789:1") is None

    def test_malformed_file_warns_and_routes_nothing(self, caplog):
        _write_project_map("{not valid json")
        mgr = _make_manager()
        with caplog.at_level(logging.WARNING):
            assert mgr._match_project_route("telegram:group:-100123456789:1") is None
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_mapping_reloads_on_mtime_change(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        assert mgr._match_project_route("discord:999888777") is None

        path = get_hermes_home() / "honcho-projects.json"
        _write_project_map({
            "projects": {"myproject": {"sessions": {"discord-999888777": "discord"}}}
        })
        # Force a visible mtime change regardless of filesystem granularity.
        stat = path.stat()
        os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))

        assert mgr._match_project_route("discord:999888777") == ("myproject", "discord")


# ---------------------------------------------------------------------------
# Routed session creation and pinned child clients
# ---------------------------------------------------------------------------


class TestRoutedSessionCreation:
    def _routed_manager(self, project_client: MagicMock) -> HonchoSessionManager:
        mgr = _make_manager()
        mgr._build_project_client = MagicMock(return_value=project_client)
        return mgr

    def test_mapped_key_creates_session_in_project_workspace(self):
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = self._routed_manager(project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")

        assert session.workspace == "myproject"
        assert session.key == "slack"
        assert session.honcho_session_id == "slack"
        project_client.session.assert_called_once_with("slack")
        default_client.session.assert_not_called()
        mgr._build_project_client.assert_called_once_with("myproject")

    def test_mapped_key_hits_child_cache_on_repeat_lookup(self):
        _write_project_map(_MAP)
        project_client = MagicMock()
        mgr = self._routed_manager(project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=MagicMock(),
        ):
            first = mgr.get_or_create("slack:group:C0EXAMPLE123")
            second = mgr.get_or_create("slack:group:C0EXAMPLE123")

        assert first is second
        project_client.session.assert_called_once()

    def test_unmapped_key_uses_default_client(self):
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = self._routed_manager(project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            session = mgr.get_or_create("discord:999888777")

        assert session.workspace is None
        assert session.key == "discord:999888777"
        default_client.session.assert_called_once_with("discord-999888777")
        project_client.session.assert_not_called()
        assert mgr._project_managers == {}

    def test_child_client_stays_pinned_across_property_access(self):
        # The honcho property refreshes through the global singleton on every
        # access; project children must keep their per-workspace client or
        # writes would silently snap back to the default workspace.
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = self._routed_manager(project_client)

        child = mgr._project_manager("myproject")
        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            assert child.honcho is project_client
            assert child.honcho is project_client  # second access stays pinned
            assert mgr.honcho is default_client


# ---------------------------------------------------------------------------
# Write routing via the session workspace stamp
# ---------------------------------------------------------------------------


class TestCrossInstanceWriteRouting:
    def _stamped_session(self) -> HonchoSession:
        session = HonchoSession(
            key="slack",
            user_peer_id="user-slack-C0EXAMPLE123",
            assistant_peer_id="hermes-assistant",
            honcho_session_id="slack",
            workspace="myproject",
        )
        session.add_message("user", "hello there")
        return session

    def test_flush_session_routes_by_stamp_on_a_different_instance(self):
        # The gateway runs several manager instances; a session created by one
        # must flush through ANY other instance's child for its workspace, not
        # through that instance's default client. _flush_session is the
        # per-turn write path invoked directly by the plugin.
        _write_project_map(_MAP)
        session = self._stamped_session()

        default_client = MagicMock()
        project_client = MagicMock()
        other_mgr = _make_manager()
        other_mgr._build_project_client = MagicMock(return_value=project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            assert other_mgr._flush_session(session) is True

        project_client.session.assert_called_once_with("slack")
        default_client.session.assert_not_called()
        assert all(m["_synced"] for m in session.messages)
        # The flush lands in the child's cache, not the parent's.
        assert "slack" in other_mgr._project_managers["myproject"]._cache
        assert "slack" not in other_mgr._cache

    def test_save_routes_by_stamp(self):
        _write_project_map(_MAP)
        session = self._stamped_session()

        default_client = MagicMock()
        project_client = MagicMock()
        mgr = _make_manager()  # write_frequency="turn" → save flushes inline
        mgr._build_project_client = MagicMock(return_value=project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            mgr.save(session)

        project_client.session.assert_called_once_with("slack")
        default_client.session.assert_not_called()
        assert all(m["_synced"] for m in session.messages)

    def test_unstamped_session_flushes_through_default_client(self):
        _write_project_map(_MAP)
        session = HonchoSession(
            key="discord:999888777",
            user_peer_id="user-discord-999888777",
            assistant_peer_id="hermes-assistant",
            honcho_session_id="discord-999888777",
        )
        session.add_message("user", "hello there")

        default_client = MagicMock()
        mgr = _make_manager()
        mgr._build_project_client = MagicMock()

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=default_client,
        ):
            assert mgr._flush_session(session) is True

        default_client.session.assert_called_once_with("discord-999888777")
        mgr._build_project_client.assert_not_called()
        assert mgr._project_managers == {}

    def test_flush_all_fans_out_to_project_children(self):
        _write_project_map(_MAP)
        session = self._stamped_session()

        project_client = MagicMock()
        mgr = _make_manager()
        mgr._build_project_client = MagicMock(return_value=project_client)

        with patch(
            "plugins.memory.honcho.session.get_honcho_client",
            return_value=MagicMock(),
        ):
            mgr._flush_session(session)
            session.add_message("user", "one more")
            mgr.flush_all()

        assert all(m["_synced"] for m in session.messages)
