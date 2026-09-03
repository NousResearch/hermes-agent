"""Tests for per-channel project workspace routing (honcho-projects.json).

Covers plugins/memory/honcho/session.py routing: terminal-segment pattern
matching, per-workspace child managers, cross-instance write routing via the
session workspace stamp, unmapped-key passthrough, malformed-mapping-file
tolerance, and the workspace-scoped client identity a routed child acquires
through the shared, OAuth-refreshing get_honcho_client(config) seam.
"""

import json
import logging
import os
from contextlib import contextmanager

import pytest
from unittest.mock import MagicMock, patch

from hermes_constants import get_hermes_home
from plugins.memory.honcho.client import HonchoClientConfig, reset_honcho_client
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Keep honcho-projects.json inside the test's tmp dir.

    Without this the mapping file would be written into the developer's real
    $HERMES_HOME and leak between tests (and into their live gateway).
    """
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_honcho_client()
    yield
    reset_honcho_client()


def _write_project_map(payload) -> None:
    """Write $HERMES_HOME/honcho-projects.json (str payloads written raw)."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "honcho-projects.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


@contextmanager
def _patched_clients(default_client=None, workspace_clients=None):
    """Patch the single client-acquisition seam the session manager uses.

    Default and routed managers both resolve through
    ``get_honcho_client(config)``; what distinguishes a routed child is that
    its config carries the routed ``workspace_id``. The fake dispatches on
    exactly that field, so a test asserting a child never touches the default
    client is really asserting the routing, not the mock wiring.
    """
    workspace_clients = workspace_clients if workspace_clients is not None else {}
    default = default_client if default_client is not None else MagicMock()

    def _acquire(config=None):
        return workspace_clients.get(getattr(config, "workspace_id", None), default)

    with patch(
        "plugins.memory.honcho.session.get_honcho_client",
        side_effect=_acquire,
    ) as acquire_mock:
        yield acquire_mock


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
# Routed session creation and per-workspace child clients
# ---------------------------------------------------------------------------


class TestRoutedSessionCreation:
    def test_mapped_key_creates_session_in_project_workspace(self):
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = _make_manager()

        with _patched_clients(default_client, {"myproject": project_client}) as acquire:
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")

        assert session.workspace == "myproject"
        assert session.key == "slack"
        assert session.honcho_session_id == "slack"
        project_client.session.assert_called_once_with("slack")
        default_client.session.assert_not_called()
        assert {c.args[0].workspace_id for c in acquire.call_args_list} == {"myproject"}

    def test_mapped_key_hits_child_cache_on_repeat_lookup(self):
        _write_project_map(_MAP)
        project_client = MagicMock()
        mgr = _make_manager()

        with _patched_clients(workspace_clients={"myproject": project_client}):
            first = mgr.get_or_create("slack:group:C0EXAMPLE123")
            second = mgr.get_or_create("slack:group:C0EXAMPLE123")

        assert first is second
        project_client.session.assert_called_once()

    def test_unmapped_key_uses_default_client(self):
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = _make_manager()

        with _patched_clients(default_client, {"myproject": project_client}) as acquire:
            session = mgr.get_or_create("discord:999888777")

        assert session.workspace is None
        assert session.key == "discord:999888777"
        default_client.session.assert_called_once_with("discord-999888777")
        project_client.session.assert_not_called()
        assert all(
            c.args[0].workspace_id != "myproject" for c in acquire.call_args_list
        )
        assert mgr._project_managers == {}

    def test_child_resolves_its_own_workspace_on_every_access(self):
        # The honcho property re-acquires on every access so a long-lived
        # manager cannot outlive its access token. A routed child must
        # re-acquire with its own workspace-scoped config, otherwise the
        # default client would silently snap its writes back to the default
        # workspace.
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = _make_manager()

        child = mgr._project_manager("myproject")
        with _patched_clients(default_client, {"myproject": project_client}) as acquire:
            assert child.honcho is project_client
            assert child.honcho is project_client
            assert mgr.honcho is default_client

        # Every access went through the one refreshing acquisition path; the
        # config passed is what selects the workspace, and the parent's own
        # acquisition is unaffected by the child's.
        assert acquire.call_count == 3
        child_calls = [
            c for c in acquire.call_args_list if c.args[0].workspace_id == "myproject"
        ]
        assert len(child_calls) == 2
        assert all(c.args[0] is child._config for c in child_calls)


# ---------------------------------------------------------------------------
# Routed-child client identity (reviewer regression, #68567)
# ---------------------------------------------------------------------------


class TestRoutedChildClientIdentity:
    """A routed child must acquire a client bound to its own workspace.

    The earlier revision of this PR carried a bespoke
    ``get_honcho_client_for_workspace`` with its own cache, which pinned a
    Bearer and so missed OAuth rotation (sweeper review, 2026-07-30). Upstream
    now caches clients per identity with ``workspace_id`` in the cache key
    (client.py::_client_cache_key), so routing only has to hand the child a
    config carrying the routed workspace: refresh is then the same code path
    the default workspace already uses. These tests pin that contract.
    """

    def test_child_config_carries_routed_workspace(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        child = mgr._project_manager("myproject")
        assert child._config.workspace_id == "myproject"
        assert child._project_workspace == "myproject"

    def test_parent_config_is_not_mutated(self):
        # replace() must copy: mutating the shared config in place would
        # migrate the DEFAULT workspace onto the first routed project.
        _write_project_map(_MAP)
        mgr = _make_manager()
        before = mgr._config.workspace_id
        mgr._project_manager("myproject")
        assert mgr._config.workspace_id == before

    def test_separate_workspaces_get_separate_configs(self):
        _write_project_map(_MAP)
        mgr = _make_manager()
        a = mgr._project_manager("myproject")
        b = mgr._project_manager("otherproject")
        assert a._config is not b._config
        assert {a._config.workspace_id, b._config.workspace_id} == {
            "myproject",
            "otherproject",
        }

    def test_child_acquires_through_the_shared_refreshing_seam(self):
        # Every access re-acquires (no pinned client), and it goes through
        # get_honcho_client — the function that owns OAuth refresh — with the
        # child's own workspace-scoped config.
        _write_project_map(_MAP)
        project_client = MagicMock()
        mgr = _make_manager()
        child = mgr._project_manager("myproject")

        with _patched_clients(workspace_clients={"myproject": project_client}) as acquire:
            assert child.honcho is project_client
            assert child.honcho is project_client

        assert acquire.call_count == 2
        for call in acquire.call_args_list:
            assert call.args[0].workspace_id == "myproject"
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

        with _patched_clients(default_client, {"myproject": project_client}):
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

        with _patched_clients(default_client, {"myproject": project_client}):
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

        with _patched_clients(default_client) as acquire:
            assert mgr._flush_session(session) is True

        default_client.session.assert_called_once_with("discord-999888777")
        assert all(
            c.args[0].workspace_id != "myproject" for c in acquire.call_args_list
        )
        assert mgr._project_managers == {}

    def test_flush_all_fans_out_to_project_children(self):
        _write_project_map(_MAP)
        session = self._stamped_session()

        project_client = MagicMock()
        mgr = _make_manager()

        with _patched_clients(workspace_clients={"myproject": project_client}):
            mgr._flush_session(session)
            session.add_message("user", "one more")
            mgr.flush_all()

        assert all(m["_synced"] for m in session.messages)


# ---------------------------------------------------------------------------
# Shutdown under the lazy async-writer lifecycle
# ---------------------------------------------------------------------------


class TestShutdownFansOutToChildren:
    def test_shutdown_drains_child_writers_started_lazily(self):
        # Under the lazy lifecycle the writer thread only exists once a write
        # is enqueued, and shutdown() must still flush when it never started.
        _write_project_map(_MAP)
        project_client = MagicMock()
        mgr = HonchoSessionManager(
            config=HonchoClientConfig(api_key="test-key", write_frequency="async")
        )

        with _patched_clients(workspace_clients={"myproject": project_client}):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")
            session.add_message("user", "hello there")
            child = mgr._project_managers["myproject"]
            # No writer thread has started yet on either manager.
            assert mgr._async_thread is None
            assert child._async_thread is None

            mgr.save(session)  # enqueues on the child, starting its writer
            mgr.shutdown()

        assert child._async_thread is None or not child._async_thread.is_alive()
        assert all(m["_synced"] for m in session.messages)
