"""Tests for per-channel project workspace routing (honcho-projects.json).

Covers plugins/memory/honcho/session.py routing: terminal-segment pattern
matching, per-workspace child managers, cross-instance write routing via the
session workspace stamp, unmapped-key passthrough, malformed-mapping-file
tolerance, and OAuth token rotation for routed workspaces
(plugins/memory/honcho/client.py::get_honcho_client_for_workspace).
"""

import json
import logging
import os
from contextlib import contextmanager

import pytest
from unittest.mock import MagicMock, patch

from hermes_constants import get_hermes_home
from plugins.memory.honcho import client as honcho_client
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client_for_workspace,
    reset_honcho_client,
)
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
    """Patch both client-acquisition seams the session manager uses.

    ``get_honcho_client`` serves the default workspace; routed children
    resolve through ``get_honcho_client_for_workspace`` on every access (that
    is what keeps their OAuth Bearer fresh), so the fake mirrors it by
    returning the same per-workspace object each time.
    """
    workspace_clients = workspace_clients if workspace_clients is not None else {}

    def _for_workspace(workspace, config=None):
        return workspace_clients[workspace]

    with patch(
        "plugins.memory.honcho.session.get_honcho_client",
        return_value=default_client if default_client is not None else MagicMock(),
    ) as default_mock, patch(
        "plugins.memory.honcho.session.get_honcho_client_for_workspace",
        side_effect=_for_workspace,
    ) as workspace_mock:
        yield default_mock, workspace_mock


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

        with _patched_clients(default_client, {"myproject": project_client}) as (
            _default_mock,
            workspace_mock,
        ):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")

        assert session.workspace == "myproject"
        assert session.key == "slack"
        assert session.honcho_session_id == "slack"
        project_client.session.assert_called_once_with("slack")
        default_client.session.assert_not_called()
        assert {c.args[0] for c in workspace_mock.call_args_list} == {"myproject"}

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

        with _patched_clients(default_client, {"myproject": project_client}) as (
            _default_mock,
            workspace_mock,
        ):
            session = mgr.get_or_create("discord:999888777")

        assert session.workspace is None
        assert session.key == "discord:999888777"
        default_client.session.assert_called_once_with("discord-999888777")
        project_client.session.assert_not_called()
        workspace_mock.assert_not_called()
        assert mgr._project_managers == {}

    def test_child_resolves_its_own_workspace_on_every_access(self):
        # The honcho property re-acquires on every access so a long-lived
        # manager cannot outlive its access token. A routed child must
        # re-acquire through the *workspace-aware* path, otherwise the default
        # singleton would silently snap its writes back to the default
        # workspace.
        _write_project_map(_MAP)
        default_client = MagicMock()
        project_client = MagicMock()
        mgr = _make_manager()

        child = mgr._project_manager("myproject")
        with _patched_clients(default_client, {"myproject": project_client}) as (
            _default_mock,
            workspace_mock,
        ):
            assert child.honcho is project_client
            assert child.honcho is project_client
            assert mgr.honcho is default_client

        # Both accesses went through the refreshing acquisition path, carrying
        # the manager's config so the workspace client resolves the same host.
        assert workspace_mock.call_count == 2
        for call in workspace_mock.call_args_list:
            assert call.args[0] == "myproject"
            assert call.args[1] is mgr._config


# ---------------------------------------------------------------------------
# OAuth token rotation for routed workspaces (reviewer regression)
# ---------------------------------------------------------------------------


class _FakeHttp:
    """Stand-in for the SDK's HTTP client (oauth.apply_token_to_client target)."""

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key


class _FakeHonchoClient:
    def __init__(self, workspace_id: str, api_key: str | None) -> None:
        self.workspace_id = workspace_id
        self._http = _FakeHttp(api_key)


class _NoHttpHonchoClient:
    """A client whose Bearer cannot be rotated in place (SDK shape change)."""

    def __init__(self, workspace_id: str, api_key: str | None) -> None:
        self.workspace_id = workspace_id


@contextmanager
def _fake_oauth(tokens, client_cls=_FakeHonchoClient):
    """Drive the real client.py OAuth path with a scripted token sequence.

    ``tokens`` is a list of ``(token, refreshed)`` pairs returned by successive
    ``oauth.ensure_fresh_token`` calls (the last pair repeats). Clients are
    built by a fake so no SDK/network work happens.
    """
    remaining = list(tokens)
    built: list = []

    def _ensure_fresh_token(path, host, raw=None, **kwargs):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    def _build_client(config):
        client = client_cls(config.workspace_id, config.api_key)
        built.append(client)
        return client, 30.0

    with patch(
        "plugins.memory.honcho.oauth.ensure_fresh_token",
        side_effect=_ensure_fresh_token,
    ), patch.object(honcho_client, "_build_client", side_effect=_build_client):
        yield built


class TestRoutedWorkspaceOAuthRotation:
    """A routed workspace must pick up a refreshed OAuth credential.

    Regression for the pinned-client bug: the child manager used to hold the
    client it was constructed with forever, so an OAuth deployment kept
    presenting the original bearer after the 1h access token expired.
    """

    def test_workspace_client_bearer_rotates_in_place(self):
        with _fake_oauth([("token-initial", True), ("token-rotated", True)]) as built:
            first = get_honcho_client_for_workspace("myproject", _make_config())
            assert first.workspace_id == "myproject"
            assert first._http.api_key == "token-initial"

            second = get_honcho_client_for_workspace("myproject", _make_config())

        # Same cached client (one client per workspace), fresh bearer on it.
        assert second is first
        assert len(built) == 1
        assert second._http.api_key == "token-rotated"

    def test_routed_child_manager_picks_up_refreshed_token(self):
        # End-to-end through the session manager: the property that used to
        # return a pinned client now resolves through the refreshing path.
        _write_project_map(_MAP)
        mgr = _make_manager()
        child = mgr._project_manager("myproject")

        with _fake_oauth([("token-initial", True), ("token-rotated", True)]):
            first = child.honcho
            assert first._http.api_key == "token-initial"
            second = child.honcho

        assert second is first
        # Per-workspace correctness is preserved alongside the refresh.
        assert second.workspace_id == "myproject"
        assert second._http.api_key == "token-rotated"

    def test_rotation_fallback_rebuilds_only_the_workspace_client(self):
        # When the SDK shape prevents in-place rotation, the *workspace* slot
        # is reset and rebuilt with the fresh token. The default singleton
        # must be left alone.
        default_client = object()
        honcho_client._honcho_client_slot.get(lambda: default_client)

        with _fake_oauth(
            [("token-initial", True), ("token-rotated", True)],
            client_cls=_NoHttpHonchoClient,
        ) as built:
            first = get_honcho_client_for_workspace("myproject", _make_config())
            second = get_honcho_client_for_workspace("myproject", _make_config())

        assert second is not first
        assert len(built) == 2
        assert second.workspace_id == "myproject"
        assert honcho_client._honcho_client_slot.peek() is default_client

    def test_separate_workspaces_get_separate_clients(self):
        with _fake_oauth([("token-initial", False)]) as built:
            one = get_honcho_client_for_workspace("myproject", _make_config())
            two = get_honcho_client_for_workspace("otherproject", _make_config())

        assert one is not two
        assert {c.workspace_id for c in built} == {"myproject", "otherproject"}

    def test_empty_workspace_falls_back_to_default_singleton(self):
        with _fake_oauth([("token-initial", False)]) as built:
            default = honcho_client.get_honcho_client(_make_config())
            resolved = get_honcho_client_for_workspace("", _make_config())

        # No second client: unrouted traffic keeps sharing the one singleton.
        assert resolved is default
        assert len(built) == 1
        assert resolved.workspace_id == "hermes"

    def test_caller_config_workspace_is_not_mutated(self):
        config = _make_config()
        with _fake_oauth([("token-initial", True)]):
            get_honcho_client_for_workspace("myproject", config)

        # The session manager shares one config object across default and
        # routed traffic; rebinding its workspace would break the default.
        assert config.workspace_id == "hermes"


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

        with _patched_clients(default_client) as (_default_mock, workspace_mock):
            assert mgr._flush_session(session) is True

        default_client.session.assert_called_once_with("discord-999888777")
        workspace_mock.assert_not_called()
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
