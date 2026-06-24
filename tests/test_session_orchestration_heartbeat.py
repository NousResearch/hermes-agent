"""
Unit tests for T008 — heartbeat edit-in-place.

Coverage
--------
1. First cadence tick (heartbeat_counter == cadence, no status_message_id):
   POSTs a new status message and persists the returned message_id via registry.
2. Subsequent cadence tick (heartbeat_counter == 2*cadence, status_message_id set):
   PATCHes in-place (PATCH, not POST); same message_id; no new notification.
3. Non-cadence tick: _on_heartbeat_tick is a no-op (no POST, no PATCH).
4. No feed_channel_id configured: no network call, early return.
5. No notification (push_turn_change) is fired on any heartbeat tick.
6. edit_status_message in feed.py uses PATCH endpoint with correct url + body.

All tests use fakes — no live Discord or registry calls (except registry tests
that use in-memory SQLite via tmp_path).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, call

import pytest

import session_orchestration.feed as feed_mod
import session_orchestration.watcher as watcher_mod
from session_orchestration.feed import edit_status_message, clear_last_notified
from session_orchestration.watcher import _on_heartbeat_tick, _HEARTBEAT_CADENCE


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeDiscordRequest:
    """Records calls to _discord_request and returns a configurable result."""

    def __init__(self):
        self.calls: List[Tuple[str, str, str]] = []  # (method, path, body_content)
        self._call_count = 0

    def __call__(self, method: str, path: str, token: str, *, body: dict = None):
        self._call_count += 1
        content = (body or {}).get("content", "")
        self.calls.append((method, path, content))
        if method == "POST":
            return {"id": f"fake-msg-id-{self._call_count}"}
        # PATCH returns the updated message (id not critical here)
        return {"id": self._get_message_id_from_path(path)}

    def _get_message_id_from_path(self, path: str) -> str:
        # /channels/<ch>/messages/<id> → extract last segment
        return path.rsplit("/", 1)[-1]


class _FakeGetBotToken:
    def __call__(self) -> str:
        return "fake-token"


def _make_row(
    task_id: str = "task-hb-001",
    agent: str = "claude",
    state: str = "RUNNING",
    heartbeat_counter: int = 0,
    status_message_id: Optional[str] = None,
    last_output_ts: Optional[float] = None,
    project: str = "my-project",
    run_id: str = "run-001",
    repo: str = "repo-abc",
    source: str = "spawn",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "state": state,
        "heartbeat_counter": heartbeat_counter,
        "status_message_id": status_message_id,
        "last_output_ts": last_output_ts,
        "project": project,
        "run_id": run_id,
        "repo": repo,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Tests: feed.edit_status_message
# ---------------------------------------------------------------------------


class TestEditStatusMessage:
    """edit_status_message issues a PATCH to the correct Discord endpoint."""

    def test_patch_issued_with_correct_path_and_content(self):
        fake_req = _FakeDiscordRequest()
        with (
            patch("tools.discord_tool._discord_request", fake_req),
            patch("tools.discord_tool._get_bot_token", _FakeGetBotToken()),
        ):
            ok = edit_status_message(
                "ch-111", "msg-999", "hello heartbeat", token="fake-token"
            )

        assert ok is True
        assert len(fake_req.calls) == 1
        method, path, content = fake_req.calls[0]
        assert method == "PATCH", f"expected PATCH, got {method}"
        assert "/channels/ch-111/messages/msg-999" in path
        assert "hello heartbeat" in content

    def test_post_not_issued(self):
        """edit_status_message must never issue a POST."""
        fake_req = _FakeDiscordRequest()
        with (
            patch("tools.discord_tool._discord_request", fake_req),
            patch("tools.discord_tool._get_bot_token", _FakeGetBotToken()),
        ):
            edit_status_message("ch-222", "msg-888", "some content", token="fake-token")

        methods = [m for m, _, _ in fake_req.calls]
        assert "POST" not in methods, "edit_status_message must use PATCH, not POST"

    def test_missing_channel_id_returns_false(self):
        fake_req = _FakeDiscordRequest()
        with (
            patch("tools.discord_tool._discord_request", fake_req),
            patch("tools.discord_tool._get_bot_token", _FakeGetBotToken()),
        ):
            ok = edit_status_message("", "msg-111", "content", token="fake-token")

        assert ok is False
        assert fake_req.calls == []

    def test_missing_message_id_returns_false(self):
        fake_req = _FakeDiscordRequest()
        with (
            patch("tools.discord_tool._discord_request", fake_req),
            patch("tools.discord_tool._get_bot_token", _FakeGetBotToken()),
        ):
            ok = edit_status_message("ch-333", "", "content", token="fake-token")

        assert ok is False
        assert fake_req.calls == []

    def test_discord_exception_returns_false(self):
        def _failing_req(*args, **kwargs):
            raise RuntimeError("network error")

        with (
            patch("tools.discord_tool._discord_request", _failing_req),
            patch("tools.discord_tool._get_bot_token", _FakeGetBotToken()),
        ):
            ok = edit_status_message("ch-444", "msg-777", "content", token="fake-token")

        assert ok is False


# ---------------------------------------------------------------------------
# Tests: _on_heartbeat_tick cadence gating
# ---------------------------------------------------------------------------


class TestHeartbeatCadenceGating:
    """Non-cadence ticks do nothing."""

    def _run_with_no_feed_calls(self, counter: int) -> List:
        """Run _on_heartbeat_tick with given counter; assert no _post_discord_message."""
        post_calls = []

        def _fake_post(channel_id, content, *, token=None):
            post_calls.append((channel_id, content))
            return "fake-id"

        def _fake_edit(channel_id, message_id, content, *, token=None):
            post_calls.append(("EDIT", channel_id, message_id, content))
            return True

        row = _make_row(heartbeat_counter=counter)
        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch.object(feed_mod, "edit_status_message", _fake_edit),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=MagicMock(feed_channel_id="ch-feed"),
            ),
        ):
            _on_heartbeat_tick("task-gate", row)

        return post_calls

    def test_counter_zero_is_skipped(self):
        calls = self._run_with_no_feed_calls(0)
        assert calls == [], "counter=0 must not fire heartbeat"

    def test_non_cadence_counter_skipped(self):
        for n in [1, 2, 3, 4]:
            calls = self._run_with_no_feed_calls(n)
            assert calls == [], f"counter={n} must not fire heartbeat"

    def test_cadence_counter_fires(self):
        """counter == _HEARTBEAT_CADENCE should trigger the heartbeat."""
        post_calls = []

        def _fake_post(channel_id, content, *, token=None):
            post_calls.append((channel_id, content))
            return "fake-msg-cadence"

        row = _make_row(heartbeat_counter=_HEARTBEAT_CADENCE, status_message_id=None)

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-cadence"

        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
            patch("session_orchestration.watcher._heartbeat_registry_ref") as mock_ref,
        ):
            _on_heartbeat_tick("task-cadence", row)

        assert len(post_calls) == 1, "cadence tick must POST one status message"
        assert mock_ref.called, "status_message_id must be persisted"


# ---------------------------------------------------------------------------
# Tests: first heartbeat POSTs + persists message_id
# ---------------------------------------------------------------------------


class TestFirstHeartbeatPost:
    """First cadence tick POSTs a new status message and stores message_id."""

    def test_first_heartbeat_posts_and_persists(self):
        post_calls = []

        def _fake_post(channel_id, content, *, token=None):
            post_calls.append((channel_id, content))
            return "new-status-msg-id"

        row = _make_row(
            task_id="t-hb-first",
            heartbeat_counter=_HEARTBEAT_CADENCE,
            status_message_id=None,
            last_output_ts=time.time() - 120,  # 2 min ago
        )

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-hb"

        persisted_calls = []

        def _fake_ref(task_id, row_arg, msg_id):
            persisted_calls.append((task_id, msg_id))

        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
            patch(
                "session_orchestration.watcher._heartbeat_registry_ref",
                _fake_ref,
            ),
        ):
            _on_heartbeat_tick("t-hb-first", row)

        # POST was issued once
        assert len(post_calls) == 1, "first heartbeat must POST exactly once"
        channel, content = post_calls[0]
        assert channel == "ch-hb"
        assert "t-hb-first" in content or "RUNNING" in content

        # message_id was persisted
        assert len(persisted_calls) == 1
        assert persisted_calls[0] == ("t-hb-first", "new-status-msg-id")

    def test_first_heartbeat_does_not_push_turn_change(self):
        """No notification is fired on heartbeat — edit only."""
        push_calls = []

        def _fake_push_turn_change(*args, **kwargs):
            push_calls.append(args)
            return True

        def _fake_post(channel_id, content, *, token=None):
            return "msg-id"

        row = _make_row(
            task_id="t-hb-nonotif",
            heartbeat_counter=_HEARTBEAT_CADENCE,
            status_message_id=None,
        )

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-hb-nonotif"

        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch.object(feed_mod, "push_turn_change", _fake_push_turn_change),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
            patch("session_orchestration.watcher._heartbeat_registry_ref"),
        ):
            _on_heartbeat_tick("t-hb-nonotif", row)

        assert push_calls == [], "heartbeat must never call push_turn_change"


# ---------------------------------------------------------------------------
# Tests: subsequent heartbeats PATCH in-place
# ---------------------------------------------------------------------------


class TestSubsequentHeartbeatPatch:
    """Subsequent cadence ticks PATCH the existing status message (not POST)."""

    def test_subsequent_heartbeat_patches_not_posts(self):
        patch_calls = []
        post_calls = []

        def _fake_post(channel_id, content, *, token=None):
            post_calls.append((channel_id, content))
            return "should-not-be-called"

        def _fake_edit(channel_id, message_id, content, *, token=None):
            patch_calls.append((channel_id, message_id, content))
            return True

        row = _make_row(
            task_id="t-hb-patch",
            heartbeat_counter=_HEARTBEAT_CADENCE * 2,  # second cadence tick
            status_message_id="existing-status-msg",
        )

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-hb-patch"

        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch.object(feed_mod, "edit_status_message", _fake_edit),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
        ):
            _on_heartbeat_tick("t-hb-patch", row)

        # PATCH was called; POST was NOT
        assert len(patch_calls) == 1, "subsequent heartbeat must PATCH once"
        assert post_calls == [], "subsequent heartbeat must not POST"

        # Correct channel and message_id
        channel, msg_id, content = patch_calls[0]
        assert channel == "ch-hb-patch"
        assert msg_id == "existing-status-msg"

    def test_subsequent_heartbeat_same_message_id(self):
        """The same message_id is always used on subsequent heartbeats."""
        edit_targets = []

        def _fake_edit(channel_id, message_id, content, *, token=None):
            edit_targets.append(message_id)
            return True

        row = _make_row(
            task_id="t-hb-sameid",
            heartbeat_counter=_HEARTBEAT_CADENCE * 3,
            status_message_id="pinned-msg-id",
        )

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-hb-sameid"

        with (
            patch.object(feed_mod, "edit_status_message", _fake_edit),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
        ):
            _on_heartbeat_tick("t-hb-sameid", row)

        assert edit_targets == ["pinned-msg-id"]

    def test_subsequent_heartbeat_no_notification(self):
        """No push_turn_change is fired on subsequent heartbeats."""
        push_calls = []

        def _fake_push(*args, **kwargs):
            push_calls.append(args)
            return True

        def _fake_edit(channel_id, message_id, content, *, token=None):
            return True

        row = _make_row(
            task_id="t-hb-nopush",
            heartbeat_counter=_HEARTBEAT_CADENCE * 2,
            status_message_id="existing-msg",
        )

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = "ch-hb-nopush"

        with (
            patch.object(feed_mod, "edit_status_message", _fake_edit),
            patch.object(feed_mod, "push_turn_change", _fake_push),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
        ):
            _on_heartbeat_tick("t-hb-nopush", row)

        assert push_calls == [], "no notification must be fired on heartbeat"


# ---------------------------------------------------------------------------
# Tests: no feed_channel_id → early return
# ---------------------------------------------------------------------------


class TestNoFeedChannelId:
    """No network call when feed_channel_id is absent."""

    def test_no_call_when_no_channel(self):
        post_calls = []
        edit_calls = []

        def _fake_post(channel_id, content, *, token=None):
            post_calls.append(channel_id)
            return "id"

        def _fake_edit(channel_id, message_id, content, *, token=None):
            edit_calls.append(channel_id)
            return True

        row = _make_row(heartbeat_counter=_HEARTBEAT_CADENCE)

        fake_cfg = MagicMock()
        fake_cfg.feed_channel_id = None  # no channel configured

        with (
            patch.object(feed_mod, "_post_discord_message", _fake_post),
            patch.object(feed_mod, "edit_status_message", _fake_edit),
            patch(
                "session_orchestration.config.load_session_orchestration_config",
                return_value=fake_cfg,
            ),
        ):
            _on_heartbeat_tick("t-hb-nochannel", row)

        assert post_calls == []
        assert edit_calls == []
