"""Tests for the Mini App resume-request marker contract + gateway watcher.

Mirrors tests/gateway/test_external_drain_control.py's structure for the
analogous drain marker -- same reasoning (no HTTP control channel into a
running gateway; a marker file + gateway-side watcher is the only safe way
for the dashboard to trigger a live in-process action), different shape
(multiple keyed pending requests instead of one presence flag).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.resume_control as rc
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestMarkerContract:
    def test_absent_by_default(self, home):
        assert rc.read_resume_requests() == {}
        assert rc.pending_resume_requests() == {}

    def test_write_then_present(self, home):
        entry = rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa", principal="dashboard")
        assert entry["target_session_id"] == "20260707_000000_aaaaaa"
        assert entry["principal"] == "dashboard"
        body = rc.read_resume_requests()
        assert body["telegram:dm:123"]["target_session_id"] == "20260707_000000_aaaaaa"

    def test_pending_requests_maps_key_to_target(self, home):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        pending = rc.pending_resume_requests()
        assert pending == {"telegram:dm:123": "20260707_000000_aaaaaa"}

    def test_multiple_keys_coexist(self, home):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        rc.write_resume_request("telegram:dm:456", "20260707_111111_bbbbbb")
        pending = rc.pending_resume_requests()
        assert pending == {
            "telegram:dm:123": "20260707_000000_aaaaaa",
            "telegram:dm:456": "20260707_111111_bbbbbb",
        }

    def test_clear_removes_only_that_key(self, home):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        rc.write_resume_request("telegram:dm:456", "20260707_111111_bbbbbb")
        assert rc.clear_resume_request("telegram:dm:123") is True
        pending = rc.pending_resume_requests()
        assert pending == {"telegram:dm:456": "20260707_111111_bbbbbb"}

    def test_clear_idempotent(self, home):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        assert rc.clear_resume_request("telegram:dm:123") is True
        assert rc.clear_resume_request("telegram:dm:123") is False

    def test_rewrite_same_key_replaces_target(self, home):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        rc.write_resume_request("telegram:dm:123", "20260707_222222_cccccc")
        pending = rc.pending_resume_requests()
        assert pending == {"telegram:dm:123": "20260707_222222_cccccc"}

    def test_path_respects_hermes_home(self, home):
        assert rc.resume_requests_path() == home / ".miniapp_resume_requests.json"

    def test_corrupt_file_reads_as_empty(self, home):
        rc.resume_requests_path().write_text("{not valid json", encoding="utf-8")
        assert rc.read_resume_requests() == {}
        assert rc.pending_resume_requests() == {}

    def test_non_object_top_level_reads_as_empty(self, home):
        rc.resume_requests_path().write_text("[1, 2, 3]", encoding="utf-8")
        assert rc.read_resume_requests() == {}

    def test_entry_missing_target_is_skipped_by_pending(self, home):
        import json
        rc.resume_requests_path().write_text(
            json.dumps({"telegram:dm:123": {"principal": "x"}}), encoding="utf-8"
        )
        assert rc.pending_resume_requests() == {}


class TestEpochStaleness:
    def test_request_from_current_instantiation_is_honoured(self, home, monkeypatch):
        monkeypatch.setattr(rc, "current_instantiation_epoch", lambda: "epoch-A")
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        assert rc.pending_resume_requests() == {"telegram:dm:123": "20260707_000000_aaaaaa"}

    def test_request_from_prior_instantiation_is_dropped(self, home, monkeypatch):
        # Mirrors the drain marker's NS-570 fix: a request written before a
        # machine restart (durable HERMES_HOME volume) must not fire against
        # a freshly-restarted gateway with a different instantiation epoch.
        monkeypatch.setattr(rc, "current_instantiation_epoch", lambda: "epoch-OLD")
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")

        monkeypatch.setattr(rc, "current_instantiation_epoch", lambda: "epoch-NEW")
        # Still physically present in the raw file...
        assert "telegram:dm:123" in rc.read_resume_requests()
        # ...but dropped from the pending mapping a watcher would act on.
        assert rc.pending_resume_requests() == {}

    def test_legacy_entry_without_epoch_still_honoured(self, home, monkeypatch):
        import json
        rc.resume_requests_path().write_text(
            json.dumps({"telegram:dm:123": {"target_session_id": "20260707_000000_aaaaaa"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(rc, "current_instantiation_epoch", lambda: "epoch-NEW")
        assert rc.pending_resume_requests() == {"telegram:dm:123": "20260707_000000_aaaaaa"}

    def test_unavailable_current_epoch_disables_staleness_check(self, home, monkeypatch):
        rc.write_resume_request("telegram:dm:123", "20260707_000000_aaaaaa")
        monkeypatch.setattr(rc, "current_instantiation_epoch", lambda: "")
        assert rc.pending_resume_requests() == {"telegram:dm:123": "20260707_000000_aaaaaa"}


# ---------------------------------------------------------------------------
# Gateway-side: _apply_session_switch (extracted from /resume) + the watcher
# ---------------------------------------------------------------------------


class TestApplySessionSwitch:
    def test_returns_none_for_unknown_session_key(self):
        runner, _ = make_restart_runner()
        runner._apply_session_switch = GatewayRunner._apply_session_switch.__get__(runner, GatewayRunner)
        # async_session_store is a read-only property that wraps whatever
        # self.session_store currently is (AsyncSessionStore(self.session_store),
        # each of its methods offloaded via asyncio.to_thread) -- mock the
        # underlying sync store and let the real property do the wrapping,
        # rather than trying to replace the property itself (no setter).
        runner.session_store = MagicMock()
        runner.session_store.switch_session.return_value = None
        runner._release_running_agent_state = MagicMock()
        result = asyncio.run(runner._apply_session_switch("telegram:dm:nope", "target-id"))
        assert result is None
        runner.session_store.switch_session.assert_called_once_with("telegram:dm:nope", "target-id")

    def test_applies_switch_and_clears_related_state(self):
        # Per-dict clearing (model/reasoning overrides, pending notes,
        # last-resolved-model cache, security state) is the conversation-
        # scope funnel's job, not this method's — see
        # test_conversation_scope_funnel.py for that behavior. This only
        # pins that _apply_session_switch routes through the funnel with
        # reason="resume", alongside its other switch mechanics.
        runner, _ = make_restart_runner()
        runner._apply_session_switch = GatewayRunner._apply_session_switch.__get__(runner, GatewayRunner)
        new_entry = MagicMock()
        runner.session_store = MagicMock()
        runner.session_store.switch_session.return_value = new_entry
        runner._release_running_agent_state = MagicMock()
        runner._clear_conversation_scope = MagicMock()
        runner._evict_cached_agent = MagicMock()

        result = asyncio.run(runner._apply_session_switch("telegram:dm:123", "target-id"))

        assert result is new_entry
        runner._release_running_agent_state.assert_called_once_with("telegram:dm:123")
        runner.session_store.switch_session.assert_called_once_with("telegram:dm:123", "target-id")
        runner._clear_conversation_scope.assert_called_once_with("telegram:dm:123", reason="resume")
        runner._evict_cached_agent.assert_called_once_with("telegram:dm:123")


class TestResumeControlWatcher:
    def _runner_with_watcher(self, home):
        runner, _ = make_restart_runner()
        runner._resume_control_watcher = GatewayRunner._resume_control_watcher.__get__(runner, GatewayRunner)
        runner._apply_session_switch = AsyncMock(return_value=MagicMock())
        return runner

    async def _run_one_tick(self, runner, monkeypatch):
        # Stop the loop after exactly one iteration by making the tick-end
        # sleep flip _running off, regardless of whether anything was
        # actually pending that tick (a no-pending tick must not hang).
        async def _stop_after_sleep(_interval):
            runner._running = False

        monkeypatch.setattr(asyncio, "sleep", _stop_after_sleep)
        await runner._resume_control_watcher(interval=0)

    def test_applies_and_clears_pending_request(self, home, monkeypatch):
        rc.write_resume_request("telegram:dm:123", "target-id")
        runner = self._runner_with_watcher(home)
        runner._running = True
        asyncio.run(self._run_one_tick(runner, monkeypatch))
        runner._apply_session_switch.assert_called_once_with("telegram:dm:123", "target-id")
        assert rc.read_resume_requests() == {}

    def test_clears_even_when_switch_returns_none(self, home, monkeypatch):
        rc.write_resume_request("telegram:dm:nope", "target-id")
        runner = self._runner_with_watcher(home)
        runner._apply_session_switch = AsyncMock(return_value=None)
        runner._running = True
        asyncio.run(self._run_one_tick(runner, monkeypatch))
        assert rc.read_resume_requests() == {}

    def test_clears_even_when_switch_raises(self, home, monkeypatch):
        rc.write_resume_request("telegram:dm:err", "target-id")
        runner = self._runner_with_watcher(home)
        runner._apply_session_switch = AsyncMock(side_effect=RuntimeError("boom"))
        runner._running = True
        asyncio.run(self._run_one_tick(runner, monkeypatch))
        assert rc.read_resume_requests() == {}

    def test_no_pending_requests_is_a_noop(self, home, monkeypatch):
        runner = self._runner_with_watcher(home)
        runner._running = True
        asyncio.run(self._run_one_tick(runner, monkeypatch))
        runner._apply_session_switch.assert_not_called()
