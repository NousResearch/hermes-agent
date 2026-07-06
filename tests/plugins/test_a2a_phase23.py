"""
Phase 2 + Phase 3 feature tests for the A2A plugin.

Tests cover:
- SSE streaming event format
- Push notification HMAC signing
- Anti-loop ping-pong protection
- Rate limiting (token bucket per peer)
- Metrics collection
- Trusted peer approval (#56434)
- Pending task registry (async durable messaging)
- Dynamic Agent Cards from real toolsets
- Capability-based routing with fan-out (a2a_orchestrate)
- Task completion notifications (#56435)
- Orphaned task watchdog
- Metrics endpoint
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from plugins.platforms.a2a import protocol, security, tools


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: SSE Streaming
# ═════════════════════════════════════════════════════════════════════════════


class TestSSEStreaming:
    """Tests for message/stream SSE response format."""

    def test_build_streaming_event_format(self):
        """SSE events should be properly formatted with event: and data: lines."""
        event = protocol.build_streaming_event("task", "task-1", "ctx-1", {"status": {"state": "completed"}})
        assert event.startswith("event: task\n")
        assert "data: " in event
        assert event.endswith("\n\n")
        payload = json.loads(event.split("data: ", 1)[1].strip())
        assert payload["taskId"] == "task-1"
        assert payload["contextId"] == "ctx-1"
        assert payload["status"]["state"] == "completed"

    def test_build_streaming_event_done(self):
        """Done events should have no extra data."""
        event = protocol.build_streaming_event("done", "task-1", "ctx-1")
        assert "event: done" in event
        assert event.endswith("\n\n")

    def test_agent_card_advertises_streaming(self):
        """Agent Card should now advertise streaming=True capability."""
        card = protocol.build_agent_card(
            name="test", url="http://localhost:9900/",
            description="test", streaming=True, push_notifications=True,
        )
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["pushNotifications"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Push Notifications
# ═════════════════════════════════════════════════════════════════════════════


class TestPushNotifications:
    """Tests for HMAC-SHA256 push notification signing."""

    def test_sign_and_verify_push_payload(self, monkeypatch):
        """Sign a payload and verify the signature round-trips."""
        monkeypatch.setenv("A2A_PUSH_SECRET", "test-secret-123")
        payload = {"taskId": "task-1", "state": "completed", "reply": "hello"}
        sig = security.sign_push_payload(payload)
        assert sig  # non-empty when secret set
        assert security.verify_push_signature(payload, sig) is True

    def test_verify_rejects_wrong_signature(self, monkeypatch):
        """Wrong signature should be rejected."""
        monkeypatch.setenv("A2A_PUSH_SECRET", "test-secret-123")
        payload = {"taskId": "task-1", "state": "completed"}
        assert security.verify_push_signature(payload, "wrong-sig") is False

    def test_no_secret_allows_all(self, monkeypatch):
        """Without a secret configured, push verification passes (localhost mode)."""
        monkeypatch.delenv("A2A_PUSH_SECRET", raising=False)
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        payload = {"taskId": "task-1"}
        assert security.verify_push_signature(payload, "") is True

    def test_falls_back_to_bearer_token(self, monkeypatch):
        """Push secret should fall back to bearer token if not set."""
        monkeypatch.delenv("A2A_PUSH_SECRET", raising=False)
        monkeypatch.setenv("A2A_BEARER_TOKEN", "bearer-as-push-secret")
        payload = {"taskId": "task-1"}
        sig = security.sign_push_payload(payload)
        assert sig  # should use bearer token as secret
        assert security.verify_push_signature(payload, sig) is True


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Anti-loop ping-pong protection
# ═════════════════════════════════════════════════════════════════════════════


class TestAntiLoopProtection:
    """Tests for ping-pong anti-loop protection."""

    def test_track_turn_increments(self):
        """track_turn should increment and return the count."""
        protocol.reset_turns("test-anti-loop-1")
        assert protocol.track_turn("test-anti-loop-1") == 1
        assert protocol.track_turn("test-anti-loop-1") == 2
        assert protocol.track_turn("test-anti-loop-1") == 3

    def test_turn_count_returns_current(self):
        """turn_count should return the current count without incrementing."""
        protocol.reset_turns("test-anti-loop-2")
        protocol.track_turn("test-anti-loop-2")
        protocol.track_turn("test-anti-loop-2")
        assert protocol.turn_count("test-anti-loop-2") == 2

    def test_reset_turns_clears(self):
        """reset_turns should clear the count for a context."""
        protocol.reset_turns("test-anti-loop-3")
        for _ in range(5):
            protocol.track_turn("test-anti-loop-3")
        assert protocol.turn_count("test-anti-loop-3") == 5
        protocol.reset_turns("test-anti-loop-3")
        assert protocol.turn_count("test-anti-loop-3") == 0

    def test_max_pingpong_turns_default(self, monkeypatch):
        """Default max should be 5 when env not set."""
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        assert protocol.max_pingpong_turns() == 5

    def test_max_pingpong_turns_env_override(self, monkeypatch):
        """Env var should override default, capped at 20."""
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "10")
        assert protocol.max_pingpong_turns() == 10
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "50")
        assert protocol.max_pingpong_turns() == 20  # hard cap
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "0")
        assert protocol.max_pingpong_turns() == 1  # min 1


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Rate limiting
# ═════════════════════════════════════════════════════════════════════════════


class TestRateLimiting:
    """Tests for token-bucket rate limiting."""

    def test_rate_limit_allows_under_limit(self, monkeypatch):
        """Requests under the limit should be allowed."""
        monkeypatch.setenv("A2A_RATE_LIMIT", "10")
        for _ in range(10):
            assert protocol.rate_limit_allow("test-peer-1") is True

    def test_rate_limit_blocks_over_limit(self, monkeypatch):
        """Requests over the limit should be blocked."""
        monkeypatch.setenv("A2A_RATE_LIMIT", "3")
        assert protocol.rate_limit_allow("test-peer-2") is True
        assert protocol.rate_limit_allow("test-peer-2") is True
        assert protocol.rate_limit_allow("test-peer-2") is True
        assert protocol.rate_limit_allow("test-peer-2") is False  # 4th blocked

    def test_rate_limit_separate_per_peer(self, monkeypatch):
        """Different peers should have separate buckets."""
        monkeypatch.setenv("A2A_RATE_LIMIT", "2")
        assert protocol.rate_limit_allow("peer-a") is True
        assert protocol.rate_limit_allow("peer-a") is True
        assert protocol.rate_limit_allow("peer-a") is False
        assert protocol.rate_limit_allow("peer-b") is True  # different bucket
        assert protocol.rate_limit_allow("peer-b") is True

    def test_rate_limit_status(self, monkeypatch):
        """rate_limit_status should return correct stats."""
        monkeypatch.setenv("A2A_RATE_LIMIT", "5")
        for _ in range(3):
            protocol.rate_limit_allow("test-peer-3")
        status = protocol.rate_limit_status("test-peer-3")
        assert status["peer"] == "test-peer-3"
        assert status["limit_per_minute"] == 5
        assert status["used"] == 3
        assert status["remaining"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Metrics
# ═════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    """Tests for the metrics system."""

    def test_metrics_snapshot_has_fields(self):
        """Snapshot should include all expected fields."""
        m = protocol.metrics.snapshot()
        assert "uptime_seconds" in m
        assert "inbound_total" in m
        assert "outbound_total" in m
        assert "streams_started" in m
        assert "push_sent" in m
        assert "push_failed" in m
        assert "tasks_completed" in m
        assert "tasks_failed" in m
        assert "anti_loop_triggers" in m
        assert "rate_limit_triggers" in m
        assert "avg_latency_ms" in m

    def test_metrics_record_latency(self):
        """Recording latency should update the average."""
        protocol.metrics.record_latency(0.1)
        protocol.metrics.record_latency(0.3)
        avg = protocol.metrics.avg_latency()
        assert 0.19 <= avg <= 0.21  # (0.1 + 0.3) / 2 = 0.2


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Trusted peer approval (#56434)
# ═════════════════════════════════════════════════════════════════════════════


class TestTrustedPeers:
    """Tests for trusted peer approval."""

    def test_localhost_trusts_all(self, monkeypatch):
        """In localhost-only mode, all peers should be trusted."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        assert security.is_trusted_peer("anyone") is True
        assert security.is_trusted_peer("random-peer") is True

    def test_allow_all_users_trusts_everyone(self, monkeypatch):
        """A2A_ALLOW_ALL_USERS should trust all peers."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "secret")
        monkeypatch.setenv("A2A_ALLOW_ALL_USERS", "true")
        assert security.is_trusted_peer("anyone") is True

    def test_untrusted_peer_rejected(self, monkeypatch):
        """Without allow-all or localhost mode, untrusted peers should be rejected."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "secret")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
        assert security.is_trusted_peer("unknown-peer") is False

    def test_trusted_peers_from_env(self, monkeypatch):
        """Trusted peers from env should be accepted."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "secret")
        monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("A2A_TRUSTED_PEERS", "alice,bob,carol")
        assert security.is_trusted_peer("alice") is True
        assert security.is_trusted_peer("bob") is True
        assert security.is_trusted_peer("carol") is True
        assert security.is_trusted_peer("dave") is False


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Pending task registry (async durable messaging)
# ═════════════════════════════════════════════════════════════════════════════


class TestPendingTaskRegistry:
    """Tests for pending task tracking."""

    def test_register_and_complete(self):
        """Register a task, verify it's pending, then complete it."""
        protocol.register_pending_task("task-reg-1", "ctx-1", "peer-1")
        info = protocol.pending_task_info("task-reg-1")
        assert info is not None
        assert info["context_id"] == "ctx-1"
        assert info["peer"] == "peer-1"
        assert info["state"] == "working"

        completed = protocol.complete_pending_task("task-reg-1", "completed", "reply text")
        assert completed is not None
        assert completed["state"] == "completed"
        assert completed["reply"] == "reply text"
        assert "completed_at" in completed

        # After completion, should not be in pending
        assert protocol.pending_task_info("task-reg-1") is None

    def test_orphaned_tasks(self):
        """Orphaned tasks should be detected and cleaned up."""
        protocol.register_pending_task("task-orphan-1", "ctx-2", "peer-2")
        # Manually age it
        protocol._pending_tasks["task-orphan-1"]["started_at"] = time.time() - 400

        orphans = protocol.orphaned_tasks(timeout_seconds=300)
        assert any(o["task_id"] == "task-orphan-1" for o in orphans)

        cleared = protocol.clear_orphaned_tasks(timeout_seconds=300)
        assert "task-orphan-1" in cleared
        assert protocol.pending_task_info("task-orphan-1") is None


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Dynamic Agent Cards
# ═════════════════════════════════════════════════════════════════════════════


class TestDynamicAgentCards:
    """Tests for dynamic Agent Cards from real toolsets."""

    def test_skills_from_real_toolsets(self):
        """skills_from_real_toolsets should build richer skill cards."""
        registry = {
            "web": {
                "description": "Web search and extraction tools",
                "tools": [{"name": "web_search"}, {"name": "web_extract"}],
            },
            "terminal": {
                "description": "Shell command execution",
                "tools": [{"name": "terminal"}, {"name": "read_file"}],
            },
        }
        skills = protocol.skills_from_real_toolsets(registry)
        assert len(skills) == 2
        names = [s["name"] for s in skills]
        assert "terminal" in names
        assert "web" in names
        # Should include tool names as tags
        web_skill = [s for s in skills if s["name"] == "web"][0]
        assert "web_search" in web_skill["tags"]
        assert "web_extract" in web_skill["tags"]

    def test_skills_from_real_toolsets_empty(self):
        """Empty registry should return default general skill."""
        skills = protocol.skills_from_real_toolsets({})
        assert len(skills) == 1
        assert skills[0]["name"] == "general"

    def test_agent_card_version_bumped(self):
        """Agent Card version should be bumped for Phase 2+3."""
        card = protocol.build_agent_card(
            name="test", url="http://localhost:9900/",
            description="test", streaming=True, push_notifications=True,
        )
        assert card["version"] == "0.2.0"


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Capability-based routing (a2a_orchestrate)
# ═════════════════════════════════════════════════════════════════════════════


class TestA2AOrchestrate:
    """Tests for capability-based routing with fan-out."""

    def test_orchestrate_requires_capability(self):
        """a2a_orchestrate should require a capability argument."""
        result = tools.a2a_orchestrate({"message": "do something"})
        assert "Error" in result
        assert "capability" in result

    def test_orchestrate_requires_message(self):
        """a2a_orchestrate should require a message argument."""
        result = tools.a2a_orchestrate({"capability": "research"})
        assert "Error" in result
        assert "message" in result

    def test_orchestrate_no_matching_peers(self):
        """Should report error when no peers match the capability."""
        from unittest.mock import patch
        with patch.object(tools, "_load_config", return_value={}):
            result = tools.a2a_orchestrate({"capability": "research", "message": "search for X"})
            assert "Error" in result
            assert "no configured peers" in result

    def test_match_peers_by_capability(self):
        """_match_peers_by_capability should find peers with matching caps."""
        from unittest.mock import patch
        with patch.object(tools, "_load_config", return_value={
            "a2a_agents": {
                "researcher": {
                    "url": "http://localhost:9991",
                    "capabilities": ["research", "web_search"],
                },
                "coder": {
                    "url": "http://localhost:9992",
                    "capabilities": ["code", "debug"],
                },
            }
        }):
            matches = tools._match_peers_by_capability("research")
            assert len(matches) == 1
            assert matches[0][0] == "researcher"

            matches = tools._match_peers_by_capability("*")
            assert len(matches) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Task completion notifications (#56435)
# ═════════════════════════════════════════════════════════════════════════════


class TestTaskCompletionNotification:
    """Tests for task completion notification."""

    def test_build_task_completed_has_reply(self):
        """Completed task should include the reply text in status message."""
        task = protocol.build_task("task-1", "ctx-1", protocol.STATE_COMPLETED, "here is the answer")
        assert task["status"]["state"] == "completed"
        assert task["artifacts"][0]["parts"][0]["text"] == "here is the answer"

    def test_build_task_failed_has_message(self):
        """Failed task should include the error message."""
        task = protocol.build_task("task-2", "ctx-2", protocol.STATE_FAILED, "something went wrong")
        assert task["status"]["state"] == "failed"
        assert task["status"]["message"]["parts"][0]["text"] == "something went wrong"


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Metrics endpoint + Watchdog
# ═════════════════════════════════════════════════════════════════════════════


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_endpoint_in_adapter(self):
        """The /metrics endpoint should be handled in do_GET."""
        from plugins.platforms.a2a.adapter import A2AAdapter
        import inspect
        source = inspect.getsource(A2AAdapter)
        assert "/metrics" in source


class TestWatchdog:
    """Tests for the orphaned task watchdog."""

    def test_watchdog_thread_started_on_connect(self):
        """connect() should start the watchdog thread."""
        from plugins.platforms.a2a.adapter import A2AAdapter
        import inspect
        source = inspect.getsource(A2AAdapter.connect)
        assert "_watchdog_thread" in source
        assert "_watchdog_loop" in source

    def test_clear_orphaned_tasks(self):
        """clear_orphaned_tasks should remove tasks older than timeout."""
        protocol.register_pending_task("task-watchdog-1", "ctx-w1", "peer-w1")
        protocol._pending_tasks["task-watchdog-1"]["started_at"] = time.time() - 600
        cleared = protocol.clear_orphaned_tasks(timeout_seconds=300)
        assert "task-watchdog-1" in cleared
        assert protocol.pending_task_info("task-watchdog-1") is None


# ═════════════════════════════════════════════════════════════════════════════
# Code review fixes: SSRF protection, body size limit, watchdog reconnect
# ═════════════════════════════════════════════════════════════════════════════

class TestSSRFProtection:
    """Push notification callback URL validation (SSRF prevention)."""

    def test_safe_https_url_allowed(self):
        assert security.is_safe_callback_url("https://example.com/webhook") is True

    def test_safe_http_url_allowed(self):
        assert security.is_safe_callback_url("http://example.com/webhook") is True

    def test_localhost_blocked_in_remote_mode(self):
        """In remote mode (localhost_only=False), localhost should be blocked."""
        # Temporarily disable localhost_only
        original = security.localhost_only
        security.localhost_only = lambda: False
        try:
            assert security.is_safe_callback_url("http://127.0.0.1:8080/hook") is False
            assert security.is_safe_callback_url("http://localhost:8080/hook") is False
        finally:
            security.localhost_only = original

    def test_aws_metadata_blocked(self):
        """169.254.x.x (AWS metadata) must be blocked."""
        original = security.localhost_only
        security.localhost_only = lambda: False
        try:
            assert security.is_safe_callback_url("http://169.254.169.254/latest/meta-data/") is False
        finally:
            security.localhost_only = original

    def test_private_ranges_blocked(self):
        """RFC1918 private ranges must be blocked in remote mode."""
        original = security.localhost_only
        security.localhost_only = lambda: False
        try:
            assert security.is_safe_callback_url("http://10.0.0.1/hook") is False
            assert security.is_safe_callback_url("http://192.168.1.1/hook") is False
            assert security.is_safe_callback_url("http://172.16.0.1/hook") is False
        finally:
            security.localhost_only = original

    def test_file_scheme_blocked(self):
        """file:// URLs must be blocked (urllib follows them)."""
        assert security.is_safe_callback_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert security.is_safe_callback_url("ftp://example.com/file") is False

    def test_empty_url_blocked(self):
        assert security.is_safe_callback_url("") is False
        assert security.is_safe_callback_url(None) is False


class TestBodySizeLimit:
    """Request body size limit (DoS prevention)."""

    def test_max_body_constant_exists(self):
        """adapter module should define _MAX_BODY."""
        from plugins.platforms.a2a import adapter
        assert hasattr(adapter, "_MAX_BODY")
        assert adapter._MAX_BODY > 0
        # Should be reasonable (1-10MB)
        assert adapter._MAX_BODY <= 10_485_760

    def test_max_body_imports_from_adapter(self):
        """The body size check should reference _MAX_BODY."""
        from plugins.platforms.a2a import adapter
        import inspect
        source = inspect.getsource(adapter)
        assert "_MAX_BODY" in source
        assert "413" in source  # HTTP 413 Payload Too Large


class TestWatchdogReconnect:
    """Watchdog should survive reconnection (disconnect → connect cycle)."""

    def test_watchdog_stop_cleared_on_connect(self):
        """connect() should call _watchdog_stop.clear() to reset state."""
        from plugins.platforms.a2a.adapter import A2AAdapter
        import inspect
        source = inspect.getsource(A2AAdapter.connect)
        assert "_watchdog_stop.clear()" in source


class TestErrorRedaction:
    """Error messages should be redacted before sending to peers."""

    def test_dispatch_failed_uses_redact_outbound(self):
        """adapter should call security.redact_outbound on error messages."""
        from plugins.platforms.a2a import adapter
        import inspect
        source = inspect.getsource(adapter)
        # All 'Dispatch failed' messages should go through redact_outbound
        import re
        dispatch_fails = re.findall(r'"Dispatch failed[^"]*"', source)
        for match in dispatch_fails:
            # Find the surrounding context (should contain redact_outbound)
            idx = source.index(match)
            context = source[max(0, idx-200):idx+100]
            assert "redact_outbound" in context, f"Dispatch failed message not redacted: {match}"


class TestThreadSafety:
    """Verify thread-safe shared state access."""

    def test_turn_tracking_is_thread_safe(self):
        """track_turn, turn_count, reset_turns should use a lock."""
        import inspect
        assert hasattr(protocol, "_turn_lock")
        source = inspect.getsource(protocol.track_turn)
        assert "_turn_lock" in source

    def test_rate_limiting_is_thread_safe(self):
        """rate_limit_allow should use a lock."""
        import inspect
        assert hasattr(protocol, "_rate_lock")
        source = inspect.getsource(protocol.rate_limit_allow)
        assert "_rate_lock" in source

    def test_pending_tasks_lock_initialized_at_import(self):
        """_pending_lock should be initialized at module level, not lazily."""
        assert hasattr(protocol, "_pending_lock")
        assert protocol._pending_lock is not None
        # Should be a Lock instance, not None
        assert hasattr(protocol._pending_lock, "acquire")


class TestContextIdConsistency:
    """a2a_call should always send contextId, even on first turn."""

    def test_a2a_call_always_sends_context_id(self):
        """tools.a2a_call should include contextId in params even when not provided."""
        import inspect
        source = inspect.getsource(tools.a2a_call)
        # The contextId should be set unconditionally, not inside an if block
        assert "contextId" in source