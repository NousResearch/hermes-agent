"""Focused regression tests for the five PR #92494 review findings.

1. TOOL SCHEMA (3837526560): register_tools must pass schema["function"]
   (the function sub-schema) to ToolRegistry, not the full wrapper.

2. NO-PEER FALSE SUCCESS (3837526562): send() must report failure when no
   peer is registered for the context, not silently succeed.

3. LOOPBACK TASK REMAINS WORKING (3837526565): fire-and-forget loopback
   pushes must complete the task immediately to prevent TaskStore entries
   from lingering in TASK_STATE_WORKING.

4. CONFIGURED PEER AUTH LOST (3837526568): _refine_peer_identity must
   resolve a sender URL that matches a configured peer back to the config
   key (preserving bearer auth), not return the URL string.

5. BOUNDED PERSISTENCE MERGE (3837526570): _merge_context_peers and
   _merge_context_sessions must insert/refresh the new entry and evict
   the oldest when at capacity, not drop the new entry.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.platforms.a2a import adapter as adapter_mod, protocol, security as security_mod, tools as a2a_tools
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_adapter() -> adapter_mod.A2AAdapter:
    from gateway.config import PlatformConfig
    return adapter_mod.A2AAdapter(PlatformConfig(enabled=True))


def _run_async(coro):
    """Run an async coroutine synchronously (no pytest-asyncio dependency)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═════════════════════════════════════════════════════════════════════════════
# 1. Tool schema — must pass schema["function"] not the full wrapper
# ═════════════════════════════════════════════════════════════════════════════


class TestToolSchemaRegistration:
    """register_tools must pass the function sub-schema to ToolRegistry."""

    def test_schema_is_function_sub_schema(self):
        """Each tool's stored schema should be the function dict, not the wrapper."""
        registry = ToolRegistry()

        class _Context:
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                registry.register(
                    name=name, toolset=toolset, schema=schema,
                    handler=handler, **kwargs,
                )

        a2a_tools.register_tools(_Context())

        for name in a2a_tools._HANDLERS:
            entry = registry.get_entry(name)
            assert entry is not None, f"{name} not registered"
            assert "name" in entry.schema, (
                f"{name}: schema should be function sub-schema with 'name' key"
            )
            assert "description" in entry.schema, (
                f"{name}: schema should have 'description' key"
            )
            assert "parameters" in entry.schema, (
                f"{name}: schema should have 'parameters' key"
            )
            assert "type" not in entry.schema, (
                f"{name}: schema has 'type' key — likely the full wrapper"
            )
            assert "function" not in entry.schema, (
                f"{name}: schema has 'function' key — likely the full wrapper"
            )

    def test_descriptions_round_trip(self, monkeypatch):
        """tool_describe must return non-empty descriptions."""
        import tools.tool_search as ts

        registry = ToolRegistry()

        class _Context:
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                registry.register(
                    name=name, toolset=toolset, schema=schema,
                    handler=handler, **kwargs,
                )

        # Gate must pass so get_definitions includes the tools.
        monkeypatch.setattr(a2a_tools, "_a2a_tools_available", lambda: True)
        a2a_tools.register_tools(_Context())
        definitions = registry.get_definitions({"a2a_call"})

        original = ts.is_deferrable_tool_name
        ts.is_deferrable_tool_name = lambda name, _dt=None: name == "a2a_call"
        try:
            described = json.loads(
                ts.dispatch_tool_describe(
                    {"names": ["a2a_call"]}, current_tool_defs=definitions,
                )
            )
        finally:
            ts.is_deferrable_tool_name = original

        tool_info = described["tools"]["a2a_call"]
        assert tool_info["description"], "a2a_call description must not be empty"
        assert tool_info["parameters"]["required"] == ["agent", "message"]
        assert set(tool_info["parameters"]["properties"]) == {
            "agent", "message", "context_id",
        }

    def test_all_tools_have_parameters(self):
        """Every registered tool must have a non-empty parameters schema."""
        registry = ToolRegistry()

        class _Context:
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                registry.register(
                    name=name, toolset=toolset, schema=schema,
                    handler=handler, **kwargs,
                )

        a2a_tools.register_tools(_Context())

        for name in a2a_tools._HANDLERS:
            entry = registry.get_entry(name)
            assert entry is not None
            params = entry.schema.get("parameters")
            assert isinstance(params, dict), (
                f"{name}: parameters must be a dict, got {type(params)}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 2. No-peer false success — send() must fail when no peer is registered
# ═════════════════════════════════════════════════════════════════════════════


class TestNoPeerFalseSuccess:
    """send() must report failure when no peer is registered for the context."""

    def test_send_no_peer_returns_failure(self):
        """send() with no registered peer must return success=False."""
        adapter = _bare_adapter()
        with adapter._context_peers_lock:
            adapter._context_peers.clear()

        result = _run_async(adapter.send(
            chat_id="ctx-no-peer-test",
            content="test reply",
            metadata={"notify": True},
        ))
        assert result.success is False, (
            f"send() with no registered peer must return success=False, "
            f"got success={result.success}"
        )
        assert "no peer" in (result.error or "").lower(), (
            f"Error message should mention 'no peer', got: {result.error}"
        )

    def test_send_with_peer_does_not_hit_no_peer_guard(self):
        """send() with a registered peer must not trigger the no-peer failure."""
        adapter = _bare_adapter()
        with adapter._context_peers_lock:
            adapter._context_peers["ctx-with-peer"] = "test-peer"
        adapter._push_out_of_band = MagicMock()

        # Non-final send (no notify) — should succeed regardless
        result = _run_async(adapter.send(
            chat_id="ctx-with-peer",
            content="progress update",
            metadata={},
        ))
        assert result.success is True


# ═════════════════════════════════════════════════════════════════════════════
# 3. Loopback task remains WORKING — fire-and-forget must complete task
# ═════════════════════════════════════════════════════════════════════════════


class TestLoopbackTaskState:
    """Fire-and-forget loopback pushes must not leave tasks in WORKING."""

    def test_fire_and_forget_completes_task(self, monkeypatch):
        """_push_loopback_in_process with want_reply=False must complete the task."""
        adapter = _bare_adapter()
        adapter.host = "127.0.0.1"
        adapter.port = 9900

        mock_pending = {
            "task_id": "task-loopback-test",
            "context_id": "ctx-loopback-test",
            "peer": "ip:127.0.0.1",
            "future": Future(),
            "created_iso": "2026-01-01T00:00:00.000Z",
            "started": time.time(),
        }
        monkeypatch.setattr(
            adapter, "_prepare_task",
            lambda params, peer, **kw: (None, mock_pending),
        )
        mock_finalize = MagicMock()
        monkeypatch.setattr(adapter, "_finalize_task", mock_finalize)

        adapter._push_loopback_in_process(
            "ctx-loopback-test", "ip:127.0.0.1", "test text",
            want_reply=False,
        )

        mock_finalize.assert_called_once()
        call_args = mock_finalize.call_args
        assert call_args[0][0] is mock_pending
        assert call_args[0][1] == protocol.STATE_COMPLETED

    def test_want_reply_does_not_finalize(self, monkeypatch):
        """_push_loopback_in_process with want_reply=True must NOT finalize."""
        adapter = _bare_adapter()
        adapter.host = "127.0.0.1"
        adapter.port = 9900

        mock_pending = {
            "task_id": "task-reply-test",
            "context_id": "ctx-reply-test",
            "peer": "ip:127.0.0.1",
            "future": Future(),
            "created_iso": "2026-01-01T00:00:00.000Z",
            "started": time.time(),
        }
        monkeypatch.setattr(
            adapter, "_prepare_task",
            lambda params, peer, **kw: (None, mock_pending),
        )
        mock_finalize = MagicMock()
        monkeypatch.setattr(adapter, "_finalize_task", mock_finalize)

        adapter._push_loopback_in_process(
            "ctx-reply-test", "ip:127.0.0.1", "reply text",
            want_reply=True,
        )

        mock_finalize.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Configured peer auth lost — URL must resolve to config key
# ═════════════════════════════════════════════════════════════════════════════


class TestConfiguredPeerAuthRetained:
    """_refine_peer_identity must resolve a sender URL to a configured peer key."""

    def test_url_matching_config_returns_key(self, monkeypatch):
        """When sender URL matches a configured peer, return the config key."""
        adapter = _bare_adapter()

        peers_cfg = {
            "my-researcher": {
                "url": "http://192.168.1.100:9901",
                "auth": {"type": "bearer", "token": "sk-secret"},
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )

        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "test",
                sender={"url": "http://192.168.1.100:9901", "agentId": "unknown-agent"},
            ),
        }

        result = adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-test")
        assert result == "my-researcher", (
            f"Expected 'my-researcher' (config key), got {result!r}"
        )

    def test_url_not_in_config_returns_url(self, monkeypatch):
        """When sender URL host is acceptable but doesn't match a peer URL, return URL."""
        adapter = _bare_adapter()

        peers_cfg = {
            "other-peer": {
                "url": "http://192.168.1.100:9901",
                "auth": {"type": "bearer", "token": "sk-other"},
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )

        # Same host but different port — acceptable (host in config) but no
        # exact URL match → returns the URL string as-is.
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "test",
                sender={"url": "http://192.168.1.100:9999"},
            ),
        }

        result = adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-test")
        assert result == "http://192.168.1.100:9999", (
            f"Unmatched URL should be returned as-is, got {result!r}"
        )

    def test_configured_key_name_still_works(self, monkeypatch):
        """When sender agentId matches a config key, return that key."""
        adapter = _bare_adapter()

        peers_cfg = {
            "my-coder": {
                "url": "http://localhost:9902",
                "auth": {"type": "bearer", "token": "sk-coder"},
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )

        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "test",
                sender={"agentId": "my-coder", "url": "http://localhost:9902"},
            ),
        }

        result = adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-test")
        assert result == "my-coder"

    def test_non_ip_identity_not_refined(self, monkeypatch):
        """Bearer-authenticated identities (not ip:) are not refined."""
        adapter = _bare_adapter()
        params = {"message": protocol.text_message(protocol.ROLE_USER, "test", sender={"agentId": "bearer-peer"})}
        result = adapter._refine_peer_identity("bearer-peer", params, "ctx-test")
        assert result == "bearer-peer"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Bounded persistence merge — insert/refresh + evict oldest at cap
# ═════════════════════════════════════════════════════════════════════════════


class TestBoundedPersistenceMerge:
    """_merge_context_peers and _merge_context_sessions must evict oldest, not drop new."""

    def test_merge_peers_inserts_at_cap(self):
        """New entry must be inserted even when at capacity (oldest evicted)."""
        cap = adapter_mod._MAX_CONTEXT_PEERS
        existing = OrderedDict((f"ctx-{i}", f"peer-{i}") for i in range(cap))
        extra = {"ctx-new": "peer-new"}

        result = adapter_mod._merge_context_peers(existing, extra)

        assert len(result) == cap
        assert "ctx-new" in result, "New entry must be present"
        assert result["ctx-new"] == "peer-new"
        assert "ctx-0" not in result, "Oldest entry (ctx-0) must be evicted"

    def test_merge_peers_refreshes_existing(self):
        """Existing entry must be refreshed (moved to end) without growing."""
        cap = adapter_mod._MAX_CONTEXT_PEERS
        existing = OrderedDict((f"ctx-{i}", f"peer-{i}") for i in range(cap))
        extra = {"ctx-5": "peer-5-updated"}

        result = adapter_mod._merge_context_peers(existing, extra)

        assert len(result) == cap
        assert result["ctx-5"] == "peer-5-updated"
        assert list(result.keys())[-1] == "ctx-5"

    def test_merge_peers_under_cap_no_eviction(self):
        """Below capacity, all entries are preserved."""
        existing = {"ctx-1": "p1", "ctx-2": "p2"}
        extra = {"ctx-3": "p3"}
        result = adapter_mod._merge_context_peers(existing, extra)
        assert len(result) == 3
        assert set(result.keys()) == {"ctx-1", "ctx-2", "ctx-3"}

    def test_merge_sessions_inserts_at_cap(self):
        """Session merge: new entry inserted, oldest evicted at cap."""
        cap = adapter_mod._MAX_CONTEXT_PEERS
        existing = OrderedDict(
            (f"ctx-{i}", {"platform": "discord", "chat_id": f"chat-{i}"})
            for i in range(cap)
        )
        extra = {"ctx-new": {"platform": "telegram", "chat_id": "chat-new"}}

        result = adapter_mod._merge_context_sessions(existing, extra)

        assert len(result) == cap
        assert "ctx-new" in result
        assert result["ctx-new"]["platform"] == "telegram"
        assert "ctx-0" not in result, "Oldest session must be evicted"

    def test_merge_sessions_refreshes_existing(self):
        """Session merge: existing entry refreshed, count unchanged."""
        cap = adapter_mod._MAX_CONTEXT_PEERS
        existing = OrderedDict(
            (f"ctx-{i}", {"platform": "discord", "chat_id": f"chat-{i}"})
            for i in range(cap)
        )
        extra = {"ctx-3": {"platform": "slack", "chat_id": "chat-3-updated"}}

        result = adapter_mod._merge_context_sessions(existing, extra)

        assert len(result) == cap
        assert result["ctx-3"]["platform"] == "slack"
        assert result["ctx-3"]["chat_id"] == "chat-3-updated"

    def test_merge_empty_extra_no_change(self):
        """Merging empty extra into existing produces no change."""
        existing = {"ctx-1": "p1"}
        result = adapter_mod._merge_context_peers(existing, {})
        assert result == {"ctx-1": "p1"}

    def test_merge_empty_existing_populates(self):
        """Merging into empty produces just the extra entries."""
        extra = {"ctx-1": "p1", "ctx-2": "p2"}
        result = adapter_mod._merge_context_peers({}, extra)
        assert result == {"ctx-1": "p1", "ctx-2": "p2"}


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Real TaskStore terminal state (not mocked)
# ═════════════════════════════════════════════════════════════════════════════


class TestLoopbackTaskStateIntegration:
    """Integration: fire-and-forget loopback must reach real TaskStore terminal state.

    Uses the real TaskStore (not mocked _finalize_task) to verify the
    complete() path actually transitions the task record.
    """

    def test_fire_and_forget_reaches_real_taskstore_completed(self, monkeypatch):
        """_finalize_task must transition a real TaskStore entry to COMPLETED."""
        adapter = _bare_adapter()
        adapter.host = "127.0.0.1"
        adapter.port = 9900

        # Create a real task in the adapter's real TaskStore.
        task_id = "task-real-store-test"
        context_id = "ctx-real-store-test"
        peer = "ip:127.0.0.1"
        adapter.tasks.create(task_id, context_id, peer)

        # Verify it starts SUBMITTED (not terminal).
        rec = adapter.tasks.get(task_id)
        assert rec is not None, "Task should exist after create"
        assert rec["state"] == protocol.STATE_SUBMITTED

        # Build a pending dict matching what _prepare_task returns.
        pending = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "future": Future(),
            "created_iso": "2026-01-01T00:00:00.000Z",
            "started": time.time(),
        }

        # Call the real _finalize_task (no mock).
        state, reply = adapter._finalize_task(
            pending, protocol.STATE_COMPLETED, "integration reply",
            audit_direction="push",
        )

        # Verify the real TaskStore record is now terminal.
        rec_after = adapter.tasks.get(task_id)
        assert rec_after is not None, "Task should still exist after finalize"
        assert rec_after["state"] == protocol.STATE_COMPLETED, (
            f"Expected TASK_STATE_COMPLETED in real TaskStore, "
            f"got {rec_after['state']!r}"
        )
        assert rec_after["reply"] == "integration reply"
        assert "completed_at" in rec_after

        # Also verify the return value matches.
        assert state == protocol.STATE_COMPLETED
        assert reply == "integration reply"

    def test_want_reply_stays_pending_in_real_taskstore(self, monkeypatch):
        """When want_reply=True, _finalize_task must NOT be called — task stays pending."""
        adapter = _bare_adapter()
        adapter.host = "127.0.0.1"
        adapter.port = 9900

        task_id = "task-want-reply-test"
        context_id = "ctx-want-reply-test"
        peer = "ip:127.0.0.1"
        adapter.tasks.create(task_id, context_id, peer)

        pending = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "future": Future(),
            "created_iso": "2026-01-01T00:00:00.000Z",
            "started": time.time(),
        }

        mock_prepare = MagicMock(return_value=(None, pending))
        monkeypatch.setattr(adapter, "_prepare_task", mock_prepare)
        mock_finalize = MagicMock()
        monkeypatch.setattr(adapter, "_finalize_task", mock_finalize)

        # want_reply=True means _finalize_task must NOT be called.
        adapter._push_loopback_in_process(context_id, peer, "reply text", want_reply=True)
        mock_finalize.assert_not_called()

        # Task should still be in SUBMITTED state in the real TaskStore.
        rec = adapter.tasks.get(task_id)
        assert rec is not None
        assert rec["state"] == protocol.STATE_SUBMITTED, (
            f"Task should remain SUBMITTED when want_reply=True, "
            f"got {rec['state']!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# EVIDENCE GAP 1: Real _push_loopback_in_process() path
# Exercises the real _prepare_task → _finalize_task path without mocking
# either method. Only disk I/O side effects (audit, persist_message) are
# stubbed; all business logic runs for real.
# ═════════════════════════════════════════════════════════════════════════════


class TestRealLoopbackPath:
    """Real _push_loopback_in_process with real _prepare_task and _finalize_task.

    The adapter's loop and message handler are set so _prepare_task proceeds
    past the gateway-readiness check.  security.audit and protocol.persist_message
    are stubbed to avoid filesystem pollution; wrap_inbound and redact_outbound
    (both pure) run for real.
    """

    def _make_real_adapter(self, monkeypatch):
        """Build an adapter with real _prepare_task/_finalize_task ready to run."""
        adapter = _bare_adapter()
        adapter.host = "127.0.0.1"
        adapter.port = 9900
        # Provide a real event loop (not running — dispatch is fire-and-forget)
        loop = asyncio.new_event_loop()
        adapter._loop = loop
        # Noop message handler so _prepare_task passes the readiness check
        async def _noop_handler(event):
            return None
        adapter._message_handler = _noop_handler
        # Stub disk I/O to keep the test environment clean
        monkeypatch.setattr(security_mod, "audit", lambda *a, **kw: None)
        monkeypatch.setattr(protocol, "persist_message", lambda *a, **kw: None)
        return adapter, loop

    @staticmethod
    def _drain_loop(loop):
        """Cancel pending tasks and close the loop without RuntimeWarning.

        ``_push_loopback_in_process`` schedules ``handle_message`` via
        ``run_coroutine_threadsafe`` on a non-running loop.  Closing the
        loop directly would trigger *coroutine was never awaited* because
        the loop's ``close`` drains ``_ready`` which still holds the
        pending callback.  We run one iteration to create the Task from
        the callback, cancel the Task, run one more iteration to finalize
        the cancellation, then close cleanly.
        """
        # Pass 1: process the Handle callback that run_coroutine_threadsafe
        # placed in _ready — this creates the asyncio Task from the coroutine.
        if loop._ready:
            loop._run_once()
        for task in asyncio.all_tasks(loop):
            task.cancel()
        # Pass 2: finalize the cancellation so the Task is properly done
        # and won't warn on destruction.
        if loop._ready:
            loop._run_once()
        loop.close()

    def test_fire_and_forget_real_path_completes_task(self, monkeypatch):
        """Real _prepare_task + _finalize_task: fire-and-forget reaches COMPLETED."""
        adapter, loop = self._make_real_adapter(monkeypatch)
        captured_ids: list[str] = []
        original_create = adapter.tasks.create
        def _capture_create(*args, **kwargs):
            rec = original_create(*args, **kwargs)
            captured_ids.append(rec["task_id"])
            return rec
        adapter.tasks.create = _capture_create
        try:
            adapter._push_loopback_in_process(
                "ctx-real-loop-ff", "ip:127.0.0.1", "fire-and-forget text",
                want_reply=False,
            )
        finally:
            self._drain_loop(loop)

        # The real _prepare_task created a task in TaskStore; _finalize_task
        # must have transitioned it to COMPLETED.
        assert len(captured_ids) == 1, f"Expected 1 task created, got {len(captured_ids)}"
        task_id = captured_ids[0]
        rec = adapter.tasks.get(task_id)
        assert rec is not None, "Task must exist in TaskStore after finalize"
        assert rec["state"] == protocol.STATE_COMPLETED, (
            f"Task must be STATE_COMPLETED after fire-and-forget finalize, "
            f"got {rec['state']!r}"
        )
        # Pending map should be empty after finalize
        with adapter._pending_lock:
            assert len(adapter._pending) == 0, (
                "Pending map should be empty after fire-and-forget finalize"
            )

    def test_want_reply_real_path_stays_pending(self, monkeypatch):
        """Real _prepare_task: want_reply=True leaves task pending (not finalized)."""
        adapter, loop = self._make_real_adapter(monkeypatch)
        captured_ids: list[str] = []
        original_create = adapter.tasks.create
        def _capture_create(*args, **kwargs):
            rec = original_create(*args, **kwargs)
            captured_ids.append(rec["task_id"])
            return rec
        adapter.tasks.create = _capture_create
        try:
            adapter._push_loopback_in_process(
                "ctx-real-loop-reply", "ip:127.0.0.1", "reply text",
                want_reply=True,
            )
        finally:
            self._drain_loop(loop)

        # With want_reply=True, _finalize_task must NOT be called.
        # TaskStore record must remain in a non-terminal state (WORKING,
        # set by _prepare_task after dispatch).
        assert len(captured_ids) == 1, f"Expected 1 task created, got {len(captured_ids)}"
        task_id = captured_ids[0]
        rec = adapter.tasks.get(task_id)
        assert rec is not None, "Task must exist in TaskStore"
        assert rec["state"] not in protocol.TERMINAL_STATES, (
            f"Task must NOT be terminal when want_reply=True, "
            f"got {rec['state']!r}"
        )
        # Pending map should have exactly one entry with a non-resolved future
        with adapter._pending_lock:
            assert len(adapter._pending) == 1, (
                "Pending map should have 1 entry when want_reply=True"
            )
            for tid, (ctx_id, fut) in adapter._pending.items():
                assert ctx_id == "ctx-real-loop-reply"
                assert not fut.done(), "Future should not be resolved yet"


# ═════════════════════════════════════════════════════════════════════════════
# EVIDENCE GAP 2: Configured peer → _resolve_peer() → _auth_header() chain
# Exercises the full URL → config key → _resolve_peer() → _auth_header() path
# and asserts the generated Authorization header is the expected bearer form.
# ═════════════════════════════════════════════════════════════════════════════


class TestResolvePeerAuthHeaderChain:
    """Configured peer URL → _resolve_peer() → _auth_header() must produce bearer."""

    def test_resolve_peer_returns_config_auth(self, monkeypatch):
        """_resolve_peer(key) must return the configured auth dict."""
        synthetic_token = "test-synthetic-token-no-real-secret"
        peers_cfg = {
            "researcher-peer": {
                "url": "http://192.168.1.50:9901",
                "auth": {"type": "bearer", "token": synthetic_token},
                "timeout": 60,
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )

        entry = a2a_tools._resolve_peer("researcher-peer")
        assert entry is not None, "_resolve_peer must return an entry"
        assert entry["url"] == "http://192.168.1.50:9901"
        assert entry["auth"]["type"] == "bearer"
        assert entry["auth"]["token"] == synthetic_token

    def test_auth_header_produces_bearer(self):
        """_auth_header must return the expected Authorization: Bearer <token> form."""
        synthetic_token = "test-synthetic-token-no-real-secret"
        auth = {"type": "bearer", "token": synthetic_token}
        headers = a2a_tools._auth_header(auth)
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {synthetic_token}"

    def test_auth_header_empty_for_no_bearer(self):
        """_auth_header must return empty dict when auth has no bearer token."""
        assert a2a_tools._auth_header({}) == {}
        assert a2a_tools._auth_header({"type": "api_key"}) == {}
        assert a2a_tools._auth_header({"type": "bearer", "token": ""}) == {}

    def test_full_chain_url_to_auth_header(self, monkeypatch):
        """End-to-end: URL matches config → resolve returns auth → header is bearer."""
        synthetic_token = "test-synthetic-token-no-real-secret"
        peers_cfg = {
            "my-coder": {
                "url": "http://192.168.1.100:9902",
                "auth": {"type": "bearer", "token": synthetic_token},
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )

        # Step 1: _refine_peer_identity resolves URL to config key
        adapter = _bare_adapter()
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "test",
                sender={"url": "http://192.168.1.100:9902", "agentId": "unknown-agent"},
            ),
        }
        refined = adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-chain")
        assert refined == "my-coder"

        # Step 2: _resolve_peer returns auth for the config key
        entry = a2a_tools._resolve_peer(refined)
        assert entry is not None
        assert entry["auth"]["type"] == "bearer"

        # Step 3: _auth_header produces the expected Authorization header
        headers = a2a_tools._auth_header(entry.get("auth") or {})
        assert headers["Authorization"] == f"Bearer {synthetic_token}"

    def test_url_not_in_config_resolves_to_url_no_auth(self, monkeypatch):
        """URL not matching any config key → _resolve_peer(url) returns no auth."""
        peers_cfg = {
            "other-peer": {
                "url": "http://10.0.0.1:9901",
                "auth": {"type": "bearer", "token": "other-token"},
            },
        }
        monkeypatch.setattr(
            a2a_tools, "_load_config",
            lambda: {"a2a_agents": peers_cfg},
        )
        # URL with a host not in config → _resolve_peer returns bare entry with empty auth
        entry = a2a_tools._resolve_peer("http://192.168.1.99:9999")
        assert entry is not None
        assert entry["auth"] == {}
        assert a2a_tools._auth_header(entry["auth"]) == {}
