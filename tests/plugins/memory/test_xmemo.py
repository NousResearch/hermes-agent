"""Tests for the XMemo memory provider plugin."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest

from plugins.memory.xmemo import XMemoMemoryProvider
from plugins.memory.xmemo.client import XMemoClient
from plugins.memory.xmemo.config import save_config


class FakeXMemoClient:
    """Fake synchronous XMemo REST client for unit tests."""

    def __init__(
        self,
        search_results: Optional[List[Dict[str, Any]]] = None,
        recall_context: Optional[Dict[str, Any]] = None,
    ):
        self.search_results = search_results or []
        self.recall_context_response = recall_context or {}
        self.captured_calls: List[Dict[str, Any]] = []

    def _record(self, method: str, **kwargs):
        self.captured_calls.append({"method": method, **kwargs})

    def health(self):
        self._record("health")
        return {"status": "ok"}

    def recall_context(self, **kwargs):
        self._record("recall_context", **kwargs)
        return self.recall_context_response

    def search(self, **kwargs):
        self._record("search", **kwargs)
        return self.search_results

    def remember(self, **kwargs):
        self._record("remember", **kwargs)
        return {"id": "mem-test-123"}

    def update_state(self, **kwargs):
        self._record("update_state", **kwargs)
        return {"state_key": kwargs.get("state_key", "active_task"), "id": "state-123"}

    def record_event(self, **kwargs):
        self._record("record_event", **kwargs)
        return {"id": "event-123"}

    def create_restart_snapshot(self, **kwargs):
        self._record("create_restart_snapshot", **kwargs)
        return {"id": "snapshot-123"}

    def create_reminder(self, **kwargs):
        self._record("create_reminder", **kwargs)
        return {"id": "reminder-123"}

    def list_reminders(self, **kwargs):
        self._record("list_reminders", **kwargs)
        return self.search_results

    def complete_reminder(self, **kwargs):
        self._record("complete_reminder", **kwargs)
        return {"id": kwargs.get("todo_id", "reminder-123")}

    def mark_used(self, **kwargs):
        self._record("mark_used", **kwargs)
        # Reject bucket/scope like the real Memory OS endpoint does.
        if "bucket" in kwargs or "scope" in kwargs:
            raise ValueError("MemoryUsageRequest does not accept bucket/scope")
        return {"id": kwargs.get("memory_id", "mem-123")}

    def forget(self, **kwargs):
        self._record("forget", **kwargs)
        return {"id": kwargs.get("memory_id", "mem-123")}

    def close(self):
        self._record("close")


@pytest.fixture
def provider_with_config(monkeypatch, tmp_path):
    """Create an initialized provider with a fake client."""
    monkeypatch.setenv("XMEMO_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XMEMO_AGENT_INSTANCE_ID", "test-instance")

    provider = XMemoMemoryProvider()
    provider.initialize("test-session")
    return provider


class TestAvailability:
    """is_available must be fast, network-free, and side-effect-free."""

    def test_not_available_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.delenv("MEMORY_OS_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is False

    def test_available_with_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is True

    def test_available_with_legacy_env_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.setenv("MEMORY_OS_API_KEY", "legacy-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is True

    def test_json_api_key_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.delenv("MEMORY_OS_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Write a stale api_key into xmemo.json
        save_config({"api_key": "stale-key"}, str(tmp_path))

        provider = XMemoMemoryProvider()
        assert provider.is_available() is False

    def test_is_available_does_not_create_xmemo_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.is_available()

        assert not (tmp_path / "xmemo.json").exists()


class TestLifecycle:
    """Initialization and shutdown behavior."""

    def test_initialize_loads_config(self, provider_with_config):
        assert provider_with_config._config["api_key"] == "test-key"
        assert provider_with_config._config["agent_id"] == "hermes"
        assert provider_with_config._session_id == "test-session"
        assert provider_with_config._auto_write_enabled is True

    def test_initialize_non_primary_disables_auto_write(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session", agent_context="cron")
        assert provider._auto_write_enabled is False

    def test_system_prompt_block_active(self, provider_with_config):
        block = provider_with_config.system_prompt_block()
        assert "XMemo Memory" in block
        assert "xmemo_search" in block
        assert "xmemo_remember" in block

    def test_system_prompt_block_empty_when_not_configured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XMEMO_KEY", raising=False)
        monkeypatch.delenv("MEMORY_OS_API_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")
        assert provider.system_prompt_block() == ""

    def test_name_property(self):
        provider = XMemoMemoryProvider()
        assert provider.name == "xmemo"


class TestToolGating:
    """Default tool surface is narrow; optional tools are gated."""

    def test_default_tools(self, provider_with_config):
        names = {s["name"] for s in provider_with_config.get_tool_schemas()}
        assert names == {
            "xmemo_recall_context",
            "xmemo_search",
            "xmemo_remember",
            "xmemo_update_state",
        }

    def test_workflow_tools_gated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"enable_workflow_tools": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")
        names = {s["name"] for s in provider.get_tool_schemas()}
        assert "xmemo_create_reminder" in names
        assert "xmemo_record_event" in names
        assert "xmemo_forget" not in names

    def test_destructive_tools_gated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"enable_destructive_tools": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")
        names = {s["name"] for s in provider.get_tool_schemas()}
        assert "xmemo_forget" in names

    def test_tool_schemas_read_config_before_initialize(self, monkeypatch, tmp_path):
        """MemoryManager.add_provider() calls get_tool_schemas() before initialize()."""
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({
            "enable_workflow_tools": True,
            "enable_destructive_tools": True,
        }, str(tmp_path))

        provider = XMemoMemoryProvider()
        # Do NOT call initialize(); this mirrors MemoryManager.add_provider timing.
        names = {s["name"] for s in provider.get_tool_schemas()}
        assert "xmemo_create_reminder" in names
        assert "xmemo_forget" in names

    def test_memory_manager_routing_matches_schemas(self, monkeypatch, tmp_path):
        """MemoryManager must index the same tools returned by get_tool_schemas()."""
        from agent.memory_manager import MemoryManager

        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({
            "enable_workflow_tools": True,
            "enable_destructive_tools": True,
        }, str(tmp_path))

        provider = XMemoMemoryProvider()
        manager = MemoryManager()
        manager.add_provider(provider)

        schema_names = {s["name"] for s in provider.get_tool_schemas()}
        routed_names = set(manager._tool_to_provider.keys())
        assert schema_names == routed_names
        for name in schema_names:
            assert manager._tool_to_provider[name] is provider


class TestTools:
    """Tool handlers route to the correct API calls."""

    def test_search_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            search_results=[
                {"content": "user prefers dark mode", "memory_type": "semantic", "similarity": 0.92},
            ]
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_search", {"query": "preferences"})
        )

        assert result["count"] == 1
        assert result["results"][0]["content"] == "user prefers dark mode"
        assert fake.captured_calls[0]["method"] == "search"
        assert fake.captured_calls[0]["query"] == "preferences"

    def test_search_tool_missing_query(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(provider_with_config.handle_tool_call("xmemo_search", {}))
        assert "error" in result

    def test_remember_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_remember",
                {"content": "user likes small PRs", "path": "hermes/preferences"},
            )
        )

        assert result["result"] == "Saved to XMemo."
        assert result["memory_id"] == "mem-test-123"
        assert fake.captured_calls[0]["method"] == "remember"
        assert fake.captured_calls[0]["content"] == "user likes small PRs"

    def test_update_state_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_update_state",
                {"current_task": "Implement XMemo plugin", "next_action": "Write tests"},
            )
        )

        assert result["result"] == "Working state saved to XMemo."
        assert fake.captured_calls[0]["method"] == "update_state"
        assert fake.captured_calls[0]["current_task"] == "Implement XMemo plugin"

    def test_recall_context_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            recall_context={
                "context_text": "User prefers concise answers.",
                "items": [{"content": "User prefers concise answers."}],
            }
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_recall_context", {"query": "style preferences"}
            )
        )

        assert "concise answers" in result["context"]
        assert fake.captured_calls[0]["method"] == "recall_context"

    def test_record_event_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_record_event",
                {"content": "Migrated to new memory backend", "event_type": "milestone"},
            )
        )

        assert result["result"] == "Event recorded in XMemo timeline."
        assert fake.captured_calls[0]["method"] == "record_event"

    def test_create_reminder_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_create_reminder",
                {"content": "Write migration docs", "due_at": "2026-06-20T10:00:00Z"},
            )
        )

        assert result["result"] == "Reminder saved to XMemo."
        assert fake.captured_calls[0]["method"] == "create_reminder"

    def test_complete_reminder_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_complete_reminder",
                {"todo_id": "reminder-123", "note": "Done in PR #42"},
            )
        )

        assert result["result"] == "Reminder marked completed."
        assert fake.captured_calls[0]["method"] == "complete_reminder"

    def test_mark_used_tool(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_mark_used",
                {"memory_id": "mem-456", "context": "Used to answer style question"},
            )
        )

        assert result["result"] == "Memory usage recorded in XMemo."
        assert fake.captured_calls[0]["method"] == "mark_used"

    def test_forget_tool_requires_exact_id(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call(
                "xmemo_forget",
                {"memory_id": "mem-789", "reason": "Outdated preference"},
            )
        )

        assert result["result"] == "Memory deleted from XMemo."
        assert fake.captured_calls[0]["method"] == "forget"
        assert fake.captured_calls[0]["memory_id"] == "mem-789"

    def test_forget_tool_rejects_old_target_param(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_forget", {"target": "mem-789"})
        )
        assert "error" in result


class TestPrefetch:
    """Background recall and prefetch behavior."""

    def test_queue_prefetch_populates_result(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(
            recall_context={
                "context_text": "User is working on the XMemo integration.",
                "items": [{"content": "User is working on the XMemo integration."}],
            }
        )
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.queue_prefetch("current task")
        key = provider_with_config._session_id or "__default__"
        provider_with_config._prefetch_threads[key].join(timeout=2)

        result = provider_with_config.prefetch("current task")
        assert "XMemo integration" in result
        assert fake.captured_calls[0]["method"] == "recall_context"

    def test_prefetch_skips_trivial_prompts(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.queue_prefetch("ok")
        assert not provider_with_config._prefetch_threads
        assert provider_with_config.prefetch("ok") == ""

    def test_prefetch_is_session_isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("session-A")

        results = {"session-A": "Recall A", "session-B": "Recall B"}

        class SessionFakeClient:
            def __init__(self, session_id):
                self.session_id = session_id

            def recall_context(self, **kwargs):
                return {"context_text": results[self.session_id]}

            def close(self):
                pass

        def make_client():
            return SessionFakeClient(provider._session_id)

        monkeypatch.setattr(provider, "_get_client", make_client)

        provider.queue_prefetch("query A", session_id="session-A")
        provider._prefetch_threads["session-A"].join(timeout=1.0)
        assert "session-A" in provider._prefetch_results

        provider.on_session_switch("session-B", reset=True)
        assert "session-A" not in provider._prefetch_results

        provider.queue_prefetch("query B", session_id="session-B")
        provider._prefetch_threads["session-B"].join(timeout=1.0)

        result_b = provider.prefetch("query B", session_id="session-B")
        assert "Recall B" in result_b
        assert "Recall A" not in result_b

        result_a = provider.prefetch("query A", session_id="session-A")
        assert result_a == ""


class TestSyncTurn:
    """Turn synchronization respects gating and capture policy."""

    def test_sync_turn_skips_low_signal_by_default(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.sync_turn("hello", "hi there", session_id="s1")
        # No internal thread now; call is synchronous.
        assert fake.captured_calls == []

    def test_sync_turn_default_does_not_write_high_signal(self, provider_with_config, monkeypatch):
        """capture_timeline=false means NO automatic timeline writes."""
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.sync_turn(
            "remember that I prefer small PRs", "got it", session_id="s1"
        )
        assert fake.captured_calls == []

    def test_sync_turn_disabled_for_non_primary_context(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"capture_timeline": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session", agent_context="cron")

        fake = FakeXMemoClient()
        monkeypatch.setattr(provider, "_get_client", lambda: fake)

        provider.sync_turn("this is a decision", "ok", session_id="s1")
        assert fake.captured_calls == []

    def test_sync_turn_capture_timeline_writes_high_signal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"capture_timeline": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")

        fake = FakeXMemoClient()
        monkeypatch.setattr(provider, "_get_client", lambda: fake)

        provider.sync_turn("remember that I prefer small PRs", "got it", session_id="s1")
        assert len(fake.captured_calls) == 1
        assert fake.captured_calls[0]["method"] == "record_event"
        assert fake.captured_calls[0]["session_id"] == "s1"

    def test_sync_turn_capture_timeline_skips_low_signal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"capture_timeline": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")

        fake = FakeXMemoClient()
        monkeypatch.setattr(provider, "_get_client", lambda: fake)

        provider.sync_turn("hello", "hi there", session_id="s1")
        assert fake.captured_calls == []

    def test_sync_turn_redacts_long_tokens(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_config({"capture_timeline": True}, str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session")

        fake = FakeXMemoClient()
        monkeypatch.setattr(provider, "_get_client", lambda: fake)

        secret = "sk-" + "a" * 50
        provider.sync_turn(f"remember this token {secret}", "ok", session_id="s1")
        content = fake.captured_calls[0]["content"]
        assert secret not in content
        assert "[REDACTED]" in content


class TestMemoryWriteMirror:
    """Built-in memory writes are mirrored to XMemo."""

    def test_on_memory_write_add_mirrors_to_remember(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.on_memory_write("add", "memory", "user prefers dark mode")

        assert len(fake.captured_calls) == 1
        assert fake.captured_calls[0]["method"] == "remember"
        assert fake.captured_calls[0]["content"] == "user prefers dark mode"

    def test_on_memory_write_remove_is_not_mirrored(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient()
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.on_memory_write("remove", "memory", "old fact")
        assert fake.captured_calls == []


class TestSessionSwitch:
    """Session lifecycle hooks keep provider state correct."""

    def test_session_switch_updates_id_and_clears_cache(self, provider_with_config, monkeypatch):
        fake = FakeXMemoClient(recall_context={"context_text": "old session context"})
        monkeypatch.setattr(provider_with_config, "_get_client", lambda: fake)

        provider_with_config.queue_prefetch("query")
        provider_with_config.on_session_switch("session-2", reset=True)

        assert provider_with_config._session_id == "session-2"
        assert provider_with_config.prefetch("query") == ""


class TestSessionEndSnapshot:
    """on_session_end snapshot must complete before shutdown closes the client."""

    def test_shutdown_waits_for_snapshot_thread(self, provider_with_config, monkeypatch):
        captured = {}

        class SlowClient:
            def create_restart_snapshot(self, **kwargs):
                time.sleep(0.2)
                captured["called"] = True
                return {"id": "snapshot-123"}

            def close(self):
                captured["closed"] = True

        slow_client = SlowClient()
        provider_with_config._client = slow_client

        provider_with_config.on_session_end([])
        assert provider_with_config._snapshot_thread is not None

        provider_with_config.shutdown()

        assert captured.get("called") is True
        assert captured.get("closed") is True


class TestCircuitBreaker:
    """Consecutive failures should pause API calls temporarily."""

    def test_circuit_breaker_trips(self, provider_with_config, monkeypatch):
        class FailingClient:
            def search(self, **kwargs):
                raise RuntimeError("network down")

            def close(self):
                pass

        monkeypatch.setattr(provider_with_config, "_get_client", lambda: FailingClient())

        for _ in range(6):
            provider_with_config.handle_tool_call("xmemo_search", {"query": "x"})

        assert provider_with_config._is_breaker_open() is True

        result = json.loads(
            provider_with_config.handle_tool_call("xmemo_search", {"query": "y"})
        )
        assert "temporarily unavailable" in result["error"]


class TestConfigSchema:
    """Setup wizard integration."""

    def test_config_schema_has_required_fields(self):
        provider = XMemoMemoryProvider()
        schema = provider.get_config_schema()
        keys = {field["key"] for field in schema}
        assert "api_key" in keys
        assert "base_url" in keys
        assert "scope" in keys
        assert "enable_workflow_tools" in keys
        assert "enable_destructive_tools" in keys
        assert "capture_timeline" in keys

    def test_save_config_does_not_persist_api_key(self, tmp_path):
        provider = XMemoMemoryProvider()
        provider.save_config({"api_key": "secret", "scope": "hermes/test"}, str(tmp_path))

        config_file = tmp_path / "xmemo.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "api_key" not in data
        assert data["scope"] == "hermes/test"

    def test_save_config_removes_stale_api_key(self, tmp_path):
        config_file = tmp_path / "xmemo.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps({"api_key": "stale", "bucket": "work"}))

        provider = XMemoMemoryProvider()
        provider.save_config({"bucket": "private"}, str(tmp_path))

        data = json.loads(config_file.read_text())
        assert "api_key" not in data
        assert data["bucket"] == "private"


class TestSetupWizard:
    """post_setup() and cli.py write config files correctly."""

    def test_post_setup_writes_config_and_env(self, monkeypatch, tmp_path, capsys):
        from hermes_cli.config import load_config, save_config as save_global_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        save_global_config({"memory": {}})

        monkeypatch.setattr(
            "plugins.memory.xmemo.cli._curses_select", lambda title, choices, default=0: default
        )
        monkeypatch.setattr(
            "plugins.memory.xmemo.cli.masked_secret_prompt", lambda prompt: "xmemo-token-123"
        )
        answers = iter(["", "", "", ""])
        monkeypatch.setattr("sys.stdin.readline", lambda: next(answers) + "\n")

        provider = XMemoMemoryProvider()
        config = load_config()
        provider.post_setup(str(tmp_path), config)

        updated = load_config()
        assert updated.get("memory", {}).get("provider") == "xmemo"

        env_file = tmp_path / ".env"
        assert env_file.exists()
        assert "XMEMO_KEY=xmemo-token-123" in env_file.read_text()

        xmemo_file = tmp_path / "xmemo.json"
        assert xmemo_file.exists()
        xmemo_data = json.loads(xmemo_file.read_text())
        assert "api_key" not in xmemo_data
        assert xmemo_data["bucket"] == "work"

        captured = capsys.readouterr()
        assert "Memory provider: xmemo" in captured.out


class TestProfileIsolation:
    """Different Hermes profiles should use different XMemo scopes."""

    def test_scope_derived_from_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XMEMO_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        provider = XMemoMemoryProvider()
        provider.initialize("test-session", agent_identity="coder")

        assert provider._config["scope"] == "hermes/coder"


class TestRestContract:
    """Verify REST path/method/payload against XMemo API spec."""

    def test_mark_used_uses_usage_endpoint(self):
        requests: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"id": "mem-123"})

        client = XMemoClient(
            base_url="https://xmemo.dev",
            api_key="test-key",
            transport=httpx.MockTransport(handler),
        )
        client.mark_used("mem-123", context="used in answer")
        client.close()

        assert len(requests) == 1
        assert requests[0].method == "POST"
        assert requests[0].url.path == "/v1/memories/mem-123/usage"
        body = json.loads(requests[0].content)
        assert body["context"] == "used in answer"
        assert body["action"] == "used"
        assert "bucket" not in body
        assert "scope" not in body

    def test_forget_uses_memories_forget_endpoint(self):
        requests: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"id": "mem-123"})

        client = XMemoClient(
            base_url="https://xmemo.dev",
            api_key="test-key",
            transport=httpx.MockTransport(handler),
        )
        client.forget("mem-123", reason="outdated")
        client.close()

        assert len(requests) == 1
        assert requests[0].method == "POST"
        assert requests[0].url.path == "/v1/memories/mem-123/forget"
        body = json.loads(requests[0].content)
        assert body["reason"] == "outdated"

    def test_headers_include_api_key_and_agent_ids(self):
        requests: List[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={})

        client = XMemoClient(
            base_url="https://xmemo.dev",
            api_key="test-key",
            agent_id="hermes",
            agent_instance_id="instance-1",
            transport=httpx.MockTransport(handler),
        )
        client.health()
        client.close()

        assert requests[0].headers["X-API-Key"] == "test-key"
        assert requests[0].headers["X-Memory-OS-Agent-ID"] == "hermes"
        assert requests[0].headers["X-Memory-OS-Agent-Instance-ID"] == "instance-1"


class TestUserPluginLoad:
    """Provider can be loaded from $HERMES_HOME/plugins/ as an external plugin."""

    def test_load_from_user_plugins_dir(self, monkeypatch, tmp_path):
        from plugins.memory import _MEMORY_PLUGINS_DIR, load_memory_provider

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        src = Path(__file__).parent.parent.parent.parent / "plugins" / "memory" / "xmemo"
        dst_parent = tmp_path / "plugins"
        dst_parent.mkdir(parents=True, exist_ok=True)
        dst = dst_parent / "xmemo"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

        # Force re-discovery by clearing cached modules for the synthetic namespace
        to_remove = [m for m in sys.modules if m.startswith("_hermes_user_memory.xmemo")]
        for m in to_remove:
            del sys.modules[m]

        # Bundled provider takes precedence, so hide it to exercise the real
        # external-plugin path for the canonical name "xmemo".
        monkeypatch.setattr("plugins.memory._MEMORY_PLUGINS_DIR", tmp_path / "nonexistent")

        provider = load_memory_provider("xmemo")
        assert provider is not None
        assert provider.name == "xmemo"
