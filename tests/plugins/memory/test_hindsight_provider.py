"""Tests for the Hindsight memory provider plugin.

Tests cover config loading, tool handlers (tags, max_tokens, types),
prefetch (auto_recall, preamble, query truncation), sync_turn (auto_retain,
turn counting, tags), and schema completeness.
"""

import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.memory_setup import _CANCELLED
from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    RECALL_SCHEMA,
    REFLECT_SCHEMA,
    RETAIN_SCHEMA,
    _load_config,
    _load_simple_env,
    _build_embedded_profile_env,
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _resolve_bank_id_template,
    _sanitize_bank_segment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Ensure no stale env vars or Windows home state leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT", "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)

    # On Windows pathlib.Path.home() resolves USERPROFILE/HOMEDRIVE+HOMEPATH,
    # not the POSIX HOME alias that these tests historically monkeypatched.
    # Patch the actual API and keep all legacy profile writes in tmp_path.
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))


def _make_mock_client():
    """Create a mock Hindsight client with async methods."""
    async def _aretain(
        bank_id,
        content,
        timestamp=None,
        context=None,
        document_id=None,
        metadata=None,
        entities=None,
        tags=None,
        update_mode=None,
        retain_async=None,
    ):
        return SimpleNamespace(ok=True)

    client = MagicMock()
    client.aretain = AsyncMock(side_effect=_aretain)
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[
                SimpleNamespace(text="Memory 1"),
                SimpleNamespace(text="Memory 2"),
            ]
        )
    )
    client.areflect = AsyncMock(
        return_value=SimpleNamespace(text="Synthesized answer")
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _provider_for_mode(tmp_path, monkeypatch, mode: str):
    """Create an initialized provider without pre-seeding its client."""
    config = {
        "mode": mode,
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    return provider


def _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, mode: str):
    """Cloud/local-external clients must ensure lazy deps before importing."""
    import builtins

    provider = _provider_for_mode(tmp_path, monkeypatch, mode)
    ensure_calls = []

    def fake_ensure(feature, prompt=True):
        ensure_calls.append((feature, prompt))

    class FakeHindsight:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hindsight_client":
            if ensure_calls != [("memory.hindsight", False)]:
                raise ModuleNotFoundError("No module named 'hindsight_client'")
            return SimpleNamespace(Hindsight=FakeHindsight)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    client = provider._get_client()

    assert ensure_calls == [("memory.hindsight", False)]
    assert isinstance(client, FakeHindsight)
    assert client.kwargs == {
        "base_url": "http://localhost:9999",
        "timeout": 120.0,
        "api_key": "test-key",
    }


class _FakeSessionDB:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    def get_messages_as_conversation(self, session_id):
        return list(self._messages)


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create an initialized HindsightMemoryProvider with a mock client."""
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    p._client = _make_mock_client()
    return p


@pytest.fixture()
def provider_with_config(tmp_path, monkeypatch):
    """Create a provider factory that accepts custom config overrides."""
    def _make(**overrides):
        config = {
            "mode": "cloud",
            "apiKey": "test-key",
            "api_url": "http://localhost:9999",
            "bank_id": "test-bank",
            "budget": "mid",
            "memory_mode": "hybrid",
        }
        config.update(overrides)
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        p._client = _make_mock_client()
        return p
    return _make


def test_normalize_retain_tags_accepts_csv_and_dedupes():
    assert _normalize_retain_tags("agent:fakeassistantname, source_system:hermes-agent, agent:fakeassistantname") == [
        "agent:fakeassistantname",
        "source_system:hermes-agent",
    ]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_retain_schema_has_content(self):
        assert RETAIN_SCHEMA["name"] == "hindsight_retain"
        assert "content" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "tags" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "content" in RETAIN_SCHEMA["parameters"]["required"]


    def test_get_tool_schemas_returns_three(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}

    def test_context_mode_returns_no_tools(self, provider_with_config):
        p = provider_with_config(memory_mode="context")
        assert p.get_tool_schemas() == []


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_cloud_client_lazy_installs_dependency_before_import(self, tmp_path, monkeypatch):
        _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, "cloud")


    def test_default_values(self, provider):
        assert provider._auto_retain is True
        assert provider._auto_recall is True
        assert provider._retain_every_n_turns == 1
        assert provider._recall_max_tokens == 4096
        assert provider._recall_max_input_chars == 800
        assert provider._tags is None
        assert provider._observation_scopes is None
        assert provider._recall_tags is None
        # Default recall narrowed to observation-only; world/experience are
        # aggregate facts that often crowd out concrete-event signal during
        # auto-recall. Users opt back in via the recall_types config key.
        assert provider._recall_types == ["observation"]
        assert provider._bank_mission == ""
        assert provider._bank_retain_mission is None
        assert provider._retain_context == "conversation between Hermes Agent and the User"

    def test_recall_types_default_is_observation_only(self, provider):
        """Auto-recall must filter to observation by default."""
        assert provider._recall_types == ["observation"]


    @pytest.mark.parametrize("scope", ["per_tag", "shared"])
    def test_observation_scopes_keyword_config(self, provider_with_config, scope):
        p = provider_with_config(observation_scopes=scope)
        assert p._observation_scopes == scope


    def test_custom_config_values(self, provider_with_config):
        p = provider_with_config(
            retain_tags=["tag1", "tag2"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
            recall_tags=["recall-tag"],
            recall_tags_match="all",
            auto_retain=False,
            auto_recall=False,
            retain_every_n_turns=3,
            retain_context="custom-ctx",
            bank_retain_mission="Extract key facts",
            recall_max_tokens=2048,
            recall_types=["world", "experience"],
            recall_prompt_preamble="Custom preamble:",
            recall_max_input_chars=500,
            bank_mission="Test agent mission",
        )
        assert p._tags == ["tag1", "tag2"]
        assert p._retain_tags == ["tag1", "tag2"]
        assert p._retain_source == "hermes"
        assert p._retain_user_prefix == "User (fakeusername)"
        assert p._retain_assistant_prefix == "Assistant (fakeassistantname)"
        assert p._recall_tags == ["recall-tag"]
        assert p._recall_tags_match == "all"
        assert p._auto_retain is False
        assert p._auto_recall is False
        assert p._retain_every_n_turns == 3
        assert p._retain_context == "custom-ctx"
        assert p._bank_retain_mission == "Extract key facts"
        assert p._recall_max_tokens == 2048
        assert p._recall_types == ["world", "experience"]
        assert p._recall_prompt_preamble == "Custom preamble:"
        assert p._recall_max_input_chars == 500
        assert p._bank_mission == "Test agent mission"


    def test_embedded_profile_env_includes_idle_timeout_from_config(self):
        env = _build_embedded_profile_env({
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "idle_timeout": 0,
        })

        assert env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"


    def test_get_client_passes_idle_timeout_to_hindsight_embedded(self, monkeypatch):
        captured = {}

        class FakeHindsightEmbedded:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeHindsightEmbedded))
        monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

        p = HindsightMemoryProvider()
        p._mode = "local_embedded"
        p._config = {
            "profile": "hermes",
            "llm_provider": "openai_compatible",
            "llm_api_key": "test-key",
            "llm_model": "test-model",
            "idle_timeout": 0,
        }
        p._llm_base_url = "http://localhost:8060/v1"

        p._get_client()

        assert captured["idle_timeout"] == 0
        assert captured["llm_provider"] == "openai"


class TestPostSetup:
    def test_setup_cancel_at_mode_picker_writes_nothing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)

        save_config = MagicMock()
        which = MagicMock(return_value="/usr/bin/uv")
        run = MagicMock()
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: _CANCELLED)
        monkeypatch.setattr("shutil.which", which)
        monkeypatch.setattr("subprocess.run", run)
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("getpass.getpass", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("hermes_cli.config.save_config", save_config)

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {"provider": "builtin"}})

        save_config.assert_not_called()
        which.assert_not_called()
        run.assert_not_called()
        assert not (hermes_home / ".env").exists()
        assert not (hermes_home / "hindsight" / "config.json").exists()
        assert not (user_home / ".hindsight" / "profiles" / "hermes.env").exists()


    def test_local_embedded_setup_materializes_profile_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))

        selections = iter([1, 0])  # local_embedded, openai
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-local-test")
        saved_configs = []
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved_configs.append(cfg.copy()))

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {}})

        assert saved_configs[-1]["memory"]["provider"] == "hindsight"
        env_text = (hermes_home / ".env").read_text()
        assert "HINDSIGHT_LLM_API_KEY=sk-local-test\n" in env_text
        assert "HINDSIGHT_TIMEOUT=120\n" in env_text
        assert "HINDSIGHT_IDLE_TIMEOUT=300\n" in env_text

        profile_env = user_home / ".hindsight" / "profiles" / "hermes.env"
        assert profile_env.exists()
        assert profile_env.read_text() == (
            "HINDSIGHT_API_LLM_PROVIDER=openai\n"
            "HINDSIGHT_API_LLM_API_KEY=sk-local-test\n"
            "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
            "HINDSIGHT_API_LOG_LEVEL=info\n"
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=300\n"
        )


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_retain_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "user likes dark mode"}
        ))
        assert result["result"] == "Memory stored successfully."
        provider._client.aretain_batch.assert_called_once()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        item = call_kwargs["items"][0]
        assert item["content"] == "user likes dark mode"
        # bank_id/retain_async are call-level args, never item keys.
        assert "bank_id" not in item
        assert "retain_async" not in item


    def test_recall_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "dark mode"}
        ))
        assert "Memory 1" in result["result"]
        assert "Memory 2" in result["result"]


    def test_reflect_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_reflect", {"query": "summarize"}
        ))
        assert result["result"] == "Synthesized answer"


    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_unknown", {}
        ))
        assert "error" in result


    def test_local_embedded_recall_reconnects_after_idle_shutdown(self, provider, monkeypatch):
        first_client = _make_mock_client()
        first_client.arecall.side_effect = RuntimeError("Cannot connect to host 127.0.0.1:8888")
        second_client = _make_mock_client()
        second_client.arecall.return_value = SimpleNamespace(
            results=[SimpleNamespace(text="Recovered memory")]
        )
        clients = iter([first_client, second_client])

        provider._mode = "local_embedded"
        provider._client = first_client
        monkeypatch.setattr(provider, "_get_client", lambda: next(clients))

        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "test"}
        ))

        assert result["result"] == "1. Recovered memory"
        assert provider._client is second_client
        first_client.arecall.assert_called_once()
        second_client.arecall.assert_called_once()


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_when_no_result(self, provider):
        assert provider.prefetch("test") == ""


    def test_queue_prefetch_skipped_in_tools_mode(self, provider_with_config):
        p = provider_with_config(memory_mode="tools")
        p.queue_prefetch("test")
        # Should not start a thread
        assert p._prefetch_thread is None

    def test_prefetch_waits_for_pending_retain_before_recall(self, provider):
        """The background prefetch must wait for queued retains to drain so the
        next turn's recall observes the just-completed turn (no retain race)."""
        import threading

        order = []
        release = threading.Event()

        async def _slow_retain(*args, **kwargs):
            release.wait(timeout=5.0)
            order.append("retain")

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        provider._client.aretain_batch = AsyncMock(side_effect=_slow_retain)
        provider._client.arecall = AsyncMock(side_effect=_recall)

        # Enqueue a slow retain, then immediately queue the next-turn prefetch.
        provider.sync_turn("hello", "world")
        provider.queue_prefetch("next turn query")

        # Let the prefetch thread start and reach the drain barrier.
        time.sleep(0.2)
        assert order == [], "recall ran before the pending retain drained"

        # Release the retain; the prefetch should now proceed AFTER it.
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        provider._retain_queue.join()
        assert order and order[0] == "retain"
        assert "recall" in order

    def test_prefetch_wait_for_retain_can_be_disabled(self, provider_with_config):
        p = provider_with_config(prefetch_waits_for_retain=False)
        p._client = _make_mock_client()
        assert p._prefetch_waits_for_retain is False


class TestPrefetchServerRetainVisibility:
    """PR #62871 review follow-up: draining the local writer queue is not a
    read-after-write signal for async retains. With ``retain_async=True`` the
    server accepts the write and returns an ``operation_id`` that stays
    ``pending`` until the write is durable/recall-visible. The background
    prefetch must gate on server-side operation completion, not just the local
    queue, before recalling.
    """

    def _client_with_ops(self, statuses):
        """Mock client whose aretain_batch returns an async operation_id and
        whose operations.get_operation_status yields *statuses* in order
        (last value repeats)."""
        client = _make_mock_client()
        client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-1", operation_ids=None)
        )
        seq = list(statuses)

        async def _status(**kwargs):
            value = seq.pop(0) if len(seq) > 1 else seq[0]
            return SimpleNamespace(status=value)

        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(side_effect=_status)
        return client

    def test_tracks_async_operation_id_from_retain(self, provider):
        provider._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-async-1", operation_ids=None)
        )
        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert "op-async-1" in provider._pending_retain_ops

    def test_tracks_multiple_operation_ids(self, provider):
        provider._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(
                operation_id=None, operation_ids=["op-a", "op-b"]
            )
        )
        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert {"op-a", "op-b"} <= provider._pending_retain_ops

    def test_sync_retain_tracks_no_ops(self, provider_with_config):
        p = provider_with_config(retain_async=False)
        p._client = _make_mock_client()
        p._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-x", operation_ids=None)
        )
        p.sync_turn("hello", "world")
        p._retain_queue.join()
        # retain_async=False → no server-side op to wait on.
        assert p._pending_retain_ops == set()

    def test_prefetch_waits_for_server_completion_before_recall(self, provider):
        """Recall must not run until the tracked async op reports completed."""
        order = []

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        provider._client = self._client_with_ops(["pending", "pending", "completed"])
        provider._client.arecall = AsyncMock(side_effect=_recall)

        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert "op-1" in provider._pending_retain_ops

        provider.queue_prefetch("next turn query")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        # Recall ran, the op was polled to completion, and the pending set
        # was cleared (so a later prefetch won't re-poll it).
        assert order == ["recall"]
        assert provider._client.operations.get_operation_status.await_count >= 3
        assert provider._pending_retain_ops == set()

    def test_prefetch_proceeds_after_server_wait_timeout(self, provider_with_config):
        """A wedged/never-completing async op must not hang prefetch forever;
        it recalls anyway once the drain budget is exhausted."""
        p = provider_with_config(prefetch_retain_drain_timeout=0.3)
        order = []

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        p._client = self._client_with_ops(["pending"])  # never completes
        p._client.arecall = AsyncMock(side_effect=_recall)

        p.sync_turn("hello", "world")
        p._retain_queue.join()

        start = time.monotonic()
        p.queue_prefetch("next turn query")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        elapsed = time.monotonic() - start

        assert order == ["recall"], "prefetch should recall after the timeout"
        assert elapsed < 3.0, "prefetch must not block well past the drain budget"

    def test_timed_out_ops_are_dropped_not_repolled(self, provider_with_config):
        """Ops unresolved at deadline must be EVICTED so a permanently failing
        status endpoint can't make every later prefetch re-burn the full
        timeout on a growing pending set (unbounded session-wide degradation
        + reply-path join penalty)."""
        p = provider_with_config(prefetch_retain_drain_timeout=0.3)
        p._client = self._client_with_ops(["pending"])  # never completes
        p._client.arecall = AsyncMock(
            return_value=SimpleNamespace(results=[SimpleNamespace(text="m")])
        )

        p.sync_turn("hello", "world")
        p._retain_queue.join()
        assert p._pending_retain_ops, "op should be tracked before the wait"

        # First prefetch burns the budget and must DROP the wedged op.
        p.queue_prefetch("q1")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        assert p._pending_retain_ops == set(), (
            "unresolved ops must be evicted at deadline, not retained"
        )

        # A later prefetch with nothing pending must be near-instant.
        start = time.monotonic()
        p.queue_prefetch("q2")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        assert time.monotonic() - start < 0.25, (
            "second prefetch re-polled dropped ops — eviction regressed"
        )

    def test_operation_notfound_treated_as_complete(self, provider):
        """A NotFound (completed+evicted) op is treated as done, not pending."""
        from hindsight_client_api.exceptions import NotFoundException

        client = _make_mock_client()
        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(
            side_effect=NotFoundException(status=404, reason="gone")
        )
        provider._client = client

        assert provider._is_retain_op_complete("bank", "op-gone") is True

    def test_transient_status_error_keeps_waiting(self, provider):
        """A transient status-check error means 'unknown', so keep waiting."""
        client = _make_mock_client()
        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(
            side_effect=RuntimeError("temporary blip")
        )
        provider._client = client

        assert provider._is_retain_op_complete("bank", "op-1") is False


# ---------------------------------------------------------------------------
# sync_turn tests
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_sync_turn_retains_metadata_rich_turn(self, provider_with_config):
        p = provider_with_config(
            retain_tags=["conv", "session1"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
        )
        p.initialize(
            session_id="session-1",
            platform="discord",
            user_id="fakeusername-123",
            user_name="fakeusername",
            chat_id="1485316232612941897",
            chat_name="fakeassistantname-forums",
            chat_type="thread",
            thread_id="1491249007475949698",
            agent_identity="fakeassistantname",
        )
        p._client = _make_mock_client()

        p.sync_turn("hello", "hi there")
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["document_id"].startswith("session-1-")
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        item = call_kwargs["items"][0]
        assert item["context"] == "conversation between Hermes Agent and the User"
        assert item["tags"] == ["conv", "session1", "session:session-1"]
        content = json.loads(item["content"])
        assert len(content) == 1
        assert content[0][0]["role"] == "user"
        assert content[0][0]["content"] == "User (fakeusername): hello"
        assert content[0][1]["role"] == "assistant"
        assert content[0][1]["content"] == "Assistant (fakeassistantname): hi there"
        assert item["metadata"]["source"] == "hermes"
        assert item["metadata"]["session_id"] == "session-1"
        assert item["metadata"]["platform"] == "discord"
        assert item["metadata"]["user_id"] == "fakeusername-123"
        assert item["metadata"]["user_name"] == "fakeusername"
        assert item["metadata"]["chat_id"] == "1485316232612941897"
        assert item["metadata"]["chat_name"] == "fakeassistantname-forums"
        assert item["metadata"]["chat_type"] == "thread"
        assert item["metadata"]["thread_id"] == "1491249007475949698"
        assert item["metadata"]["agent_identity"] == "fakeassistantname"
        assert item["metadata"]["turn_index"] == "1"
        assert item["metadata"]["message_count"] == "2"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00", content[0][0]["timestamp"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", item["metadata"]["retained_at"])


    def test_resume_creates_new_document(self, tmp_path, monkeypatch):
        """Resuming a session (re-initializing) gets a new document_id
        so previously stored content is not overwritten."""
        config = {"mode": "cloud", "apiKey": "k", "api_url": "http://x", "bank_id": "b"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p1 = HindsightMemoryProvider()
        p1.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Sleep just enough that the microsecond timestamp differs
        import time
        time.sleep(0.001)

        p2 = HindsightMemoryProvider()
        p2.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Same session, but each process gets its own document_id
        assert p1._document_id != p2._document_id
        assert p1._document_id.startswith("resumed-session-")
        assert p2._document_id.startswith("resumed-session-")


# ---------------------------------------------------------------------------
# Shutdown / writer tests
# ---------------------------------------------------------------------------


class TestShutdownRace:
    def test_sync_turn_uses_single_writer_thread(self, provider):
        """All retains run through one long-lived writer thread."""
        provider.sync_turn("a", "b")
        provider._retain_queue.join()
        first_writer = provider._writer_thread
        assert first_writer is not None
        assert first_writer.is_alive()

        provider.sync_turn("c", "d")
        provider._retain_queue.join()
        # Same thread reused — no ad-hoc thread per call.
        assert provider._writer_thread is first_writer
        assert provider._client.aretain_batch.call_count == 2


    def test_shutdown_drains_pending_retains(self, provider):
        """Shutdown must wait for queued retains to complete, not abandon them.

        Otherwise the LAST in-flight turn — typically the most important —
        is silently lost.
        """
        client = provider._client
        provider.sync_turn("a", "b")
        provider.sync_turn("c", "d")
        provider.shutdown()
        # Both retains drained before shutdown returned.
        assert client.aretain_batch.call_count == 2
        assert provider._retain_queue.empty()


# ---------------------------------------------------------------------------
# on_session_switch — flush + prefetch reset behavior
# ---------------------------------------------------------------------------


class TestSessionSwitchBufferFlush:
    def test_buffered_turns_flushed_before_clear(self, provider_with_config):
        """retain_every_n_turns > 1 must not silently drop partial buffers
        on session switch. Whatever's in _session_turns at switch time
        should land in the OLD document under the OLD session id."""
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        old_doc = p._document_id

        # Two turns buffered, no retain yet (boundary is at turn 3). The
        # writer hasn't been started either — sync_turn's early return
        # skips _ensure_writer when no retain is due.
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None
        p._client.aretain_batch.assert_not_called()

        # Switch — flush should fire under OLD document_id via the writer queue.
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        kw = p._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        item = kw["items"][0]
        # Both buffered turns must be present in the flushed payload.
        content = json.loads(item["content"])
        flat = json.dumps(content)
        assert "turn1-user" in flat
        assert "turn2-user" in flat
        # Old session id must appear in lineage tags / metadata.
        assert "session:test-session" in item["tags"]
        assert item["metadata"]["session_id"] == "test-session"

        # And the new session must start with a clean slate.
        assert p._session_id == "new-sid"
        assert p._session_turns == []
        assert p._turn_counter == 0
        assert p._document_id != old_doc
        assert p._document_id.startswith("new-sid-")


    def test_in_flight_prefetch_thread_drained_on_switch(self, provider, monkeypatch):
        """on_session_switch must wait for an in-flight prefetch from the
        old session to settle before clearing _prefetch_result, otherwise
        the thread can race and re-populate the field after the clear."""
        import threading

        gate = threading.Event()
        finished = threading.Event()

        def _slow_prefetch():
            gate.wait(timeout=5.0)
            with provider._prefetch_lock:
                provider._prefetch_result = "old-session recall"
            finished.set()

        provider._prefetch_thread = threading.Thread(target=_slow_prefetch, daemon=True)
        provider._prefetch_thread.start()

        # Release the prefetch worker so it writes _prefetch_result, then
        # call on_session_switch — it must join the thread before clearing.
        gate.set()
        provider.on_session_switch("new-sid")

        assert finished.is_set(), "switch returned before prefetch thread settled"
        assert provider._prefetch_result == ""

    def test_flush_serializes_behind_pending_retains_via_writer_queue(
        self, provider_with_config
    ):
        """The flush closure must ride the same _retain_queue sync_turn
        uses, so it lands FIFO behind any still-queued old-session
        retains rather than racing them on a separate thread.

        Regression guard: an earlier draft spawned a raw threading.Thread
        for flush, overwriting _sync_thread and racing the writer against
        the same document_id.
        """
        import threading as _threading

        p = provider_with_config(retain_every_n_turns=2, retain_async=False)

        # Block the first writer job until we've enqueued the flush
        # behind it. This proves ordering — the flush MUST wait.
        gate = _threading.Event()
        call_order: list[str] = []

        def _aretain_batch_tracking(**kw):
            idx = kw["items"][0]["metadata"].get("turn_index", "")
            call_order.append(str(idx))
            if idx == "2":
                # First retain blocks until we've enqueued the flush.
                gate.wait(timeout=5.0)

        p._client.aretain_batch = AsyncMock(side_effect=_aretain_batch_tracking)

        # Turn 1+2 → boundary hit → retain enqueued (will block).
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")

        # One more buffered turn so flush has something to land.
        p.sync_turn("turn3-user", "turn3-asst")

        # Switch while the first retain is still blocked on `gate`.
        p.on_session_switch("new-sid", parent_session_id="test-session")

        # Release the first retain. Flush must have been enqueued
        # BEHIND it, and run second.
        gate.set()
        p._retain_queue.join()

        # The flush carries all buffered turns; sync_turn's retain #2
        # carried the batch at boundary time. Two distinct calls.
        assert p._client.aretain_batch.call_count == 2
        # First call landed while buffer was [t1, t2]; flush landed
        # after we added t3. So the second call must be strictly after.
        assert call_order[0] == "2"
        # Flush retain has turn_index matching the buffered count at
        # switch time (3 turns accumulated, _turn_index was set to 3
        # by the last sync_turn).
        assert call_order[1] == "3"


# ---------------------------------------------------------------------------
# update_mode='append' capability probe + retain dispatch
# ---------------------------------------------------------------------------


class TestUpdateModeAppendCapability:
    def _clear_capability_cache(self):
        from plugins.memory.hindsight import _append_capability_cache, _append_capability_lock
        with _append_capability_lock:
            _append_capability_cache.clear()

    def test_legacy_api_falls_back_to_per_process_doc_id(self, provider, monkeypatch):
        """API returns no /version (or pre-0.5.0) — sync_turn must use the
        per-process unique doc_id and NOT pass update_mode."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: None,
        )
        old_doc = provider._document_id
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        assert kw["document_id"].startswith("test-session-")
        item = kw["items"][0]
        assert "update_mode" not in item

    def test_modern_api_uses_stable_doc_id_with_append(self, provider, monkeypatch):
        """API on >=0.5.0 — retain uses stable session_id and sets update_mode='append'."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        # Stable: just the session id, no per-process timestamp suffix.
        assert kw["document_id"] == "test-session"
        item = kw["items"][0]
        assert item["update_mode"] == "append"


    def test_session_switch_flush_picks_capability_against_old_session(
        self, provider_with_config, monkeypatch
    ):
        """When the API supports append, the flush on /reset must land
        in the OLD session's stable document, not a per-process id."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        kw = p._client.aretain_batch.call_args.kwargs
        # Flush goes to the OLD session's stable doc, not new-sid's.
        assert kw["document_id"] == "test-session"
        assert kw["items"][0]["update_mode"] == "append"


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_hybrid_mode_prompt(self, provider):
        block = provider.system_prompt_block()
        assert "Hindsight Memory" in block
        assert "hindsight_recall" in block
        assert "automatically injected" in block


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_has_all_new_fields(self, provider):
        schema = provider.get_config_schema()
        keys = {f["key"] for f in schema}
        expected_keys = {
            "mode", "api_url", "api_key", "llm_provider", "llm_api_key",
            "llm_model", "bank_id", "bank_id_template", "bank_mission", "bank_retain_mission",
            "recall_budget", "memory_mode", "recall_prefetch_method",
            "retain_tags", "retain_source",
            "retain_user_prefix", "retain_assistant_prefix",
            "recall_tags", "recall_tags_match",
            "auto_recall", "auto_retain",
            "retain_every_n_turns", "retain_async", "retain_context",
            "recall_max_tokens", "recall_max_input_chars",
            "recall_prompt_preamble",
        }
        assert expected_keys.issubset(keys), f"Missing: {expected_keys - keys}"


# ---------------------------------------------------------------------------
# bank_id_template tests
# ---------------------------------------------------------------------------


class TestBankIdTemplate:
    def test_sanitize_bank_segment_passthrough(self):
        assert _sanitize_bank_segment("hermes") == "hermes"
        assert _sanitize_bank_segment("my-agent_1") == "my-agent_1"


    def test_resolve_empty_template_uses_fallback(self):
        result = _resolve_bank_id_template(
            "", fallback="hermes", profile="coder"
        )
        assert result == "hermes"


    def test_resolve_sanitizes_placeholder_values(self):
        result = _resolve_bank_id_template(
            "user-{user}", fallback="hermes",
            profile="", workspace="", platform="",
            user="josh@example.com", session="",
        )
        assert result == "user-josh-example-com"


    def test_provider_uses_bank_id_template_from_config(self, tmp_path, monkeypatch):
        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "fallback-bank",
            "bank_id_template": "hermes-{profile}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="coder",
            agent_workspace="hermes",
        )
        assert p._bank_id == "hermes-coder"
        assert p._bank_id_template == "hermes-{profile}"


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_API_KEY", "test-key")
        p = HindsightMemoryProvider()
        assert p.is_available()


    def test_local_mode_unavailable_when_runtime_import_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_MODE", "local")

        def _raise(_name):
            raise RuntimeError(
                "NumPy was built with baseline optimizations: (x86_64-v2)"
            )

        monkeypatch.setattr(
            "plugins.memory.hindsight.importlib.import_module",
            _raise,
        )
        p = HindsightMemoryProvider()
        assert not p.is_available()

    def test_initialize_disables_local_mode_when_runtime_import_fails(self, tmp_path, monkeypatch):
        config = {"mode": "local_embedded"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        def _raise(_name):
            raise RuntimeError("x86_64-v2 unsupported")

        monkeypatch.setattr(
            "plugins.memory.hindsight.importlib.import_module",
            _raise,
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        assert p._mode == "disabled"


class TestSharedEventLoopLifecycle:
    """Regression tests for #11923 — Hindsight leaking aiohttp ClientSession /
    TCPConnector objects in long-running gateway processes.

    Root cause: the module-global ``_loop`` / ``_loop_thread`` pair is shared
    across every HindsightMemoryProvider instance in the process (the plugin
    loader builds one provider per AIAgent, and the gateway builds one AIAgent
    per concurrent chat session). When a session ended, ``shutdown()`` stopped
    the shared loop, which orphaned every *other* live provider's aiohttp
    ClientSession on a dead loop. Those sessions were never closed and surfaced
    as ``Unclosed client session`` / ``Unclosed connector`` errors.
    """

    def test_shutdown_does_not_stop_shared_event_loop(self, provider_with_config):
        from plugins.memory import hindsight as hindsight_mod

        async def _noop():
            return 1

        # Prime the shared loop by scheduling a trivial coroutine — mirrors
        # the first time any real async call (arecall/aretain/areflect) runs.
        assert hindsight_mod._run_sync(_noop()) == 1

        loop_before = hindsight_mod._loop
        thread_before = hindsight_mod._loop_thread
        assert loop_before is not None and loop_before.is_running()
        assert thread_before is not None and thread_before.is_alive()

        # Build two independent providers (two concurrent chat sessions).
        provider_a = provider_with_config()
        provider_b = provider_with_config()

        # End session A.
        provider_a.shutdown()

        # Module-global loop/thread must still be the same live objects —
        # provider B (and any other sibling provider) is still relying on them.
        assert hindsight_mod._loop is loop_before, (
            "shutdown() swapped out the shared event loop — sibling providers "
            "would have their aiohttp ClientSession orphaned (#11923)"
        )
        assert hindsight_mod._loop.is_running(), (
            "shutdown() stopped the shared event loop — sibling providers' "
            "aiohttp sessions would leak (#11923)"
        )
        assert hindsight_mod._loop_thread is thread_before
        assert hindsight_mod._loop_thread.is_alive()

        # Provider B can still dispatch async work on the shared loop.
        async def _still_working():
            return 42

        assert hindsight_mod._run_sync(_still_working()) == 42

        provider_b.shutdown()

    def test_client_aclose_called_on_cloud_mode_shutdown(self, provider):
        """Per-provider session cleanup still runs even though the shared
        loop is preserved. Each provider's own aiohttp session is closed
        via ``self._client.aclose()``; only the (empty) shared loop survives.
        """
        assert provider._client is not None
        mock_client = provider._client

        provider.shutdown()

        mock_client.aclose.assert_called_once()
        assert provider._client is None


class TestShutdown:
    def test_local_embedded_shutdown_closes_inner_async_client_on_shared_loop(self, provider):
        inner_client = _make_mock_client()
        embedded = MagicMock()
        embedded._client = inner_client
        embedded.close = MagicMock()

        provider._mode = "local_embedded"
        provider._client = embedded

        provider.shutdown()

        inner_client.aclose.assert_awaited_once()
        embedded.close.assert_called_once()
        assert embedded._client is None
        assert provider._client is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """hindsight/config.json must be written with 0o600 so API key is not world-readable."""
    provider = HindsightMemoryProvider()
    provider.save_config({"api_key": "hd-test-key"}, str(tmp_path))
    config_file = tmp_path / "hindsight" / "config.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"


class TestLoadSimpleEnv:
    def test_bom_first_key_is_recognized(self, tmp_path):
        """A Notepad-edited .env carries a BOM; the first key must still parse
        instead of becoming '\ufeffHINDSIGHT_LLM_API_KEY'."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿HINDSIGHT_LLM_API_KEY=sk-test\n".encode("utf-8"))
        values = _load_simple_env(env_path)
        assert values.get("HINDSIGHT_LLM_API_KEY") == "sk-test"


class TestPostSetupEnvEncoding:
    def _run_cloud_post_setup(self, tmp_path, monkeypatch):
        """Drive post_setup through the cloud path with piped stdin."""
        import io

        monkeypatch.setattr("hermes_cli.memory_setup._curses_select",
                            lambda *a, **kw: 0)  # cloud mode
        monkeypatch.setattr("hermes_cli.config.save_config", lambda c: None)
        # Skip the dependency install (now routed through lazy_deps, NS-605).
        import tools.lazy_deps as lazy_deps_mod
        monkeypatch.setattr(
            lazy_deps_mod, "install_specs",
            lambda *a, **kw: lazy_deps_mod.InstallSpecsResult(ok=True),
        )
        # First line: API key prompt (readline). Second line: API URL (input).
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-new\n\n"))

        provider = HindsightMemoryProvider()
        provider.post_setup(str(tmp_path), {"memory": {}})

    def test_bom_first_key_updated_in_place(self, tmp_path, monkeypatch):
        """The setup writer reads the existing .env BOM-tolerantly, so a
        BOM'd first key is matched and rewritten, not duplicated."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿HINDSIGHT_API_KEY=old\n".encode("utf-8"))

        self._run_cloud_post_setup(tmp_path, monkeypatch)

        content = env_path.read_text(encoding="utf-8")
        assert content.count("HINDSIGHT_API_KEY=") == 1
        assert "HINDSIGHT_API_KEY=sk-new" in content
        assert "old" not in content
        assert "﻿" not in content


class TestClientAutoUpgradeRoutesThroughLazyDeps:
    """The initialize()-time hindsight-client auto-upgrade must go through
    lazy_deps.install_specs() (environment-aware, durable-target on sealed
    hosted venvs) — never a direct `uv pip install --python sys.executable`
    subprocess, which fails with EROFS/EACCES on immutable images (NS-605)."""

    def _init_with_outdated_client(self, tmp_path, monkeypatch, outcome):
        import importlib.metadata as md
        import subprocess as subprocess_mod
        import tools.lazy_deps as lazy_deps_mod

        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"mode": "cloud"}))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        # Simulate an installed-but-outdated client.
        monkeypatch.setattr(md, "version", lambda name: "0.0.1")

        calls = []
        monkeypatch.setattr(
            lazy_deps_mod, "install_specs",
            lambda specs, **kw: calls.append(tuple(specs)) or outcome,
        )

        # Regression guard: no direct pip subprocess may run.
        def _no_subprocess(*a, **kw):  # pragma: no cover - fails loudly
            raise AssertionError(f"unexpected subprocess.run during auto-upgrade: {a}")
        monkeypatch.setattr(subprocess_mod, "run", _no_subprocess)

        provider = HindsightMemoryProvider()
        provider.initialize(session_id="s", hermes_home=str(tmp_path), platform="cli")
        return calls

    def test_upgrade_uses_install_specs_not_subprocess(self, tmp_path, monkeypatch):
        from plugins.memory.hindsight import _MIN_CLIENT_VERSION
        from tools.lazy_deps import InstallSpecsResult

        calls = self._init_with_outdated_client(
            tmp_path, monkeypatch, InstallSpecsResult(ok=True)
        )
        assert calls == [(f"hindsight-client>={_MIN_CLIENT_VERSION}",)]

    def test_blocked_upgrade_is_nonfatal_and_surfaces_reason(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging
        from tools.lazy_deps import InstallSpecsResult

        with caplog.at_level(logging.WARNING):
            calls = self._init_with_outdated_client(
                tmp_path, monkeypatch,
                InstallSpecsResult(ok=False, blocked=True,
                                   reason="runtime installs are disabled on this deployment"),
            )
        assert len(calls) == 1  # attempted exactly once, init still completed
        assert any("runtime installs are disabled" in r.getMessage()
                   for r in caplog.records)
