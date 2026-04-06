"""Tests for the Pallium memory plugin.

Covers: config loading, is_available, initialize, tool schemas, tool routing,
sync_turn payloads, on_memory_write mirroring, circuit breaker, prefetch threading.

No Pallium server required — all HTTP calls are mocked.
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import sys
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.pallium import PalliumMemoryProvider, _load_config, register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME per test."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture
def provider():
    return PalliumMemoryProvider()


@pytest.fixture
def initialized_provider(provider, tmp_path):
    hermes_home = tmp_path / ".hermes"
    provider.initialize("test-session-1", hermes_home=str(hermes_home), platform="cli")
    return provider


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:

    def test_defaults_when_no_config_file(self, tmp_path):
        cfg = _load_config(str(tmp_path / "nonexistent"))
        assert cfg["base_url"] == "http://localhost:8000"
        assert cfg["actor_ref"] == "hermes-user"
        assert cfg["container_ref"] == "hermes"

    def test_loads_from_json_file(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text(json.dumps({
            "base_url": "http://myhost:9000",
            "actor_ref": "alice",
            "container_ref": "myproject",
        }))
        cfg = _load_config(str(hermes_home))
        assert cfg["base_url"] == "http://myhost:9000"
        assert cfg["actor_ref"] == "alice"
        assert cfg["container_ref"] == "myproject"

    def test_partial_override_keeps_defaults(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text(json.dumps({"actor_ref": "bob"}))
        cfg = _load_config(str(hermes_home))
        assert cfg["actor_ref"] == "bob"
        assert cfg["base_url"] == "http://localhost:8000"  # default preserved

    def test_malformed_json_falls_back_to_defaults(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text("{bad json}")
        cfg = _load_config(str(hermes_home))
        assert cfg["base_url"] == "http://localhost:8000"


# ---------------------------------------------------------------------------
# Availability and config schema
# ---------------------------------------------------------------------------

class TestAvailability:

    def test_is_available_true_by_default(self, provider):
        # Default config always has a base_url
        assert provider.is_available() is True

    def test_get_config_schema_fields(self, provider):
        schema = provider.get_config_schema()
        keys = [f["key"] for f in schema]
        assert "base_url" in keys
        assert "actor_ref" in keys
        assert "container_ref" in keys

    def test_no_required_fields(self, provider):
        schema = provider.get_config_schema()
        required = [f for f in schema if f.get("required")]
        assert required == []

    def test_save_config_writes_json(self, provider, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        provider.save_config({"base_url": "http://newhost:8888"}, str(hermes_home))
        saved = json.loads((hermes_home / "pallium.json").read_text())
        assert saved["base_url"] == "http://newhost:8888"

    def test_save_config_merges_existing(self, provider, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text(json.dumps({"actor_ref": "carol"}))
        provider.save_config({"base_url": "http://newhost:8888"}, str(hermes_home))
        saved = json.loads((hermes_home / "pallium.json").read_text())
        assert saved["actor_ref"] == "carol"
        assert saved["base_url"] == "http://newhost:8888"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_initialize_sets_fields(self, provider, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text(json.dumps({
            "base_url": "http://pallium:8000",
            "actor_ref": "dave",
            "container_ref": "project-x",
        }))
        provider.initialize("sess-42", hermes_home=str(hermes_home), platform="cli")
        assert provider._base_url == "http://pallium:8000"
        assert provider._actor_ref == "dave"
        assert provider._container_ref == "project-x"
        assert provider._session_id == "sess-42"

    def test_initialize_strips_trailing_slash(self, provider, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)
        (hermes_home / "pallium.json").write_text(json.dumps({"base_url": "http://pallium:8000/"}))
        provider.initialize("s", hermes_home=str(hermes_home))
        assert provider._base_url == "http://pallium:8000"

    def test_system_prompt_block_mentions_container(self, initialized_provider):
        block = initialized_provider.system_prompt_block()
        assert "Pallium" in block
        assert "hermes" in block  # default container_ref


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TestTools:

    def test_tool_schemas_returned(self, provider):
        schemas = provider.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert "pallium_query" in names
        assert "pallium_remember" in names

    def test_pallium_query_has_required_query_param(self, provider):
        schema = next(s for s in provider.get_tool_schemas() if s["name"] == "pallium_query")
        assert "query" in schema["parameters"]["required"]

    def test_pallium_remember_has_required_content_param(self, provider):
        schema = next(s for s in provider.get_tool_schemas() if s["name"] == "pallium_remember")
        assert "content" in schema["parameters"]["required"]

    def test_handle_tool_call_query_missing_param(self, initialized_provider):
        result = json.loads(initialized_provider.handle_tool_call("pallium_query", {}))
        assert "error" in result

    def test_handle_tool_call_remember_missing_param(self, initialized_provider):
        result = json.loads(initialized_provider.handle_tool_call("pallium_remember", {}))
        assert "error" in result

    def test_handle_unknown_tool(self, initialized_provider):
        result = json.loads(initialized_provider.handle_tool_call("unknown_tool", {}))
        assert "error" in result

    def test_handle_tool_call_query_posts_correct_payload(self, initialized_provider):
        mock_response = {
            "injectable_blocks": [
                {"title": "decision", "text": "We chose PostgreSQL.", "memory_type": "decision"}
            ]
        }
        with patch("plugins.memory.pallium._http_post", return_value=mock_response) as mock_post:
            result = json.loads(initialized_provider.handle_tool_call(
                "pallium_query", {"query": "why postgresql?"}
            ))

        mock_post.assert_called_once()
        _, payload = mock_post.call_args[0]
        assert payload["text"] == "why postgresql?"
        assert payload["container_ref"] == "hermes"
        assert "results" in result
        assert result["results"][0]["text"] == "We chose PostgreSQL."

    def test_handle_tool_call_remember_posts_item(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post", return_value={}) as mock_post:
            result = json.loads(initialized_provider.handle_tool_call(
                "pallium_remember", {"content": "Use UTC for all timestamps", "kind": "decision"}
            ))

        mock_post.assert_called_once()
        _, payload = mock_post.call_args[0]
        # /items receives a list; the item is the first element
        item = payload[0]
        assert item["content"] == "Use UTC for all timestamps"
        assert item["role"] == "assistant"  # decision → assistant
        assert item["container_ref"] == "hermes"
        assert result["result"] == "Stored."

    def test_remember_note_uses_user_role(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post", return_value={}) as mock_post:
            initialized_provider.handle_tool_call(
                "pallium_remember", {"content": "Just a note", "kind": "note"}
            )
        _, payload = mock_post.call_args[0]
        assert payload[0]["role"] == "user"

    def test_query_no_blocks_returns_no_results_message(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post", return_value={"injectable_blocks": []}):
            result = json.loads(initialized_provider.handle_tool_call(
                "pallium_query", {"query": "anything"}
            ))
        assert "No relevant memories" in result["result"]


# ---------------------------------------------------------------------------
# sync_turn
# ---------------------------------------------------------------------------

class TestSyncTurn:

    def test_sync_turn_posts_both_messages(self, initialized_provider):
        posted = []

        def fake_post(url, payload, **kwargs):
            posted.append(payload)
            return {}

        with patch("plugins.memory.pallium._http_post", side_effect=fake_post):
            initialized_provider.sync_turn("Hello", "Hi there")
            # Wait for background thread
            if initialized_provider._sync_thread:
                initialized_provider._sync_thread.join(timeout=5.0)

        # One call to /items with a list of 2 items
        assert len(posted) == 1
        items = posted[0]
        assert len(items) == 2
        roles = {p["role"] for p in items}
        assert roles == {"user", "assistant"}
        contents = {p["content"] for p in items}
        assert "Hello" in contents
        assert "Hi there" in contents

    def test_sync_turn_includes_session_and_turn_in_source_id(self, initialized_provider):
        posted = []

        def fake_post(url, payload, **kwargs):
            posted.append(payload)
            return {}

        with patch("plugins.memory.pallium._http_post", side_effect=fake_post):
            initialized_provider.sync_turn("msg1", "reply1")
            if initialized_provider._sync_thread:
                initialized_provider._sync_thread.join(timeout=5.0)

        items = posted[0]
        source_ids = {p["source_id"] for p in items}
        # All source_ids contain session id and turn count
        for sid in source_ids:
            assert "test-session-1" in sid
            assert "1" in sid  # turn 1

    def test_sync_turn_increments_turn_count(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post", return_value={}):
            initialized_provider.sync_turn("a", "b")
            if initialized_provider._sync_thread:
                initialized_provider._sync_thread.join(timeout=5.0)
            initialized_provider.sync_turn("c", "d")
            if initialized_provider._sync_thread:
                initialized_provider._sync_thread.join(timeout=5.0)
        assert initialized_provider._turn_count == 2

    def test_sync_turn_is_non_blocking(self, initialized_provider):
        """sync_turn should return immediately, not block on HTTP."""
        slow_event = threading.Event()

        def slow_post(url, payload, **kwargs):
            slow_event.wait(timeout=10)
            return {}

        start = time.monotonic()
        with patch("plugins.memory.pallium._http_post", side_effect=slow_post):
            initialized_provider.sync_turn("test", "test")
        elapsed = time.monotonic() - start

        assert elapsed < 0.5  # returned immediately
        slow_event.set()  # unblock background thread
        if initialized_provider._sync_thread:
            initialized_provider._sync_thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# on_memory_write
# ---------------------------------------------------------------------------

class TestOnMemoryWrite:

    def test_mirrors_add_action(self, initialized_provider):
        posted = []

        def fake_post(url, payload, **kwargs):
            posted.append(payload)
            return {}

        with patch("plugins.memory.pallium._http_post", side_effect=fake_post):
            initialized_provider.on_memory_write("add", "memory", "Important fact")
            # Find and join the thread
            threads = [t for t in threading.enumerate() if "pallium-memwrite" in t.name]
            for t in threads:
                t.join(timeout=5.0)

        assert len(posted) == 1
        item = posted[0][0]  # /items receives a list; unwrap
        assert item["content"] == "Important fact"
        assert item["source_type"] == "hermes_agent-memory"

    def test_mirrors_user_profile_writes(self, initialized_provider):
        posted = []

        def fake_post(url, payload, **kwargs):
            posted.append(payload)
            return {}

        with patch("plugins.memory.pallium._http_post", side_effect=fake_post):
            initialized_provider.on_memory_write("add", "user", "User name: Alice")
            threads = [t for t in threading.enumerate() if "pallium-memwrite" in t.name]
            for t in threads:
                t.join(timeout=5.0)

        item = posted[0][0]
        assert item["source_type"] == "hermes_user-profile"
        assert item["role"] == "user"

    def test_ignores_remove_action(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post") as mock_post:
            initialized_provider.on_memory_write("remove", "memory", "something")
            time.sleep(0.05)
        mock_post.assert_not_called()

    def test_ignores_empty_content(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post") as mock_post:
            initialized_provider.on_memory_write("add", "memory", "")
            time.sleep(0.05)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_breaker_opens_after_threshold_failures(self, initialized_provider):
        assert not initialized_provider._is_breaker_open()
        for _ in range(5):
            initialized_provider._record_failure()
        assert initialized_provider._is_breaker_open()

    def test_breaker_resets_after_success(self, initialized_provider):
        for _ in range(5):
            initialized_provider._record_failure()
        assert initialized_provider._is_breaker_open()
        initialized_provider._record_success()
        assert initialized_provider._consecutive_failures == 0

    def test_breaker_blocks_tool_calls(self, initialized_provider):
        for _ in range(5):
            initialized_provider._record_failure()

        with patch("plugins.memory.pallium._http_post") as mock_post:
            result = json.loads(initialized_provider.handle_tool_call(
                "pallium_query", {"query": "test"}
            ))
        mock_post.assert_not_called()
        assert "unavailable" in result["error"].lower()

    def test_breaker_blocks_prefetch(self, initialized_provider):
        for _ in range(5):
            initialized_provider._record_failure()

        with patch("plugins.memory.pallium._http_post") as mock_post:
            initialized_provider.queue_prefetch("test query")
            time.sleep(0.05)
        mock_post.assert_not_called()

    def test_breaker_cooldown_resets(self, initialized_provider):
        """After cooldown expires, breaker resets on next check."""
        for _ in range(5):
            initialized_provider._record_failure()
        # Backdate the cooldown so it's already expired
        initialized_provider._breaker_open_until = time.monotonic() - 1
        assert not initialized_provider._is_breaker_open()
        assert initialized_provider._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

class TestPrefetch:

    def test_prefetch_returns_empty_when_no_result(self, initialized_provider):
        result = initialized_provider.prefetch("anything")
        assert result == ""

    def test_queue_prefetch_then_prefetch(self, initialized_provider):
        mock_response = {
            "injectable_blocks": [
                {"title": "decision", "text": "Always use UTC."}
            ]
        }
        with patch("plugins.memory.pallium._http_post", return_value=mock_response):
            initialized_provider.queue_prefetch("timezone handling")
            if initialized_provider._prefetch_thread:
                initialized_provider._prefetch_thread.join(timeout=5.0)

        result = initialized_provider.prefetch("timezone handling")
        assert "Pallium Memory" in result
        assert "Always use UTC" in result

    def test_prefetch_clears_cache_after_read(self, initialized_provider):
        with initialized_provider._prefetch_lock:
            initialized_provider._prefetch_result = "cached content"

        initialized_provider.prefetch("query")
        with initialized_provider._prefetch_lock:
            assert initialized_provider._prefetch_result == ""


# ---------------------------------------------------------------------------
# on_pre_compress
# ---------------------------------------------------------------------------

class TestOnPreCompress:

    def test_returns_empty_string(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post", return_value={}):
            result = initialized_provider.on_pre_compress([
                {"role": "user", "content": "old message"},
                {"role": "assistant", "content": "old reply"},
            ])
        assert result == ""

    def test_empty_messages_does_nothing(self, initialized_provider):
        with patch("plugins.memory.pallium._http_post") as mock_post:
            initialized_provider.on_pre_compress([])
            time.sleep(0.05)
        mock_post.assert_not_called()

    def test_ingests_user_and_assistant_messages(self, initialized_provider):
        posted = []

        def fake_post(url, payload, **kwargs):
            posted.append(payload)
            return {}

        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "tool", "content": "tool output"},  # should be skipped
        ]

        with patch("plugins.memory.pallium._http_post", side_effect=fake_post):
            initialized_provider.on_pre_compress(messages)
            threads = [t for t in threading.enumerate() if "pallium-compress" in t.name]
            for t in threads:
                t.join(timeout=5.0)

        # Each accepted message is posted as a single-item list
        roles = {p[0]["role"] for p in posted}
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" not in roles


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_shutdown_joins_threads(self, initialized_provider):
        """shutdown() joins background threads without error."""
        with patch("plugins.memory.pallium._http_post", return_value={}):
            initialized_provider.sync_turn("a", "b")
        initialized_provider.shutdown()  # should not raise or hang


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class TestPluginRegistration:

    def test_register_function_exists(self):
        assert callable(register)

    def test_register_adds_provider(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_memory_provider.assert_called_once()
        provider = ctx.register_memory_provider.call_args[0][0]
        assert provider.name == "pallium"
        assert isinstance(provider, PalliumMemoryProvider)

    def test_provider_name(self):
        p = PalliumMemoryProvider()
        assert p.name == "pallium"

    def test_integrates_with_memory_manager(self):
        """Plugin wires correctly into MemoryManager."""
        from agent.memory_manager import MemoryManager
        from agent.builtin_memory_provider import BuiltinMemoryProvider

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        mgr.add_provider(PalliumMemoryProvider())

        assert "pallium" in mgr.provider_names
        assert mgr.has_tool("pallium_query")
        assert mgr.has_tool("pallium_remember")
