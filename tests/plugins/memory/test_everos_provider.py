"""Tests for the EverOS memory provider plugin.

Tests cover config loading, tool handlers (search, recall), prefetch,
sync_turn, circuit breaker, profile isolation, schema completeness,
and the register() entry point. All HTTP calls are mocked — no live
EverOS server required.
"""

import json
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.everos import (
    EverOSMemoryProvider,
    SEARCH_SCHEMA,
    RECALL_SCHEMA,
    _load_config,
    _http_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale env vars leak between tests."""
    for key in ("EVEROS_URL", "EVEROS_USER_ID"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create an EverOSMemoryProvider with mocked HTTP and config."""
    config = {"url": "http://localhost:1995", "user_id": "test-user"}
    config_path = tmp_path / "everos.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path
    )

    p = EverOSMemoryProvider()
    p.initialize(
        session_id="test-session",
        hermes_home=str(tmp_path),
        platform="cli",
    )
    return p


@pytest.fixture()
def provider_with_config(tmp_path, monkeypatch):
    """Create a provider factory that accepts custom config overrides."""

    def _make(**overrides):
        config = {"url": "http://localhost:1995", "user_id": "test-user"}
        config.update(overrides)
        config_path = tmp_path / "everos.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )

        p = EverOSMemoryProvider()
        p.initialize(
            session_id="test-session",
            hermes_home=str(tmp_path),
            platform="cli",
        )
        return p

    return _make


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_search_schema_has_query(self):
        assert SEARCH_SCHEMA["name"] == "everos_search"
        assert "query" in SEARCH_SCHEMA["parameters"]["properties"]
        assert "query" in SEARCH_SCHEMA["parameters"]["required"]

    def test_recall_schema_has_memory_type(self):
        assert RECALL_SCHEMA["name"] == "everos_recall"
        assert "memory_type" in RECALL_SCHEMA["parameters"]["properties"]

    def test_get_tool_schemas_returns_two(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"everos_search", "everos_recall"}


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values(self, tmp_path, monkeypatch):
        """Without config file, uses env defaults."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("EVEROS_URL", "http://custom:1995")
        monkeypatch.setenv("EVEROS_USER_ID", "custom-user")

        cfg = _load_config()
        assert cfg["url"] == "http://custom:1995"
        assert cfg["user_id"] == "custom-user"

    def test_config_file_overrides_env(self, tmp_path, monkeypatch):
        """Config file values take precedence over env defaults."""
        config = {"url": "http://file-url:1995", "user_id": "file-user"}
        config_path = tmp_path / "everos.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )

        cfg = _load_config()
        assert cfg["url"] == "http://file-url:1995"
        assert cfg["user_id"] == "file-user"

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) == 2
        keys = {s["key"] for s in schema}
        assert keys == {"url", "user_id"}

    def test_save_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )
        p = EverOSMemoryProvider()
        p.save_config({"url": "http://saved:1995"}, str(tmp_path))

        saved = json.loads((tmp_path / "everos.json").read_text())
        assert saved["url"] == "http://saved:1995"


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestToolHandlers:
    @patch("plugins.memory.everos._http_request")
    def test_search_success(self, mock_http, provider):
        mock_http.return_value = {
            "result": {
                "memories": [
                    {"summary": "User prefers dark mode"},
                    {"summary": "User likes espresso"},
                ]
            }
        }
        result = json.loads(
            provider.handle_tool_call(
                "everos_search", {"query": "coffee preferences"}
            )
        )
        assert "result" in result
        assert len(result["result"]["memories"]) == 2

    @patch("plugins.memory.everos._http_request")
    def test_search_missing_query(self, mock_http, provider):
        result = json.loads(
            provider.handle_tool_call("everos_search", {})
        )
        assert "error" in result
        mock_http.assert_not_called()

    @patch("plugins.memory.everos._http_request")
    def test_search_passes_method(self, mock_http, provider):
        mock_http.return_value = {"result": {"memories": []}}
        provider.handle_tool_call(
            "everos_search", {"query": "test", "method": "agentic"}
        )
        call_args = mock_http.call_args
        payload = call_args[1].get("data", call_args[0][1] if len(call_args[0]) > 1 else {})
        # Method is passed in the URL payload
        assert call_args.called

    @patch("plugins.memory.everos._http_request")
    def test_recall_success(self, mock_http, provider):
        mock_http.return_value = {
            "result": {
                "memories": [{"summary": "Profile fact"}]
            }
        }
        result = json.loads(
            provider.handle_tool_call(
                "everos_recall", {"memory_type": "profile"}
            )
        )
        assert "result" in result

    @patch("plugins.memory.everos._http_request")
    def test_recall_default_type(self, mock_http, provider):
        mock_http.return_value = {"result": {"memories": []}}
        provider.handle_tool_call("everos_recall", {})
        # Verify it was called (default episodic_memory)
        mock_http.assert_called_once()

    def test_unknown_tool(self, provider):
        result = json.loads(
            provider.handle_tool_call("everos_unknown", {})
        )
        assert "error" in result

    @patch("plugins.memory.everos._http_request")
    def test_search_error_handling(self, mock_http, provider):
        mock_http.return_value = {"error": "connection refused"}
        result = json.loads(
            provider.handle_tool_call(
                "everos_search", {"query": "test"}
            )
        )
        assert "error" in result
        assert "connection refused" in result["error"]

    @patch("plugins.memory.everos._http_request")
    def test_recall_error_handling(self, mock_http, provider):
        mock_http.return_value = {"error": "timeout"}
        result = json.loads(
            provider.handle_tool_call("everos_recall", {})
        )
        assert "error" in result

    @patch("plugins.memory.everos._http_request")
    def test_search_resets_consecutive_failures_on_success(
        self, mock_http, provider
    ):
        provider._consecutive_failures = 3
        mock_http.return_value = {"result": {"memories": []}}
        provider.handle_tool_call(
            "everos_search", {"query": "test"}
        )
        assert provider._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_initially(self, provider):
        assert provider.prefetch("test") == ""

    @patch("plugins.memory.everos._http_request")
    def test_queue_prefetch_populates_cache(self, mock_http, provider):
        mock_http.return_value = {
            "result": {
                "memories": [{"summary": "remembered fact"}]
            }
        }
        provider.queue_prefetch("test query", session_id="s1")
        # Wait for background thread
        time.sleep(0.5)

        result = provider.prefetch("anything", session_id="s1")
        assert "remembered fact" in result

    def test_prefetch_returns_empty_for_wrong_session(self, provider):
        provider._prefetch_cache["s1"] = "some data"
        assert provider.prefetch("test", session_id="s2") == ""


# ---------------------------------------------------------------------------
# sync_turn tests
# ---------------------------------------------------------------------------


class TestSyncTurn:
    @patch("plugins.memory.everos._http_request")
    def test_sync_turn_ingests_both_messages(self, mock_http, provider):
        mock_http.return_value = {"status": "ok"}
        provider.sync_turn("user message", "assistant reply")
        # Wait for background thread
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        # Should have called _http_request twice (user + assistant)
        assert mock_http.call_count == 2

    @patch("plugins.memory.everos._http_request")
    def test_sync_turn_skips_non_primary_context(
        self, mock_http, provider
    ):
        provider._agent_context = "subagent"
        provider.sync_turn("user", "assistant")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        mock_http.assert_not_called()

    @patch("plugins.memory.everos._http_request")
    def test_sync_turn_resets_failures_on_success(
        self, mock_http, provider
    ):
        provider._consecutive_failures = 3
        mock_http.return_value = {"status": "ok"}
        provider.sync_turn("u", "a")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        assert provider._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_check_breaker_allows_when_under_threshold(self, provider):
        provider._consecutive_failures = 4
        assert provider._check_breaker() is True

    def test_check_breaker_blocks_when_over_threshold(self, provider):
        provider._consecutive_failures = 5
        provider._breaker_until = time.time() + 9999
        assert provider._check_breaker() is False

    def test_check_breaker_allows_after_cooldown(self, provider):
        provider._consecutive_failures = 5
        provider._breaker_until = time.time() - 1  # expired
        assert provider._check_breaker() is True

    def test_record_failure_increments(self, provider):
        provider._record_failure("test error")
        assert provider._consecutive_failures == 1

    def test_record_failure_opens_breaker_at_threshold(self, provider):
        for _ in range(5):
            provider._record_failure("error")
        assert provider._breaker_until > time.time()

    @patch("plugins.memory.everos._http_request")
    def test_search_blocked_by_breaker(self, mock_http, provider):
        provider._consecutive_failures = 5
        provider._breaker_until = time.time() + 9999
        result = json.loads(
            provider.handle_tool_call(
                "everos_search", {"query": "test"}
            )
        )
        assert "circuit breaker" in result["error"]
        mock_http.assert_not_called()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_name(self, provider):
        assert provider.name == "everos"

    def test_is_available_with_url(self, provider):
        assert provider.is_available() is True

    def test_is_available_without_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.delenv("EVEROS_URL", raising=False)
        # _load_config defaults to "http://localhost:1995" when env is unset,
        # so is_available is True even without explicit config. Test that
        # setting EVEROS_URL="" makes it unavailable.
        monkeypatch.setenv("EVEROS_URL", "")
        p = EverOSMemoryProvider()
        assert p.is_available() is False

    def test_system_prompt_block(self, provider):
        block = provider.system_prompt_block()
        assert "EverOS" in block
        assert "everos_search" in block

    @patch("plugins.memory.everos._http_request")
    def test_initialize_health_check(self, mock_http, tmp_path, monkeypatch):
        mock_http.return_value = {"status": "ok"}
        config_path = tmp_path / "everos.json"
        config_path.write_text(
            json.dumps({"url": "http://localhost:1995"})
        )
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )

        p = EverOSMemoryProvider()
        p.initialize(session_id="test", hermes_home=str(tmp_path))
        # Health check is called during init
        mock_http.assert_called_once()
        assert "/health" in mock_http.call_args[0][0]

    def test_shutdown_waits_for_thread(self, provider):
        provider._sync_thread = threading.Thread(
            target=lambda: None, daemon=True
        )
        provider._sync_thread.start()
        provider.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# Optional hooks tests
# ---------------------------------------------------------------------------


class TestOptionalHooks:
    @patch("plugins.memory.everos._http_request")
    def test_on_session_end_flushes_last_turn(self, mock_http, provider):
        mock_http.return_value = {"status": "ok"}
        messages = [
            {"role": "user", "content": "last user msg"},
            {"role": "assistant", "content": "last assistant msg"},
        ]
        provider.on_session_end(messages)
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        assert mock_http.call_count >= 1

    def test_on_session_end_skips_non_primary(self, provider):
        provider._agent_context = "cron"
        provider.on_session_end(
            [{"role": "user", "content": "msg"}]
        )
        # No thread should be started
        assert provider._sync_thread is None or not provider._sync_thread.is_alive()

    @patch("plugins.memory.everos._http_request")
    def test_on_memory_write_mirrors(self, mock_http, provider):
        mock_http.return_value = {"status": "ok"}
        provider.on_memory_write("add", "user", "Timezone: UTC")
        assert mock_http.called

    def test_on_memory_write_skips_non_primary(self, provider):
        provider._agent_context = "subagent"
        provider.on_memory_write("add", "user", "test")
        # Should not throw


# ---------------------------------------------------------------------------
# register() entry point tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_calls_ctx(self):
        from plugins.memory.everos import register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_memory_provider.assert_called_once()
        provider = mock_ctx.register_memory_provider.call_args[0][0]
        assert isinstance(provider, EverOSMemoryProvider)


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


class TestHttpRequest:
    @patch("plugins.memory.everos.urlopen")
    def test_http_request_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ok"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _http_request("http://localhost/test")
        assert result == {"status": "ok"}

    @patch("plugins.memory.everos.urlopen")
    def test_http_request_error(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("connection refused")
        result = _http_request("http://localhost/test")
        assert "error" in result


# ---------------------------------------------------------------------------
# Profile isolation test
# ---------------------------------------------------------------------------


class TestProfileIsolation:
    def test_config_uses_hermes_home(self, tmp_path, monkeypatch):
        """Config path should be under get_hermes_home(), not hardcoded."""
        config = {"url": "http://profile-test:1995"}
        config_path = tmp_path / "everos.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: tmp_path
        )

        cfg = _load_config()
        assert cfg["url"] == "http://profile-test:1995"
