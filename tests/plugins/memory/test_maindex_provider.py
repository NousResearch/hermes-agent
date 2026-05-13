"""MaindexMemoryProvider: lifecycle hooks, tool dispatch, circuit breaker, config."""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.maindex import (
    MaindexMemoryProvider,
    MaindexClient,
    _load_config,
    register,
    SEARCH_SCHEMA,
    KEEP_SCHEMA,
    RECALL_SCHEMA,
    UPDATE_SCHEMA,
    FORGET_SCHEMA,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _event_side_effect(return_value=None):
    """Return (event, side_effect) for deterministic background-thread tests.

    The side_effect sets the event when called, so the test can wait on
    the event instead of sleeping.
    """
    event = threading.Event()

    def _fn(*args, **kwargs):
        event.set()
        return return_value if return_value is not None else {}

    return event, _fn


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def provider():
    """Fresh MaindexMemoryProvider with a mock client pre-injected.

    Returns (provider, mock_client) so tests can set up return values
    on mock_client methods and assert calls.
    """
    p = MaindexMemoryProvider()
    mock_client = MagicMock()
    p._client = mock_client
    p._session_id = "sess-001"
    p._config = {"api_key": "k", "token": "", "collection": ""}
    return p, mock_client


# ── Availability ─────────────────────────────────────────────────────────


class TestAvailability:

    def test_available_with_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "k")
        monkeypatch.delenv("MAINDEX_TOKEN", raising=False)
        assert MaindexMemoryProvider().is_available() is True

    def test_available_with_token(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.delenv("MAINDEX_API_KEY", raising=False)
        monkeypatch.setenv("MAINDEX_TOKEN", "t")
        assert MaindexMemoryProvider().is_available() is True

    def test_unavailable_without_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.delenv("MAINDEX_API_KEY", raising=False)
        monkeypatch.delenv("MAINDEX_TOKEN", raising=False)
        assert MaindexMemoryProvider().is_available() is False


# ── _load_config ─────────────────────────────────────────────────────────


class TestLoadConfig:

    def test_reads_all_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "env-key")
        monkeypatch.setenv("MAINDEX_TOKEN", "env-tok")
        monkeypatch.setenv("MAINDEX_COLLECTION", "env-col")
        cfg = _load_config()
        assert cfg == {"api_key": "env-key", "token": "env-tok", "collection": "env-col"}

    def test_file_values_override_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "env-key")
        monkeypatch.delenv("MAINDEX_TOKEN", raising=False)
        monkeypatch.delenv("MAINDEX_COLLECTION", raising=False)
        (tmp_path / "maindex.json").write_text(
            json.dumps({"api_key": "file-key", "collection": "file-col"})
        )
        cfg = _load_config()
        assert cfg["api_key"] == "file-key"
        assert cfg["collection"] == "file-col"

    def test_empty_file_values_do_not_clobber_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "env-key")
        monkeypatch.delenv("MAINDEX_TOKEN", raising=False)
        monkeypatch.delenv("MAINDEX_COLLECTION", raising=False)
        (tmp_path / "maindex.json").write_text(json.dumps({"api_key": ""}))
        cfg = _load_config()
        assert cfg["api_key"] == "env-key"

    def test_corrupt_json_file_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "env-key")
        monkeypatch.delenv("MAINDEX_TOKEN", raising=False)
        monkeypatch.delenv("MAINDEX_COLLECTION", raising=False)
        (tmp_path / "maindex.json").write_text("NOT VALID JSON{{{")
        cfg = _load_config()
        assert cfg["api_key"] == "env-key"


# ── save_config ──────────────────────────────────────────────────────────


class TestSaveConfig:

    def test_writes_json_file(self, tmp_path):
        p = MaindexMemoryProvider()
        p.save_config({"api_key": "new-key"}, str(tmp_path))
        data = json.loads((tmp_path / "maindex.json").read_text())
        assert data["api_key"] == "new-key"

    def test_merges_with_existing_file(self, tmp_path):
        (tmp_path / "maindex.json").write_text(
            json.dumps({"api_key": "old", "collection": "kept"})
        )
        p = MaindexMemoryProvider()
        p.save_config({"api_key": "new"}, str(tmp_path))
        data = json.loads((tmp_path / "maindex.json").read_text())
        assert data["api_key"] == "new"
        assert data["collection"] == "kept"


# ── Initialize & shutdown ────────────────────────────────────────────────


class TestInitialize:

    def test_wires_config_to_client(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys.modules["hermes_constants"], "get_hermes_home", lambda: tmp_path,
        )
        monkeypatch.setenv("MAINDEX_API_KEY", "init-key")
        monkeypatch.setenv("MAINDEX_TOKEN", "init-tok")
        monkeypatch.setenv("MAINDEX_COLLECTION", "init-col")

        p = MaindexMemoryProvider()
        p.initialize("sess-x")

        assert p._session_id == "sess-x"
        assert p._collection == "init-col"
        assert p._client is not None
        assert p._client._client.headers["x-api-key"] == "init-key"
        assert p._client._client.headers["authorization"] == "Bearer init-tok"
        p.shutdown()


class TestShutdown:

    def test_closes_client_and_nullifies(self, provider):
        p, mock_client = provider
        p.shutdown()
        mock_client.close.assert_called_once()
        assert p._client is None


# ── System prompt ────────────────────────────────────────────────────────


class TestSystemPrompt:

    def test_includes_provider_name_and_tool_hints(self, provider):
        p, _ = provider
        block = p.system_prompt_block()
        assert "Maindex Memory" in block
        assert "maindex_search" in block

    def test_includes_collection_when_set(self, provider):
        p, _ = provider
        p._collection = "my-project"
        assert "my-project" in p.system_prompt_block()

    def test_omits_collection_line_when_empty(self, provider):
        p, _ = provider
        p._collection = ""
        assert "Default collection" not in p.system_prompt_block()


# ── Tool schemas ─────────────────────────────────────────────────────────


class TestToolSchemas:

    def test_returns_exactly_five_schemas(self, provider):
        p, _ = provider
        assert len(p.get_tool_schemas()) == 5

    def test_schema_names_match_expected_set(self, provider):
        p, _ = provider
        names = {s["name"] for s in p.get_tool_schemas()}
        assert names == {
            "maindex_search", "maindex_keep", "maindex_recall",
            "maindex_update", "maindex_forget",
        }

    def test_update_schema_exposes_metadata_fields(self, provider):
        p, _ = provider
        update = next(s for s in p.get_tool_schemas() if s["name"] == "maindex_update")
        props = set(update["parameters"]["properties"])
        for field in ("canon_status", "kind", "confidence", "verification_status"):
            assert field in props, f"{field} missing from update schema"


# ── Tool routing ─────────────────────────────────────────────────────────


class TestToolRouting:

    def test_dispatches_each_tool_to_correct_handler(self, provider):
        p, mock = provider
        mock.search.return_value = {"items": []}
        mock.keep.return_value = {"id": "x", "shortId": "mem-1"}
        mock.recall.return_value = {
            "id": "x", "shortId": "mem-1", "headline": "H",
            "body": "", "tags": [], "kind": "note", "canonStatus": "",
            "collections": [], "createdAt": "", "updatedAt": "",
        }
        mock.update.return_value = {"id": "x", "shortId": "mem-1"}
        mock.forget.return_value = {"ok": True}

        p.handle_tool_call("maindex_search", {"query": "test"})
        mock.search.assert_called_once()

        p.handle_tool_call("maindex_keep", {"headline": "H"})
        mock.keep.assert_called_once()

        p.handle_tool_call("maindex_recall", {"memory_id": "mem-1"})
        mock.recall.assert_called_once()

        p.handle_tool_call("maindex_update", {"memory_id": "m", "mode": "body_replace"})
        mock.update.assert_called_once()

        p.handle_tool_call("maindex_forget", {"memory_id": "m"})
        mock.forget.assert_called_once()

    def test_unknown_tool_returns_error(self, provider):
        p, _ = provider
        result = json.loads(p.handle_tool_call("maindex_nope", {}))
        assert "Unknown tool" in result["error"]

    def test_no_client_returns_error(self):
        p = MaindexMemoryProvider()
        result = json.loads(p.handle_tool_call("maindex_search", {"query": "x"}))
        assert "error" in result

    def test_breaker_open_returns_unavailable(self, provider):
        p, _ = provider
        for _ in range(5):
            p._record_failure()
        result = json.loads(p.handle_tool_call("maindex_search", {"query": "x"}))
        assert "temporarily unavailable" in result["error"]


# ── Circuit breaker ──────────────────────────────────────────────────────


class TestCircuitBreaker:

    def test_stays_closed_under_threshold(self, provider):
        p, _ = provider
        for _ in range(4):
            p._record_failure()
        assert p._is_breaker_open() is False

    def test_trips_at_threshold(self, provider):
        p, _ = provider
        for _ in range(5):
            p._record_failure()
        assert p._is_breaker_open() is True
        assert p._consecutive_failures == 5

    def test_resets_after_cooldown_expires(self, provider):
        p, _ = provider
        for _ in range(5):
            p._record_failure()
        future = p._breaker_open_until + 1
        with patch.object(time, "monotonic", return_value=future):
            assert p._is_breaker_open() is False
        assert p._consecutive_failures == 0

    def test_success_resets_failure_counter(self, provider):
        p, _ = provider
        for _ in range(3):
            p._record_failure()
        assert p._consecutive_failures == 3
        p._record_success()
        assert p._consecutive_failures == 0

    def test_tool_error_increments_failure_counter(self, provider):
        p, mock = provider
        mock.search.side_effect = Exception("API down")
        p.handle_tool_call("maindex_search", {"query": "test"})
        assert p._consecutive_failures == 1


# ── Tool: search ─────────────────────────────────────────────────────────


class TestToolSearch:

    def test_requires_query(self, provider):
        p, _ = provider
        result = json.loads(p._tool_search({}))
        assert "error" in result

    def test_formats_results_with_short_id(self, provider):
        p, mock = provider
        mock.search.return_value = {
            "items": [{
                "shortId": "mem-1a", "headline": "Auth design",
                "body": "OAuth2 with PKCE", "tags": ["domain:auth"],
                "score": 0.92,
            }],
        }
        result = json.loads(p._tool_search({"query": "auth"}))
        assert result["count"] == 1
        r = result["results"][0]
        assert r["id"] == "mem-1a"
        assert r["headline"] == "Auth design"
        assert r["body"] == "OAuth2 with PKCE"
        assert r["tags"] == ["domain:auth"]
        assert r["score"] == 0.92

    def test_truncates_body_at_500_chars(self, provider):
        p, mock = provider
        mock.search.return_value = {
            "items": [{"shortId": "m", "headline": "H", "body": "B" * 1000}],
        }
        result = json.loads(p._tool_search({"query": "test"}))
        assert len(result["results"][0]["body"]) == 500

    def test_returns_no_results_message(self, provider):
        p, mock = provider
        mock.search.return_value = {"items": []}
        result = json.loads(p._tool_search({"query": "nonexistent"}))
        assert "No relevant memories" in result["result"]

    def test_applies_collection_fallback(self, provider):
        p, mock = provider
        p._collection = "project-x"
        mock.search.return_value = {"items": []}
        p._tool_search({"query": "test"})
        _, kwargs = mock.search.call_args
        assert kwargs["collection"] == "project-x"

    def test_explicit_collection_overrides_fallback(self, provider):
        p, mock = provider
        p._collection = "project-x"
        mock.search.return_value = {"items": []}
        p._tool_search({"query": "test", "collection": "other"})
        _, kwargs = mock.search.call_args
        assert kwargs["collection"] == "other"

    def test_caps_limit_at_50(self, provider):
        p, mock = provider
        mock.search.return_value = {"items": []}
        p._tool_search({"query": "test", "limit": 200})
        _, kwargs = mock.search.call_args
        assert kwargs["limit"] == 50

    def test_records_failure_on_api_exception(self, provider):
        p, mock = provider
        mock.search.side_effect = Exception("timeout")
        p._tool_search({"query": "test"})
        assert p._consecutive_failures == 1


# ── Tool: keep ───────────────────────────────────────────────────────────


class TestToolKeep:

    def test_requires_headline(self, provider):
        p, _ = provider
        result = json.loads(p._tool_keep({}))
        assert "error" in result

    def test_attaches_session_and_source(self, provider):
        p, mock = provider
        mock.keep.return_value = {"id": "uuid", "shortId": "mem-1a", "headline": "H"}
        result = json.loads(p._tool_keep({"headline": "Important fact"}))
        assert result["result"] == "Memory stored."
        assert result["id"] == "mem-1a"
        _, kwargs = mock.keep.call_args
        assert kwargs["conversations"][0]["conversation_key"] == "sess-001"
        assert kwargs["source"]["origin"] == "hermes"

    def test_applies_collection_fallback(self, provider):
        p, mock = provider
        p._collection = "my-col"
        mock.keep.return_value = {"id": "x", "shortId": "mem-1"}
        p._tool_keep({"headline": "Test"})
        _, kwargs = mock.keep.call_args
        assert kwargs["collections"] == ["my-col"]

    def test_explicit_collections_override_fallback(self, provider):
        p, mock = provider
        p._collection = "my-col"
        mock.keep.return_value = {"id": "x", "shortId": "mem-1"}
        p._tool_keep({"headline": "Test", "collections": ["other"]})
        _, kwargs = mock.keep.call_args
        assert kwargs["collections"] == ["other"]

    def test_forwards_optional_fields(self, provider):
        p, mock = provider
        mock.keep.return_value = {"id": "x", "shortId": "mem-1"}
        p._tool_keep({"headline": "H", "body": "B", "tags": ["t1"], "kind": "fact"})
        args, kwargs = mock.keep.call_args
        assert args[0] == "H"
        assert kwargs["body"] == "B"
        assert kwargs["tags"] == ["t1"]
        assert kwargs["kind"] == "fact"

    def test_records_failure_on_api_exception(self, provider):
        p, mock = provider
        mock.keep.side_effect = Exception("timeout")
        p._tool_keep({"headline": "H"})
        assert p._consecutive_failures == 1


# ── Tool: recall ─────────────────────────────────────────────────────────


class TestToolRecall:

    def test_requires_memory_id(self, provider):
        p, _ = provider
        result = json.loads(p._tool_recall({}))
        assert "error" in result

    def test_formats_full_response(self, provider):
        p, mock = provider
        mock.recall.return_value = {
            "id": "uuid-123", "shortId": "mem-1a",
            "headline": "Auth decision", "body": "Use OAuth2",
            "tags": ["domain:auth"], "kind": "decision",
            "canonStatus": "accepted",
            "collections": [{"slug": "infra"}],
            "createdAt": "2025-01-01", "updatedAt": "2025-06-01",
        }
        result = json.loads(p._tool_recall({"memory_id": "mem-1a"}))
        assert result["id"] == "mem-1a"
        assert result["headline"] == "Auth decision"
        assert result["canon_status"] == "accepted"
        assert result["kind"] == "decision"
        assert result["created"] == "2025-01-01"
        assert result["updated"] == "2025-06-01"

    def test_records_failure_on_api_exception(self, provider):
        p, mock = provider
        mock.recall.side_effect = Exception("not found")
        result = json.loads(p._tool_recall({"memory_id": "mem-1"}))
        assert "error" in result
        assert p._consecutive_failures == 1


# ── Tool: update ─────────────────────────────────────────────────────────


class TestToolUpdate:

    def test_requires_memory_id(self, provider):
        p, _ = provider
        result = json.loads(p._tool_update({"mode": "body_replace"}))
        assert "error" in result
        assert "memory_id" in result["error"]

    def test_requires_mode(self, provider):
        p, _ = provider
        result = json.loads(p._tool_update({"memory_id": "mem-1"}))
        assert "error" in result
        assert "mode" in result["error"]

    def test_forwards_all_metadata_fields(self, provider):
        """Prior bug: only headline/body/tags were forwarded,
        ignoring canon_status, kind, confidence, verification_status."""
        p, mock = provider
        mock.update.return_value = {"id": "x", "shortId": "mem-1"}
        p._tool_update({
            "memory_id": "mem-1", "mode": "revision_only",
            "canon_status": "accepted", "kind": "decision",
            "confidence": 95, "verification_status": "verified",
            "tags": ["important"],
        })
        mock.update.assert_called_once_with(
            "mem-1", "revision_only",
            canon_status="accepted", kind="decision",
            confidence=95, verification_status="verified",
            tags=["important"],
        )

    def test_omits_absent_fields(self, provider):
        p, mock = provider
        mock.update.return_value = {"id": "x", "shortId": "mem-1"}
        p._tool_update({"memory_id": "mem-1", "mode": "headline_replace", "headline": "New"})
        _, kwargs = mock.update.call_args
        assert "headline" in kwargs
        assert "body" not in kwargs
        assert "canon_status" not in kwargs

    def test_records_failure_on_api_exception(self, provider):
        p, mock = provider
        mock.update.side_effect = Exception("conflict")
        result = json.loads(p._tool_update({"memory_id": "m", "mode": "body_replace"}))
        assert "error" in result
        assert p._consecutive_failures == 1


# ── Tool: forget ─────────────────────────────────────────────────────────


class TestToolForget:

    def test_requires_memory_id(self, provider):
        p, _ = provider
        result = json.loads(p._tool_forget({}))
        assert "error" in result

    def test_returns_success_message(self, provider):
        p, mock = provider
        mock.forget.return_value = {"ok": True}
        result = json.loads(p._tool_forget({"memory_id": "mem-1"}))
        assert result["result"] == "Memory deleted."

    def test_records_failure_on_api_exception(self, provider):
        p, mock = provider
        mock.forget.side_effect = Exception("server error")
        result = json.loads(p._tool_forget({"memory_id": "mem-1"}))
        assert "error" in result
        assert p._consecutive_failures == 1


# ── sync_turn ────────────────────────────────────────────────────────────


class TestSyncTurn:

    def test_skips_short_user_content(self, provider):
        p, mock = provider
        p.sync_turn("hi", "hello")
        if p._sync_thread:
            p._sync_thread.join(timeout=1.0)
        mock.keep.assert_not_called()

    def test_skips_when_breaker_open(self, provider):
        p, mock = provider
        for _ in range(5):
            p._record_failure()
        p.sync_turn("A" * 50, "response text")
        if p._sync_thread:
            p._sync_thread.join(timeout=1.0)
        mock.keep.assert_not_called()

    def test_skips_when_no_client(self):
        p = MaindexMemoryProvider()
        p.sync_turn("A" * 50, "response")
        assert p._sync_thread is None

    def test_stores_turn_as_note(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.sync_turn(
            "Tell me about Python classes and inheritance patterns",
            "Python supports single and multiple inheritance...",
        )
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert kwargs["kind"] == "note"
        assert "source:hermes" in kwargs["tags"]
        assert "User:" in kwargs["body"]
        assert "Assistant:" in kwargs["body"]

    def test_appends_ellipsis_to_long_headline(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.sync_turn("A" * 300, "response")
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert kwargs["headline"].endswith("...")
        assert len(kwargs["headline"]) <= 204

    def test_includes_collection_when_configured(self, provider):
        p, mock = provider
        p._collection = "my-col"
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.sync_turn("A long enough user message here!", "response text")
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert kwargs["collections"] == ["my-col"]


# ── queue_prefetch + prefetch ────────────────────────────────────────────


class TestPrefetch:

    def test_queue_prefetch_skips_short_query(self, provider):
        p, mock = provider
        p.queue_prefetch("hi")
        assert p._prefetch_thread is None
        mock.search.assert_not_called()

    def test_queue_prefetch_skips_breaker_open(self, provider):
        p, mock = provider
        for _ in range(5):
            p._record_failure()
        p.queue_prefetch("a long enough query to pass the length check")
        assert p._prefetch_thread is None

    def test_roundtrip_stores_and_retrieves_results(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({
            "items": [
                {"shortId": "mem-1a", "headline": "Auth design", "body": "OAuth2 flow"},
            ],
        })
        mock.search.side_effect = side_effect

        p.queue_prefetch("how does authentication work in our system?")
        assert done.wait(timeout=2.0)

        result = p.prefetch("irrelevant")
        assert "mem-1a" in result
        assert "Auth design" in result

    def test_prefetch_clears_result_after_read(self, provider):
        p, _ = provider
        with p._prefetch_lock:
            p._prefetch_result = "cached data"
        p.prefetch("q")
        assert p._prefetch_result == ""

    def test_prefetch_returns_empty_when_nothing_queued(self, provider):
        p, _ = provider
        assert p.prefetch("q") == ""


# ── on_memory_write ──────────────────────────────────────────────────────


class TestOnMemoryWrite:

    def test_skips_delete_action(self, provider):
        p, mock = provider
        p.on_memory_write("delete", "user", "some content that is long enough")
        time.sleep(0.05)
        mock.keep.assert_not_called()

    def test_skips_empty_content(self, provider):
        p, mock = provider
        p.on_memory_write("add", "user", "")
        time.sleep(0.05)
        mock.keep.assert_not_called()

    def test_skips_breaker_open(self, provider):
        p, mock = provider
        for _ in range(5):
            p._record_failure()
        p.on_memory_write("add", "user", "User prefers dark mode")
        time.sleep(0.05)
        mock.keep.assert_not_called()

    def test_stores_user_profile_as_fact(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.on_memory_write("add", "user", "User prefers dark mode and vim keybindings")
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert "User profile" in kwargs["headline"]
        assert kwargs["kind"] == "fact"
        assert "hermes:user" in kwargs["tags"]

    def test_stores_agent_memory_label(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.on_memory_write("replace", "agent", "Agent learned a new pattern")
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert "Agent memory" in kwargs["headline"]


# ── on_pre_compress ──────────────────────────────────────────────────────


class TestOnPreCompress:

    def test_skips_empty_messages(self, provider):
        p, mock = provider
        result = p.on_pre_compress([])
        assert result == ""
        time.sleep(0.05)
        mock.keep.assert_not_called()

    def test_skips_when_no_client(self):
        p = MaindexMemoryProvider()
        result = p.on_pre_compress([{"role": "user", "content": "hello"}])
        assert result == ""

    def test_filters_non_user_assistant_roles(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.on_pre_compress([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a language"},
            {"role": "tool", "content": "tool output"},
        ])
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert "system:" not in kwargs["body"]
        assert "tool:" not in kwargs["body"]
        assert "user:" in kwargs["body"]
        assert "assistant:" in kwargs["body"]

    def test_stores_as_summary_kind(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        p.on_pre_compress([
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ])
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert kwargs["kind"] == "summary"
        assert "hermes:compression" in kwargs["tags"]

    def test_limits_to_last_10_messages(self, provider):
        p, mock = provider
        done, side_effect = _event_side_effect({"id": "x"})
        mock.keep.side_effect = side_effect
        messages = [
            {"role": "user", "content": f"msg-{i}"} for i in range(20)
        ]
        p.on_pre_compress(messages)
        assert done.wait(timeout=2.0)
        _, kwargs = mock.keep.call_args
        assert "msg-10" in kwargs["body"]
        assert "msg-0" not in kwargs["body"]

    def test_skips_messages_with_empty_content(self, provider):
        p, mock = provider
        result = p.on_pre_compress([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},
        ])
        assert result == ""
        time.sleep(0.05)
        mock.keep.assert_not_called()


# ── Plugin entry point ───────────────────────────────────────────────────


class TestRegister:

    def test_registers_provider_instance(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_memory_provider.assert_called_once()
        arg = ctx.register_memory_provider.call_args[0][0]
        assert isinstance(arg, MaindexMemoryProvider)
