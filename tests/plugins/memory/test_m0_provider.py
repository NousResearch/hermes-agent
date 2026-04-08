"""Tests for seekdb M0 memory provider (HTTP client mocked).

Optional **live API** tests run only when both are set::

    export M0_INTEGRATION_TESTS=1
    export M0_API_KEY=ak_...

CI and default ``pytest`` runs stay mock-only (live class is skipped).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import pytest

from plugins.memory.m0 import (
    M0MemoryProvider,
    _load_m0_config,
    _m0_api_key,
    _save_m0_config,
)


class FakeM0Client:
    """Replaces _M0Client — no network."""

    def __init__(self, api_key: str, base_url: str, timeout: float):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.search_payload = {
            "memories": [{"id": 1, "content": "User prefers dark mode", "score": 0.91}],
            "rewritten_queries": None,
        }
        self.capture_calls: list = []
        self.store_calls: list = []
        self.deleted_ids: list = []

    def search(self, query, *, limit=10, rewrite=False, context=None):
        return dict(self.search_payload)

    def capture(self, messages):
        self.capture_calls.append(list(messages))
        return {"ok": True}

    def store(self, content, metadata=None):
        self.store_calls.append({"content": content, "metadata": metadata or {}})
        return {"id": 42}

    def list_memories(self):
        return [{"id": 1, "content": "alpha"}, {"id": 2, "content": "beta"}]

    def get_memory(self, memory_id):
        return {"id": memory_id, "content": "one memory", "metadata": {}}

    def update_memory(self, memory_id, content, metadata=None):
        return {"id": memory_id, "content": content}

    def delete_memory(self, memory_id):
        self.deleted_ids.append(memory_id)


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("M0_API_KEY", "ak_test_fake")
    monkeypatch.setattr("plugins.memory.m0._M0Client", FakeM0Client)
    p = M0MemoryProvider()
    p.initialize("session-m0", hermes_home=str(tmp_path), platform="cli")
    return p


def test_is_available_false_without_key(monkeypatch):
    monkeypatch.delenv("M0_API_KEY", raising=False)
    p = M0MemoryProvider()
    assert p.is_available() is False


def test_load_save_m0_json(tmp_path):
    _save_m0_config({"recall_limit": 7, "auto_capture": False}, str(tmp_path))
    cfg = _load_m0_config(str(tmp_path))
    assert cfg["recall_limit"] == 7
    assert cfg["auto_capture"] is False
    assert cfg["auto_recall"] is True


def test_get_config_schema_has_key():
    p = M0MemoryProvider()
    keys = [x.get("key") for x in p.get_config_schema()]
    assert "api_key" in keys
    assert "base_url" in keys


def test_save_config_writes_base_url(tmp_path):
    p = M0MemoryProvider()
    p.save_config({"base_url": "https://example.test"}, str(tmp_path))
    cfg = _load_m0_config(str(tmp_path))
    assert cfg["base_url"] == "https://example.test"


def test_system_prompt_lists_tools(provider):
    text = provider.system_prompt_block()
    assert "m0_search" in text
    assert "m0_store" in text


def test_prefetch_formats_context(provider):
    out = provider.prefetch("dark mode preferences")
    assert "<m0-context>" in out
    assert "dark mode" in out


def test_prefetch_empty_when_auto_recall_off(monkeypatch, tmp_path):
    monkeypatch.setenv("M0_API_KEY", "ak_x")
    monkeypatch.setattr("plugins.memory.m0._M0Client", FakeM0Client)
    _save_m0_config({"auto_recall": False}, str(tmp_path))
    p = M0MemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path), platform="cli")
    assert p.prefetch("anything") == ""


def test_sync_turn_capture_background(provider):
    u = "x" * 50
    a = "y" * 50
    provider.sync_turn(u, a)
    if provider._sync_thread:
        provider._sync_thread.join(timeout=2.0)
    assert len(provider._client.capture_calls) == 1  # type: ignore[attr-defined]
    msgs = provider._client.capture_calls[0]  # type: ignore[attr-defined]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_sync_turn_skips_trivial_user(monkeypatch, tmp_path):
    monkeypatch.setenv("M0_API_KEY", "ak_x")
    monkeypatch.setattr("plugins.memory.m0._M0Client", FakeM0Client)
    p = M0MemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path), platform="cli")
    p.sync_turn("thanks", "assistant reply " * 5)
    if p._sync_thread:
        p._sync_thread.join(timeout=2.0)
    assert p._client.capture_calls == []  # type: ignore[attr-defined]


def test_on_memory_write_store(provider):
    provider.on_memory_write("add", "MEMORY.md", "remember this fact")
    if provider._write_thread:
        provider._write_thread.join(timeout=2.0)
    assert len(provider._client.store_calls) == 1  # type: ignore[attr-defined]


def test_tool_search(provider):
    raw = provider.handle_tool_call("m0_search", {"query": "prefs", "limit": 3})
    data = json.loads(raw)
    assert data["count"] == 1
    assert "dark mode" in data["results"][0]["content"]


def test_tool_store(provider):
    raw = provider.handle_tool_call("m0_store", {"content": "hello m0"})
    data = json.loads(raw)
    assert data["saved"] is True


def test_tool_list(provider):
    raw = provider.handle_tool_call("m0_list", {})
    data = json.loads(raw)
    assert data["count"] == 2


def test_tool_get(provider):
    raw = provider.handle_tool_call("m0_get", {"id": 9})
    data = json.loads(raw)
    assert data["id"] == 9


def test_tool_update(provider):
    raw = provider.handle_tool_call(
        "m0_update", {"id": 1, "content": "updated text", "metadata": {"k": "v"}}
    )
    data = json.loads(raw)
    assert data["updated"] is True


def test_tool_delete(provider):
    raw = provider.handle_tool_call("m0_delete", {"id": 3})
    data = json.loads(raw)
    assert data["deleted"] is True
    assert provider._client.deleted_ids == [3]  # type: ignore[attr-defined]


def test_tool_errors(provider):
    err = json.loads(provider.handle_tool_call("m0_search", {}))
    assert "error" in err
    err2 = json.loads(provider.handle_tool_call("m0_store", {"content": ""}))
    assert "error" in err2


def test_handle_unknown_tool(provider):
    err = json.loads(provider.handle_tool_call("m0_nope", {}))
    assert "error" in err


def test_not_configured_returns_error(monkeypatch, tmp_path):
    monkeypatch.delenv("M0_API_KEY", raising=False)
    monkeypatch.setattr("plugins.memory.m0._M0Client", FakeM0Client)
    p = M0MemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path), platform="cli")
    err = json.loads(p.handle_tool_call("m0_search", {"query": "x"}))
    assert "error" in err


def test_get_tool_schemas_count():
    p = M0MemoryProvider()
    assert len(p.get_tool_schemas()) == 6


def test_load_memory_provider_m0():
    from plugins.memory import load_memory_provider

    p = load_memory_provider("m0")
    assert p is not None
    assert p.name == "m0"


# ---------------------------------------------------------------------------
# Optional: real m0.seekdb.ai HTTP (opt-in — CI must not set M0_INTEGRATION_TESTS)
# ---------------------------------------------------------------------------


def _m0_live_tests_enabled() -> bool:
    flag = os.environ.get("M0_INTEGRATION_TESTS", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    key = (os.environ.get("M0_API_KEY") or "").strip()
    return bool(key) and key.startswith("ak_")


needs_m0_live = pytest.mark.skipif(
    not _m0_live_tests_enabled(),
    reason="Live seekdb M0: set M0_INTEGRATION_TESTS=1 and M0_API_KEY (skipped in CI).",
)


def _extract_store_id(store_resp: object) -> Optional[int]:
    if not isinstance(store_resp, dict):
        return None
    rid = store_resp.get("id")
    if rid is None and isinstance(store_resp.get("data"), dict):
        rid = store_resp["data"].get("id")
    try:
        return int(rid) if rid is not None else None
    except (TypeError, ValueError):
        return None


def _search_hits_with_marker(client, marker: str, *, attempts: int = 8, delay_s: float = 0.75):
    from plugins.memory.m0 import _parse_memory_list_payload

    last_memories: list = []
    for _ in range(attempts):
        data = client.search(marker, limit=15, rewrite=False, context=[])
        last_memories = data.get("memories") if isinstance(data, dict) else []
        if not isinstance(last_memories, list):
            last_memories = _parse_memory_list_payload(data)
        for m in last_memories:
            if not isinstance(m, dict):
                continue
            text = str(m.get("content") or m.get("memory") or "")
            if marker in text:
                return m, last_memories
        time.sleep(delay_s)
    return None, last_memories


@needs_m0_live
class TestM0LiveApiOptional:
    """Hits production (or M0_BASE_URL) — requires explicit env opt-in."""

    @pytest.fixture
    def live_client(self):
        from plugins.memory.m0 import _DEFAULT_BASE_URL, _M0Client

        key = (os.environ.get("M0_API_KEY") or "").strip()
        base = (os.environ.get("M0_BASE_URL") or "").strip() or _DEFAULT_BASE_URL
        return _M0Client(key, base, 45.0)

    def test_instance_status(self, live_client):
        st = live_client.instance_status()
        assert st.get("status") == "ready"

    def test_store_search_delete_roundtrip(self, live_client):
        marker = f"hermes_m0_live_{uuid.uuid4().hex}"
        mem_id: Optional[int] = None
        try:
            raw = live_client.store(marker, {"source": "hermes_pytest"})
            mem_id = _extract_store_id(raw)

            hit, _mems = _search_hits_with_marker(live_client, marker)
            assert hit is not None, "search did not return stored marker after retries"

            if mem_id is None:
                try:
                    mem_id = int(hit["id"])
                except (TypeError, ValueError, KeyError):
                    pytest.fail("could not resolve memory id for cleanup")

            live_client.delete_memory(mem_id)
            mem_id = None
            time.sleep(1.0)
            hit2, _ = _search_hits_with_marker(live_client, marker, attempts=6, delay_s=1.0)
            assert hit2 is None, "memory still searchable after delete"
        finally:
            if mem_id is not None:
                try:
                    live_client.delete_memory(mem_id)
                except Exception:
                    pass

    def test_provider_tools_live(self, monkeypatch, tmp_path):
        from plugins.memory.m0 import M0MemoryProvider

        key = (os.environ.get("M0_API_KEY") or "").strip()
        monkeypatch.setenv("M0_API_KEY", key)
        _save_m0_config(
            {"auto_capture": False, "auto_recall": False, "api_timeout": 45.0},
            str(tmp_path),
        )
        p = M0MemoryProvider()
        p.initialize("pytest-live", hermes_home=str(tmp_path), platform="cli")
        marker = f"hermes_m0_tool_{uuid.uuid4().hex}"
        mem_id: Optional[int] = None
        try:
            add_raw = p.handle_tool_call(
                "m0_store",
                {"content": marker, "metadata": {"source": "hermes_pytest"}},
            )
            add_data = json.loads(add_raw)
            assert "error" not in add_data, add_raw
            mem_id = _extract_store_id(add_data.get("response"))

            search_raw = p.handle_tool_call("m0_search", {"query": marker, "limit": 10})
            search_data = json.loads(search_raw)
            assert "error" not in search_data, search_raw
            results = search_data.get("results") or []
            texts = [str(r.get("content") or "") for r in results]
            assert any(marker in t for t in texts), search_data

            if mem_id is None and results:
                try:
                    mem_id = int(results[0].get("id"))
                except (TypeError, ValueError):
                    pass
            if mem_id is not None:
                del_raw = p.handle_tool_call("m0_delete", {"id": mem_id})
                assert "error" not in json.loads(del_raw)
        finally:
            p.shutdown()
            if mem_id is not None:
                try:
                    from plugins.memory.m0 import _DEFAULT_BASE_URL, _M0Client

                    base = (os.environ.get("M0_BASE_URL") or "").strip() or _DEFAULT_BASE_URL
                    _M0Client(key, base, 15.0).delete_memory(mem_id)
                except Exception:
                    pass
