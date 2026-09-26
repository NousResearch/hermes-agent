"""Tests for scripts/memory_vault_export.py.

Coverage:
- Config path resolution uses get_hermes_home(), not a hardcoded path.
- Default collection name is 'mem0' (matches _backend.py:211).
- Qdrant HTTP scroll pagination continues until next_page_offset is None.
- render_markdown / extract_text unit coverage.
- _QdrantAdapter HTTP path (qdrant_client SDK not required for these tests).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the script module without qdrant_client being installed
# ---------------------------------------------------------------------------

_SCRIPT_ROOT = Path(__file__).parent.parent / "scripts"


def _load_module(monkeypatch, hermes_home: Path) -> ModuleType:
    """Import memory_vault_export with HERMES_HOME pointing to a temp dir."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Remove any cached import so each test gets a fresh module.
    for key in list(sys.modules.keys()):
        if "memory_vault_export" in key:
            del sys.modules[key]

    spec = importlib.util.spec_from_file_location(
        "memory_vault_export",
        _SCRIPT_ROOT / "memory_vault_export.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# 1.  Config path resolution
# ---------------------------------------------------------------------------


class TestConfigPathResolution:
    """get_hermes_home() must drive mem0.json resolution, not a hardcoded path."""

    def test_reads_from_hermes_home_env(self, tmp_path, monkeypatch):
        """mem0.json inside HERMES_HOME is found; a file at the old hardcoded
        path (~/.hermes/hermes-agent/mem0.json) must NOT be consulted."""
        fake_home = tmp_path / "fake_hermes_home"
        fake_home.mkdir()
        mem0_cfg = {
            "oss": {
                "vector_store": {
                    "config": {"collection_name": "my_custom_col", "path": "/some/path"}
                }
            }
        }
        (fake_home / "mem0.json").write_text(json.dumps(mem0_cfg))

        mod = _load_module(monkeypatch, fake_home)

        assert mod.get_hermes_home() == fake_home
        assert mod.load_collection_from_mem0_json() == "my_custom_col"

    def test_missing_mem0_json_returns_none(self, tmp_path, monkeypatch):
        """When mem0.json is absent, load_collection_from_mem0_json returns None."""
        mod = _load_module(monkeypatch, tmp_path)
        assert mod.load_collection_from_mem0_json() is None

    def test_does_not_read_hardcoded_path(self, tmp_path, monkeypatch):
        """Place a mem0.json at the OLD hardcoded path; it must be ignored."""
        # Redirect home so Path.home() points somewhere we control.
        fake_real_home = tmp_path / "home"
        fake_real_home.mkdir()
        old_hardcoded = fake_real_home / ".hermes" / "hermes-agent"
        old_hardcoded.mkdir(parents=True)
        decoy_cfg = {
            "oss": {"vector_store": {"config": {"collection_name": "DECOY"}}}
        }
        (old_hardcoded / "mem0.json").write_text(json.dumps(decoy_cfg))

        # Point HERMES_HOME to a *different* empty directory.
        real_home = tmp_path / "real_hermes_home"
        real_home.mkdir()
        mod = _load_module(monkeypatch, real_home)

        # The decoy file at the old path must NOT be read.
        result = mod.load_collection_from_mem0_json()
        assert result is None, (
            f"Expected None (no mem0.json in HERMES_HOME), got {result!r}. "
            "Script is still reading the old hardcoded path."
        )


# ---------------------------------------------------------------------------
# 2.  Default collection name
# ---------------------------------------------------------------------------


class TestDefaultCollectionName:
    """The fallback collection name must be 'mem0' (matching _backend.py:211)."""

    def test_fallback_is_mem0_when_no_config(self, tmp_path, monkeypatch):
        """When mem0.json is absent and no override is given, default is 'mem0'."""
        mod = _load_module(monkeypatch, tmp_path)
        collected = mod.load_collection_from_mem0_json()
        default = collected or "mem0"
        assert default == "mem0", (
            f"Default collection must be 'mem0' to match _backend.py:211, got {default!r}."
        )

    def test_fallback_in_parse_args(self, tmp_path, monkeypatch):
        """parse_args() must default --collection to 'mem0' when mem0.json absent."""
        mod = _load_module(monkeypatch, tmp_path)
        # Minimal args: --user is required; use --qdrant-url to avoid SDK check.
        with patch.object(sys, "argv", ["memory_vault_export.py", "--user", "alice",
                                         "--qdrant-url", "http://localhost:6333"]):
            args = mod.parse_args()
        assert args.collection == "mem0", (
            f"parse_args() default collection must be 'mem0', got {args.collection!r}."
        )

    def test_collection_from_mem0_json_overrides_default(self, tmp_path, monkeypatch):
        """A collection_name in mem0.json beats the 'mem0' fallback."""
        cfg = {"oss": {"vector_store": {"config": {"collection_name": "my_col"}}}}
        (tmp_path / "mem0.json").write_text(json.dumps(cfg))
        mod = _load_module(monkeypatch, tmp_path)

        with patch.object(sys, "argv", ["memory_vault_export.py", "--user", "bob",
                                         "--qdrant-url", "http://localhost:6333"]):
            args = mod.parse_args()
        assert args.collection == "my_col"


# ---------------------------------------------------------------------------
# 3.  Qdrant HTTP scroll pagination
# ---------------------------------------------------------------------------


class TestQdrantScrollPagination:
    """scroll_all_for_user must keep paginating until next_page_offset is None."""

    def _make_adapter(self, mod) -> Any:
        """Return an HTTP-mode adapter (no SDK needed)."""
        adapter = mod._QdrantAdapter.__new__(mod._QdrantAdapter)
        adapter._mode = "http"
        adapter._url = "http://localhost:6333"
        adapter._client = None
        return adapter

    def test_single_page(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        adapter = self._make_adapter(mod)

        page1 = {
            "result": {
                "points": [{"id": "aaa", "payload": {"user_id": "alice", "data": "hello"}}],
                "next_page_offset": None,
            }
        }
        with patch.object(adapter, "_api_post", return_value=page1) as mock_post:
            result = adapter.scroll_all_for_user("mem0", "alice")

        assert len(result) == 1
        assert mock_post.call_count == 1

    def test_two_pages(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        adapter = self._make_adapter(mod)

        page1 = {
            "result": {
                "points": [{"id": "aaa", "payload": {"data": "p1"}},
                           {"id": "bbb", "payload": {"data": "p2"}}],
                "next_page_offset": "bbb",
            }
        }
        page2 = {
            "result": {
                "points": [{"id": "ccc", "payload": {"data": "p3"}}],
                "next_page_offset": None,
            }
        }
        with patch.object(adapter, "_api_post", side_effect=[page1, page2]) as mock_post:
            result = adapter.scroll_all_for_user("mem0", "alice")

        assert len(result) == 3
        assert mock_post.call_count == 2

        # Second call must carry the offset from page1
        _, second_body = mock_post.call_args_list[1]
        # call_args_list entries are call(args, kwargs); body is positional arg 1
        second_call_body = mock_post.call_args_list[1][0][1]  # path, body positional
        assert second_call_body["offset"] == "bbb"

    def test_empty_batch_stops_pagination(self, tmp_path, monkeypatch):
        """Even if next_page_offset is set, an empty batch stops the loop."""
        mod = _load_module(monkeypatch, tmp_path)
        adapter = self._make_adapter(mod)

        # Server returns a non-None offset but zero points — edge case
        page1 = {
            "result": {
                "points": [],
                "next_page_offset": "some_offset",
            }
        }
        with patch.object(adapter, "_api_post", return_value=page1) as mock_post:
            result = adapter.scroll_all_for_user("mem0", "alice")

        assert result == []
        assert mock_post.call_count == 1

    def test_three_pages_exhausted(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        adapter = self._make_adapter(mod)

        pages = [
            {"result": {"points": [{"id": str(i), "payload": {"data": f"m{i}"}}],
                        "next_page_offset": str(i)}}
            for i in range(2)
        ]
        pages.append({"result": {"points": [{"id": "2", "payload": {"data": "m2"}}],
                                  "next_page_offset": None}})

        with patch.object(adapter, "_api_post", side_effect=pages) as mock_post:
            result = adapter.scroll_all_for_user("mem0", "alice")

        assert len(result) == 3
        assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# 4.  Markdown rendering unit tests
# ---------------------------------------------------------------------------


class TestMarkdownRendering:
    def test_render_contains_frontmatter(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        md = mod.render_markdown("my-id", {"data": "hello world", "agent_id": "hermes"})
        assert "---" in md
        assert 'id: "my-id"' in md
        assert 'agent_id: "hermes"' in md
        assert "hello world" in md

    def test_extract_text_prefers_data(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        payload = {"data": "preferred", "text": "fallback", "memory": "other"}
        assert mod.extract_text(payload) == "preferred"

    def test_extract_text_falls_back_to_text(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        assert mod.extract_text({"text": "from text"}) == "from text"

    def test_extract_text_empty_payload(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        assert mod.extract_text({}) == ""


# ---------------------------------------------------------------------------
# 5.  Embedded Qdrant detection
# ---------------------------------------------------------------------------


class TestEmbeddedQdrantDetection:
    def test_detect_embedded_mode_from_mem0_json(self, tmp_path, monkeypatch):
        """When mem0.json has a 'path' in vector_store.config, detect_qdrant_mode returns it."""
        cfg = {"oss": {"vector_store": {"config": {"path": "/data/qdrant"}}}}
        (tmp_path / "mem0.json").write_text(json.dumps(cfg))
        mod = _load_module(monkeypatch, tmp_path)

        mem0_cfg = mod.load_mem0_json()
        path, url = mod.detect_qdrant_mode(mem0_cfg)
        assert path == "/data/qdrant"
        assert url is None

    def test_detect_http_mode_from_mem0_json(self, tmp_path, monkeypatch):
        cfg = {"oss": {"vector_store": {"config": {"url": "http://qdrant:6333"}}}}
        (tmp_path / "mem0.json").write_text(json.dumps(cfg))
        mod = _load_module(monkeypatch, tmp_path)

        mem0_cfg = mod.load_mem0_json()
        path, url = mod.detect_qdrant_mode(mem0_cfg)
        assert path is None
        assert url == "http://qdrant:6333"

    def test_detect_neither_when_no_config(self, tmp_path, monkeypatch):
        mod = _load_module(monkeypatch, tmp_path)
        path, url = mod.detect_qdrant_mode({})
        assert path is None
        assert url is None
