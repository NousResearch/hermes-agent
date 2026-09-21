"""Tests for kensei-super-memory-mcp — enhanced memory MCP tools.

Three layers:
1. Unit tests — handler functions with mocked Mnemosyne
2. Tool registration tests — verify FastMCP tool definitions
3. Integration tests — end-to-end through FastMCP's tool manager
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from importlib.util import find_spec

if find_spec("mcp.server.fastmcp") is None:
    pytestmark = pytest.mark.skip(reason="legacy FastMCP server API is unavailable with MCP 2.0")

# Pre-import the real mcp SDK before adding the scripts directory to sys.path.
# The scripts dir contains a local mcp/ package that shadows the installed SDK.
import mcp as _real_mcp
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
# Pre-import every real mcp submodule the codebase uses so a later
# ``import mcp.shared`` / ``import mcp.types`` cannot resolve from the shadow
# package once scripts/ is on sys.path (the top-level restore alone leaves
# submodule resolution vulnerable to the on-disk shadow).
import mcp.shared  # noqa: F401
import mcp.shared.auth  # noqa: F401
import mcp.types  # noqa: F401
sys.path.insert(0, str(_SCRIPTS_DIR))
# Restore the real mcp in sys.modules so the shadowed package doesn't win.
sys.modules["mcp"] = _real_mcp

# Remove scripts/ from sys.path immediately: it shadows the real mcp package
# on disk, and leaving it on the global path poisons every later
# ``import mcp.shared``/``mcp.types`` during full-suite collection. The
# kensei_super_memory_mcp imports are lazy (inside test functions); an
# autouse fixture re-adds the dir at test time.
try:
    sys.path.remove(str(_SCRIPTS_DIR))
except ValueError:
    pass

# ---------------------------------------------------------------------------
# Mock Mnemosyne at module level so create_app() doesn't raise.
# The script checks _MNEMOSYNE_AVAILABLE at import time; we inject stubs
# before the import so the flag becomes True.
# ---------------------------------------------------------------------------
_mnemosyne_stub = MagicMock()
sys.modules["mnemosyne"] = _mnemosyne_stub
sys.modules["mnemosyne.core"] = MagicMock()
sys.modules["mnemosyne.core.memory"] = MagicMock()
sys.modules["mnemosyne.core.beam"] = MagicMock()
sys.modules["mnemosyne.core.triples"] = MagicMock()
sys.modules["mnemosyne.core.canonical"] = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Prevent accidental writes to real Mnemosyne data."""
    monkeypatch.setenv("KENSEI_MEMORY_BANK", "test_bank")
    monkeypatch.setenv("MNEMOSYNE_HOME", "/tmp/mnemosyne_test")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes_test")


@pytest.fixture(autouse=True)
def _scripts_on_path_for_lazy_imports():
    """Re-add scripts/ to sys.path for the lazy kensei_super_memory_mcp imports.

    The module-level path entry is removed at import time so the on-disk mcp
    shadow never poisons full-suite collection; the lazy imports inside test
    functions still need the dir at call time.
    """
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    yield
    try:
        sys.path.remove(str(_SCRIPTS_DIR))
    except ValueError:
        pass


@pytest.fixture
def mock_triple_store():
    """A MagicMock that behaves like TripleStore."""
    store = MagicMock()
    store.add = MagicMock(return_value=None)
    store.query = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_canonical_store():
    """A MagicMock that behaves like CanonicalStore."""
    store = MagicMock()
    store.remember = MagicMock(return_value=None)
    return store


@pytest.fixture
def mock_beam():
    """A MagicMock that behaves like BeamMemory."""
    beam = MagicMock()
    beam.db_path = "/tmp/mnemosyne_test/test_bank/mnemosyne.db"
    beam.get_working_memories = MagicMock(return_value=[])
    beam.get_working_stats = MagicMock(return_value={})
    beam.clear_working = MagicMock(return_value=None)
    beam.get_episodic_stats = MagicMock(return_value={})
    beam.get_episodic_memories = MagicMock(return_value=[])
    beam.clear_episodic = MagicMock(return_value=None)
    return beam


@pytest.fixture
def mock_mnemosyne(mock_beam):
    """A MagicMock that behaves like Mnemosyne."""
    mem = MagicMock()
    mem.beam = mock_beam
    mem.db_path = "/tmp/mnemosyne_test/test_bank/mnemosyne.db"
    mem.remember = MagicMock(return_value=None)
    return mem


# ---------------------------------------------------------------------------
# Unit tests: _handle_mem0_extract
# ---------------------------------------------------------------------------


class TestMem0Extract:
    def test_empty_text_returns_error(self):
        """Empty text should return an error status."""
        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="")
        assert result["status"] == "error"
        assert "text is required" in result["message"]

    def test_whitespace_text_returns_error(self):
        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="   \n  ")
        assert result["status"] == "error"

    @patch("kensei_super_memory_mcp._call_llm", return_value=None)
    def test_llm_failure_returns_error(self, mock_llm):
        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="Alice works at Acme Corp.")
        assert result["status"] == "error"
        assert "LLM extraction failed" in result["message"]

    @patch("kensei_super_memory_mcp._call_llm", return_value="not valid json")
    def test_unparseable_llm_response(self, mock_llm):
        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="Alice works at Acme Corp.")
        assert result["status"] == "error"
        assert "unparseable JSON" in result["message"]

    @patch("kensei_super_memory_mcp._call_llm")
    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_successful_extraction(self, mock_get_ts, mock_llm):
        """Valid LLM response should parse and store triples."""
        mock_llm.return_value = json.dumps({
            "entities": ["Alice", "Acme Corp"],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"}
            ],
        })
        mock_store = MagicMock()
        mock_store.add = MagicMock(return_value=None)
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(
            text="Alice works at Acme Corp.",
            bank="test_bank",
            source="test_extract",
            confidence=0.9,
        )

        assert result["status"] == "ok"
        assert result["entities"] == ["Alice", "Acme Corp"]
        assert result["triples_extracted"] == 1
        assert result["triples_stored"] == 1
        mock_store.add.assert_called_once_with(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
            source="test_extract",
            confidence=0.9,
        )

    @patch("kensei_super_memory_mcp._call_llm")
    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_handles_markdown_wrapped_json(self, mock_get_ts, mock_llm):
        """LLM response wrapped in ```json should be parsed correctly."""
        mock_llm.return_value = "```json\n{\"entities\": [\"Bob\"], \"triples\": []}\n```"
        mock_store = MagicMock()
        mock_store.add = MagicMock(return_value=None)
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="Bob is here.")
        assert result["status"] == "ok"
        assert result["entities"] == ["Bob"]

    @patch("kensei_super_memory_mcp._call_llm")
    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_invalid_triple_skipped(self, mock_get_ts, mock_llm):
        """Triples with missing fields should be skipped with an error logged."""
        mock_llm.return_value = json.dumps({
            "entities": [],
            "triples": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
                {"subject": "", "predicate": "is", "object": ""},  # invalid
            ],
        })
        mock_store = MagicMock()
        mock_store.add = MagicMock(return_value=None)
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_mem0_extract

        result = _handle_mem0_extract(text="Alice works at Acme Corp.")
        assert result["status"] == "ok"
        assert result["triples_stored"] == 1
        assert result["errors"] is not None
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# Unit tests: _handle_cognee_cognify
# ---------------------------------------------------------------------------


class TestCogneeCognify:
    def test_empty_text_returns_error(self):
        from kensei_super_memory_mcp import _handle_cognee_cognify

        result = _handle_cognee_cognify(text="")
        assert result["status"] == "error"

    @patch("kensei_super_memory_mcp._call_llm", return_value=None)
    def test_llm_failure_returns_error(self, mock_llm):
        from kensei_super_memory_mcp import _handle_cognee_cognify

        result = _handle_cognee_cognify(text="Some text.")
        assert result["status"] == "error"

    @patch("kensei_super_memory_mcp._call_llm")
    @patch("kensei_super_memory_mcp._get_canonical_store")
    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_successful_cognify(self, mock_get_ts, mock_get_cs, mock_llm):
        """Entities should be stored in canonical store, relations in triple store."""
        mock_llm.return_value = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "description": "A software engineer"},
                {"name": "Acme Corp", "type": "org", "description": "A tech company"},
            ],
            "relations": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
            ],
        })
        mock_cs = MagicMock()
        mock_cs.remember = MagicMock(return_value=None)
        mock_get_cs.return_value = mock_cs

        mock_ts = MagicMock()
        mock_ts.add = MagicMock(return_value=None)
        mock_get_ts.return_value = mock_ts

        from kensei_super_memory_mcp import _handle_cognee_cognify

        result = _handle_cognee_cognify(
            text="Alice is a software engineer at Acme Corp.",
            bank="test_bank",
            source="test_cognify",
            confidence=0.7,
        )

        assert result["status"] == "ok"
        assert result["entities_extracted"] == 2
        assert result["entities_stored"] == 2
        assert result["relations_extracted"] == 1
        assert result["relations_stored"] == 1

        # Verify canonical store was called for each entity
        assert mock_cs.remember.call_count == 2

        # Verify triple store was called for the relation
        mock_ts.add.assert_called_once_with(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
            source="test_cognify",
            confidence=0.7,
        )


# ---------------------------------------------------------------------------
# Unit tests: _handle_zep_temporal_query
# ---------------------------------------------------------------------------


class TestZepTemporalQuery:
    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_empty_query_returns_all(self, mock_get_ts):
        """No filters should return all results."""
        mock_store = MagicMock()
        mock_store.query = MagicMock(return_value=[
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
        ])
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_zep_temporal_query

        result = _handle_zep_temporal_query(bank="test_bank")
        assert result["status"] == "ok"
        assert result["count"] == 1

    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_filters_by_subject(self, mock_get_ts):
        mock_store = MagicMock()
        mock_store.query = MagicMock(return_value=[
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp"},
        ])
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_zep_temporal_query

        result = _handle_zep_temporal_query(subject="Alice", bank="test_bank")
        assert result["status"] == "ok"
        mock_store.query.assert_called_once_with(
            subject="Alice",
            predicate=None,
            object=None,
            as_of=None,
        )

    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_excludes_invalidated_by_default(self, mock_get_ts):
        """Facts with valid_until set should be excluded unless include_invalidated=True."""
        mock_store = MagicMock()
        mock_store.query = MagicMock(return_value=[
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp",
             "valid_from": "2026-01-01", "valid_until": None},
            {"subject": "Alice", "predicate": "works_at", "object": "Old Corp",
             "valid_from": "2025-01-01", "valid_until": "2026-01-01"},
        ])
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_zep_temporal_query

        result = _handle_zep_temporal_query(bank="test_bank")
        assert result["count"] == 1  # only the current fact

    @patch("kensei_super_memory_mcp._get_triple_store")
    def test_include_invalidated_returns_all(self, mock_get_ts):
        mock_store = MagicMock()
        mock_store.query = MagicMock(return_value=[
            {"subject": "Alice", "predicate": "works_at", "object": "Acme Corp",
             "valid_from": "2026-01-01", "valid_until": None},
            {"subject": "Alice", "predicate": "works_at", "object": "Old Corp",
             "valid_from": "2025-01-01", "valid_until": "2026-01-01"},
        ])
        mock_get_ts.return_value = mock_store

        from kensei_super_memory_mcp import _handle_zep_temporal_query

        result = _handle_zep_temporal_query(
            bank="test_bank", include_invalidated=True
        )
        assert result["count"] == 2


# ---------------------------------------------------------------------------
# Unit tests: _handle_simplemem_compress
# ---------------------------------------------------------------------------


class TestSimplememCompress:
    @patch("kensei_super_memory_mcp._create_instance")
    def test_no_entries_returns_ok(self, mock_create):
        """No entries to compress should return ok with zero counts."""
        mock_beam = MagicMock()
        mock_beam.get_working_memories = MagicMock(return_value=[])
        mock_mem = MagicMock()
        mock_mem.beam = mock_beam
        mock_create.return_value = mock_mem

        from kensei_super_memory_mcp import _handle_simplemem_compress

        result = _handle_simplemem_compress(bank="test_bank", target="working")
        assert result["status"] == "ok"
        assert result["entries_before"] == 0

    @patch("kensei_super_memory_mcp._create_instance")
    def test_single_entry_skips_compression(self, mock_create):
        """Only 1 entry should skip compression (nothing to merge)."""
        mock_beam = MagicMock()
        mock_beam.get_working_memories = MagicMock(return_value=[
            {"id": "1", "content": "Alice works at Acme Corp."},
        ])
        mock_mem = MagicMock()
        mock_mem.beam = mock_beam
        mock_create.return_value = mock_mem

        from kensei_super_memory_mcp import _handle_simplemem_compress

        result = _handle_simplemem_compress(bank="test_bank", target="working")
        assert result["status"] == "ok"
        assert result["compressed"] == 0

    @patch("kensei_super_memory_mcp._call_llm", return_value=None)
    @patch("kensei_super_memory_mcp._create_instance")
    def test_llm_failure_returns_error(self, mock_create, mock_llm):
        mock_beam = MagicMock()
        mock_beam.get_working_memories = MagicMock(return_value=[
            {"id": "1", "content": "Alice works at Acme Corp."},
            {"id": "2", "content": "Bob works at Beta Inc."},
        ])
        mock_mem = MagicMock()
        mock_mem.beam = mock_beam
        mock_create.return_value = mock_mem

        from kensei_super_memory_mcp import _handle_simplemem_compress

        result = _handle_simplemem_compress(bank="test_bank", target="working")
        assert result["status"] == "error"

    @patch("kensei_super_memory_mcp._call_llm")
    @patch("kensei_super_memory_mcp._create_instance")
    def test_dry_run_does_not_write(self, mock_create, mock_llm):
        """dry_run=True should return compressed entries without writing."""
        mock_llm.return_value = json.dumps(["Compressed entry 1", "Compressed entry 2"])
        mock_beam = MagicMock()
        mock_beam.get_working_memories = MagicMock(return_value=[
            {"id": "1", "content": "Alice works at Acme Corp."},
            {"id": "2", "content": "Bob works at Beta Inc."},
            {"id": "3", "content": "Alice is a software engineer."},
        ])
        mock_mem = MagicMock()
        mock_mem.beam = mock_beam
        mock_create.return_value = mock_mem

        from kensei_super_memory_mcp import _handle_simplemem_compress

        result = _handle_simplemem_compress(
            bank="test_bank", target="working", dry_run=True
        )
        assert result["status"] == "ok"
        assert "DRY RUN" in result["message"]
        assert result["entries_before"] == 3
        assert result["entries_after"] == 2
        assert result["compressed"] == 1
        # Verify no write happened
        mock_beam.clear_working.assert_not_called()
        mock_mem.remember.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests: _call_llm
# ---------------------------------------------------------------------------


class TestCallLlm:
    def test_ollama_fallback(self):
        """When auxiliary client is unavailable, should try Ollama fallback."""
        from kensei_super_memory_mcp import _call_llm

        # With no Ollama running, this should return None gracefully
        result = _call_llm("test prompt", max_tokens=10, timeout=2)
        assert result is None


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify the FastMCP app registers all 4 tools correctly."""

    def test_create_app_imports(self):
        """create_app() should raise RuntimeError if MCP SDK is missing."""
        from kensei_super_memory_mcp import create_app

        # Both MCP and Mnemosyne are available in test env, so this should work
        app = create_app()
        assert app is not None
        assert app.name == "kensei-super-memory"

    def test_tool_names_registered(self):
        """All 4 tools should be registered on the FastMCP app."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        # FastMCP stores tools in _tool_manager
        tool_names = {t.name for t in app._tool_manager._tools.values()}
        assert "mem0_extract" in tool_names
        assert "cognee_cognify" in tool_names
        assert "zep_temporal_query" in tool_names
        assert "simplemem_compress" in tool_names

    def test_tool_descriptions_non_empty(self):
        """Each tool should have a meaningful description."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        for tool in app._tool_manager._tools.values():
            assert tool.description, f"Tool {tool.name} has empty description"
            assert len(tool.description) > 20, (
                f"Tool {tool.name} description too short"
            )

    def test_tool_parameters_defined(self):
        """Each tool should have the expected parameters."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        tools = {t.name: t for t in app._tool_manager._tools.values()}

        # mem0_extract should have text, bank, source, confidence
        mem0_params = set(tools["mem0_extract"].parameters.get("properties", {}).keys())
        assert "text" in mem0_params
        assert "bank" in mem0_params
        assert "confidence" in mem0_params

        # zep_temporal_query should have subject, predicate, object, as_of
        zep_params = set(tools["zep_temporal_query"].parameters.get("properties", {}).keys())
        assert "subject" in zep_params
        assert "predicate" in zep_params
        assert "object" in zep_params
        assert "as_of" in zep_params
        assert "include_invalidated" in zep_params

        # simplemem_compress should have bank, target, max_entries, dry_run
        compress_params = set(tools["simplemem_compress"].parameters.get("properties", {}).keys())
        assert "bank" in compress_params
        assert "target" in compress_params
        assert "max_entries" in compress_params
        assert "dry_run" in compress_params


# ---------------------------------------------------------------------------
# Integration tests: end-to-end through FastMCP
# ---------------------------------------------------------------------------


class TestMcpIntegration:
    """End-to-end tests calling tools through FastMCP's tool manager."""

    def test_mem0_extract_tool_returns_json(self):
        """Calling mem0_extract with empty text should return JSON error."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        tool = app._tool_manager._tools["mem0_extract"]

        # Call with empty text
        result = tool.fn(text="")
        parsed = json.loads(result)
        assert parsed["status"] == "error"

    def test_zep_temporal_query_tool_returns_json(self):
        """Calling zep_temporal_query should return valid JSON."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        tool = app._tool_manager._tools["zep_temporal_query"]

        result = tool.fn(bank="test_bank")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert "count" in parsed
        assert "results" in parsed

    def test_simplemem_compress_tool_returns_json(self):
        """Calling simplemem_compress should return valid JSON."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        tool = app._tool_manager._tools["simplemem_compress"]

        result = tool.fn(bank="test_bank", target="working")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"

    def test_cognee_cognify_tool_returns_json(self):
        """Calling cognee_cognify with empty text should return JSON error."""
        from kensei_super_memory_mcp import create_app

        app = create_app()
        tool = app._tool_manager._tools["cognee_cognify"]

        result = tool.fn(text="")
        parsed = json.loads(result)
        assert parsed["status"] == "error"


# ---------------------------------------------------------------------------
# CLI entry point tests
# ---------------------------------------------------------------------------


class TestCliEntryPoint:
    def test_main_creates_app(self):
        """main() should create the app and call mcp.run()."""
        from kensei_super_memory_mcp import main

        with patch("kensei_super_memory_mcp.create_app") as mock_create, \
             patch("kensei_super_memory_mcp.logger") as mock_logger:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            with patch("sys.argv", ["kensei-super-memory-mcp.py"]):
                main()

            mock_create.assert_called_once()
            mock_app.run.assert_called_once_with(transport="stdio")

    def test_main_sse_mode(self):
        """--sse flag should use SSE transport."""
        from kensei_super_memory_mcp import main

        with patch("kensei_super_memory_mcp.create_app") as mock_create, \
             patch("kensei_super_memory_mcp.logger") as mock_logger:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            with patch("sys.argv", ["kensei-super-memory-mcp.py", "--sse", "--port", "8082"]):
                main()

            mock_app.run.assert_called_once_with(
                transport="sse", host="127.0.0.1", port=8082
            )
