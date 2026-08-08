"""Tests for LSP query tools (goToDefinition, findReferences, hover, etc.).

Tests verify:
1. The mock LSP server responds correctly to query requests
2. The LSPService.query_lsp_sync() bridge works end-to-end
3. The tools/lsp_tools.py handlers produce correctly formatted output
4. Graceful degradation when LSP is unavailable
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _patch_server(script: str = "query"):
    """Return a context manager that patches ``pyright`` with the mock server.

    ``script`` is passed as ``MOCK_LSP_SCRIPT`` to the mock, which
    determines what behaviours it exposes.  We use ``"query"`` for the
    query-capable mock.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[target_index]

    # Always resolve mock server path relative to THIS file (test_query_tools.py)
    mock_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mock_lsp_server.py")

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, mock_server_path],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock pyright (query-capable)",
    )

    class _Patch:
        def __enter__(self):
            SERVERS[target_index] = replacement
            return self

        def __exit__(self, *args):
            SERVERS[target_index] = original

    return _Patch()


@pytest.fixture
def mock_query_server(tmp_path, monkeypatch):
    """Set up a minimal git workspace + mock pyright that responds to queries."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    monkeypatch.chdir(str(repo))

    with _patch_server("query"):
        yield repo


@pytest.fixture
def service(mock_query_server):
    """Create an LSPService pointed at the mock workspace."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=10.0,
        install_strategy="auto",
        binary_overrides={},
        env_overrides={},
        init_overrides={},
    )
    yield svc
    svc.shutdown()


# ---------------------------------------------------------------------------
# Test: LSPService.query_lsp_sync bridge
# ---------------------------------------------------------------------------


def test_go_to_definition(service, mock_query_server):
    """query_lsp_sync with textDocument/definition returns definition locations."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("class Foo:\n    pass\n")

    result = service.query_lsp_sync(
        str(test_file),
        "textDocument/definition",
        {
            "textDocument": {"uri": "file://" + str(test_file)},
            "position": {"line": 0, "character": 6},
        },
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1
    first = result[0]
    assert "uri" in first or "targetUri" in first
    assert "range" in first or "targetRange" in first


def test_find_references(service, mock_query_server):
    """query_lsp_sync with textDocument/references returns reference locations."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("x = 42\nprint(x)\n")

    result = service.query_lsp_sync(
        str(test_file),
        "textDocument/references",
        {
            "textDocument": {"uri": "file://" + str(test_file)},
            "position": {"line": 0, "character": 0},
            "context": {"includeDeclaration": True},
        },
    )

    assert result is not None
    assert isinstance(result, list)


def test_hover(service, mock_query_server):
    """query_lsp_sync with textDocument/hover returns type info."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("x = 42\n")

    result = service.query_lsp_sync(
        str(test_file),
        "textDocument/hover",
        {
            "textDocument": {"uri": "file://" + str(test_file)},
            "position": {"line": 0, "character": 0},
        },
    )

    assert result is not None
    # Hover result has 'contents' field
    assert "contents" in result


def test_document_symbols(service, mock_query_server):
    """query_lsp_sync with textDocument/documentSymbol returns symbol list."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("class Foo:\n    def bar(self):\n        pass\n")

    result = service.query_lsp_sync(
        str(test_file),
        "textDocument/documentSymbol",
        {"textDocument": {"uri": "file://" + str(test_file)}},
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1


def test_workspace_symbols(service, mock_query_server):
    """query_lsp_sync with workspace/symbol returns matching symbols."""
    # workspace/symbol needs a real file in the workspace to enable LSP
    test_file = mock_query_server / "test.py"
    test_file.write_text("class Foo:\n    pass\n")

    result = service.query_lsp_sync(
        str(test_file),
        "workspace/symbol",
        {"query": "Foo"},
    )

    assert result is not None
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test: tools/lsp_tools.py handlers
# ---------------------------------------------------------------------------


def test_tool_go_to_definition_handler(service, mock_query_server):
    """lsp_go_to_definition handler returns correctly formatted JSON."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("class Foo:\n    pass\n")

    # Import the handler (which uses get_service() internally)
    # We'll test via the module directly by mocking get_service()
    from tools import lsp_tools

    # Mock the LSP service in the tool's namespace so _get_lsp_service() finds it
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_go_to_definition({
            "path": str(test_file),
            "line": 0,
            "character": 6,
        })
        data = json.loads(result_str)
        assert data["error"] == ""
        assert "Definitions" in data["result"] or "no" in data["result"]
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_find_references_handler(service, mock_query_server):
    """lsp_find_references handler returns correctly formatted JSON."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("x = 42\nprint(x)\n")

    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_find_references({
            "path": str(test_file),
            "line": 1,
            "character": 6,
        })
        data = json.loads(result_str)
        assert data["error"] == ""
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_hover_handler(service, mock_query_server):
    """lsp_hover handler returns correctly formatted JSON."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("x = 42\n")

    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_hover({
            "path": str(test_file),
            "line": 0,
            "character": 0,
        })
        data = json.loads(result_str)
        assert data["error"] == ""
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_document_symbols_handler(service, mock_query_server):
    """lsp_document_symbols handler returns correctly formatted JSON."""
    test_file = mock_query_server / "test.py"
    test_file.write_text("class Foo:\n    def bar(self):\n        pass\n")

    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_document_symbols({
            "path": str(test_file),
        })
        data = json.loads(result_str)
        assert data["error"] == ""
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_workspace_symbols_handler(service, mock_query_server):
    """lsp_workspace_symbols handler returns correctly formatted JSON."""
    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_workspace_symbols({
            "query": "Foo",
        })
        data = json.loads(result_str)
        assert data["error"] == ""
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_go_to_definition_missing_path(service):
    """lsp_go_to_definition returns error when path is missing."""
    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_go_to_definition({
            "line": 0,
            "character": 0,
        })
        data = json.loads(result_str)
        assert data["error"] != ""
    finally:
        lsp_tools._get_lsp_service = orig_get


def test_tool_hover_missing_path(service):
    """lsp_hover returns error when path is missing."""
    from tools import lsp_tools
    orig_get = lsp_tools._get_lsp_service
    lsp_tools._get_lsp_service = lambda: service
    try:
        result_str = lsp_tools._handle_hover({
            "line": 0,
            "character": 0,
        })
        data = json.loads(result_str)
        assert data["error"] != ""
    finally:
        lsp_tools._get_lsp_service = orig_get


# ---------------------------------------------------------------------------
# Test: graceful degradation when LSP is unavailable
# ---------------------------------------------------------------------------


def test_query_lsp_unavailable_no_git(tmp_path, monkeypatch):
    """query_lsp_sync returns None when no git workspace is detected."""
    # No .git directory
    work_dir = tmp_path / "nows"
    work_dir.mkdir()
    monkeypatch.chdir(str(work_dir))

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=5.0,
        install_strategy="auto",
        binary_overrides={},
        env_overrides={},
        init_overrides={},
    )
    try:
        test_file = work_dir / "test.py"
        test_file.write_text("x = 1\n")
        result = svc.query_lsp_sync(
            str(test_file),
            "textDocument/definition",
            {"textDocument": {"uri": "file://" + str(test_file)}, "position": {"line": 0, "character": 0}},
        )
        assert result is None
    finally:
        svc.shutdown()


def test_query_lsp_disabled(monkeypatch, tmp_path):
    """query_lsp_sync returns None when LSP is disabled in config."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(str(repo))

    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=5.0,
        install_strategy="auto",
        binary_overrides={},
        env_overrides={},
        init_overrides={},
    )
    try:
        test_file = repo / "test.py"
        test_file.write_text("x = 1\n")
        result = svc.query_lsp_sync(
            str(test_file),
            "textDocument/definition",
            {"textDocument": {"uri": "file://" + str(test_file)}, "position": {"line": 0, "character": 0}},
        )
        assert result is None
    finally:
        svc.shutdown()