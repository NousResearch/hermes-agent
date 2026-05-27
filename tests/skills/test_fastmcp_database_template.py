from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest


class _FakeFastMCP:
    def __init__(self, _name: str) -> None:
        pass

    def tool(self, fn):
        return fn


@pytest.fixture
def template_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=_FakeFastMCP))
    path = Path("optional-skills/mcp/fastmcp/templates/database_server.py")
    spec = importlib.util.spec_from_file_location("database_server_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_interrupts_expensive_select(template_module):
    conn = sqlite3.connect(":memory:")
    template_module._connect = lambda: conn
    template_module.MAX_PROGRESS_CALLS = 1
    template_module.PROGRESS_GRANULARITY = 1

    with pytest.raises(ValueError, match="execution budget"):
        template_module.query(
            "SELECT (WITH RECURSIVE cnt(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM cnt WHERE x<1000000) SELECT sum(x) FROM cnt)",
            limit=1,
        )


def test_query_returns_limited_rows(template_module):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.executemany("INSERT INTO t (id) VALUES (?)", [(1,), (2,), (3,)])
    template_module._connect = lambda: conn
    template_module.MAX_PROGRESS_CALLS = 10000
    template_module.PROGRESS_GRANULARITY = 1000

    result = template_module.query("SELECT id FROM t ORDER BY id", limit=2)

    assert result["limit"] == 2
    assert result["columns"] == ["id"]
    assert result["rows"] == [{"id": 1}, {"id": 2}]
