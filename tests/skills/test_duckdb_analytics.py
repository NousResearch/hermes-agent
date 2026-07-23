"""Smoke tests for skills/data-science/duckdb-analytics/scripts/duckdb_run.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_duckdb_run():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "skills/data-science/duckdb-analytics/scripts/duckdb_run.py"
    spec = importlib.util.spec_from_file_location("duckdb_run", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_simple_select_json(capsys):
    pytest.importorskip("duckdb")
    mod = _load_duckdb_run()
    assert mod.main(["-q", "SELECT 40 + 2 AS x", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "42" in captured.out and "x" in captured.out


def test_parquet_alias_view(tmp_path, capsys):
    pytest.importorskip("duckdb")
    import duckdb as ddb

    pq = tmp_path / "sample.parquet"
    con = ddb.connect(":memory:")
    con.execute("COPY (SELECT 1 AS id, 'north' AS region) TO ? (FORMAT PARQUET)", [str(pq)])
    con.close()

    mod = _load_duckdb_run()
    alias_spec = f"t={pq}"
    assert mod.main(["--parquet", alias_spec, "-q", "SELECT region FROM t", "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert "north" in out
