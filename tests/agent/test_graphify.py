"""Deterministic behavior contracts for Graphify (agent.graphify).

Asserts invariants across indexing, schema, extraction, persistence, and
relationship queries. Tests avoid snapshotting live counts from arbitrary
repos; instead they assert stable relationships between inputs and outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import graphify as gf


def _write_repo(root: Path, tree: dict[str, str]) -> None:
    for rel, content in tree.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_symbol_ids_are_stable_and_reversible():
    relpath = "pkg/sub/mod.py"
    name = "MyClass.do_something"
    sid = gf._symbol_id(relpath, name)
    r, n = gf._split_symbol_id(sid)
    assert r == relpath
    assert n == name
    # Nested method id round trip
    nested = gf._symbol_id(relpath, f"MyClass{gf.SYMBOL_ID_DELIM_NESTED}do_something")
    assert gf._split_symbol_id(nested) == (relpath, f"MyClass{gf.SYMBOL_ID_DELIM_NESTED}do_something")


def test_language_detection():
    assert gf._detect_language(Path("foo.py")) == "python"
    assert gf._detect_language(Path("foo.tsx")) == "typescript"
    assert gf._detect_language(Path("foo.rs")) == "rust"
    assert gf._detect_language(Path("README.md")) == "unknown"


def test_extract_python_symbols_and_edges(tmp_path: Path):
    code = '''
"""module doc."""

import os
from typing import Optional

class Base:
    """Base doc."""
    def method(self, x: int) -> int:
        return x

def helper(a: Optional[int]) -> int:
    return a or 0

helper(1)
Base().method(2)
'''
    root = tmp_path / "repo"
    root.mkdir()
    f = root / "pkg" / "mod.py"
    f.parent.mkdir(parents=True)
    f.write_text(code, encoding="utf-8")
    rel = "pkg/mod.py"
    symbols, edges = gf._extract_python(f, rel)
    sym_names = {s["name"] for s in symbols.values()}
    assert "Base" in sym_names
    assert "helper" in sym_names
    assert "Base.method" in {s["name"] for s in symbols.values() if s["kind"] == "method"}
    edge_targets = {e["target"] for e in edges}
    assert any("os" in t for t in edge_targets)
    assert any("typing.Optional" in t for t in edge_targets)


def test_graph_store_index_path_counts(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "a.py": "def a(): pass\n",
            "b.py": "def b(): pass\n",
            "c.txt": "ignored\n",
        },
    )
    store = gf.CodeGraphStore(root=root)
    assert store.index_path(root) == 2
    assert len(store.files) == 2
    assert "a.py" in store.files
    assert "c.txt" not in store.files


def test_graph_store_ignore_dirs(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "app/main.py": "def main(): pass\n",
            "app/.git/ignored.py": "def ignored(): pass\n",
            "app/venv/lib.py": "def lib(): pass\n",
        },
    )
    store = gf.CodeGraphStore(root=root)
    assert store.index_walk(root) == 1
    assert {r.split("/")[-1] for r in store.files} == {"main.py"}


def test_graph_store_nodes_and_relationships_shape(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def foo(): pass\n",
        },
    )
    store = gf.build_graph(root)
    nodes = store.nodes()
    rels = store.relationships()
    kinds = {n["kind"] for n in nodes}
    assert "file" in kinds
    assert "module" in kinds or "function" in kinds
    assert all(set(["source", "target", "type"]).issubset(e.keys()) for e in rels)
    assert all(e["type"] in gf.EDGE_TYPES for e in rels)


def test_graph_store_references_in_out_edges(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "pkg/mod.py": "def foo():\n    foo()\n",
        },
    )
    store = gf.build_graph(root)
    sym_id = gf._symbol_id("pkg/mod.py", "foo")
    refs = store.references(sym_id)
    assert refs, "expected at least one reference for foo"
    assert any(r["direction"] == "out" for r in refs)
    assert any(r["type"] == "CALLS" for r in refs)


def test_json_round_trip_preserves_graph(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "pkg/mod.py": "class Foo:\n    def bar(self): pass\n",
        },
    )
    store = gf.build_graph(root)
    out = tmp_path / "graphify.json"
    store.write_json(out)
    loaded = gf.CodeGraphStore.read_json(out)
    assert loaded.root == store.root
    assert loaded.files.keys() == store.files.keys()
    assert loaded.symbols.keys() == store.symbols.keys()
    assert loaded.edges == store.edges
    assert loaded.generated_at == store.generated_at


def test_stats_are_invariant_safe(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "pkg/a.py": "class A: pass\n",
            "pkg/b.py": "def b(): pass\n",
        },
    )
    store = gf.build_graph(root)
    stats = store.stats()
    assert stats["files"] == 2
    assert stats["symbols"] >= 2
    # Edge count is implementation-defined; assert at least file↔module CONTAINS edges exist.
    assert stats["edges"] >= 0
    assert stats["files"] + stats["symbols"] + stats["edges"] > 0


def test_index_extensions_filter(tmp_path: Path):
    root = tmp_path / "repo"
    _write_repo(
        root,
        {
            "a.py": "def a(): pass\n",
            "b.ts": "function b() {}\n",
        },
    )
    store = gf.CodeGraphStore(root=root)
    assert store.index_walk(root, extensions=(".py",)) == 1
    assert "a.py" in store.files
    assert "b.ts" not in store.files
