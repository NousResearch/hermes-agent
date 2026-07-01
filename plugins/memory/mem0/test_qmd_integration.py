"""Integration tests for the QMD fold-in wiring inside Mem0MemoryProvider.

Drives the REAL prefetch / queue_prefetch / handle_tool_call paths with the mem0 client
and qmd_recall.qmd_query stubbed (no live daemon, no live mem0). Run from the repo root:
  venv/bin/python -m pytest plugins/memory/mem0/test_qmd_integration.py -v -o addopts=""
"""
import json
import time

import pytest

from plugins.memory.mem0 import qmd_recall
import plugins.memory.mem0 as mem0pkg
from plugins.memory.mem0 import Mem0MemoryProvider, SEARCH_SCHEMA


class _StubClient:
    def __init__(self, rows):
        self._rows = rows

    def search(self, **kwargs):
        return list(self._rows)


def _provider(qmd_enabled=False, mem0_rows=None, qmd_hits=None, monkeypatch=None,
              prefetch_enabled=True, search_enabled=True):
    p = Mem0MemoryProvider()
    # minimal init state (skip initialize() network/config machinery)
    p._config = {}
    p._rerank = False
    p._keyword_search = None
    p._temporal_search = False
    p._consecutive_failures = 0
    p._breaker_open_until = 0
    p._qmd_cfg = qmd_recall.load_qmd_config(
        {"enabled": qmd_enabled, "prefetch_enabled": prefetch_enabled, "search_enabled": search_enabled}
    )
    p._qmd_enabled = qmd_enabled
    # mirror initialize()'s sub-lane derivation
    p._qmd_prefetch_enabled = qmd_enabled and prefetch_enabled
    p._qmd_search_enabled = qmd_enabled and search_enabled
    p._get_client = lambda: _StubClient(mem0_rows or [])
    # neutralize forgotten-filter + read_filters so the stub rows flow through
    p._drop_forgotten = lambda rows: rows
    p._read_filters = lambda: {}
    if monkeypatch is not None:
        monkeypatch.setattr(qmd_recall, "qmd_query", lambda *a, **k: list(qmd_hits or []))
    return p


_HIT = {"file": "obsidian/DNS-PRD.md", "title": "DNS Block Portal", "score": 0.93, "line": 1, "docid": "#ec41f3"}


def _run_prefetch(p, query):
    p.queue_prefetch(query)
    if p._prefetch_thread:
        p._prefetch_thread.join(timeout=5)
    return p.prefetch(query)


# ---- AC1: disabled output byte-identical to legacy shape ------------------
def test_prefetch_disabled_is_legacy_shape(monkeypatch):
    p = _provider(qmd_enabled=False, mem0_rows=[{"memory": "fact one"}], monkeypatch=monkeypatch)
    out = _run_prefetch(p, "where did we decide the dns split")
    assert out == "## Mem0 Memory\n- fact one"  # exactly the pre-change render


def test_prefetch_disabled_empty_is_empty(monkeypatch):
    p = _provider(qmd_enabled=False, mem0_rows=[], monkeypatch=monkeypatch)
    assert _run_prefetch(p, "where did we decide the dns split") == ""


# ---- AC2: lookup prefetch injects both blocks -----------------------------
def test_prefetch_lookup_injects_both(monkeypatch):
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one"}], qmd_hits=[_HIT], monkeypatch=monkeypatch)
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Mem0 Memory\n- fact one" in out
    assert "## Local Docs (QMD)" in out
    assert "obsidian/DNS-PRD.md" in out
    assert out.index("## Mem0 Memory") < out.index("## Local Docs")  # mem0 first


# ---- AC3: non-lookup utterance skips QMD ----------------------------------
def test_prefetch_non_lookup_skips_qmd(monkeypatch):
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return [_HIT]
    monkeypatch.setattr(qmd_recall, "qmd_query", _spy)
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one"}])
    p._qmd_enabled = True
    out = _run_prefetch(p, "ship it")
    assert calls["n"] == 0                 # gate short-circuited before the call
    assert out == "## Mem0 Memory\n- fact one"


# ---- AC9: slow QMD never drops the mem0 block -----------------------------
def test_prefetch_slow_qmd_keeps_mem0(monkeypatch):
    def _slow(*a, **k):
        time.sleep(3.0)
        return [_HIT]
    monkeypatch.setattr(qmd_recall, "qmd_query", _slow)
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one"}])
    p._qmd_enabled = True
    p.queue_prefetch("where did we decide the local dns split")
    # the mem0 block is committed before QMD runs; read it without waiting for the slow leg
    time.sleep(0.5)
    out = p.prefetch("where did we decide the local dns split")
    assert "## Mem0 Memory\n- fact one" in out  # mem0 present despite slow QMD


# ---- AC4: mem0_search additive docs key -----------------------------------
def test_search_adds_docs_key(monkeypatch):
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}],
                  qmd_hits=[_HIT], monkeypatch=monkeypatch)
    out = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "results" in out and "docs" in out
    assert out["results"][0]["memory"] == "fact one"
    assert out["docs"][0]["file"] == "obsidian/DNS-PRD.md"


def test_search_disabled_is_legacy_shape(monkeypatch):
    p = _provider(qmd_enabled=False, mem0_rows=[{"memory": "fact one", "score": 0.9}], monkeypatch=monkeypatch)
    out = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert out == {"results": [{"memory": "fact one", "score": 0.9}], "count": 1}  # no docs key


def test_search_qmd_fail_keeps_mem0(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("qmd down")
    # _qmd_pointers swallows; but prove the search still returns mem0 even if qmd raises
    monkeypatch.setattr(qmd_recall, "qmd_query", _boom)
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}])
    p._qmd_enabled = True
    out = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert out["results"][0]["memory"] == "fact one"
    assert "docs" not in out  # qmd failed -> no docs, mem0 intact


def test_provider_forwards_semantic_floor_config(monkeypatch):
    seen = {}

    def _spy(*a, **k):
        seen.update(k)
        return [_HIT]

    monkeypatch.setattr(qmd_recall, "qmd_query", _spy)
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}])
    p._qmd_enabled = True
    p._qmd_cfg = qmd_recall.load_qmd_config({
        "enabled": True,
        "use_rerank_score_floor": True,
        "rerank_score_min": 0.62,
    })
    json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert seen["use_rerank_score_floor"] is True
    assert seen["rerank_score_min"] == 0.62


# ---- AC12 / INV-4a: mem0 over its own budget => QMD leg skipped ----------
def test_mem0_over_budget_skips_qmd(monkeypatch):
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return [_HIT]
    monkeypatch.setattr(qmd_recall, "qmd_query", _spy)

    class _SlowClient:
        def search(self, **k):
            time.sleep(1.2)            # exceed the tiny mem0_budget below
            return [{"memory": "slow fact"}]
    p = Mem0MemoryProvider()
    p._config = {}; p._rerank = False; p._keyword_search = None; p._temporal_search = False
    p._consecutive_failures = 0; p._breaker_open_until = 0
    p._drop_forgotten = lambda r: r; p._read_filters = lambda: {}
    p._qmd_cfg = qmd_recall.load_qmd_config({"enabled": True, "mem0_budget_s": 0.5})
    p._qmd_enabled = True
    p._get_client = lambda: _SlowClient()
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Mem0 Memory" in out          # mem0 still returned
    assert calls["n"] == 0                   # QMD skipped: mem0 blew its budget
    assert "## Local Docs" not in out


# ---- INV-1: no QMD output ever written back to mem0 -----------------------
def test_no_write_path_from_qmd(monkeypatch):
    writes = {"n": 0}

    class _WriteSpyClient:
        def search(self, **k):
            return [{"memory": "fact"}]

        def add(self, *a, **k):
            writes["n"] += 1

        def update(self, *a, **k):
            writes["n"] += 1
    monkeypatch.setattr(qmd_recall, "qmd_query", lambda *a, **k: [_HIT])
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact"}])
    p._qmd_enabled = True
    p._get_client = lambda: _WriteSpyClient()
    _run_prefetch(p, "where did we decide the local dns split")
    json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert writes["n"] == 0                   # QMD recall never triggers a mem0 write


# ---- INV-8 / AC7: tool schema byte-unchanged ------------------------------
def test_search_schema_unchanged():
    # the model-facing schema must not gain a field (prompt-cache safety)
    props = SEARCH_SCHEMA.get("function", SEARCH_SCHEMA).get("parameters", {}).get("properties", {})
    assert "docs" not in props
    assert "qmd" not in props


# ---- sub-lane toggles: prefetch_enabled / search_enabled ------------------
def test_prefetch_off_search_on(monkeypatch):
    """prefetch_enabled:false kills the every-turn QMD block but keeps mem0_search docs."""
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}],
                  qmd_hits=[_HIT], monkeypatch=monkeypatch,
                  prefetch_enabled=False, search_enabled=True)
    # prefetch: NO QMD block
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Mem0 Memory\n- fact one" in out
    assert "## Local Docs (QMD)" not in out
    # search: STILL fans out to QMD docs
    s = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "docs" in s and s["docs"][0]["file"] == "obsidian/DNS-PRD.md"


def test_search_off_prefetch_on(monkeypatch):
    """search_enabled:false kills the mem0_search docs fan-out but keeps prefetch injection."""
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}],
                  qmd_hits=[_HIT], monkeypatch=monkeypatch,
                  prefetch_enabled=True, search_enabled=False)
    # prefetch: QMD block present
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Local Docs (QMD)" in out
    # search: NO docs key (legacy shape)
    s = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "docs" not in s


def test_master_off_ignores_sub_gates(monkeypatch):
    """enabled:false wins even if the sub-gates are true — both lanes off."""
    p = _provider(qmd_enabled=False, mem0_rows=[{"memory": "fact one", "score": 0.9}],
                  qmd_hits=[_HIT], monkeypatch=monkeypatch,
                  prefetch_enabled=True, search_enabled=True)
    assert p._qmd_prefetch_enabled is False
    assert p._qmd_search_enabled is False
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Local Docs (QMD)" not in out
    s = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "docs" not in s


def test_sub_gates_default_true(monkeypatch):
    """flipping only enabled:true (sub-gates unspecified) → both lanes on (backward-compat)."""
    p = _provider(qmd_enabled=True, mem0_rows=[{"memory": "fact one", "score": 0.9}],
                  qmd_hits=[_HIT], monkeypatch=monkeypatch)  # prefetch/search default True
    assert p._qmd_prefetch_enabled is True
    assert p._qmd_search_enabled is True
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert "## Local Docs (QMD)" in out
    s = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "docs" in s
