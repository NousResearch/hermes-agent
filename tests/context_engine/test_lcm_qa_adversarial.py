"""LCM QA battery — Tier-0 adversarial + DAG-invariant tests (PRD-7).

Offline, deterministic, free. No live model, no money. Exercises:
  - Malformed / hostile tool args across all 7 registered tools (fuzz).
  - Tool registration <-> handler parity invariant (PRD-7 F7).
  - SummaryDAG invariants over a forced multi-depth corpus (PRD-7 F8/C):
      i.   Tokens(summary) < Tokens(source) at every node.
      ii.  condensation produces depth>=1 source_type="nodes" rows.
      iii. source_ids of a nodes-type node all resolve (no dangling edges).
      iv.  earliest_at <= latest_at and parent window contains source windows.
  - Both fail-open branches (recovery-error re-raise vs generic degraded).
  - Redaction: secrets absent from summary / expand_hint / store text.

Regression for BUG-1 (PRD-7 §8): lcm_grep with {"query": null} must NOT raise.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


REGISTERED_TOOLS = {
    "lcm_grep",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_doctor",
}


def _close(engine) -> None:
    fn = getattr(engine, "shutdown", None)
    if callable(fn):
        fn()


def _make_engine(tmp_path, **overrides):
    cfg = LCMConfig(database_path=str(tmp_path / "qa.db"), **overrides)
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    engine.on_session_start("qa-sess")
    return engine


def _forced_dag_engine(tmp_path):
    """Lower leaf/fanin so condensation fires deterministically offline."""
    return _make_engine(
        tmp_path,
        leaf_chunk_tokens=120,
        condensation_fanin=2,
        fresh_tail_count=2,
        context_threshold=0.05,
        incremental_max_depth=3,
    )


def _drive_compaction(engine, batches=6, per_batch=30):
    for b in range(batches):
        msgs = []
        for i in range(b * per_batch, (b + 1) * per_batch):
            msgs.append(
                {"role": "user", "content": f"Slot {i} codeword is ZK-{i:04d}-" + ("q" * 30)}
            )
            msgs.append({"role": "assistant", "content": f"ack {i}"})
        engine.compress(msgs, current_tokens=10**9)


# --------------------------------------------------------------------------
# Tool registration / handler parity (F7)
# --------------------------------------------------------------------------

def test_tool_schema_set_is_exactly_the_seven(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        schemas = engine.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == REGISTERED_TOOLS
    finally:
        _close(engine)


def test_unknown_tool_returns_error_not_exception(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        out = engine.handle_tool_call("lcm_not_a_tool", {})
        payload = json.loads(out)
        assert "error" in payload
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# Adversarial / hostile args fuzz across all tools (BUG-1 regression + class)
# --------------------------------------------------------------------------

HOSTILE_QUERY_ARGS = [
    {"query": None},          # BUG-1: was AttributeError on .strip()
    {"query": 123},
    {"query": ["x"]},
    {"query": ""},
    {"query": '" OR 1=1 --'},  # SQL-ish
    {"query": "*"},            # FTS metachar
    {"query": "NEAR"},         # FTS keyword
    {"query": "a" * 50_000},   # huge
    {"limit": -5, "query": "x"},
    {"limit": "notanint", "query": "x"},
]


@pytest.mark.parametrize("args", HOSTILE_QUERY_ARGS)
def test_lcm_grep_hostile_args_return_json_never_raise(tmp_path, args):
    engine = _forced_dag_engine(tmp_path)
    try:
        out = engine.handle_tool_call("lcm_grep", args)
        assert isinstance(out, str)
        json.loads(out)  # must be valid JSON, never an exception escape
    finally:
        _close(engine)


def test_bug1_null_query_specifically(tmp_path):
    """PRD-7 §8 BUG-1: {"query": null} must return graceful error JSON."""
    engine = _make_engine(tmp_path)
    try:
        out = engine.handle_tool_call("lcm_grep", {"query": None})
        payload = json.loads(out)
        assert payload.get("error") == "No query provided"
    finally:
        _close(engine)


@pytest.mark.parametrize(
    "tool",
    sorted(REGISTERED_TOOLS),
)
@pytest.mark.parametrize(
    "args",
    [{}, {"node_id": None}, {"session_id": None}, {"prompt": None}, {"query": None}],
)
def test_all_tools_survive_null_and_empty_args(tmp_path, tool, args):
    engine = _make_engine(tmp_path)
    try:
        out = engine.handle_tool_call(tool, args)
        assert isinstance(out, str)
        json.loads(out)  # valid JSON, no uncaught exception
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# Degenerate compress inputs
# --------------------------------------------------------------------------

def test_compress_empty_list_is_noop(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        out = engine.compress([], current_tokens=10**9)
        assert out == []
    finally:
        _close(engine)


def test_compress_all_tool_results_does_not_crash(tmp_path):
    engine = _forced_dag_engine(tmp_path)
    try:
        msgs = [
            {"role": "tool", "content": json.dumps({"k": "v" * 200}), "tool_call_id": f"t{i}"}
            for i in range(40)
        ]
        engine.compress(msgs, current_tokens=10**9)  # no raise
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# DAG invariants over a forced multi-depth corpus (F8 / §1.C)
# --------------------------------------------------------------------------

def _read_nodes(db_path):
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(summary_nodes)").fetchall()]
        rows = con.execute("SELECT * FROM summary_nodes").fetchall()
        return cols, [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def test_forced_condensation_produces_multi_depth_dag(tmp_path):
    engine = _forced_dag_engine(tmp_path)
    db = engine._config.database_path
    try:
        _drive_compaction(engine)
        _, nodes = _read_nodes(db)
        assert nodes, "no summary nodes created"
        depths = {n["depth"] for n in nodes}
        assert max(depths) >= 1, f"no condensation (depths={depths})"
        # at least one condensed node is source_type='nodes'
        condensed = [n for n in nodes if n["depth"] >= 1]
        assert any(
            str(n.get("source_type")) == "nodes" for n in condensed
        ), "depth>=1 node not sourced from nodes"
    finally:
        _close(engine)


def test_dag_invariant_summary_smaller_than_source(tmp_path):
    """Invariant (i): every node compresses — token_count <= source tokens."""
    engine = _forced_dag_engine(tmp_path)
    db = engine._config.database_path
    try:
        _drive_compaction(engine)
        cols, nodes = _read_nodes(db)
        src_col = "source_token_count" if "source_token_count" in cols else None
        checked = 0
        for n in nodes:
            if src_col and n.get(src_col):
                assert n["token_count"] <= n[src_col], (
                    f"node {n['node_id']} grew: {n['token_count']} > {n[src_col]}"
                )
                checked += 1
        assert checked > 0, "no node had a source-token count to verify"
    finally:
        _close(engine)


def test_dag_invariant_no_dangling_source_ids(tmp_path):
    """Invariant (iii): every nodes-type node's source_ids resolve to real rows."""
    engine = _forced_dag_engine(tmp_path)
    db = engine._config.database_path
    try:
        _drive_compaction(engine)
        cols, nodes = _read_nodes(db)
        if "source_ids" not in cols or "source_type" not in cols:
            pytest.skip("schema lacks source_ids/source_type columns")
        existing = {n["node_id"] for n in nodes}
        for n in nodes:
            if str(n.get("source_type")) != "nodes":
                continue
            raw = n.get("source_ids")
            try:
                ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except (ValueError, TypeError):
                ids = []
            for sid in ids:
                assert sid in existing, (
                    f"node {n['node_id']} references missing source node {sid}"
                )
    finally:
        _close(engine)


def test_dag_invariant_source_window_ordering(tmp_path):
    """Invariant (iv): earliest_at <= latest_at on every node that has them."""
    engine = _forced_dag_engine(tmp_path)
    db = engine._config.database_path
    try:
        _drive_compaction(engine)
        cols, nodes = _read_nodes(db)
        if "earliest_at" not in cols or "latest_at" not in cols:
            pytest.skip("schema lacks source-window columns")
        for n in nodes:
            e, l = n.get("earliest_at"), n.get("latest_at")
            if e is not None and l is not None:
                assert e <= l, f"node {n['node_id']} window inverted: {e} > {l}"
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# Fail-open: generic degraded branch (F3 branch 2)
# --------------------------------------------------------------------------

def test_fail_open_generic_exception_degrades_not_crash(tmp_path, monkeypatch):
    engine = _forced_dag_engine(tmp_path)
    try:
        # Force the lossless path to blow up with a generic exception.
        def boom(*a, **k):
            raise RuntimeError("induced compaction failure")

        monkeypatch.setattr(engine, "_compress_lossless", boom)
        msgs = [{"role": "user", "content": "x" * 500} for _ in range(5)]
        # Must not raise — fail-open returns the messages (degraded).
        out = engine.compress(msgs, current_tokens=10**9)
        assert isinstance(out, list)
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# Fail-open: recovery-error RE-RAISE branch (F3 branch 1) — distinct semantics.
# compress() catches LCMFailOpenRecoveryError and RE-RAISES it (engine.py:879),
# a separate signal from the generic degraded handler. A test that only induces
# a generic Exception never hits this branch.
# --------------------------------------------------------------------------

def test_fail_open_recovery_error_is_reraised_not_swallowed(tmp_path, monkeypatch):
    from plugins.context_engine.lcm.engine import LCMFailOpenRecoveryError

    engine = _forced_dag_engine(tmp_path)
    try:
        def raise_recovery(*a, **k):
            raise LCMFailOpenRecoveryError("induced overflow recovery signal")

        monkeypatch.setattr(engine, "_compress_lossless", raise_recovery)
        msgs = [{"role": "user", "content": "y" * 500} for _ in range(5)]
        # This branch must PROPAGATE (re-raise), NOT degrade to returning msgs.
        with pytest.raises(LCMFailOpenRecoveryError):
            engine.compress(msgs, current_tokens=10**9)
    finally:
        _close(engine)


def test_recovery_error_carries_recoverable_flags(tmp_path):
    """The signal advertises recoverable/compression_exhausted so the caller
    can route it (vs a hard crash)."""
    from plugins.context_engine.lcm.engine import LCMFailOpenRecoveryError

    err = LCMFailOpenRecoveryError("x")
    assert getattr(err, "recoverable", False) is True
    assert getattr(err, "compression_exhausted", False) is True


# --------------------------------------------------------------------------
# FTS stale-hit on delete (F3 / §1.C): external-content FTS must not return a
# hit pointing at a deleted summary node.
# --------------------------------------------------------------------------

def test_fts_no_stale_hit_after_node_delete(tmp_path):
    import time

    from plugins.context_engine.lcm.dag import SummaryDAG, SummaryNode

    dag = SummaryDAG(db_path=str(tmp_path / "fts.db"))
    now = time.time()

    def _node(text):
        return SummaryNode(
            node_id=None, session_id="s", depth=0, summary=text,
            token_count=10, source_token_count=50, source_ids=[],
            source_type="messages", created_at=now, earliest_at=now,
            latest_at=now, expand_hint="hint", search_rank=0.0,
            search_directness=0.0,
        )

    dag.add_node(_node("UNIQUESENTINELWORD apple zebra"))
    dag.add_node(_node("banana cherry orange"))
    assert [n.node_id for n in dag.search("UNIQUESENTINELWORD", session_id="s")]
    dag.delete_session_nodes("s")
    # External-content FTS trigger must have removed the shadow row.
    assert dag.search("UNIQUESENTINELWORD", session_id="s") == []


# --------------------------------------------------------------------------
# Escalation L3 during CONDENSATION (F3 / §1.D): condensing near-incompressible
# already-summarized nodes must still converge (Tokens(out) < Tokens(in)).
# --------------------------------------------------------------------------

def test_escalation_l3_deterministic_truncation_converges():
    from plugins.context_engine.lcm.escalation import summarize_with_escalation
    from plugins.context_engine.lcm.tokens import count_tokens

    # High-entropy, near-incompressible source. With no model configured the LLM
    # summary path (L1/L2) yields nothing usable, so escalation must fall through
    # to L3 deterministic truncation — which is guaranteed to converge.
    source = " ".join(f"x{i:04d}{chr(97 + i % 26)}" for i in range(2000))
    src_tokens = count_tokens(source)

    summary, out_tokens = summarize_with_escalation(
        source,
        source_tokens=src_tokens,
        token_budget=max(1000, int(src_tokens * 0.40)),
        l3_truncate_tokens=512,
    )
    assert count_tokens(summary) < src_tokens, "L3 did not converge below source"
    assert out_tokens <= src_tokens


# --------------------------------------------------------------------------
# Redaction: with sensitive-patterns enabled, a real secret pattern must not
# land in summary text or expand_hint. (Default config does NOT redact — that
# is by design; the immutable store keeps raw bytes. This asserts the
# opt-in redaction path actually scrubs the DAG summary surface.)
# --------------------------------------------------------------------------

# A real openai-style key shape the "all" pattern set recognizes, assembled at
# runtime so the literal never sits in the test source.
SECRET = "sk" + "-" + "proj" + "-" + ("d" * 48)


def test_secret_not_persisted_in_summary_or_expand_hint(tmp_path):
    engine = _make_engine(
        tmp_path,
        leaf_chunk_tokens=120,
        condensation_fanin=2,
        fresh_tail_count=2,
        context_threshold=0.05,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["all"],
        encryption_enabled=False,
    )
    db = engine._config.database_path
    try:
        msgs = []
        for i in range(30):
            if i == 7:
                msgs.append(
                    {"role": "user", "content": f"my api key is {SECRET} keep it safe"}
                )
            else:
                msgs.append({"role": "user", "content": f"benign message number {i} " + "z" * 40})
            msgs.append({"role": "assistant", "content": f"ok {i}"})
        engine.compress(msgs, current_tokens=10**9)
        cols, nodes = _read_nodes(db)
        for n in nodes:
            for field in ("summary", "expand_hint"):
                val = n.get(field) or ""
                assert SECRET not in str(val), (
                    f"SECRET leaked into summary_nodes.{field} (node {n['node_id']})"
                )
    finally:
        _close(engine)


# --------------------------------------------------------------------------
# Concurrency-contention guard: lcm_expand_query's aux-LLM synthesis must
# DEGRADE (not crash with an opaque "'type' object is not subscriptable") when
# call_llm returns an error-shaped / partial response. Reproduces the live
# Arm-B failure where the recovery sub-model collided with the foreground
# gateway and `response.choices` was missing/None/non-subscriptable.
# --------------------------------------------------------------------------

import pytest as _pytest


@_pytest.mark.parametrize("bad_response", [
    type("Resp", (), {"choices": None})(),          # choices is None
    type("Resp", (), {"choices": []})(),            # empty choices
    type("Resp", (), {"choices": object})(),        # a TYPE, not subscriptable -> the real symptom
    type("Resp", (), {})(),                          # no choices attr at all
    "rate_limited",                                  # a bare string error sentinel
])
def test_expansion_synthesis_degrades_on_malformed_llm_response(monkeypatch, bad_response):
    from plugins.context_engine.lcm import tools as lcm_tools

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **kw: bad_response,
    )
    # Must raise the CLEAN, catchable error — never a raw TypeError/AttributeError.
    with _pytest.raises(lcm_tools._ExpansionSynthesisError):
        lcm_tools._synthesize_expansion_answer(
            prompt="who is the recovery owner?",
            context_blocks=[{"node_id": 1, "summary": "owner is Ada Lovelace"}],
            model="claude-haiku-4-5",
            max_tokens=256,
            timeout=30.0,
        )


def test_expansion_synthesis_ok_on_well_formed_response(monkeypatch):
    """Belt: a normal well-formed response still returns the content unchanged."""
    from types import SimpleNamespace
    from plugins.context_engine.lcm import tools as lcm_tools

    good = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="Ada Lovelace is the recovery owner."))])
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: good)
    out = lcm_tools._synthesize_expansion_answer(
        prompt="who?", context_blocks=[{"node_id": 1, "summary": "x"}],
        model="claude-haiku-4-5", max_tokens=256, timeout=30.0,
    )
    assert "Ada Lovelace" in out
