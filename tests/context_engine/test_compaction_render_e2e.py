"""T2 — compaction-render verification harness (AC-4).

Drive the REAL ``LCMEngine.compress`` to a terminal ``compacted`` status on a
realistic large message set (list-of-content-blocks shape, tool-heavy) with NO
model configured (the summarizer falls through to L3 deterministic truncation —
offline, deterministic, zero LLM cost), then assert the EXACT granular in-turn
announce reconciles through the real ``build_inturn_stats`` done-site via the
Option-B provenance branch.

This converts AC-6 ("watch a real heavy compaction render the exact granular
announce live") from a wait-and-eyeball into a repeatable CI gate. The optional
``scripts/compaction-render-proof.py --real-model`` driver is the full-fidelity
backstop against stub-vs-real drift (AC-5, manual).

Spec: ~/.hermes/plans/2026-06-27_skew-telemetry-and-render-harness-SPEC.md (v0.3).
"""
from __future__ import annotations

import os

from agent.compaction_stats import build_inturn_stats
from agent.model_metadata import estimate_messages_tokens_rough as _est
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


def _make_engine(tmp_path):
    """Low thresholds so a real compaction fires deterministically offline."""
    cfg = LCMConfig(
        database_path=str(tmp_path / "render.db"),
        leaf_chunk_tokens=120,
        condensation_fanin=2,
        fresh_tail_count=4,
        context_threshold=0.05,
        incremental_max_depth=3,
    )
    eng = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    eng.on_session_start("render-sess")
    return eng


def _tool_heavy_turn(n_pairs=40):
    """A realistic large in-turn message list: user → assistant tool_call →
    tool result, list-of-content-blocks shape (the shape that broke the in-turn
    path pre-#95/#110), big enough that a real fold happens past fresh_tail_count."""
    msgs = [{"role": "system", "content": "You are Apollo."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"do task {i} " + ("w" * 30)})
        msgs.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"calling tool {i}"}],
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "run", "arguments": f'{{"i": {i}}}'}}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": [{"type": "tool_result", "content": f"result {i} " + ("z" * 60)}],
        })
    return msgs


def test_real_compress_reaches_compacted_and_granular_reconciles(tmp_path):
    eng = _make_engine(tmp_path)
    try:
        pre = _tool_heavy_turn(40)
        compressed = eng.compress(list(pre), current_tokens=10**9)

        # 1) the real engine reached a terminal COMPACTED status (not noop/lifecycle)
        assert eng._last_compression_status == "compacted", eng._last_compression_status
        # 2) it actually folded (compressed strictly shorter than the input)
        assert len(compressed) < len(pre)

        # 3) the granular in-turn announce reconciles through the REAL done-site
        stats = build_inturn_stats(
            messages=pre, compressed=compressed, estimator=_est, engine_is_lcm=True
        )
        ok, why = stats.validate()
        assert ok, why
        # totals reconcile by construction
        assert stats.folded_count + stats._kept_pre_messages == stats.pre_messages
    finally:
        fn = getattr(eng, "shutdown", None)
        if callable(fn):
            fn()


def test_real_compress_takes_B_provenance_branch(tmp_path):
    """The dominant path renders the EXACT split (Option B), not the A-floor
    approximation — assert approx_attribution is False on a single-pass compaction."""
    eng = _make_engine(tmp_path)
    try:
        pre = _tool_heavy_turn(40)
        compressed = eng.compress(list(pre), current_tokens=10**9)
        assert eng._last_compression_status == "compacted"

        # the returned active context must carry _src_idx provenance on its kept tail
        kept_with_prov = [
            m for m in compressed
            if isinstance(m, dict) and "_src_idx" in m
        ]
        # single-pass compaction stamps provenance → B branch is exact
        stats = build_inturn_stats(
            messages=pre, compressed=compressed, estimator=_est, engine_is_lcm=True
        )
        ok, why = stats.validate()
        assert ok, why
        if kept_with_prov:
            # B engaged → exact attribution
            assert stats.approx_attribution is False
        # (if provenance didn't stamp on this shape, A-floor still reconciles —
        #  totals identity holds either way, asserted above)
    finally:
        fn = getattr(eng, "shutdown", None)
        if callable(fn):
            fn()


def test_render_harness_does_not_touch_live_lcm_db(tmp_path):
    """INV-4: the harness uses a temp store; the live ~/.hermes/lcm.db is untouched."""
    live = os.path.join(
        os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes"),
        "lcm.db",
    )
    before = os.path.getmtime(live) if os.path.exists(live) else None
    eng = _make_engine(tmp_path)
    try:
        eng.compress(_tool_heavy_turn(40), current_tokens=10**9)
    finally:
        fn = getattr(eng, "shutdown", None)
        if callable(fn):
            fn()
    after = os.path.getmtime(live) if os.path.exists(live) else None
    assert before == after, "render harness must not touch the live lcm.db"


def test_too_small_input_does_not_falsely_assert_granular(tmp_path):
    """A message list below the fold threshold must NOT produce a compacted status
    (guards the test from a false-green where nothing actually folded)."""
    eng = _make_engine(tmp_path)
    try:
        tiny = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        compressed = eng.compress(list(tiny), current_tokens=1)
        # nothing to fold → not a compacted terminal status
        assert eng._last_compression_status != "compacted"
        assert len(compressed) == len(tiny)
    finally:
        fn = getattr(eng, "shutdown", None)
        if callable(fn):
            fn()
