"""Option B — provenance-stamped exact in-turn compaction partition (2026-06-27, P1-B).

PR #109 shipped the A-floor (reconciles totals, signature-approximate split bounded
≤7% of pre). B makes the split EXACT: the LCM engine stamps ``_src_idx`` (origin index
into the original ``messages``) onto each tail row before ``_assemble_context``; the
pipeline's shallow-copies (``dict(msg)``/``msg.copy()``) carry it through every
drop/strip/content-rewrite stage by construction, and synthetic tool stubs lack it. The
consumer harvests the exact pre-side kept set off the returned ``compressed`` (no
inference, no replay), then strips the key before it flows onward.

Design lineage: positional-index (v0.1) was DISPROVEN — the kept tail is a lossy
non-positional transform; provenance (v0.3) is the correct mechanism; provenance-on-
return with NO engine instance state (v0.5) avoids the process-global-singleton race.
"""
from __future__ import annotations

from agent.compaction_stats import (
    build_inturn_stats,
    harvest_provenance_partition,
    strip_provenance,
)
from agent.model_metadata import estimate_messages_tokens_rough as _est


def _pre(n=60):
    return [{"role": "user" if i % 2 else "assistant", "content": f"m{i} " + ("w" * 40)}
            for i in range(n)]


def _stamp(rows, start=0):
    """Mimic the engine stamp: shallow-copy + _src_idx = origin index."""
    return [dict(r, **{"_src_idx": start + i}) for i, r in enumerate(rows)]


# ───────────────────────── harvest_provenance_partition ─────────────────────────

def test_harvest_exact_origins():
    pre = _pre(50)
    # kept = stamped copies of the last 5 pre rows (origins 45..49)
    kept = _stamp(pre[45:], start=45)
    out = harvest_provenance_partition(pre, kept)
    assert out is not None
    idx, stub = out
    assert idx == [45, 46, 47, 48, 49]
    assert stub == 0


def test_harvest_excludes_stubs_independently():
    pre = _pre(50)
    kept = _stamp(pre[46:], start=46)  # 4 real kept (46..49)
    # inject a synthetic stub (no _src_idx) in the middle of the kept region
    stub_row = {"role": "tool", "tool_call_id": "x", "content": "[Result from earlier conversation — see context summary]"}
    kept = kept[:2] + [stub_row] + kept[2:]
    out = harvest_provenance_partition(pre, kept)
    assert out is not None
    idx, stub = out
    assert idx == [46, 47, 48, 49]   # the 4 real origins, exact
    assert stub == 1                 # stub counted INDEPENDENTLY (keyless), not derived


def test_harvest_none_when_no_provenance():
    """A non-B / non-stamped comp (no _src_idx anywhere) → None → caller uses replay/A-floor."""
    pre = _pre(20)
    kept = [dict(r) for r in pre[-3:]]  # copies, no _src_idx
    assert harvest_provenance_partition(pre, kept) is None


def test_harvest_rejects_malformed_index():
    pre = _pre(20)
    kept = _stamp(pre[-3:], start=17)
    kept[1]["_src_idx"] = 999  # out of range
    assert harvest_provenance_partition(pre, kept) is None


def test_harvest_rejects_duplicate_index():
    pre = _pre(20)
    kept = _stamp(pre[-3:], start=17)
    kept[2]["_src_idx"] = kept[0]["_src_idx"]  # duplicate origin
    assert harvest_provenance_partition(pre, kept) is None


# ───────────────────────── strip_provenance (BLK-3 wire-leak) ────────────────────

def test_strip_removes_all_src_idx():
    rows = _stamp(_pre(10))
    n = strip_provenance(rows)
    assert n == 10
    assert all("_src_idx" not in r for r in rows)
    # idempotent
    assert strip_provenance(rows) == 0


# ───────────────────────── build_inturn_stats B path (exact) ─────────────────────

def _comp_with_provenance(pre, anchor, summary, kept_origin_slice):
    kept = _stamp(pre[kept_origin_slice], start=kept_origin_slice.start)
    return [anchor, summary] + kept, kept


def test_b_path_exact_attribution_not_approx():
    """B engages on stamped kept rows → exact split, approx_attribution False."""
    pre = _pre(60)
    anchor = {"role": "system", "content": "SYS"}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] x", "_lcm_summary": True}
    comp, _ = _comp_with_provenance(pre, anchor, summary, slice(55, 60))
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why
    assert stats.approx_attribution is False          # B is EXACT, not the A-floor
    assert stats._kept_pre_messages == 5              # exact origins
    assert stats.folded_count == 55
    assert stats.folded_count + stats._kept_pre_messages == stats.pre_messages


def test_b_path_rewritten_content_row_still_in_kept_pre():
    """A kept row whose CONTENT was rewritten (objective-trim) keeps _src_idx → still
    attributed to kept_pre (the provenance survives content mutation)."""
    pre = _pre(40)
    anchor = {"role": "system", "content": "SYS"}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] y", "_lcm_summary": True}
    kept = _stamp(pre[36:], start=36)
    kept[0] = dict(kept[0]); kept[0]["content"] = "REWRITTEN by objective-trim"  # content changed, _src_idx kept
    comp = [anchor, summary] + kept
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why
    assert stats.approx_attribution is False
    assert stats._kept_pre_messages == 4   # all 4 origins, incl. the rewritten one


def test_b_path_stub_in_kept_excluded_totals_reconcile():
    """A synthetic stub in the kept region is NOT in kept_pre; totals still reconcile."""
    pre = _pre(50)
    anchor = {"role": "system", "content": "SYS"}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] z", "_lcm_summary": True}
    kept = _stamp(pre[47:], start=47)  # origins 47,48,49
    stub = {"role": "tool", "tool_call_id": "s", "content": "[Result from earlier conversation — …]"}
    comp = [anchor, summary, kept[0], stub, kept[1], kept[2]]
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why
    assert stats._kept_pre_messages == 3        # 3 real origins (stub excluded from pre-side)
    assert stats.kept_messages == 4             # comp-side display includes the stub
    assert stats.folded_count + stats._kept_pre_messages == stats.pre_messages


def test_b_path_malformed_falls_to_a_floor():
    """A corrupt _src_idx (out of range) → harvest returns None → A-floor (approx),
    never a confidently-wrong B split."""
    pre = _pre(40)
    anchor = {"role": "system", "content": "SYS"}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] q", "_lcm_summary": True}
    kept = _stamp(pre[36:], start=36)
    kept[1]["_src_idx"] = 9999  # corrupt
    comp = [anchor, summary] + kept
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why                       # A-floor still reconciles
    assert stats.approx_attribution is True   # fell to A-floor, not B
