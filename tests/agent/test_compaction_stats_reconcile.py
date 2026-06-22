"""Reconciliation guard tests for CompactionStats — the CI lint gate that makes a
non-reconciling compaction announce structurally impossible to ship.

Covers (per the approved spec, Part 5 layer-3):
- property: consistent stats -> ok; deliberately-broken -> not ok / assert raises
- producer-shape: every frozen Phase-0 fixture shape reconciles
- (c1) partition-break negative test (row in two/zero buckets)
- (c2) misclassification negative test (row moved between two VALID buckets — the
       222->222 bug class; only the cross-identity catches it)
- zero-fold + zero-clear edges accepted
"""

from __future__ import annotations

import json
import os

import pytest

from agent.compaction_stats import CompactionStats

_FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "compaction_phase0.json")


def _stats(**overrides) -> CompactionStats:
    """A baseline consistent stats object; override fields to break it."""
    base = dict(
        pre_messages=748, post_messages=34, eligible_count=332,
        kept_messages=32, summary_messages=1, anchor_messages=1,
        cleared_count=416, folded_count=300,
        pre_tokens=397767, post_tokens=17811,
        kept_tokens=11211, summary_tokens=4800, anchor_tokens=1800,
        cleared_tokens=247262, folded_tokens=139294,
    )
    base.update(overrides)
    return CompactionStats(**base)


# ---------------------------------------------------------------------------
# property: baseline reconciles; freed math
# ---------------------------------------------------------------------------

def test_baseline_reconciles():
    ok, reason = _stats().validate()
    assert ok, reason


def test_freed_tokens_and_pct():
    s = _stats()
    assert s.freed_tokens == 397767 - 17811
    assert s.freed_pct == 96


def test_freed_pct_none_when_pre_zero():
    # pre_tokens<=0 must be rejected by validate AND freed_pct guarded
    s = _stats(pre_tokens=0)
    assert s.freed_pct is None
    ok, _ = s.validate()
    assert not ok


# ---------------------------------------------------------------------------
# (c1) partition-break: a row double-counted or dropped -> message axis fails
# ---------------------------------------------------------------------------

def test_producer_rejects_partition_break_extra_row():
    # one extra cleared row (double-count): cleared+folded+kept > pre
    ok, reason = _stats(cleared_count=417).validate()
    assert not ok and "msg axis" in reason


def test_producer_rejects_partition_break_dropped_row():
    # a dropped row: cleared+folded+kept < pre
    ok, reason = _stats(kept_messages=31, post_messages=33).validate()
    assert not ok


# ---------------------------------------------------------------------------
# (c2) misclassification: move a row between two VALID buckets so the partition
# is STILL TOTAL (cleared+folded+kept==pre stays true) — the 222->222 bug class.
# Only the eligible cross-identity catches it.
# ---------------------------------------------------------------------------

def test_producer_rejects_misclassified_population():
    # move one row cleared->folded: 415+301+32==748 still TRUE (partition total),
    # but eligible was measured at 332, so kept+folded=333 != eligible -> caught.
    s = _stats(cleared_count=415, folded_count=301)
    # partition sum still total:
    assert s.cleared_count + s.folded_count + s.kept_messages == s.pre_messages
    ok, reason = s.validate()
    assert not ok and "eligible" in reason, reason


def test_assert_reconciles_raises_on_broken():
    with pytest.raises(ValueError, match="does not reconcile"):
        _stats(cleared_count=415, folded_count=301).assert_reconciles()


def test_assert_reconciles_silent_on_ok():
    _stats().assert_reconciles()  # no raise


# ---------------------------------------------------------------------------
# zero-fold + zero-clear edges accepted
# ---------------------------------------------------------------------------

def test_zero_fold_222_shape_is_valid():
    # folded==0, summary==0, kept==eligible (the literal triggering shape)
    s = CompactionStats(
        pre_messages=658, post_messages=222, eligible_count=222,
        kept_messages=222, summary_messages=0, anchor_messages=0,
        cleared_count=436, folded_count=0,
        pre_tokens=329000, post_tokens=30000,
        kept_tokens=30000, summary_tokens=0, anchor_tokens=0,
        cleared_tokens=299000, folded_tokens=0,
    )
    ok, reason = s.validate()
    assert ok, reason


def test_zero_fold_rejects_nonzero_summary():
    # folded==0 but summary_messages==1 is inconsistent
    s = CompactionStats(
        pre_messages=658, post_messages=223, eligible_count=222,
        kept_messages=222, summary_messages=1, anchor_messages=0,
        cleared_count=436, folded_count=0,
        pre_tokens=329000, post_tokens=34800,
        kept_tokens=30000, summary_tokens=4800, anchor_tokens=0,
        cleared_tokens=299000, folded_tokens=0,
    )
    ok, reason = s.validate()
    assert not ok and "zero-fold" in reason


def test_zero_clear_allchat_is_valid():
    s = CompactionStats(
        pre_messages=120, post_messages=34, eligible_count=120,
        kept_messages=32, summary_messages=1, anchor_messages=1,
        cleared_count=0, folded_count=88,
        pre_tokens=58000, post_tokens=12500,
        kept_tokens=8000, summary_tokens=3000, anchor_tokens=1500,
        cleared_tokens=0, folded_tokens=50000,
    )
    ok, reason = s.validate()
    assert ok, reason


# ---------------------------------------------------------------------------
# token-axis + freed-identity (anchor term) negative cases
# ---------------------------------------------------------------------------

def test_token_pre_axis_break():
    ok, reason = _stats(cleared_tokens=200000).validate()
    assert not ok and "token pre" in reason


def test_freed_identity_missing_anchor_would_fail():
    # If a producer forgot the anchor term, post_tokens would be wrong by anchor.
    # Simulate: anchor really 1800 but post computed without it.
    ok, reason = _stats(post_tokens=17811 - 1800).validate()
    assert not ok  # token post axis catches it


# ---------------------------------------------------------------------------
# optional cleared sub-split must sum to cleared
# ---------------------------------------------------------------------------

def test_cleared_subsplit_must_sum():
    ok, reason = _stats(
        cleared_tool_count=300, cleared_other_count=100,  # 400 != 416
    ).validate()
    assert not ok and "sub-split" in reason


def test_cleared_subsplit_ok():
    ok, reason = _stats(
        cleared_tool_count=356, cleared_other_count=60,  # 416 == cleared
        cleared_tool_tokens=200000, cleared_other_tokens=47262,
    ).validate()
    assert ok, reason


# ---------------------------------------------------------------------------
# frozen-fixture shapes all reconcile (fixture-is-oracle)
# ---------------------------------------------------------------------------

def test_all_frozen_fixture_shapes_reconcile():
    fix = json.load(open(_FIXTURE))
    for name, s in fix["shapes"].items():
        kw = {k: v for k, v in s.items() if not k.startswith("_")}
        kw.pop("session_id", None)
        kw.pop("fresh_tail_count", None)
        kw.pop("eligible_tokens", None)
        cs = CompactionStats(**kw)
        ok, reason = cs.validate()
        assert ok, f"fixture shape {name!r}: {reason}"


# ---------------------------------------------------------------------------
# producer test — build_hygiene_stats over realistic role-mixed data reconciles
# (this is where the 222->222 bug lived: the bucket-COMPUTATION code)
# ---------------------------------------------------------------------------

def _est(msgs):
    """Tiny deterministic additive estimator (chars/4) over a row subset."""
    return sum(len((m.get("content") or "")) for m in msgs) // 4


def _raw_history():
    raw = []
    for i in range(6):
        raw.append({"role": "user", "content": f"q{i} " * 20})
        raw.append({"role": "assistant", "content": f"a{i} " * 40})
        raw.append({"role": "assistant", "content": "", "tool_calls": "[{}]"})  # contentless
        raw.append({"role": "tool", "content": f"TOOL {i} " * 200, "tool_call_id": f"t{i}"})
    return raw


def test_build_hygiene_stats_reconciles_with_fold():
    from agent.compaction_stats import build_hygiene_stats
    raw = _raw_history()
    eligible = [m for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    keep = 3
    kept = eligible[-keep:]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nfolded\n[Expand for details: x]"}
    anchor = {"role": "system", "content": "SYS " * 30}
    compressed = [anchor, summary] + kept
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    # buckets reflect reality
    assert stats.pre_messages == len(raw)
    assert stats.eligible_count == len(eligible)
    assert stats.cleared_count == len(raw) - len(eligible)
    assert stats.kept_messages == keep
    assert stats.folded_count == len(eligible) - keep
    assert stats.summary_messages == 1
    assert stats.anchor_messages == 1


def test_build_hygiene_stats_zero_fold_reconciles():
    # LCM folded nothing: compressed == eligible (no summary, no anchor) — the 222->222 shape
    from agent.compaction_stats import build_hygiene_stats
    raw = _raw_history()
    eligible = [m for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    compressed = list(eligible)  # everything kept verbatim, nothing folded
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    assert stats.folded_count == 0 and stats.summary_messages == 0
    assert stats.kept_messages == stats.eligible_count


# ---------------------------------------------------------------------------
# LCM hygiene degrade fix (2026-06-22) — the fresh tail keeps RAW tool/system
# rows verbatim, which are NOT in the user/assistant-only `eligible` set.  The
# pre-fix producer assumed kept ⊆ eligible and mis-partitioned, so validate()
# failed and the announce silently degraded to two-line on every tool-heavy
# session (the live 1075→33 message Ace flagged).  These reproduce that and
# guard the fix.
# ---------------------------------------------------------------------------

def _est_formula(msgs):
    """Documented estimator formula oracle (RC-A): estimate_messages_tokens_rough
    is ceil(Σ chars / 3.5) + image tokens.  Mirrors the constant, does NOT wrap
    the production function, so the expectations are an INDEPENDENT oracle."""
    import math
    total = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
    return math.ceil(total / 3.5)


def _gateway_shaped_tool_heavy():
    """Reproduce the gateway's real shapes: raw_history is COPIES, eligible is the
    user/assistant-with-content filtered ORIGINALS, compressed = [anchor, summary]
    + a raw fresh tail that INCLUDES a tool row (LCM keeps recent turns verbatim)."""
    raw = []
    for i in range(8):
        raw.append({"role": "user", "content": f"u{i} " * 20})
        raw.append({"role": "assistant", "content": f"a{i} " * 40})
        raw.append({"role": "tool", "content": f"TOOL {i} " * 200, "tool_call_id": f"t{i}"})
    # gateway: _hyg_pre_history = [{**m} ...] (copies); _hyg_msgs = filtered originals
    raw_history = [dict(m) for m in raw]
    eligible = [m for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    # LCM fresh tail = last 3 RAW turns kept verbatim → includes a tool row
    tail = [dict(m) for m in raw[-3:]]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nfolded\n[Expand for details: x]"}
    anchor = {"role": "system", "content": "SYS " * 30}
    compressed = [anchor, summary] + tail
    return raw, raw_history, eligible, compressed, tail


def test_build_hygiene_stats_tool_in_fresh_tail_reconciles():
    """REGRESSION: LCM fresh tail contains a tool row not in eligible → must still
    reconcile (this is the exact 2026-06-22 live degrade)."""
    from agent.compaction_stats import build_hygiene_stats
    raw, raw_history, eligible, compressed, tail = _gateway_shaped_tool_heavy()
    assert any(m["role"] == "tool" for m in tail), "fixture must put a tool row in the tail"
    stats = build_hygiene_stats(
        raw_history=raw_history, eligible_msgs=eligible, compressed=compressed, estimator=_est_formula,
    )
    ok, reason = stats.validate()
    assert ok, f"tool-in-tail must reconcile, got: {reason}"
    # the message axis must cover pre exactly
    assert stats.cleared_count + stats.folded_count + stats.kept_messages == stats.pre_messages
    # the kept tail (3 rows) all survive into post
    assert stats.kept_messages == 3


def test_build_hygiene_stats_tool_in_tail_independent_oracle():
    """RC-A: bucket tokens equal hand-computed expectations from the documented
    estimator formula — an oracle independent of the production estimator AND of
    the code under test."""
    from agent.compaction_stats import build_hygiene_stats
    raw, raw_history, eligible, compressed, tail = _gateway_shaped_tool_heavy()
    stats = build_hygiene_stats(
        raw_history=raw_history, eligible_msgs=eligible, compressed=compressed, estimator=_est_formula,
    )
    # Independent oracle: partition pre by identity into kept / folded / cleared.
    # map tail (copies) back to raw by content+role since they are dict copies:
    #  kept   = raw rows whose (role, content) matches a tail row (verbatim survivors)
    #  folded = eligible rows not kept
    #  cleared= pre rows removed by the filter (tool/system/contentless) and not kept
    tail_keys = [(m["role"], m["content"]) for m in tail]
    import collections
    want = collections.Counter(tail_keys)
    kept_rows, rest = [], []
    for m in raw:
        k = (m["role"], m.get("content"))
        if want.get(k, 0) > 0:
            want[k] -= 1
            kept_rows.append(m)
        else:
            rest.append(m)
    # folded = eligible rows not in kept
    kept_ids = {id(m) for m in kept_rows}
    folded_rows = [m for m in eligible if id(m) not in kept_ids]
    cleared_rows = [m for m in rest if m not in eligible]  # tool/contentless not kept
    exp_kept = _est_formula(kept_rows)
    exp_folded = _est_formula(folded_rows)
    exp_cleared = _est_formula(cleared_rows)
    # the producer's buckets must equal these independent expectations (±estimator non-additivity)
    assert abs(stats.kept_tokens - exp_kept) <= 2, (stats.kept_tokens, exp_kept)
    assert abs(stats.folded_tokens - exp_folded) <= 2, (stats.folded_tokens, exp_folded)
    assert abs(stats.cleared_tokens - exp_cleared) <= 2, (stats.cleared_tokens, exp_cleared)


def test_build_hygiene_stats_duplicate_tool_rows_collision():
    """RC-C/B4: many byte-identical tool rows (signature collisions) must partition
    correctly by IDENTITY, not by content-signature subtraction."""
    from agent.compaction_stats import build_hygiene_stats
    raw = []
    IDENTICAL = "DUPLICATE TOOL OUTPUT " * 50
    for i in range(3):
        raw.append({"role": "user", "content": f"u{i} " * 20})
        raw.append({"role": "assistant", "content": f"a{i} " * 40})
    # 20 byte-identical tool rows
    for i in range(20):
        raw.append({"role": "tool", "content": IDENTICAL, "tool_call_id": f"t{i}"})
    raw_history = [dict(m) for m in raw]
    eligible = [m for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    # fresh tail keeps 2 of the identical tool rows verbatim + 1 chat row
    tail = [dict(raw[-1]), dict(raw[-2]), dict(eligible[-1])]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nx\n[Expand for details: y]"}
    anchor = {"role": "system", "content": "SYS " * 30}
    compressed = [anchor, summary] + tail
    stats = build_hygiene_stats(
        raw_history=raw_history, eligible_msgs=eligible, compressed=compressed, estimator=_est_formula,
    )
    ok, reason = stats.validate()
    assert ok, f"duplicate-tool-row collision must reconcile, got: {reason}"

def test_freed_check_tolerates_compounded_axis_error():
    """The freed identity is the difference of the two ±_TOKEN_TOL axis checks, so
    its bound is 2×_TOKEN_TOL. A compounded error of +_TOKEN_TOL on pre and
    -_TOKEN_TOL on post (net 2×) must still reconcile — not spuriously degrade.
    """
    from agent.compaction_stats import _TOKEN_TOL
    # Start from the baseline, then perturb pre/post axes in OPPOSITE directions
    # each by exactly _TOKEN_TOL so both axis checks pass at their edge but the
    # freed identity sees a 2×_TOKEN_TOL gap.
    s = _stats(
        # pre axis: cleared+folded+kept = pre + _TOKEN_TOL (edge-pass)
        cleared_tokens=247262 + _TOKEN_TOL,
        # post axis: kept+summary+anchor = post - _TOKEN_TOL (edge-pass)
        summary_tokens=4800 - _TOKEN_TOL,
    )
    ok, reason = s.validate()
    assert ok, f"compounded 2x-tol error must still reconcile: {reason}"


def test_freed_check_never_fails_alone_when_both_axes_pass():
    """With _FREED_TOL = 2×_TOKEN_TOL, the freed identity is mathematically
    redundant: it is the difference of the two axis checks, so whenever both axes
    pass (±_TOKEN_TOL each) the freed gap is ≤2×_TOKEN_TOL by construction and can
    never be the *sole* failure. Any value that breaks freed beyond 2× must break
    an axis first — proving the old single-_TOKEN_TOL bound was the spurious one.
    """
    from agent.compaction_stats import _TOKEN_TOL
    # Break freed by 2×+1 by perturbing only cleared_tokens: this necessarily
    # breaks the pre axis too (cleared feeds it), so freed is never the lone cause.
    s = _stats(cleared_tokens=247262 + 2 * _TOKEN_TOL + 1)
    ok, reason = s.validate()
    assert not ok
    assert "token pre" in reason  # the axis fails first, not "freed"


def test_signature_fallback_handles_identical_long_prefixes():
    """Fallback subtraction (copies path) must NOT collide on messages that share
    a long identical prefix — the [:200]-truncation bug. With copied dicts (no
    id() match), the producer falls back to the row signature; a full-content
    hash keeps distinct-suffix rows distinct so cleared/folded attribution and
    validate() stay correct.
    """
    from agent.compaction_stats import build_hygiene_stats
    prefix = "TOOLRESULT " * 40  # >200 chars, shared by every tool row
    raw = []
    for i in range(5):
        raw.append({"role": "user", "content": f"u{i} " * 20})
        raw.append({"role": "assistant", "content": f"{prefix} UNIQUE_SUFFIX_{i}"})
    # eligible = COPIES (fresh dicts) so id()-identity fails → signature path runs
    eligible = [
        {"role": m["role"], "content": m["content"]}
        for m in raw
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    # keep the last two eligible (also copies, distinct suffixes)
    kept = [{"role": m["role"], "content": m["content"]} for m in eligible[-2:]]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nx\n[Expand for details: y]"}
    compressed = [summary] + kept
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, f"long-prefix rows must reconcile via full-content signature: {reason}"
    assert stats.eligible_count == len(eligible)
    assert stats.kept_messages == 2
    assert stats.folded_count == len(eligible) - 2


# ---------------------------------------------------------------------------
# Phase 1 — folded sub-split fields + validation (tool-result sub-split spec)
# ---------------------------------------------------------------------------

def test_folded_subsplit_ok():
    """A folded sub-split whose counts AND tokens partition the folded bucket reconciles."""
    ok, reason = _stats(
        folded_tool_count=250, folded_other_count=50,            # 300 == folded_count
        folded_tool_tokens=120000, folded_other_tokens=19294,    # == folded_tokens 139294
    ).validate()
    assert ok, reason


def test_folded_subsplit_count_must_sum():
    ok, reason = _stats(
        folded_tool_count=250, folded_other_count=40,            # 290 != 300
        folded_tool_tokens=120000, folded_other_tokens=19294,
    ).validate()
    assert not ok and "folded sub-split" in reason


def test_folded_subsplit_tokens_must_sum_exactly():
    # exact tie-out (derive-by-subtraction) — even +1 off must fail
    ok, reason = _stats(
        folded_tool_count=250, folded_other_count=50,
        folded_tool_tokens=120000, folded_other_tokens=19295,    # 139295 != 139294
    ).validate()
    assert not ok and "folded sub-split tokens" in reason


def test_folded_subsplit_never_raises_on_bad_data():
    # validate() is non-raising at runtime — a broken sub-split returns (False, …)
    s = _stats(folded_tool_count=999, folded_other_count=0,
               folded_tool_tokens=1, folded_other_tokens=1)
    ok, reason = s.validate()
    assert ok is False and isinstance(reason, str)


def test_cleared_subsplit_tokens_must_sum_exactly():
    # the cleared sub-split gains the token-axis check it was missing (CHANGE-2)
    ok, reason = _stats(
        cleared_tool_count=356, cleared_other_count=60,          # 416 == cleared_count
        cleared_tool_tokens=200000, cleared_other_tokens=47263,  # 247263 != 247262
    ).validate()
    assert not ok and "cleared sub-split tokens" in reason


def test_cleared_subsplit_tokens_ok_exact():
    ok, reason = _stats(
        cleared_tool_count=356, cleared_other_count=60,
        cleared_tool_tokens=200000, cleared_other_tokens=47262,  # == cleared_tokens
    ).validate()
    assert ok, reason


# ---------------------------------------------------------------------------
# Phase 2 — _tool_other_split helper + build_inturn_stats folded sub-split
# ---------------------------------------------------------------------------

def _toolmsg(i, n=200):
    return {"role": "tool", "content": f"TOOLRESULT {i} " * n, "tool_call_id": f"t{i}"}

def _chatmsg(i, role="assistant", n=40):
    return {"role": role, "content": f"{role} turn {i} " * n}


def test_tool_other_split_derives_other_by_subtraction():
    from agent.compaction_stats import _tool_other_split
    rows = [_toolmsg(0), _toolmsg(1), _chatmsg(0), _chatmsg(1, role="user")]
    parent = _est(rows)
    tc, tt, oc, ot = _tool_other_split(rows, parent, _est)
    assert tc == 2 and oc == 2
    # tool estimated, other DERIVED → exact tie-out, parent untouched
    assert tt == _est([rows[0], rows[1]])
    assert tt + ot == parent
    assert ot == parent - tt


def test_estimator_returns_int_contract():
    # CHANGE-1: estimator returns int, so no int() wrap is needed in the helper
    from agent.model_metadata import estimate_messages_tokens_rough as real_est
    r = real_est([{"role": "tool", "content": "x" * 100}])
    assert isinstance(r, int)
    assert isinstance(real_est([]), int)


def test_build_inturn_stats_populates_folded_subsplit_exact():
    from agent.compaction_stats import build_inturn_stats
    # 6 tool + 4 chat in the population; keep the last 2 chat as the tail
    msgs = [_toolmsg(i) for i in range(6)] + [_chatmsg(i) for i in range(4)]
    kept = msgs[-2:]  # last 2 chat rows (members of msgs, so fold = 8)
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nx\n[Expand for details: y]"}
    compressed = [summary] + kept
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    # folded population = 8 (6 tool + 2 chat); kept tail = 2 chat
    assert stats.folded_tool_count == 6
    assert stats.folded_other_count == 2
    assert stats.folded_tool_tokens + stats.folded_other_tokens == stats.folded_tokens
    assert stats.folded_tool_count > 0


def test_build_inturn_stats_parent_unchanged_by_subsplit():
    """BLOCKER-2: folded_tokens must be byte-equal whether or not the sub-split runs."""
    from agent.compaction_stats import build_inturn_stats, _fold_rows
    msgs = [_toolmsg(i) for i in range(6)] + [_chatmsg(i) for i in range(4)]
    kept = msgs[-2:]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nx\n[Expand for details: y]"}
    compressed = [summary] + kept
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    # parent folded_tokens == direct estimate of the fold population (unperturbed)
    kept_rows = [m for m in compressed
                 if m.get("role") != "system"
                 and "[Recent Summary" not in (m.get("content") or "")]
    fold_pop = _fold_rows(msgs, kept_rows)
    assert stats.folded_tokens == _est(fold_pop)


def test_build_inturn_stats_roleless_row_lands_in_other_succeeds():
    # CHANGE-D (a-i): a roleless row goes to `other`, never estimated → split SUCCEEDS
    from agent.compaction_stats import build_inturn_stats
    msgs = [_toolmsg(0), {"content": "no role here"}, _chatmsg(0)]
    kept = msgs[-1:]  # the chat row (member of msgs)
    compressed = list(kept)
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    assert stats.folded_tool_count is not None  # populated, not degraded


def test_build_inturn_stats_degrades_when_tool_estimate_raises():
    # CHANGE-D (a-ii): estimator raises on the pure-tool sublist (inside _tool_other_split)
    # → sub-split degrades to None, but the parent folded_tokens (mixed list) still computes.
    from agent.compaction_stats import build_inturn_stats
    t0 = {"role": "tool", "content": "tool a " * 50, "tool_call_id": "a"}
    t1 = {"role": "tool", "content": "tool b " * 50, "tool_call_id": "b"}
    c0, c1 = _chatmsg(0), _chatmsg(1)
    msgs = [t0, t1, c0, c1]
    def boom_est(rows):
        # raise ONLY on a non-empty all-tool list (the sublist _tool_other_split builds)
        if rows and all(m.get("role") == "tool" for m in rows):
            raise ValueError("boom on pure-tool sublist")
        return _est(rows)
    kept = [c1]
    compressed = list(kept)
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=boom_est)
    ok, reason = stats.validate()
    assert ok, reason  # parent still reconciles
    assert stats.folded_tool_count is None  # sub-split degraded


def test_build_inturn_stats_subsplit_survives_fold_signature_fallback():
    # CHANGE-E/C5: duplicate-signature rows force _fold_rows' Counter fallback;
    # the helper must split the SAME post-fold list → still sums exactly.
    from agent.compaction_stats import build_inturn_stats
    dup = "TOOLRESULT " * 40
    msgs = [{"role": "tool", "content": dup, "tool_call_id": f"t{i}"} for i in range(4)] + [_chatmsg(0)]
    # kept = COPIES so id() match fails → signature fallback
    kept = [{"role": "tool", "content": dup, "tool_call_id": "t0"}]
    compressed = list(kept)
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    assert stats.folded_tool_count + stats.folded_other_count == stats.folded_count


# ---------------------------------------------------------------------------
# Phase 3 — build_hygiene_stats populates the (previously dead) cleared sub-split
# ---------------------------------------------------------------------------

def test_build_hygiene_stats_populates_cleared_subsplit_exact():
    from agent.compaction_stats import build_hygiene_stats
    # raw has tool rows (go to cleared) + contentless tool-call assistant (cleared) + chat (eligible)
    raw = []
    for i in range(5):
        raw.append({"role": "user", "content": f"u{i} " * 20})
        raw.append({"role": "assistant", "content": f"a{i} " * 40})
        raw.append({"role": "assistant", "content": "", "tool_calls": "[{}]"})   # contentless → cleared
        raw.append({"role": "tool", "content": f"TOOL {i} " * 200, "tool_call_id": f"t{i}"})  # cleared
    eligible = [{"role": m["role"], "content": m["content"]}
                for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    keep = 3
    kept = eligible[-keep:]
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)]\nx\n[Expand for details: y]"}
    anchor = {"role": "system", "content": "SYS " * 30}
    compressed = [anchor, summary] + kept
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    # cleared = tool rows + contentless assistant rows; tool count = 5
    assert stats.cleared_tool_count == 5
    assert stats.cleared_other_count == stats.cleared_count - 5
    assert stats.cleared_tool_tokens + stats.cleared_other_tokens == stats.cleared_tokens


def test_build_hygiene_stats_cleared_parent_unchanged_by_subsplit():
    from agent.compaction_stats import build_hygiene_stats, _disjoint_remainder
    raw = []
    for i in range(5):
        raw.append({"role": "user", "content": f"u{i} " * 20})
        raw.append({"role": "assistant", "content": f"a{i} " * 40})
        raw.append({"role": "tool", "content": f"TOOL {i} " * 200, "tool_call_id": f"t{i}"})
    eligible = [{"role": m["role"], "content": m["content"]}
                for m in raw if m["role"] in ("user", "assistant") and m.get("content")]
    kept = eligible[-2:]
    compressed = list(kept)
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=compressed, estimator=_est)
    cleared_rows = _disjoint_remainder(raw, eligible)
    assert stats.cleared_tokens == _est(cleared_rows)  # parent untouched
