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


def _blockmsg(role, text, n=40):
    """An API-shaped message whose ``content`` is a LIST of content blocks.

    This is the real shape the IN-TURN compaction path feeds build_inturn_stats
    (assistant text+tool_use blocks, user tool_result blocks) — NOT the flat
    string content the hygiene path uses. _is_summary_message must not crash on it.
    """
    return {"role": role, "content": [{"type": "text", "text": f"{text} " * n}]}


def _toolresult_blockmsg(i, n=60):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"tool out {i} " * n},
    ]}


def test_build_inturn_stats_handles_list_content_messages():
    """REGRESSION: in-turn messages have LIST content (API content blocks), not
    flat strings. _is_summary_message did `regex.search(content)` which raises
    TypeError on a list → build_inturn_stats raised → the in-turn announce
    silently degraded to the single-line form for EVERY real session (which all
    carry tool-call/block content). This must build + reconcile, not raise.
    """
    from agent.compaction_stats import build_inturn_stats
    # Realistic in-turn population: a system anchor + list-content chat/tool rows.
    anchor = {"role": "system", "content": "SYSTEM PROMPT " * 50}
    body = []
    for i in range(20):
        if i % 3 == 0:
            body.append(_toolresult_blockmsg(i))
        else:
            body.append(_blockmsg("assistant", f"assistant block {i}"))
    msgs = [anchor] + body
    kept = msgs[-4:]  # fresh tail (identity members of msgs)
    summary = {"role": "assistant",
               "content": "[Session Arc Summary (d1, node 7)]\nrolled up\n[Expand for details: x]"}
    compressed = [anchor, summary] + kept
    # Must NOT raise (the bug was a TypeError from list content), and must reconcile.
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    # The summary (string content) is still detected; the anchor counted once.
    assert stats.summary_messages == 1
    assert stats.anchor_messages == 1
    assert stats.kept_messages == 4


def test_is_summary_message_tolerates_non_string_content():
    """_is_summary_message must return False (not raise) on list/dict/None content,
    and still detect the marker inside a list of text blocks."""
    from agent.compaction_stats import _is_summary_message
    # Non-string shapes must not raise.
    assert _is_summary_message([{"type": "text", "text": "hi"}]) is False
    assert _is_summary_message({"type": "tool_result", "content": "x"}) is False
    assert _is_summary_message(None) is False
    # A marker living inside a text block IS detected.
    assert _is_summary_message(
        [{"type": "text", "text": "[Recent Summary (d0, node 1)] body"}]
    ) is True


def test_inturn_tool_breakout_recognizes_content_block_tool_results():
    """The folded tool/other sub-split must break out tool messages in the LIVE
    in-turn shape — a tool RESULT is a ``role=user`` message with a ``tool_result``
    content block (NOT a flat ``role=tool`` row). Without block-awareness the whole
    folded population reads as 'other' and the 'N tool-result messages' line the
    design calls for never renders on the real path.
    """
    from agent.compaction_stats import build_inturn_stats
    anchor = {"role": "system", "content": "SYS " * 50}
    body = []
    for i in range(30):
        if i % 2 == 0:
            body.append(_toolresult_blockmsg(i))            # role=user + tool_result block
        else:
            body.append(_blockmsg("assistant", f"chat {i}"))  # role=assistant + text block
    msgs = [anchor] + body
    compressed = [anchor,
                  {"role": "assistant",
                   "content": "[Session Arc Summary (d1, node 3)]\nx\n[Expand for details: y]"}] + msgs[-4:]
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, reason = stats.validate()
    assert ok, reason
    # tool-result block messages must be counted as tool, not swept into "other".
    assert stats.folded_tool_count is not None and stats.folded_tool_count > 0
    assert stats.folded_tool_tokens + stats.folded_other_tokens == stats.folded_tokens




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


# ───────────────────────────────────────────────────────────────────────────
# v0.3 — _content_to_text byte-faithful mirror of canonical LCM extractor
# Spec: plans/2026-06-22_content-to-text-robust-block-extraction-SPEC.md
# Hardens summary-marker detection on structured (list/dict) content WITHOUT a
# name-agnostic field scan: type-gated parts only, "\n" join (forecloses
# cross-part synthesis), is-None fall-through, one documented json.dumps deviation.
# ───────────────────────────────────────────────────────────────────────────

import re as _re_v03

_MARKER_RE = _re_v03.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
)


def _detect(content) -> bool:
    """Drive detection through the real consumer (_is_summary_message)."""
    from agent.compaction_stats import _is_summary_message
    return _is_summary_message(content)


# ── AC-1: detection under text-typed parts incl nested value, accumulation ──

def test_marker_detected_under_text_typed_value_nested():
    # nested {value: "<marker>"} under a text-typed part — current code never read `value`.
    assert _detect([{"type": "text", "text": {"value": "[Recent Summary (d0, node 1)] body"}}]) is True


def test_marker_accumulates_across_parts():
    # marker is whole inside the 2nd text part — must be found (not first-hit-return-dropped).
    assert _detect([
        {"type": "text", "text": "intro"},
        {"type": "text", "text": "[Session Arc Summary (d1, node 7)] x"},
    ]) is True


# ── AC-2: no false positives (structural / cross-field / cross-part); bare-string accepted ──

def test_marker_under_structural_key_is_not_detected():
    # tool_use block is NOT text-typed → contributes nothing → marker under `name` never searched.
    assert _detect([
        {"type": "tool_use", "name": "[Durable Summary (d2, node 3)] not a summary", "id": "t1"}
    ]) is False


def test_text_typed_part_marker_under_noncanonical_key_not_synthesized():
    # text-typed (type-gate ACCEPTED) but marker split across non-canonical a/b → only text/content read.
    assert _detect([
        {"type": "text", "a": "[Recent Summary (d0,", "b": "node 1)] x"}
    ]) is False


def test_marker_split_across_two_text_parts_not_synthesized():
    # "\n" join injects a newline the single-line regex cannot span → no cross-part synthesis.
    assert _detect([
        {"type": "text", "text": "x [Recent Summary (d0,"},
        {"type": "text", "text": "node 1)] y"},
    ]) is False


def test_marker_as_bare_string_element_IS_detected():
    # bare-string passthrough mirrors canonical — accepted/intended behavior, pinned.
    assert _detect(["[Durable Summary (d2, node 3)] body"]) is True


def test_tool_result_realistic_text_not_false_positive():
    assert _detect([
        {"type": "tool_result", "tool_use_id": "t1", "content": "exit_code 0, all good, 12 files"}
    ]) is False


# ── AC-4c: parity gate — core mirrors canonical for marker detection ──

def test_content_to_text_mirrors_canonical_for_marker_detection():
    from agent.compaction_stats import _content_to_text
    from plugins.context_engine.lcm.message_content import text_content_for_pattern_matching as _canon

    M = "[Recent Summary (d0, node 1)] body"
    corpus = [
        None,
        "",
        M,
        f"prefix {M} suffix",
        [{"type": "text", "text": M}],
        [{"type": "text", "text": {"value": M}}],
        [{"type": "text", "text": {"content": M}}],
        [{"type": "input_text", "text": M}],
        [{"type": "output_text", "text": M}],
        [M],                                                 # bare string element
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        [{"type": "tool_use", "name": "terminal", "id": "t1"}],   # structural, no text
        [{"type": "text", "text": "x"}, {"type": "tool_result", "content": "y"}],
        [{"type": "text", "text": ""}],
    ]
    for shape in corpus:
        core = bool(_MARKER_RE.search(_content_to_text(shape)))
        canon_txt = _canon(shape)
        canon = bool(_MARKER_RE.search(canon_txt)) if canon_txt else False
        # core's match-set ⊆ canonical's (the one allowed deviation: canonical's json.dumps
        # fallback could match where core returns ""; core must NEVER match where canonical doesn't).
        if core:
            assert canon, f"core matched but canonical did not for shape={shape!r}"


# ── AC-6 + carryover: depth/shape bounds, fall-through, defensive bare dict ──

def test_text_dict_falls_through_to_content():
    # `text` is a non-extractable dict → is-None fall-through reads `content`.
    assert _detect({"type": "text", "text": {"foo": "bar"},
                    "content": "[Recent Summary (d0, node 1)]"}) is True


def test_text_typed_part_with_non_value_content_dict():
    from agent.compaction_stats import _content_to_text
    assert _content_to_text({"type": "text", "text": {"foo": "bar"}}) == ""


def test_list_valued_text_is_out_of_bounds():
    # list-valued `text` is neither str nor dict in _extract_part_text → "" (documented out-of-bounds).
    assert _detect([{"type": "text", "text": ["[Recent Summary (d0, node 1)]"]}]) is False


def test_depth_cap_stops_at_one_level():
    from agent.compaction_stats import _content_to_text
    assert _content_to_text({"type": "text", "text": {"value": {"value": "x"}}}) == ""


def test_bare_dict_defensive():
    # D-6 defensive bare-dict branch is exercised, not dead code.
    assert _detect({"type": "text", "text": "[Recent Summary (d0, node 1)]"}) is True


# ── AC-3: never raises on any shape, incl. list-element exotics ──

import pytest as _pytest_v03


@_pytest_v03.mark.parametrize("shape", [
    None,
    42,
    object(),
    {"type": "text", "text": 5},
    {"deeply": {"nested": {"dict": "x"}}},
    [None, 42, object(), {"type": "text", "text": "ok"}],
    [{"type": "text"}],
    [{}],
])
def test_content_to_text_never_raises(shape):
    from agent.compaction_stats import _content_to_text
    out = _content_to_text(shape)
    assert isinstance(out, str)


# ── INV-6: working flat-string path returns verbatim ──

def test_flat_string_returned_verbatim():
    from agent.compaction_stats import _content_to_text
    s = "[Recent Summary (d0, node 1)] body"
    assert _content_to_text(s) == s


# ───────────────────────────────────────────────────────────────────────────
# Structural summary tagging (_lcm_summary) — tag-first / regex-fallback, with
# a tag-missing tripwire. Spec:
# plans/2026-06-22_structural-summary-tagging-and-degrade-observability-SPEC.md
# ───────────────────────────────────────────────────────────────────────────

def test_is_summary_row_tag_fast_path():
    # tagged row with NON-marker content is still a summary (structural, not regex).
    from agent.compaction_stats import _is_summary_row
    assert _is_summary_row(
        {"role": "assistant", "content": "no marker text here at all", "_lcm_summary": True}
    ) is True


def test_is_summary_row_falls_back_to_regex():
    # untagged row with marker content → detected via the regex fallback (INV-3).
    from agent.compaction_stats import _is_summary_row
    assert _is_summary_row(
        {"role": "assistant", "content": "[Recent Summary (d0, node 1)] x"}
    ) is True


def test_is_summary_row_none_content_no_raise():
    # the existing call sites' `or ""` None-guard must be preserved (Pass-1 bpp B1).
    from agent.compaction_stats import _is_summary_row
    assert _is_summary_row({"role": "assistant", "content": None}) is False


def test_is_summary_row_tag_must_be_literal_true():
    # only literal True takes the fast path — no truthy widening (INV-4).
    from agent.compaction_stats import _is_summary_row
    for bad in ("yes", 1, None, [], {"x": 1}):
        assert _is_summary_row(
            {"role": "assistant", "content": "plain non-marker text", "_lcm_summary": bad}
        ) is False, f"_lcm_summary={bad!r} must NOT take the fast path"


def _regex_miss_summary_row(tagged: bool):
    # marker text broken across content parts → the regex MISSES it (single-line
    # join can't span the part boundary), the exact loose-end-#2 failure.
    row = {"role": "assistant", "content": [
        {"type": "text", "text": "[Recent Summary (d0,"},
        {"type": "text", "text": "node 1)] x"},
    ]}
    if tagged:
        row["_lcm_summary"] = True
    return row


def test_regex_miss_fixture_tag_detects_regex_misses():
    # INV-5: the tag detects a summary the regex MISSES (proves a real defect fix).
    from agent.compaction_stats import _is_summary_row, _is_summary_message
    row = _regex_miss_summary_row(tagged=True)
    assert _is_summary_message(row["content"]) is False, "regex must MISS the split marker"
    assert _is_summary_row(row) is True, "tag must detect what the regex missed"


def test_tagged_classification_keeps_validate_valid():
    # INV-5: build_inturn_stats over a fixture containing the regex-miss summary row
    # (tagged) reconciles AND counts the summary.
    from agent.compaction_stats import build_inturn_stats
    anchor = {"role": "system", "content": "SYS " * 50}
    summary = _regex_miss_summary_row(tagged=True)
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, summary] + msgs[-3:]
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, why = stats.validate()
    assert ok, why
    assert stats.summary_messages == 1


def test_mixed_tagged_and_untagged_summaries_reconcile():
    # RQ-2: a single build pass over a MIX of tagged + untagged-but-marker summary
    # rows (the real first-post-deploy state) reconciles and counts both.
    from agent.compaction_stats import build_inturn_stats
    anchor = {"role": "system", "content": "SYS " * 50}
    tagged = _regex_miss_summary_row(tagged=True)                 # tag-only (regex misses)
    untagged = {"role": "assistant",
                "content": "[Session Arc Summary (d1, node 7)] y"}  # regex-only
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, tagged, untagged] + msgs[-3:]
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    ok, why = stats.validate()
    assert ok, why
    assert stats.summary_messages == 2


def test_tag_missing_tripwire_fires_once():
    # INV-7: an LCM session whose summary row lacks the tag fires on_tag_missing
    # exactly once, even across the two partition scans.
    from agent.compaction_stats import build_inturn_stats
    fires = []
    anchor = {"role": "system", "content": "SYS " * 50}
    untagged_marker = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] z"}
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, untagged_marker] + msgs[-3:]
    build_inturn_stats(
        messages=msgs, compressed=compressed, estimator=_est,
        engine_is_lcm=True, on_tag_missing=lambda: fires.append(1),
    )
    assert sum(fires) == 1, f"tripwire must fire exactly once, fired {sum(fires)}"


def test_tag_present_no_tripwire():
    # a properly-tagged LCM summary fires NO tripwire.
    from agent.compaction_stats import build_inturn_stats
    fires = []
    anchor = {"role": "system", "content": "SYS " * 50}
    tagged = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] z", "_lcm_summary": True}
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, tagged] + msgs[-3:]
    build_inturn_stats(
        messages=msgs, compressed=compressed, estimator=_est,
        engine_is_lcm=True, on_tag_missing=lambda: fires.append(1),
    )
    assert sum(fires) == 0, "a tagged summary must not fire the tag-missing tripwire"


def test_tag_missing_silent_when_engine_not_lcm():
    # built-in-engine sessions (engine_is_lcm=False) never fire the tripwire even
    # with an untagged marker row.
    from agent.compaction_stats import build_inturn_stats
    fires = []
    anchor = {"role": "system", "content": "SYS " * 50}
    untagged_marker = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] z"}
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, untagged_marker] + msgs[-3:]
    build_inturn_stats(
        messages=msgs, compressed=compressed, estimator=_est,
        engine_is_lcm=False, on_tag_missing=lambda: fires.append(1),
    )
    assert sum(fires) == 0


# ───────────────────────────────────────────────────────────────────────────
# Hygiene reconcile fix — comp-side vs pre-side kept tokens (2026-06-22 live bug)
# Spec: plans/2026-06-22_hygiene-reconcile-fix-and-stats-watcher-SPEC.md
# The post identity (kept+summary+anchor==post=estimator(comp)) must use a
# COMP-side kept-token measurement; the pre identity (cleared+folded+kept==pre)
# uses the PRE-side kept rows. Conflating them was the live reconcile failure.
# ───────────────────────────────────────────────────────────────────────────

def _hyg_sanitized_tail():
    """A hygiene shape where the comp kept tail does NOT signature-match its raw
    original (LCM sanitized it) — pre-side kept rows differ from comp-side."""
    marker = "[Session Arc Summary (d1, node 7)] " + ("S " * 80)
    # raw history: folded chat + two BIG kept-tail originals
    raw_kept_1 = {"role": "user", "content": "KEPT ONE " + ("x " * 400)}
    raw_kept_2 = {"role": "assistant", "content": "KEPT TWO " + ("y " * 400)}
    pre = [{"role": "user", "content": f"folded {i} " + ("z " * 40)} for i in range(8)] + [
        raw_kept_1, raw_kept_2,
    ]
    # comp: summary + SANITIZED (smaller, non-sig-matching) kept tail
    comp = [
        {"role": "assistant", "content": marker, "_lcm_summary": True},
        {"role": "user", "content": "KEPT ONE small"},
        {"role": "assistant", "content": "KEPT TWO small"},
    ]
    return pre, comp


def test_hygiene_reconciles_when_kept_tail_sanitized():
    """RED on pre-fix: post identity used pre-side kept (28) but comp post is small,
    so kept(pre) != comp kept → fails. GREEN: comp-side kept_tokens reconciles."""
    from agent.compaction_stats import build_hygiene_stats
    pre, comp = _hyg_sanitized_tail()
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why


def test_hygiene_reconciles_when_kept_pre_larger_than_comp():
    """The live 16:34 shape: the comp kept tail is SANITIZED smaller than its raw
    original, so pre-side and comp-side kept-token measurements diverge. The fix
    measures the POST identity over the comp-side tail → reconciles."""
    from agent.compaction_stats import build_hygiene_stats
    # raw kept originals are BIG; the comp tail is the SAME rows but content-cleaned
    # (sanitized) so they no longer signature-match → pre-side kept rows fold, and
    # the post identity must rest on the comp-side tail, not the big pre originals.
    big1 = "KEPT ALPHA " + ("q " * 2000)
    big2 = "KEPT BETA " + ("w " * 2000)
    pre = [{"role": "user", "content": f"f{i} " + ("z " * 30)} for i in range(6)] + [
        {"role": "user", "content": big1}, {"role": "assistant", "content": big2},
    ]
    comp = [
        {"role": "assistant", "content": "[Recent Summary (d0, node 1)] " + ("s " * 200), "_lcm_summary": True},
        {"role": "user", "content": "KEPT ALPHA (cleaned)"},
        {"role": "assistant", "content": "KEPT BETA (cleaned)"},
    ]
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why


def test_hygiene_pre_identity_still_holds():
    """The pre token identity (cleared+folded+kept_pre==pre) must still reconcile
    after the fix — it uses the pre-side kept rows."""
    from agent.compaction_stats import build_hygiene_stats
    pre, comp = _hyg_sanitized_tail()
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    # pre identity: cleared + folded + kept_pre ≈ pre
    lhs = stats.cleared_tokens + stats.folded_tokens + (stats.kept_pre_tokens or 0)
    assert abs(lhs - stats.pre_tokens) <= 8, (lhs, stats.pre_tokens)


def test_postfix_reconciles_real_session_sanitized_tail():
    """Real-session replay oracle (Pass-2 bpp B2 — replaces the v0.2 tautology).
    Load a real session from the live lcm.db, build a SANITIZED kept tail
    (assistant content cleaned so it no longer signature-matches raw — the real
    LCM behavior), and assert the fix reconciles, with kept measured over the
    ACTUAL comp rows (not post-summary-anchor). Skips when no lcm.db (CI)."""
    import os, sqlite3
    from agent.compaction_stats import build_hygiene_stats, hygiene_eligible_msgs
    db = os.path.expanduser("~/.hermes/lcm.db")
    if not os.path.exists(db):
        import pytest
        pytest.skip("no local lcm.db (CI / fresh checkout)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT session_id, COUNT(*) n FROM messages GROUP BY session_id "
            "HAVING n>=200 ORDER BY n DESC LIMIT 1"
        ).fetchone()
        if not row:
            import pytest
            pytest.skip("no large real session available")
        sid = row[0]
        msgs = con.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY store_id", (sid,)
        ).fetchall()
    finally:
        con.close()
    history = [{"role": r[0], "content": (r[1] or "")} for r in msgs]
    eligible = hygiene_eligible_msgs(history)
    raw = [{**m} for m in history]
    fresh = [{**m} for m in eligible[-32:]]
    for m in fresh:  # sanitize the tail → no signature match against raw (real LCM)
        if m["role"] == "assistant":
            m["content"] = (m["content"] or "")[:50] + " [sanitized]"
    comp = [{"role": "user", "content": "[Recent Summary (d0, node 1)] x"}] + fresh
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, f"real-session sanitized-tail must reconcile after the fix: {why}"
    assert stats.kept_pre_tokens != stats.kept_tokens, (
        "fixture didn't exercise the pre/comp divergence — pick a session whose tail sanitizes"
    )


def test_postfix_reconciles_committed_realshape_fixture():
    """CI-runnable non-tautological floor (Pass-3 bpp #1): a COMMITTED
    content-scrubbed real-shape fixture (same-length filler → identical estimator
    inputs + signature behavior, zero real content). Runs in CI where lcm.db is
    absent. Sanitizes the kept tail → exercises the pre/comp divergence, recomputes
    kept from the ACTUAL comp rows, asserts reconcile."""
    from agent.compaction_stats import build_hygiene_stats, hygiene_eligible_msgs
    fx = os.path.join(os.path.dirname(__file__), "fixtures", "hygiene_reconcile_real_shape.json")
    with open(fx, encoding="utf-8") as fh:
        scrubbed = json.load(fh)["messages"]
    eligible = hygiene_eligible_msgs(scrubbed)
    raw = [{**m} for m in scrubbed]
    fresh = [{**m} for m in eligible[-32:]]
    for m in fresh:  # sanitize the tail → no signature match (real LCM behavior)
        if m["role"] == "assistant":
            m["content"] = (m["content"] or "")[:50] + "yyyy"
    comp = [{"role": "user", "content": "[Recent Summary (d0, node 1)] " + ("z" * 200)}] + fresh
    stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, f"committed real-shape fixture must reconcile after the fix: {why}"
    assert stats.kept_pre_tokens != stats.kept_tokens, "fixture must exercise pre/comp divergence"


def test_text_part_type_sets_agree_across_sites():
    """Part 3 drift-guard (Pass-1 bpp): the {text,input_text,output_text} set is
    defined in FOUR places (core compaction_stats, LCM message_content, LCM engine,
    gateway api_server). The 4-way dedup is WONTFIX (cross-layer, ~zero gain), but
    this asserts MEMBERSHIP equality so the duplicated constant can't silently
    drift into the '5th part-type outage' class. set()-normalized (tuple vs set
    vs frozenset all compare)."""
    from agent.compaction_stats import _TEXT_PART_TYPES as core_set
    from plugins.context_engine.lcm.message_content import _TEXT_PART_TYPES as lcm_mc
    from plugins.context_engine.lcm.engine import _VISIBLE_TEXT_PART_TYPES as lcm_eng
    from gateway.platforms.api_server import _TEXT_PART_TYPES as api_set
    canonical = {"text", "input_text", "output_text"}
    assert set(core_set) == canonical
    assert set(lcm_mc) == canonical
    assert set(lcm_eng) == canonical
    assert set(api_set) == canonical


def test_preaxis_reconciles_on_real_sessions():
    """Codifies the §0.13 investigation: the PRE identity holds on real sessions —
    the 00:48 179K gap is NOT a live bug. Guards a future regression that would
    make the pre partition non-exhaustive."""
    import os, sqlite3
    from agent.compaction_stats import build_hygiene_stats, hygiene_eligible_msgs
    db = os.path.expanduser("~/.hermes/lcm.db")
    if not os.path.exists(db):
        import pytest
        pytest.skip("no local lcm.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT session_id FROM messages GROUP BY session_id HAVING COUNT(*)>=200 "
            "ORDER BY COUNT(*) DESC LIMIT 5"
        ).fetchall()
        checked = 0
        for (sid,) in rows:
            msgs = con.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY store_id", (sid,)
            ).fetchall()
            history = [{"role": r[0], "content": (r[1] or "")} for r in msgs]
            eligible = hygiene_eligible_msgs(history)
            if len(eligible) < 4:
                continue
            raw = [{**m} for m in history]
            fresh = [{**m} for m in eligible[-32:]]
            comp = [{"role": "user", "content": "[Recent Summary (d0, node 1)] x"}] + fresh
            stats = build_hygiene_stats(raw_history=raw, eligible_msgs=eligible, compressed=comp,
                                        estimator=_est, engine_is_lcm=True)
            lhs = stats.cleared_tokens + stats.folded_tokens + (stats.kept_pre_tokens or 0)
            assert abs(lhs - stats.pre_tokens) <= 8, f"{sid}: pre axis gap {stats.pre_tokens - lhs}"
            checked += 1
        if checked == 0:
            import pytest
            pytest.skip("no eligible real sessions")
    finally:
        con.close()



def test_estimator_additive_over_comp_partition():
    """§0.10: the estimator IS row-additive within tol over the comp 3-subset
    partition — so the comp-side fix reconciles LIVE, not just synthetic. (Locks
    the premise the Pass-1 BLOCK feared was false.)"""
    summary = [{"role": "assistant", "content": "word " * 8000}]
    kept = [{"role": "user", "content": "u " * 1200}, {"role": "assistant", "content": "a " * 1100}]
    anchor = []
    comp = summary + kept + anchor
    whole = _est(comp)
    parts = _est(summary) + _est(kept) + (_est(anchor) if anchor else 0)
    assert abs(parts - whole) <= 8, (parts, whole)


def test_both_kept_populations_measured_independently():
    """Each kept field has teeth: corrupt kept_tokens ALONE → post fails; corrupt
    kept_pre_tokens ALONE → pre fails. Neither is a total-minus-others derivation."""
    from agent.compaction_stats import build_hygiene_stats
    import dataclasses
    pre, comp = _hyg_sanitized_tail()
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    assert stats.validate()[0]
    bad_post = dataclasses.replace(stats, kept_tokens=stats.kept_tokens + 5000)
    assert not bad_post.validate()[0], "corrupting comp-side kept must fail the post identity"
    bad_pre = dataclasses.replace(stats, kept_pre_tokens=(stats.kept_pre_tokens or 0) + 5000)
    assert not bad_pre.validate()[0], "corrupting pre-side kept must fail the pre identity"


def test_kept_messages_display_unchanged():
    """The announce renders kept_messages (a COUNT). Post-message-axis-fix it is the
    COMP-side count (what's actually in live context), so the granular announce shows
    the real kept-tail size — never the pre-side 0 that the sanitized-tail bug produced."""
    from agent.compaction_stats import build_hygiene_stats
    pre, comp = _hyg_sanitized_tail()
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    # comp kept tail = 2 sanitized rows → kept_messages == 2 (truth in context),
    # while the pre-side kept count is 0 (sanitized rows don't signature-match raw).
    assert stats.kept_messages == 2, "display count must be the comp-side kept tail"
    assert stats._kept_pre_messages == 0, "pre-side kept is 0 here (sanitized tail)"
    assert stats.kept_messages != stats._kept_pre_messages, "this fixture must diverge"


def test_message_axis_two_populations_measured_independently():
    """CI-caught regression (PR #101): the message axis has the SAME two-population
    split as the token axis. kept_messages must be the COMP-side count (display +
    POST identity); kept_pre_messages the PRE-side count (PRE/eligible identities).
    Corrupt each independently → the matching identity fails. Guards against a future
    collapse back to one field (which rendered 'kept 0 recent chat' on live sessions)."""
    from agent.compaction_stats import build_hygiene_stats
    import dataclasses
    pre, comp = _hyg_sanitized_tail()
    stats = build_hygiene_stats(raw_history=pre, eligible_msgs=pre, compressed=comp,
                                estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, f"sanitized-tail message axis must reconcile after the fix: {why}"
    # post_messages is MEASURED (len(comp)), not the tautological kept+summary+anchor
    assert stats.post_messages == len(comp)
    # corrupt comp-side kept count → POST message identity fails
    bad_post = dataclasses.replace(stats, kept_messages=stats.kept_messages + 3)
    assert not bad_post.validate()[0], "corrupting comp-side kept_messages must fail POST"
    # corrupt pre-side kept count → PRE/eligible message identity fails
    bad_pre = dataclasses.replace(stats, kept_pre_messages=(stats._kept_pre_messages) + 3)
    assert not bad_pre.validate()[0], "corrupting pre-side kept_pre_messages must fail PRE"


def test_inturn_kept_pre_messages_defaults_to_comp_side():
    """In-turn path doesn't set kept_pre_messages → it defaults to comp-side
    kept_messages, so the in-turn (already-correct) reconcile is unchanged."""
    from agent.compaction_stats import build_inturn_stats
    anchor = {"role": "system", "content": "SYS " * 50}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] x", "_lcm_summary": True}
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, summary] + msgs[-3:]
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    assert stats.validate()[0]
    assert stats.kept_pre_messages is None  # not set → property falls back to kept_messages
    assert stats._kept_pre_messages == stats.kept_messages


def test_inturn_kept_pre_defaults_to_comp_side():
    """In-turn path doesn't set kept_pre_tokens → it defaults to comp-side kept_tokens,
    so the in-turn (already-correct) reconcile is unchanged."""
    from agent.compaction_stats import build_inturn_stats
    anchor = {"role": "system", "content": "SYS " * 50}
    summary = {"role": "assistant", "content": "[Recent Summary (d0, node 1)] x", "_lcm_summary": True}
    body = [_blockmsg("assistant", f"chat {i}") for i in range(10)]
    msgs = [anchor] + body
    compressed = [anchor, summary] + msgs[-3:]
    stats = build_inturn_stats(messages=msgs, compressed=compressed, estimator=_est)
    assert stats.validate()[0]
    assert stats.kept_pre_tokens is None  # not set → property falls back to kept_tokens
    assert stats._kept_pre_tokens == stats.kept_tokens



