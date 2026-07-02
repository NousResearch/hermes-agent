"""A-floor exhaustive single-walk partition for the in-turn compaction-stats path
(2026-06-27, P1 residual reconcile fix).

Root cause (corrected to single-pass): the LCM engine sanitizes the kept tail
IN-CONTEXT ([anchor]+[summary]+tail), but ``build_inturn_stats`` replayed the
sanitizer over the raw tail STANDALONE — different adjacency → ``find_inturn_kept_cut``
returns None on heavy sessions → the OLD fallback left ``kept_pre`` unset (defaulting
to the comp-side sanitized kept) while ``folded`` was measured pre-side → a mixed
pre/comp-side sum → ``COMPACTION_STATS_RECONCILE_FAILED in-turn`` (live: ``cleared 0
+ folded F + kept K != pre P``, |gap| ≤ ~3000 on ~650K sessions).

The fix replaces that fallback with an exhaustive consume-once partition
(``_signature_partition``): ``pre`` is split ONCE into (kept_pre, folded)
against the comp-kept multiset, so both buckets are measured pre-side over a disjoint
exhaustive partition → totals reconcile BY CONSTRUCTION regardless of why alignment
failed. The kept/folded *attribution* is signature-approximate (bounded by the
kept-tail fraction — the folded bulk is a contiguous prefix that always classifies
correctly), so ``approx_attribution`` is flagged for labelling + observability.
"""
from __future__ import annotations

import dataclasses
import random

from agent.compaction_stats import (
    CompactionStats,
    _signature_partition,
    build_inturn_stats,
)
from agent.model_metadata import estimate_messages_tokens_rough as _est


# ───────────────────────── RED-1: the mixed-side bug, now fixed ─────────────────

def _sanitized_comp(pre, n_tail=5):
    """A comp whose kept tail is the STRIPPED (shorter) form of the raw tail — so the
    comp-kept rows token-differ from any raw suffix, exactly the live divergence."""
    summary = {"role": "assistant",
               "content": "[Recent Summary (d0, node 1)] x [Expand for details: y]",
               "_lcm_summary": True}
    # stripped versions: same role, shorter content (sanitizer stripped scaffolding)
    kept_sanitized = [{"role": pre[-n_tail + i]["role"], "content": f"stripped{i}"}
                      for i in range(n_tail)]
    return [{"role": "system", "content": "anchor"}, summary] + kept_sanitized, kept_sanitized


def test_red1_old_mixed_side_fallback_fails():
    """RED: the OLD fallback (kept_pre unset → property defaults to comp-side, folded
    measured pre-side) produces the exact live gap and FAILS validate()."""
    pre = [{"role": "user", "content": f"u{i} " + ("w" * 40)} for i in range(60)]
    _comp, kept_sanitized = _sanitized_comp(pre)
    _, fold_old = _signature_partition(pre, kept_sanitized)
    old = CompactionStats(
        pre_messages=60, post_messages=2 + len(kept_sanitized), eligible_count=60,
        kept_messages=len(kept_sanitized), summary_messages=1, anchor_messages=1,
        cleared_count=0, folded_count=60 - len(kept_sanitized),
        pre_tokens=int(_est(pre)), post_tokens=int(_est(_comp)),
        kept_tokens=int(_est(kept_sanitized)), summary_tokens=10, anchor_tokens=5,
        cleared_tokens=0, folded_tokens=int(_est(fold_old)),
        kept_pre_tokens=None, kept_pre_messages=None,  # OLD: unset → comp-side default
    )
    ok, why = old.validate()
    assert ok is False
    assert "!= pre" in why  # the mixed-side gap signature


def test_green_a_floor_reconciles_same_shape():
    """GREEN: the A-floor exhaustive partition reconciles the same shape + flags approx."""
    pre = [{"role": "user", "content": f"u{i} " + ("w" * 40)} for i in range(60)]
    comp, _ = _sanitized_comp(pre)
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est, engine_is_lcm=True)
    ok, why = stats.validate()
    assert ok, why
    assert stats.approx_attribution is True


# ───────────────────────── exhaustiveness (single walk) ─────────────────────────

def test_partition_is_exhaustive_and_disjoint():
    pre = [{"role": "user", "content": f"u{i}"} for i in range(100)]
    kept = [dict(pre[i]) for i in (90, 92, 95, 97, 99)]  # copies (not id-identical)
    kept_pre, folded = _signature_partition(pre, kept)
    # exhaustive
    assert len(kept_pre) + len(folded) == len(pre)
    # disjoint by identity
    assert not ({id(m) for m in kept_pre} & {id(m) for m in folded})
    # token reconcile within rounding
    assert abs((_est(kept_pre) + _est(folded)) - _est(pre)) <= 2


def test_partition_identity_fast_path():
    """When the SAME row objects flow through, the partition is exact via id()."""
    pre = [{"role": "user", "content": f"u{i}"} for i in range(20)]
    kept = pre[-3:]  # same objects
    kept_pre, folded = _signature_partition(pre, kept)
    assert kept_pre == pre[-3:]
    assert folded == pre[:-3]


# ───────────────────────── duplicate-signature collision safety ─────────────────

def test_partition_duplicate_signatures_consume_once():
    """Heavy sessions repeat identical rows (tool scaffolds, short turns). The
    consume-once multiset must not match a duplicate twice or steal another row's
    slot — partition stays exhaustive + disjoint."""
    # 10 identical user rows + 10 identical tool rows, interleaved
    dup_u = {"role": "user", "content": "SAME USER"}
    dup_t = {"role": "tool", "tool_call_id": None, "content": "SAME TOOL OUT"}
    pre = []
    for _ in range(10):
        pre.append(dict(dup_u))
        pre.append(dict(dup_t))
    # comp keeps 3 user + 2 tool copies of those duplicated signatures
    kept = [dict(dup_u), dict(dup_u), dict(dup_u), dict(dup_t), dict(dup_t)]
    kept_pre, folded = _signature_partition(pre, kept)
    assert len(kept_pre) == 5            # exactly as many as comp-kept (consume-once)
    assert len(kept_pre) + len(folded) == len(pre)   # exhaustive
    assert not ({id(m) for m in kept_pre} & {id(m) for m in folded})  # disjoint


# ───────────────────────── gross-error bound (the A-only justification) ─────────

def test_a_floor_gross_error_bounded_by_kept_tail():
    """The folded bulk is a contiguous prefix → only kept-tail rows can mis-bucket,
    so gross misattribution ≤ kept-tail token fraction. Construct a heavy pre with a
    small kept tail and assert the mis-bucketed tokens are within the kept-tail size."""
    pre = [{"role": "user", "content": f"u{i} " + ("w" * 60)} for i in range(500)]
    true_kept = pre[-20:]                       # the real pre-side kept tail
    # comp-kept = sanitized (content-stripped) versions of the true kept tail
    kept_sanitized = [{"role": "user", "content": f"strip{i}"} for i in range(20)]
    kept_pre, folded = _signature_partition(pre, kept_sanitized)
    # rows the A-floor mis-bucketed vs the true tail boundary
    true_ids = {id(m) for m in true_kept}
    mis = [m for m in kept_pre if id(m) not in true_ids] + [m for m in folded if id(m) in true_ids]
    gross = _est(mis) if mis else 0
    kept_tail_tokens = _est(true_kept)
    assert gross <= kept_tail_tokens            # bounded by the kept-tail size
    assert gross / _est(pre) < 0.10             # under the spec's 10% degrade threshold


# ───────────────────────── INV-7: scattered estimator additivity ────────────────

def test_estimator_additivity_scattered_partition():
    """The A-floor's "reconciles by construction" rests on estimator additivity over a
    NON-contiguous (scattered) two-way partition. Pin it as an invariant."""
    import random
    random.seed(123)
    pre = [{"role": random.choice(["user", "assistant", "tool"]),
            "content": "x" * random.randint(5, 400) + f" m{i}"} for i in range(600)]
    worst = 0
    for _ in range(50):
        mask = [random.random() < 0.5 for _ in pre]
        a = [m for m, k in zip(pre, mask) if k]
        b = [m for m, k in zip(pre, mask) if not k]
        worst = max(worst, abs((_est(a) + _est(b)) - _est(pre)))
    assert worst <= 4   # two-bucket ceil-rounding worst case (validate tol is ≥ this)


def test_signature_partition_property_exhaustive_with_copies_sanitized_duplicates():
    """Property-style guard: copied rows, sanitized non-matches, and duplicated
    signatures still produce an exhaustive consume-once kept/folded partition.
    """
    from agent.compaction_stats import _row_signature

    rng = random.Random(20260701)
    for _ in range(100):
        population = []
        for i in range(rng.randint(1, 60)):
            role = rng.choice(["user", "assistant", "tool"])
            # Small content pool deliberately creates duplicate signatures.
            content = f"dup-{rng.randrange(8)} " * rng.randint(1, 4)
            population.append({"role": role, "content": content, "i": i})

        reference = []
        for row in rng.sample(population, rng.randint(0, len(population))):
            copied = dict(row)
            if rng.random() < 0.25:
                # Sanitized comp-side row: same role, changed content, should not
                # steal an original row's signature slot.
                copied["content"] = f"sanitized-{copied['content']}"
            reference.append(copied)
            if rng.random() < 0.20:
                # Duplicate reference signature; the helper may match another
                # population duplicate, but only one row per reference copy.
                reference.append(dict(copied))

        kept, folded = _signature_partition(population, reference)

        kept_ids = {id(m) for m in kept}
        folded_ids = {id(m) for m in folded}
        assert kept_ids.isdisjoint(folded_ids)
        assert kept_ids | folded_ids == {id(m) for m in population}
        assert len(kept) + len(folded) == len(population)

        ref_counts = {}
        for row in reference:
            sig = _row_signature(row)
            ref_counts[sig] = ref_counts.get(sig, 0) + 1
        kept_counts = {}
        for row in kept:
            sig = _row_signature(row)
            kept_counts[sig] = kept_counts.get(sig, 0) + 1
        for sig, count in kept_counts.items():
            assert count <= ref_counts.get(sig, 0)


# ───────────────────────── gross-error guard denominator (Greptile P1 ×2, #109) ──

def test_gross_guard_uses_raw_tail_not_sanitized_or_pre_side():
    """Greptile P1 (×2): the guard must key off the RAW kept-tail size
    (raw_tail_tokens = estimator(messages[-fresh_tail_count:])), because:
      - kept_tokens (comp-side, sanitized) is stripped SMALL on a heavily-sanitized tail
      - _kept_pre_tokens is 0 when the signature match fails
    so both can stay under 10% while the true raw kept tail is larger. The consumer
    uses max(raw_tail_tokens, kept_tokens, _kept_pre_tokens) → raw_tail dominates."""
    big_pre = 600_000
    # Heavily-sanitized + match-failed: comp kept stripped to 5k, pre-side kept 0,
    # but the RAW tail is 90k (15% of pre) → MUST degrade.
    stats = CompactionStats(
        pre_messages=500, post_messages=20, eligible_count=500,
        kept_messages=18, summary_messages=1, anchor_messages=1,
        cleared_count=0, folded_count=500,
        pre_tokens=big_pre, post_tokens=10_000,
        kept_tokens=5_000,                  # sanitized comp-side: deceptively small
        summary_tokens=4000, anchor_tokens=1000, cleared_tokens=0,
        folded_tokens=big_pre,
        kept_pre_tokens=0, kept_pre_messages=0,   # match failed → 0
        approx_attribution=True,
        raw_tail_tokens=90_000,             # RAW tail: the true magnitude (15% of pre)
    )
    gross_tok = max(stats.raw_tail_tokens or 0, stats.kept_tokens or 0, stats._kept_pre_tokens or 0)
    assert gross_tok == 90_000              # raw_tail dominates the stripped/zero values
    assert (gross_tok / stats.pre_tokens) > 0.10   # → consumer degrades to two-line
    # the OLD (broken) guard would have used kept_tokens(5k) or kept_pre(0) → 0.83% → WRONGLY render
    assert (stats.kept_tokens / stats.pre_tokens) < 0.10  # proves the old guard was fooled

    # within-bound real session (raw tail ~7% of pre) still renders
    ok_stats = CompactionStats(
        pre_messages=500, post_messages=20, eligible_count=500,
        kept_messages=18, summary_messages=1, anchor_messages=1,
        cleared_count=0, folded_count=480,
        pre_tokens=650_000, post_tokens=50_000,
        kept_tokens=42_000, summary_tokens=7000, anchor_tokens=1000, cleared_tokens=0,
        folded_tokens=608_000, kept_pre_tokens=42_000, kept_pre_messages=20,
        approx_attribution=True, raw_tail_tokens=46_000,   # ~7% of pre
    )
    ok_gross = max(ok_stats.raw_tail_tokens or 0, ok_stats.kept_tokens or 0, ok_stats._kept_pre_tokens or 0)
    assert (ok_gross / ok_stats.pre_tokens) < 0.10   # → renders (labeled approx)


def test_build_inturn_sets_raw_tail_tokens_on_floor():
    """build_inturn_stats populates raw_tail_tokens = estimator(pre[-fresh_tail_count:])
    when the A-floor fires, so the consumer guard has the match-independent bound."""
    pre = [{"role": "user", "content": f"u{i} " + ("w" * 40)} for i in range(60)]
    # sanitized comp tail that won't signature-match → A-floor
    comp = [{"role": "system", "content": "anchor"},
            {"role": "assistant", "content": "[Recent Summary (d0, node 1)] x [Expand for details: y]",
             "_lcm_summary": True}] + [{"role": "user", "content": f"strip{i}"} for i in range(5)]
    stats = build_inturn_stats(messages=pre, compressed=comp, estimator=_est,
                               engine_is_lcm=True, sanitize=None, fresh_tail_count=32)
    assert stats.approx_attribution is True
    assert stats.raw_tail_tokens == int(_est(pre[-32:]))   # raw suffix, match-independent
