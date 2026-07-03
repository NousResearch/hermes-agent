"""Escalation ladder for the method-explorer test suite.

The tiers run cheapest-and-most-foundational FIRST and most-expensive-and-adversarial
LAST. The gauntlet runner (`run.py --gauntlet`) walks them in order and STOPS at the
first tier with any (post-retry) failure — so when something breaks you fix the
earliest, simplest cause before spending tokens on the harder tiers whose premises it
invalidates. `run.py --survey` runs ALL tiers and prints a stratified scorecard.

Ordering rationale:
  T0 validates the machinery itself (driver control logic + trace engine + the gauntlet
     gate) — offline, deterministic, ~0 tokens. If this is red, every live result below
     is noise.
  T1 validates the two load-bearing premises every generated scenario depends on: the
     phonetic-tag convention and the compact plan-tree + lean journal format.
     (test_c1 also asserts backtrack behavior; kept unified — splitting would weaken the
     gate, and the failure-triggered trace shows WHICH aspect failed.)
  T2 is the core loop (don't-fabricate, backtrack, complete records, no re-expand, resume).
  T3 is the discriminating behaviors (K5-jump vs backtrack, guard-halt vs exhaustion,
     hard-vs-soft relax, verify-correctness, driver end-to-end resume).
  T4 is adversarial + generative (lying tool, context-scoped reopen, property/metamorphic,
     traps) — the hardest and most expensive, only worth running on a green foundation.

Retry policy (per tier): live LLM tests are stochastic, so the gauntlet retries a failed
test FUNCTION once at T2-T4 (pass-on-retry = FLAKY-PASS, logged + trended). T0 gets no
retry (deterministic — a failure is real). T1 gets no retry either: the convention/format
premises should be near-deterministic in a healthy skill, so a flaky T1 is itself
evidence the premise is degrading — treat it as red.

Each tier: (index, name, one-line intent, [module names]). Reorder here to re-shape the
ladder; the runner and `--tiers` view read straight from this list.
"""

TIERS = [
    (0, "Foundation", "offline machinery: driver control logic + trace engine + gauntlet gate (~0 tokens)",
     ["test_15_driver_loop", "test_trace", "test_gauntlet"]),
    (1, "Conventions", "load-bearing premises: phonetic-tag convention + compact/lean format",
     ["test_00_builder_convention", "test_c1_compact_backtrack"]),
    (2, "Core behaviors", "the fundamental loop: no-fabricate, backtrack, records, no-reexpand, resume",
     ["test_01_anti_fabrication", "test_02_backtrack_success",
      "test_03_decision_record_completeness", "test_04_no_reexpand_tombstone",
      "test_10_resume", "test_12_resume_completed", "test_18_malformed_tree_resume",
      "test_19_sim_never_goes_real"]),
    (3, "Discriminating", "subtle distinctions: K5-jump, guard-halt, hard-vs-soft, verify, driver-resume",
     ["test_05_upstream_jump_k5", "test_06_necessity_propose_and_log",
      "test_07_guard_halt_distinct", "test_11_verify_correctness",
      "test_13_relaxation_monotonicity", "test_14_structural_blocker_relax",
      "test_16_driver_resume_integration", "test_17_sim_locality_label"]),
    (4, "Adversarial/property", "hardest + generative: lying tool, context-reopen, properties, traps",
     ["test_08_adversarial_lying_tool", "test_09_context_scoped_reopen",
      "test_properties", "test_traps"]),
]

# Tiers whose modules make NO container/model calls — free to run, deterministic.
OFFLINE_TIERS = {0}

# Tiers where the gauntlet may retry a failed test function ONCE (FLAKY-PASS semantics).
# T0 deterministic and T1 near-deterministic-by-design get none: their failures are real.
RETRY_TIERS = {2, 3, 4}

# Nominal `hermes -z` invocations per module run (1 rep, no no-op retries). The ladder's
# rationale is COST escalation, so cost labels must be call estimates, not module counts:
# test_16's driver loops ticks, test_13 runs a hard+soft pair, test_properties runs 3
# metamorphic pairs. Modules not listed default to 1 (offline tiers to 0).
EST_CALLS = {
    "test_15_driver_loop": 0, "test_trace": 0, "test_gauntlet": 0,
    "test_13_relaxation_monotonicity": 2,
    "test_17_sim_locality_label": 2,      # backtrack pair + sole-source exhaustion
    "test_16_driver_resume_integration": 4,   # 2-6 driver ticks; nominal midpoint
    "test_properties": 6,                     # 3 metamorphic pairs
    "test_traps": 2,
}


def est_calls(module, tier_idx=None):
    if module in EST_CALLS:
        return EST_CALLS[module]
    return 0 if tier_idx in OFFLINE_TIERS else 1


def tier_calls(idx):
    """Summed nominal model-call estimate for one tier."""
    _, _, _, mods = tier(idx)
    return sum(est_calls(m, idx) for m in mods)


def tier(index):
    for t in TIERS:
        if t[0] == index:
            return t
    raise KeyError(f"no tier {index}")


def modules_through(lo=0, hi=None):
    """Flat, ordered module list for tiers in [lo, hi] (hi defaults to the last tier)."""
    hi = TIERS[-1][0] if hi is None else hi
    out = []
    for idx, _name, _intent, mods in TIERS:
        if lo <= idx <= hi:
            out.extend(mods)
    return out


def cost_label(idx):
    """Human cost label for a tier: call estimate, not module count."""
    if idx in OFFLINE_TIERS:
        return "offline, ~0 tokens"
    return f"~{tier_calls(idx)} model call(s)"


def render_ladder():
    """A human-readable view of the ladder (used by `run.py --tiers`)."""
    lines = ["Escalation ladder (run low→high; the gauntlet stops at the first red tier):", ""]
    for idx, name, intent, mods in TIERS:
        retry = "no retry" if idx not in RETRY_TIERS else "retry-once on fail"
        lines.append(f"  Tier {idx} · {name}  [{cost_label(idx)} · {retry}]")
        lines.append(f"    {intent}")
        for m in mods:
            n = est_calls(m, idx)
            lines.append(f"      - {m}" + (f"  (~{n} calls)" if n > 1 else ""))
        lines.append("")
    total = sum(tier_calls(i) for i, *_ in TIERS)
    lines.append(f"  Full ladder ≈ {total} model calls (nominal; no-op retries and --reps multiply).")
    return "\n".join(lines)
