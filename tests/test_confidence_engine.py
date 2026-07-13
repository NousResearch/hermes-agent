#!/usr/bin/env python3
"""Tests for confidence_engine.py — generic core"""

import json, sys, tempfile, os
sys.path.insert(0, os.path.dirname(__file__))
from confidence_engine import (
    preprocess, assess, assess_batch, validate_profile,
    _default_profiles, _apply_decay, _map_thresholds, _detect_bonus,
)

PASS, FAIL = 0, 0

def ok(msg):
    global PASS; PASS += 1; print(f"✅ {msg}")

def no(msg):
    global FAIL; FAIL += 1; print(f"❌ {msg}")


# ── T1: user_stated → high score ──
dims = preprocess("User always prefers dark mode for all applications", "user_stated", 5)
r = assess(dims)
assert r["score"] > 0.5, f"expected >0.5, got {r['score']}"
assert r["level"] in ("long_term", "permanent"), f"expected long_term+, got {r['level']}"
ok("T1 user-stated-high-score")

# ── T2: llm_inferred → lower score ──
dims = preprocess("Agent inferred user might like Python", "llm_inferred", 30)
r = assess(dims)
assert r["score"] <= 0.8, f"inferred should score lower, got {r['score']}"
ok("T2 llm-inferred-lower")

# ── T3: old entry → decay ──
dims = preprocess("User used to prefer tabs over spaces", "llm_inferred", 400)
r = assess(dims)
assert r["score"] < 0.5, f"old entry should decay, got score={r['score']}"
ok("T3 old-entry-decay")

# ── T4: transient → accelerated decay ──
dims = preprocess("Temporary workaround for one-time issue", "llm_inferred", 10)
r = assess(dims)
assert dims["fact_type_transient"], "should detect transient"
ok("T4 transient-detection")

# ── T5: constant → bypass decay ──
dims = preprocess("User was born on January 15 1990", "user_stated", 500)
dims["fact_type_constant"] = True  # simulate safelist match
r = assess(dims)
assert r["score"] > 0.5, f"constant should not decay, got {r['score']}"
ok("T5 constant-bypass-decay")

# ── T6: bonus keyword detection ──
dims = preprocess("Iron law: never commit directly to main", "user_stated", 1)
r = assess(dims)
assert dims.get("lesson_learned") or dims.get("permanence_marker"), "should detect bonus"
ok("T6 bonus-keywords")

# ── T7: completed task → negative bonus ──
dims = preprocess("Completed: deployed v2.1 to production", "llm_inferred", 1)
r = assess(dims)
assert dims.get("completed_task"), "should detect completed_task"
assert "bonus_completed_task" in r["breakdown"], f"breakdown: {r['breakdown']}"
ok("T7 completed-negative-bonus")

# ── T8: permanent threshold ──
dims = preprocess("User always uses git from terminal, never GUI", "user_stated", 1)
r = assess(dims)
assert r["score"] >= 0.5, f"user_stated preference should score high, got {r['score']}"
ok("T8 permanent-threshold")

# ── T9: volatile entry ──
dims = preprocess("maybe we could try vscode for this one project", "llm_inferred", 365)
r = assess(dims)
assert r["level"] in ("volatile", "short_term"), f"should be low, got {r['level']}"
ok("T9 volatile-entry")

# ── T10: batch assessment ──
items = [
    {"content": "User prefers Python over JavaScript", "source": "user_stated", "age_days": 10},
    {"content": "maybe try go for concurrency?", "source": "llm_inferred", "age_days": 200},
]
results = assess_batch(items)
assert len(results) == 2
assert results[0]["score"] > results[1]["score"], "user_stated should outscore llm_inferred"
ok("T10 batch-ordering")

# ── T11: profile validation ──
try:
    validate_profile(_default_profiles()["memory_entry"])
    ok("T11 validate-default")
except Exception as e:
    no(f"T11 validate-default: {e}")

# ── T12: missing key validation ──
try:
    validate_profile({"base_weights": {}})
    no("T12 missing-key (should have raised)")
except AssertionError:
    ok("T12 missing-key-detected")

print(f"\n{PASS}/{PASS+FAIL} PASS")
sys.exit(0 if FAIL == 0 else 1)
