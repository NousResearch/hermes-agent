"""Metamorphic scenario generator.

Emits builder-generated scenario variants whose outcomes obey a known RELATION, so we
test the skill without a per-case hand-written oracle. Each case is a dict:
    {slug, scenario(dict), prompt(str), methods, expect_terminal}
A metamorphic *pair* asserts a relation between two cases' terminal states — robust to
model non-determinism because both runs share the same noise.

Reuses scenario_builder (non-overlapping tags + co-generated prompt). THRESHOLD CAUTION:
sibling-count variants stay BELOW K=5 (the upstream-jump threshold) so they don't
*legitimately* change behavior.

Self-test (no container): `python3 gen_scenarios.py` validates every generated scenario
is well-formed and oracle-consistent.
"""
import json
import os

from scenario_builder import (Method, build_prompt, build_scenario, expected_terminal,
                              validate)

BP_DIR = os.path.join(os.path.dirname(__file__), "blueprints")


def load_blueprint(name):
    bp = json.load(open(os.path.join(BP_DIR, name)))
    methods = [Method(m["tag"], m["outcome"], opens=m.get("opens"),
                      on_occurrence=m.get("on_occurrence"),
                      after_tombstones=m.get("after_tombstones"),
                      reason=m.get("reason", "")) for m in bp["methods"]]
    return bp["intent"], bp.get("meanings", {}), methods, bp.get("base_terminal")


def _case(slug, intent, methods, meanings, expect_terminal, notes=""):
    validate(methods, "tombstone")
    return {
        "slug": slug,
        "scenario": build_scenario(intent, methods, notes=notes),
        "prompt": build_prompt(intent, methods, slug, meanings=meanings),
        "methods": methods,
        "expect_terminal": expect_terminal,
    }


# Paraphrase frames (metamorphic: re-framing the intent must not change the outcome).
FRAMES = [
    "",
    "Be careful and verify the evidence: ",
    "This is decision-critical; do not skip steps: ",
    "Concise but thorough: ",
    "Prefer the cheapest working method: ",
]


def paraphrase_variants(n=2):
    """RELATION: paraphrasing the intent leaves the terminal state unchanged (success)."""
    intent, meanings, methods, base_terminal = load_blueprint("backtrack.json")
    cases = []
    for i in range(n):
        framed = FRAMES[i % len(FRAMES)] + intent
        cases.append(_case(f"prop-para-{i}", framed, methods, meanings, base_terminal,
                           notes=f"paraphrase variant {i}"))
    return cases


def reachability_flip_pair():
    """RELATION: flipping one fallback from dead -> reachable flips exhaustion -> success."""
    meanings = {"alfa": "primary source", "bravo": "mirror source",
                "charlie": "local cache", "delta": "verify valid JSON with key ok"}
    dead = [Method("alfa", "tombstone", reason="primary down"),
            Method("bravo", "tombstone", reason="mirror down"),
            Method("charlie", "tombstone", reason="cache also unavailable")]
    base = _case("prop-flip-base",
                 "obtain data with key ok; only these sources exist and all are down",
                 dead, meanings, "exhaustion", notes="reachability-flip BASE (no success path)")
    live = [Method("alfa", "tombstone", reason="primary down"),
            Method("bravo", "tombstone", reason="mirror down"),
            Method("charlie", "progress", opens=["delta"], reason="cache now available"),
            Method("delta", "success", reason="verified valid JSON with key ok")]
    flip = _case("prop-flip-live",
                 "obtain data with key ok; a local cache is also available",
                 live, meanings, "success", notes="reachability-flip VARIANT (success reachable)")
    return {"a": base, "b": flip, "relation": "reachability-flip: exhaustion -> success",
            "expect": ("exhaustion", "success")}


def add_dead_sibling_pair():
    """RELATION: adding ONE dead sibling (still < K=5) leaves the terminal state
    unchanged (success via the same recovery path) — just more cycles."""
    intent, meanings, methods, base_terminal = load_blueprint("backtrack.json")
    base = _case("prop-sib-base", intent, methods, meanings, base_terminal,
                 notes="add-dead-sibling BASE")
    # insert one more dead sibling (echo) before the recovery methods; total 3 dead < K=5.
    plus_methods = methods[:2] + [Method("echo", "tombstone", reason="secondary mirror down")] + methods[2:]
    plus_meanings = {**meanings, "echo": "fetch from the secondary mirror"}
    plus = _case("prop-sib-plus", intent, plus_methods, plus_meanings, base_terminal,
                 notes="add-dead-sibling VARIANT (+1 dead, still <K)")
    return {"a": base, "b": plus, "relation": "add-dead-sibling(<K): terminal unchanged",
            "expect": (base_terminal, base_terminal)}


if __name__ == "__main__":
    # Self-test: every generated scenario is valid + oracle-consistent (no container).
    for c in paraphrase_variants(3):
        assert c["scenario"]["rules"], "empty rules"
        assert "INTENT" in c["prompt"] and "alfa" in c["prompt"], "prompt missing tags/intent"
        assert expected_terminal(c["methods"]) == c["expect_terminal"], "oracle mismatch"
    fp = reachability_flip_pair()
    assert expected_terminal(fp["a"]["methods"]) == "exhaustion"
    assert expected_terminal(fp["b"]["methods"]) == "success"
    sp = add_dead_sibling_pair()
    assert expected_terminal(sp["a"]["methods"]) == "success"
    assert expected_terminal(sp["b"]["methods"]) == "success"
    assert len([m for m in sp["b"]["methods"] if m.outcome == "tombstone"]) < 5, "sibling variant crossed K=5"
    print("ALL GEN_SCENARIOS SELF-TESTS PASSED")
