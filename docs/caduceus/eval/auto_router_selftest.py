#!/usr/bin/env python3
"""Offline self-test for the Caduceus Auto Router core (agent/auto_router.py).

Proves the pure selection logic with a synthetic classifier — no network, no
model, no creds. Mirrors the shim's router demo (UltraCode-Shim/docs/AUTO_ROUTER.md).

Run:  HERMES_AGENT_DIR=/path/to/hermes-agent python3 eval/auto_router_selftest.py
"""
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.environ.get("HERMES_AGENT_DIR", _REPO_ROOT))

from agent import auto_router as r  # noqa: E402


def _classify_with(scores_map):
    def _c(system_prompt, user_content):
        # Wrap in prose to also exercise lenient parsing.
        return "noise " + json.dumps({"scores": scores_map, "reasoning": "x"}) + " trailing"
    return _c


def main() -> int:
    cands = [
        r.Candidate(id="cheap-fast", cost=0.3, supports_images=False,
                    card="Cheap, fast. Great at simple edits/codegen. Weak on hard multi-file refactors and debugging."),
        r.Candidate(id="mid", cost=1.0, supports_images=False, card="Solid generalist coder."),
        r.Candidate(id="strong-vision", cost=5.0, supports_images=True,
                    card="Frontier reasoning + agentic coding + images. Best for the hardest work."),
    ]
    checks = []

    r.reset_cache()
    checks.append(("easy->cheapest viable",
                   r.select("add a docstring", cands, tier="t1",
                            classify=_classify_with({"cheap-fast": 0.9, "mid": 0.92, "strong-vision": 0.95})) == "cheap-fast"))
    checks.append(("hard->escalate to strong",
                   r.select("refactor auth across 8 files", cands, tier="t2",
                            classify=_classify_with({"cheap-fast": 0.4, "mid": 0.55, "strong-vision": 0.95})) == "strong-vision"))
    checks.append(("image->vision-capable only",
                   r.select("what is in this screenshot", cands, has_images=True, tier="t3",
                            classify=_classify_with({"cheap-fast": 0.9, "mid": 0.9, "strong-vision": 0.8})) == "strong-vision"))

    r.reset_cache()
    r.select("cached task", cands, tier="tc",
             classify=_classify_with({"cheap-fast": 0.9, "mid": 0.9, "strong-vision": 0.9}))
    def _boom(s, u):
        raise RuntimeError("classifier must NOT be called on a cache hit")
    checks.append(("cache hit (no re-score)",
                   r.select("cached task", cands, classify=_boom, tier="tc") == "cheap-fast"))

    checks.append(("no classifier -> cheapest fallback",
                   r.select("x", cands, classify=None, tier="t5") == "cheap-fast"))
    checks.append(("garbage output -> fallback",
                   r.select("y", cands, classify=lambda s, u: "not json", tier="t6") == "cheap-fast"))
    checks.append(("single candidate -> no classifier call",
                   r.select("z", [cands[0]], classify=_boom, tier="t7") == "cheap-fast"))
    checks.append(("default fallback honored",
                   r.fallback_id(cands, default="mid") == "mid"))

    ok = all(p for _, p in checks)
    for name, passed in checks:
        print(f"  {'OK ' if passed else 'FAIL'} {name}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
