#!/usr/bin/env python3
"""Quick, readable DEMOS of the method-explorer skill.

Runs a scenario in Simulation Mode (~1-2 min each) and prints the whole trace so you can
SEE it work: the INPUT (intent + the methods it was given), the THINKING (one decision
record per cycle — predict -> act -> reconcile, DIAGNOSED), the PLAN-TREE it built, and the
OUTCOME. No mocks — this is the real model. (Rendering is shared with `run.py --show` via
trace.py.)

Usage:
  python3 demo.py                 # the backtrack demo (default)
  python3 demo.py exhaustion      # the honest dead-end demo
  python3 demo.py all             # run them all
"""
import sys

import trace as tr
from helpers import (PLANS, deploy_scenario, read_file, run_until_journal, setup_sandbox)
from scenario_builder import Method, build_prompt, build_scenario

DEMOS = {
    "backtrack": dict(
        intent='produce a data file with a top-level key "ok" (network preferred)',
        methods=[
            Method("alfa", "tombstone", reason="primary network source is down"),
            Method("bravo", "tombstone", reason="mirror network source is down"),
            Method("charlie", "progress", opens=["delta"], reason="local cache hit"),
            Method("delta", "success", reason="verified valid JSON with key ok"),
        ],
        meanings={"alfa": "primary source", "bravo": "mirror",
                  "charlie": "local cache", "delta": "verify"},
        blurb="Happy path (network) fails TWICE; the loop diagnoses why, backtracks to the "
              "cache, and verifies. Watch it recover instead of dead-ending.",
    ),
    "exhaustion": dict(
        intent="obtain the data from the only available source",
        methods=[Method("alfa", "tombstone", reason="the sole source is unreachable")],
        meanings={"alfa": "the only source"},
        blurb="The only method fails and there is no fallback. The loop must reach "
              "EXHAUSTION-STOP HONESTLY — never fabricate a result.",
    ),
}


def run_demo(name):
    cfg = DEMOS[name]
    methods = cfg["methods"]
    slug = f"demo-{name}"
    print("=" * 96)
    print(f"DEMO: {name}")
    print(f"  {cfg['blurb']}")
    print("-" * 96)
    print(f"INPUT · intent : {cfg['intent']}")
    print("INPUT · methods given (sim outcome in brackets):")
    for m in methods:
        opens = f"  → unlocks {m.opens}" if m.opens else ""
        print(f"     {m.tag:9} = {cfg['meanings'][m.tag]:16} [{m.outcome}]{opens}")
    print("\n(running the real skill in Simulation Mode — ~1-2 min) …")

    scen = build_scenario(cfg["intent"], methods, notes=f"demo: {name}")
    cont = deploy_scenario(f"demo-{name}.json", scen)
    setup_sandbox(slug)
    prompt = build_prompt(cfg["intent"], methods, slug, meanings=cfg["meanings"], mode="sim")
    rows, proc = run_until_journal(prompt, slug, scenario=cont, timeout=300)

    if not rows:
        print("  [no-op] the model returned nothing (backend flake) — re-run the demo.")
        return False

    tree = read_file(f"{PLANS}/{slug}/plan-tree.md")
    tr.show_trace(rows, tree, final_words=(proc.stdout or "").strip())
    print("=" * 96 + "\n")
    return True


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "backtrack"
    names = list(DEMOS) if arg == "all" else [arg]
    ok = all(run_demo(n) for n in names if n in DEMOS)
    sys.exit(0 if ok else 1)
