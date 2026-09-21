#!/usr/bin/env python3
"""
Memory Promotion Gate (P2-8) — pre-promotion contradiction check.

Runs before the memory-promotion-daily cron to detect contradictions
between new Mnemosyne facts and existing GBrain wiki pages. Outputs
a JSON verdict for each fact: promote, review, or block.

Usage:
    memory-promotion-gate.py <new_facts.json> <brain_dir>
    memory-promotion-gate.py --stdin-fact "User prefers dark mode" --target ~/brain/references/preferences.md
    memory-promotion-gate.py --batch  # reads from state file

JSON input format:
    [{"fact": "...", "importance": 0.9, "veracity": "stated"}, ...]

JSON output:
    [{"fact": "...", "verdict": "promote|review|block", "conflicts": [...]}, ...]

This is a no_agent script — runs deterministically, no LLM calls.
"""

import json
import os
import sys
from pathlib import Path

# Add repo to path so we can import hermes_cli
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hermes_cli.memory_hygiene import memory_promotion_gate, detect_contradictions


STATE_FILE = Path.home() / ".hermes" / "data" / "memory-promotion-gate-state.json"


def load_facts(path: str) -> list[dict]:
    """Load new facts from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "facts" in data:
        return data["facts"]
    return [data]


def gate_batch(facts: list[dict], brain_dir: str) -> list[dict]:
    """Run the contradiction gate on a batch of facts."""
    results = []

    for item in facts:
        fact_text = item.get("fact", "")
        importance = item.get("importance", 0.5)
        veracity = item.get("veracity", "unknown")

        # Find the target brain page
        target = _resolve_target_page(fact_text, brain_dir)
        if not target or not os.path.exists(target):
            # No target page — promote (will be filed as new page by the cron)
            results.append({
                "fact": fact_text,
                "verdict": "promote",
                "target": target or "",
                "note": "no existing page — safe to promote",
            })
            continue

        # Run the gate
        verdicts = memory_promotion_gate(
            new_facts=[fact_text],
            existing_store_path=target,
        )
        _, verdict = verdicts[0] if verdicts else (fact_text, "promote")
        results.append({
            "fact": fact_text,
            "verdict": verdict,
            "target": target,
        })

    return results


def _resolve_target_page(fact: str, brain_dir: str) -> str:
    """Map a fact to the most likely brain page using keyword matching."""
    import re

    brain = Path(brain_dir).expanduser()
    if not brain.exists():
        return ""

    # Keyword → path mapping (mirrors memory-promotion skill RESOLVER.md)
    keyword_map = {
        r"(?i)\bplenishd\b": "apps/portfolio.md",
        r"(?i)\bcoachos\b": "apps/portfolio.md",
        r"(?i)\bmatchdaymaestro\b": "apps/portfolio.md",
        r"(?i)\bkick-tionary\b": "apps/portfolio.md",
        r"(?i)\bvps|server|cron|gateway|infrastructure\b": "conventions/infrastructure.md",
        r"(?i)\baccount|login|api.key|token\b": "accounts/connected-accounts.md",
        r"(?i)\bproperty|house|mortgage\b": "properties/sahil-properties.md",
        r"(?i)\bprefer|like|use|style|voice\b": "references/preferences.md",
        r"(?i)\bproject|build|scaffold\b": "projects/",
        r"(?i)\bjob|interview|cv|resume\b": "projects/job-hunt.md",
    }

    for pattern, path in keyword_map.items():
        if re.search(pattern, fact):
            target = brain / path
            if target.exists():
                return str(target)
            # For project/ wildcard, try matching any file in that dir
            if path.endswith("/"):
                target_dir = brain / path
                if target_dir.exists():
                    # Return the most recently modified file
                    files = sorted(target_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
                    if files:
                        return str(files[0])

    return ""


def main():
    if "--batch" in sys.argv:
        # Read from state file
        if not STATE_FILE.exists():
            print(json.dumps({"status": "no pending facts"}))
            return 0
        with open(STATE_FILE) as f:
            state = json.load(f)
        facts = state.get("pending_facts", [])
        brain_dir = state.get("brain_dir", os.path.expanduser("~/brain"))
        results = gate_batch(facts, brain_dir)
        print(json.dumps(results, indent=2))
        return 0

    if "--stdin-fact" in sys.argv:
        idx = sys.argv.index("--stdin-fact")
        target_idx = sys.argv.index("--target") if "--target" in sys.argv else -1
        fact = sys.argv[idx + 1]
        target = sys.argv[target_idx + 1] if target_idx != -1 else ""
        if target:
            verdicts = memory_promotion_gate(
                new_facts=[fact],
                existing_store_path=target,
            )
        else:
            # No target — auto-resolve
            target = _resolve_target_page(fact, os.path.expanduser("~/brain"))
            if target:
                verdicts = memory_promotion_gate(
                    new_facts=[fact],
                    existing_store_path=target,
                )
            else:
                verdicts = [(fact, "promote")]
        _, verdict = verdicts[0] if verdicts else (fact, "promote")
        print(json.dumps({"fact": fact, "verdict": verdict, "target": target}))
        return 0

    if len(sys.argv) < 3:
        print("Usage: memory-promotion-gate.py <new_facts.json> <brain_dir>", file=sys.stderr)
        print("       memory-promotion-gate.py --stdin-fact '<fact>' --target <path>", file=sys.stderr)
        print("       memory-promotion-gate.py --batch", file=sys.stderr)
        return 2

    facts = load_facts(sys.argv[1])
    brain_dir = sys.argv[2]
    results = gate_batch(facts, brain_dir)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
