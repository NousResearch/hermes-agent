#!/usr/bin/env python3
"""Token budget measurement for the 3-Tier Knowledge Center.

Measures token usage for each tier combination and verifies it stays
within the 3000 token budget.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_CHARS_PER_TOKEN = 4
_BUDGET_LIMIT = 3000


def estimate_tokens(text: str) -> int:
    """Rough token estimate using char/4 heuristic."""
    return len(text) // _CHARS_PER_TOKEN


def measure_tier1() -> int:
    """Measure Tier 1: Project context pack only."""
    context_dir = Path(__file__).resolve().parents[2] / "docs" / "hermes-agent-standalone" / "context-packs"
    if not context_dir.exists():
        return 0
    # Pick first non-index pack as sample
    packs = [f for f in context_dir.glob("*.md") if f.name != "index.md"]
    if not packs:
        return 0
    text = packs[0].read_text(encoding="utf-8")
    return estimate_tokens(text)


def measure_tier2() -> int:
    """Measure Tier 2: Domain notes for a sample project."""
    vault = Path.home() / "ObsidianVault" / "HermesAgent" / "domains"
    if not vault.exists():
        return 0
    total = 0
    for domain_dir in vault.iterdir():
        if not domain_dir.is_dir():
            continue
        for note in domain_dir.glob("*.md"):
            if note.name == "README.md":
                continue
            text = note.read_text(encoding="utf-8")
            total += estimate_tokens(text)
    return total


def measure_tier3() -> int:
    """Measure Tier 3: Global playbooks."""
    vault = Path.home() / "ObsidianVault" / "HermesAgent" / "playbooks"
    if not vault.exists():
        return 0
    total = 0
    for pb in vault.glob("*.md"):
        if pb.name == "README.md":
            continue
        text = pb.read_text(encoding="utf-8")
        total += estimate_tokens(text)
    return total


def measure_memory() -> int:
    """Measure Memory system (MEMORY.md + USER.md)."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    total = 0
    for fname in ["MEMORY.md", "USER.md"]:
        f = hermes_home / fname
        if f.exists():
            text = f.read_text(encoding="utf-8")
            total += estimate_tokens(text)
    return total


def main() -> int:
    t1 = measure_tier1()
    t2 = measure_tier2()
    t3 = measure_tier3()
    mem = measure_memory()

    print("=== Token Budget Measurement ===")
    print(f"Tier 1 (Project Context Pack):  ~{t1} tokens")
    print(f"Tier 2 (Domain Notes):          ~{t2} tokens")
    print(f"Tier 3 (Global Playbooks):      ~{t3} tokens")
    print(f"Memory System:                  ~{mem} tokens")
    print()

    combo_1 = t1
    combo_2 = t1 + t2
    combo_3 = t1 + t2 + t3
    combo_all = t1 + t2 + t3 + mem

    print("=== Combinations ===")
    print(f"Tier 1 only:                    ~{combo_1} tokens")
    print(f"Tier 1 + Tier 2:                ~{combo_2} tokens")
    print(f"Tier 1 + Tier 2 + Tier 3:       ~{combo_3} tokens")
    print(f"All tiers + Memory:             ~{combo_all} tokens")
    print()

    budget_ok = combo_all <= _BUDGET_LIMIT
    print(f"Budget limit:                   {_BUDGET_LIMIT} tokens")
    print(f"Status:                         {'✅ PASS' if budget_ok else '❌ EXCEEDS BUDGET'}")
    if not budget_ok:
        excess = combo_all - _BUDGET_LIMIT
        print(f"Excess:                         ~{excess} tokens over budget")
        print("Mitigation: Reduce domain note count or truncate older notes.")

    return 0 if budget_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
