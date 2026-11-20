#!/usr/bin/env python3
"""Skill-shortlist selection eval: hit rate on work requests, false positives on everything else.

Ranks every case in ``cases.json`` with the production ``agent.skill_selection.shortlist`` against a
real skill catalog (the repo's bundled ``skills/``, optionally plus ``optional-skills/`` for a large
library) and prints exact counts. No model, no network.

    python evals/skill_selection/run.py                 # bundled skills
    python evals/skill_selection/run.py --optional      # bundled + optional-skills (~large library)
    python evals/skill_selection/run.py --verbose       # per-case suggestions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def catalog(roots: list[Path]) -> tuple[dict[str, str], tuple]:
    """``(index rows, shortlist docs)`` for every skill under ``roots``, shaped like the live index."""
    from agent.skill_selection import skill_docs_from_roots
    from agent.skill_utils import TIER_LOCAL, extract_skill_description, iter_skill_index_files, parse_frontmatter
    index: dict[str, str] = {}
    for root in roots:
        for skill_file in iter_skill_index_files(root, "SKILL.md"):
            frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8-sig"))
            name = str(frontmatter.get("name") or skill_file.parent.name)
            index.setdefault(name, f"    - {name}: {extract_skill_description(frontmatter)}")
    # Root order is precedence, as setdefault above: each root its own (non-project) tier.
    return index, skill_docs_from_roots([(TIER_LOCAL + i, root) for i, root in enumerate(roots)], index)


def evaluate(cases: dict, docs: tuple, verbose: bool = False) -> dict:
    from agent.skill_selection import shortlist
    hits = wrong = extra = shown = 0
    for case in cases["should_suggest"]:
        got = shortlist(case["request"], docs)
        ok = any(name in got for name in case["expect"])
        hits += ok
        extra += len([n for n in got if n not in case["expect"]])
        shown += len(got)
        wrong += bool(got) and not ok
        if verbose:
            print(f"  {'HIT ' if ok else 'MISS'} {case['request'][:70]!r} -> {got} (want {case['expect']})")
    false_pos = 0
    for request in cases["should_not_suggest"]:
        got = shortlist(request, docs)
        false_pos += bool(got)
        if verbose and got:
            print(f"  FP   {request[:70]!r} -> {got}")
    return {"positives": len(cases["should_suggest"]), "hits": hits, "wrong_only": wrong, "extra": extra, "shown": shown,
            "negatives": len(cases["should_not_suggest"]), "false_positives": false_pos}


def main() -> int:
    parser = argparse.ArgumentParser(description="Skill-shortlist selection eval")
    parser.add_argument("--optional", action="store_true", help="add optional-skills/ to the catalog")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    roots = [REPO / "skills"] + ([REPO / "optional-skills"] if args.optional else [])
    index, docs = catalog(roots)
    cases = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8-sig"))
    result = evaluate(cases, docs, args.verbose)
    print(f"catalog: {len(index)} skills ({', '.join(r.name for r in roots)})")
    print(f"hit rate: {result['hits']}/{result['positives']}  "
          f"(only wrong skills suggested: {result['wrong_only']}; "
          f"names per work request: {result['shown'] / max(result['positives'], 1):.1f})")
    print(f"false positives: {result['false_positives']}/{result['negatives']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
