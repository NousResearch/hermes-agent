#!/usr/bin/env python3
"""Validate the update-first skill owner map against synthetic routing cases.

This is a cheap Phase 1.5 harness, not an LLM judge. It catches the practical
failure mode MJ flagged: a new lesson should have an obvious existing owner
before an agent creates another adjacent skill.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML required: {exc}")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/+-]*", re.I)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def score_owner(prompt: str, owner: dict[str, Any]) -> int:
    # Score only the incoming prompt. Example triggers document intent, but they
    # must not affect predictions or every owner can match itself by example.
    haystack = prompt.lower()
    prompt_tokens = tokens(prompt)
    score = 0
    for term in owner.get("match_terms") or []:
        t = str(term).lower()
        if " " in t or "/" in t or "-" in t or "." in t:
            if t in haystack:
                score += 4
        elif t in prompt_tokens:
            score += 3
    for phrase in owner.get("strong_phrases") or []:
        if str(phrase).lower() in prompt.lower():
            score += 8
    # Prefer explicit guard/system skills when the prompt names the system.
    system = str(owner.get("system") or "").lower()
    if system and system in prompt.lower():
        score += 5
    return score


def predict(prompt: str, owners: list[dict[str, Any]]) -> tuple[str | None, int, list[tuple[str, int]]]:
    scored = [(o["owner_skill"], score_owner(prompt, o)) for o in owners]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_skill, best_score = scored[0] if scored else (None, 0)
    if best_score <= 0:
        return None, 0, scored[:5]
    return best_skill, best_score, scored[:5]


def validate_owner_map(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen = set()
    for i, owner in enumerate(data.get("owners") or []):
        skill = owner.get("owner_skill")
        if not skill:
            errors.append(f"owners[{i}] missing owner_skill")
            continue
        if skill in seen:
            errors.append(f"duplicate owner_skill: {skill}")
        seen.add(skill)
        for field in ("skill_type", "owns", "update_policy", "do_not_use_for", "match_terms"):
            if not owner.get(field):
                errors.append(f"{skill} missing {field}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-map", type=Path, default=Path("docs/plans/skills-owner-map.yaml"))
    parser.add_argument("--cases", type=Path, default=Path("docs/plans/skills-routing-cases.yaml"))
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    owner_map = load_yaml(args.owner_map)
    case_data = load_yaml(args.cases)
    owners = owner_map.get("owners") or []
    cases = case_data.get("cases") or []
    errors = validate_owner_map(owner_map)

    results = []
    passed = 0
    for case in cases:
        predicted, score, top = predict(case["prompt"], owners)
        ok = predicted == case["expected_owner"]
        passed += int(ok)
        results.append({
            "id": case.get("id"),
            "expected_owner": case["expected_owner"],
            "predicted_owner": predicted,
            "score": score,
            "ok": ok,
            "top": top,
        })
    pass_rate = passed / len(cases) if cases else 0.0
    report = {
        "owner_count": len(owners),
        "case_count": len(cases),
        "passed": passed,
        "pass_rate": pass_rate,
        "min_pass_rate": args.min_pass_rate,
        "errors": errors,
        "results": results,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"owners={len(owners)} cases={len(cases)} passed={passed} pass_rate={pass_rate:.0%}")
        if errors:
            print("owner map errors:")
            for e in errors:
                print(f"- {e}")
        for r in results:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"{mark} {r['id']}: expected={r['expected_owner']} predicted={r['predicted_owner']} score={r['score']}")
    return 0 if not errors and pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
