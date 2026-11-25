#!/usr/bin/env python3
"""Judge a downloaded skills-index.json and print ``status=``/``detail=``/``summary=`` output lines.

Run by ``.github/workflows/skills-index-freshness.yml`` after it downloads the live
``/docs/api/skills-index.json``; the step appends stdout to ``$GITHUB_OUTPUT``. The index is
``degraded`` when it is more than 26 hours old, a major source has collapsed below its floor, or
the total falls under 1500. ``parse-failed`` and ``invalid-shape`` cover a file that is not the
expected JSON. Every verdict exits 0 so the workflow's issue step decides what to do.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime

MAX_AGE_HOURS = 26
MIN_TOTAL = 1500
# Same as build_skills_index.py EXPECTED_FLOORS.
SOURCE_FLOORS = {
    "skills.sh": 100,
    "lobehub": 100,
    "clawhub": 50,
    "official": 50,
    "github": 30,
    "browse-sh": 50,
}


def _one_line(value) -> str:
    """Flatten remote text for ``$GITHUB_OUTPUT``, where a newline starts a new ``key=value``."""
    return str(value).replace("\r", " ").replace("\n", " ")


def evaluate(data, now: datetime) -> list[str]:
    """Return the ``key=value`` output lines for a decoded index."""
    generated_at = data.get("generated_at", "")
    total = data.get("skill_count", 0)
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        return [
            "status=invalid-shape",
            f"detail=skills field is not a list (got {type(skills).__name__})",
        ]

    by_src = Counter(_one_line(s.get("source", "")) for s in skills)

    age_hours = None
    try:
        ts = datetime.fromisoformat(generated_at)
        age_hours = (now - ts).total_seconds() / 3600
    except (TypeError, ValueError):
        pass  # an unreadable or naive timestamp skips only the age check

    issues = []
    if age_hours is not None and age_hours > MAX_AGE_HOURS:
        issues.append(f"Index is {age_hours:.1f}h old (limit {MAX_AGE_HOURS}h)")
    for src, floor in SOURCE_FLOORS.items():
        count = by_src.get(src, 0)
        if src == "skills.sh":
            count = by_src.get("skills.sh", 0) + by_src.get("skills-sh", 0)
        if count < floor:
            issues.append(f"{src}: {count} < {floor}")
    if total < MIN_TOTAL:
        issues.append(f"total skills: {total} < {MIN_TOTAL}")

    if issues:
        # GITHUB_OUTPUT doesn't allow newlines without explicit delimiter
        return ["status=degraded", f"detail={'; '.join(issues)}"]
    by_summary = ", ".join(f"{k}={v}" for k, v in by_src.most_common(8))
    return [
        "status=ok",
        f"detail=Index OK — {total} skills, generated {_one_line(generated_at)}",
        f"summary={by_summary}",
    ]


def judge_file(path: str, now: datetime) -> list[str]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return ["status=parse-failed", f"detail=JSON decode error: {_one_line(e)}"]
    return evaluate(data, now)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge a downloaded skills-index.json.")
    parser.add_argument("index", help="Path to the downloaded skills-index.json")
    args = parser.parse_args(argv)
    for line in judge_file(args.index, datetime.now(UTC)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
