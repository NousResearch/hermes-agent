"""Create a sanitized Lunar City roster from Hermes profile directories.

Only profile IDs, inferred role labels, personality archetypes, and visual tags
are emitted. SOUL.md content is never copied into the desktop asset manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "lunar-city" / "profile-assets.json"
PROFILE_ROOT = Path(os.environ.get("HERMES_PROFILE_ROOT", Path.home() / ".hermes" / "profiles"))


ROLE_RULES = (
    ("review", "review", "methodical", ("inspection", "violet")),
    ("audit", "audit", "methodical", ("inspection", "violet")),
    ("research", "research", "curious", ("research", "cyan")),
    ("science", "research", "curious", ("research", "cyan")),
    ("security", "security", "protective", ("security", "amber")),
    ("operations", "operations", "protective", ("operations", "cyan")),
    ("ops", "operations", "protective", ("operations", "cyan")),
    ("release", "release", "bold", ("release", "amber")),
    ("deploy", "release", "bold", ("release", "amber")),
    ("support", "support", "social", ("support", "green")),
    ("community", "support", "social", ("support", "green")),
    ("writer", "creative", "social", ("creative", "green")),
)


def infer(profile_id: str, soul: str) -> tuple[str, str, tuple[str, str]]:
    haystack = f"{profile_id}\n{soul}".lower()
    for needle, role, personality, tags in ROLE_RULES:
        if needle in haystack:
            return role, personality, tags
    digest = hashlib.sha256(profile_id.encode()).digest()
    fallback = (("engineering", "curious", ("engineering", "cyan")), ("general", "cautious", ("general", "violet")))
    return fallback[digest[0] % len(fallback)]


def main() -> None:
    classes = {}
    if PROFILE_ROOT.is_dir():
        for soul_path in sorted(PROFILE_ROOT.glob("*/SOUL.md")):
            profile_id = soul_path.parent.name
            role, personality, (discipline, accent) = infer(profile_id, soul_path.read_text(errors="ignore"))
            archetype = "leader" if "director" in profile_id or "maintainer" in profile_id else "worker"
            key = f"{role}:{personality}:{archetype}"
            classes.setdefault(
                key,
                {"role": role, "personality": personality, "visualTags": [discipline, accent, archetype], "count": 0},
            )["count"] += 1
    payload = {
        "schemaVersion": 1,
        "source": "hermes-profile-roster-sanitized",
        "classes": sorted(classes.values(), key=lambda value: (value["role"], value["personality"], value["visualTags"][-1])),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {sum(item['count'] for item in payload['classes'])} profiles as {len(payload['classes'])} aggregate classes to {OUTPUT}")


if __name__ == "__main__":
    main()
