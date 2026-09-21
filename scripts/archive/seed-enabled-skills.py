#!/usr/bin/env python3
"""Seed per-profile skills.enabled_skills from observed usage (Phase 2).

Reads skill.loaded events from the central activity ledger and, for each
profile, sets skills.enabled_skills = (skills actually loaded over the window)
UNION (its always_skills). This turns the shadow-mode observation window into
evidence-based allowlists, so flipping enforcement to "enforce" never blocks a
skill a profile genuinely uses.

Usage:
  python3 seed-enabled-skills.py [--days 14] [--apply] [--mode shadow|enforce]

Dry-run by default (prints the proposed allowlist per profile). --apply writes
the profile configs (comment-preserving). --mode also sets enforcement_mode.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hermes_cli.profile_activity_ledger import query_events  # noqa: E402
from ruamel.yaml import YAML  # noqa: E402
from ruamel.yaml.comments import CommentedMap  # noqa: E402

PROFILES = Path("/home/kensei/.hermes/profiles")
DEFAULT_HOME = Path("/home/kensei/.hermes")


def _config_path(profile: str) -> Path:
    return (DEFAULT_HOME if profile == "default" else PROFILES / profile) / "config.yaml"


def _always(profile: str, yaml: YAML) -> set:
    cfg_path = _config_path(profile)
    if not cfg_path.exists():
        return set()
    data = yaml.load(cfg_path.read_text()) or {}
    return set(((data.get("skills") or {}).get("always_skills")) or [])


def _profiles_with_config():
    names = []
    if (DEFAULT_HOME / "config.yaml").exists():
        names.append("default")
    for d in sorted(PROFILES.iterdir()):
        if (d / "config.yaml").exists():
            names.append(d.name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--mode", choices=["off", "shadow", "enforce"])
    args = ap.parse_args()

    since = int(time.time() - args.days * 86400)
    yaml = YAML()
    yaml.preserve_quotes = True

    results = {}
    for profile in _profiles_with_config():
        loaded = {
            e.get("object_id")
            for e in query_events(event_types=["skill.loaded"], target_profile=profile, since=since)
            if e.get("object_id")
        }
        allow = sorted(loaded | _always(profile, yaml))
        results[profile] = allow

    for profile, allow in results.items():
        print(f"{profile}: {len(allow)} enabled_skills")
        if args.apply:
            cfg_path = _config_path(profile)
            data = yaml.load(cfg_path.read_text()) or {}
            skills = data.get("skills")
            if not isinstance(skills, dict):
                skills = CommentedMap()
                data["skills"] = skills
            skills["enabled_skills"] = allow
            if args.mode and args.mode != "off":
                skills["enforcement_mode"] = args.mode
            with cfg_path.open("w") as fh:
                yaml.dump(data, fh)

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'} · {len(results)} profiles · window={args.days}d"
          + (f" · mode={args.mode}" if args.mode else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
