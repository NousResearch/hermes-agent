#!/usr/bin/env python3
"""Denji auto-promotion — promote frequently-borrowed skills to permanent enablement (Phase 5).

Reads ``skill.borrowed`` events from the central activity ledger, groups by
(profile, skill), and auto-adds skills to the profile's ``enabled_skills`` when
the borrow count exceeds the threshold. Every promotion is recorded as a
reversible ``skill.enabled_auto`` event.

Usage:
  python3 denji-auto-promote.py [--days 30] [--threshold 3] [--apply] [--verbose]

Dry-run by default (prints proposed promotions). --apply writes profile configs
and records events. Safe: never removes skills, only appends to enabled_skills.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hermes_cli.profile_activity_ledger import query_events, append_event  # noqa: E402
from ruamel.yaml import YAML  # noqa: E402

PROFILES = Path("/home/kensei/.hermes/profiles")
DEFAULT_HOME = Path("/home/kensei/.hermes")

# Event types
EV_BORROW = "skill.borrowed"
EV_AUTO_PROMOTE = "skill.enabled_auto"
EV_AUTO_PROMOTE_REVERT = "skill.enabled_auto_reverted"
SOURCE = "denji-auto-promote"

# Skill names that should never be auto-promoted (security-sensitive or profile-owner decisions).
NEVER_AUTO_PROMOTE = {
    "governance",
    "kanban-ops",
    "1password",
    "code-security",
}


def _config_path(profile: str) -> Path:
    return (DEFAULT_HOME if profile == "default" else PROFILES / profile) / "config.yaml"


def _profiles_with_config() -> list[str]:
    names = []
    if (DEFAULT_HOME / "config.yaml").exists():
        names.append("default")
    try:
        for d in sorted(PROFILES.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists():
                names.append(d.name)
    except (OSError, PermissionError) as e:
        print(f"ERROR: cannot list profiles directory: {e}")
    return names


def _borrow_counts(days: int) -> dict[str, dict[str, int]]:
    """Return {profile: {skill: count}} for borrow events in the window."""
    since = int(time.time() - days * 86400)
    try:
        events = query_events(event_types=[EV_BORROW], since=since)
    except Exception as e:
        print(f"ERROR: could not query activity ledger: {e}")
        return {}
    counts: dict[str, dict[str, int]] = {}
    for e in events:
        try:
            profile = e.get("target_profile")
            skill = e.get("object_id")
            if not profile or not skill:
                continue
            counts.setdefault(profile, {}).setdefault(skill, 0)
            counts[profile][skill] += 1
        except (AttributeError, TypeError) as e:
            print(f"  WARNING: skipping malformed event: {e}")
            continue
    return counts


def _already_enabled(profile: str, yaml_parser: YAML) -> set[str]:
    cfg_path = _config_path(profile)
    if not cfg_path.exists():
        return set()
    try:
        data = yaml_parser.load(cfg_path.read_text()) or {}
    except Exception as e:
        print(f"  SKIP {profile}: config read error ({e})")
        return set()
    skills_cfg = data.get("skills") or {}
    return set((skills_cfg.get("enabled_skills") or []) + (skills_cfg.get("always_skills") or []))


def _record_promotion(profile: str, skill: str, count: int, days: int) -> str | None:
    """Record an auto-promotion event in the ledger. Returns event_id or None on failure."""
    eid = f"auto-promote-{int(time.time())}-{skill[:20]}"
    try:
        return append_event(
            source=SOURCE,
            event_type=EV_AUTO_PROMOTE,
            event_id=eid,
            actor_profile="denji",
            target_profile=profile,
            object_type="skill",
            object_id=skill,
            summary=f"Auto-promoted {skill} for {profile} ({count} borrows in {days}d)",
            payload={
                "profile": profile,
                "skill": skill,
                "borrow_count": count,
                "window_days": days,
                "promoted_at": int(time.time()),
                "revert_event_type": EV_AUTO_PROMOTE_REVERT,
            },
        )
    except Exception as e:
        print(f"  ERROR recording promotion event for {profile}/{skill}: {e}")
        return None


def _add_to_enabled(profile: str, skill: str, yaml_parser: YAML, verbose: bool = False) -> bool:
    """Add skill to profile's enabled_skills. Returns True if written."""
    cfg_path = _config_path(profile)
    if not cfg_path.exists():
        if verbose:
            print(f"  SKIP {profile}: no config.yaml")
        return False

    data = yaml_parser.load(cfg_path.read_text()) or {}
    skills_cfg = data.get("skills")
    if not isinstance(skills_cfg, dict):
        from ruamel.yaml.comments import CommentedMap
        skills_cfg = CommentedMap()
        data["skills"] = skills_cfg

    enabled = list(skills_cfg.get("enabled_skills") or [])
    if skill in enabled or skill in (skills_cfg.get("always_skills") or []):
        if verbose:
            print(f"  SKIP {profile}/{skill}: already enabled")
        return False

    enabled.append(skill)
    enabled.sort()
    skills_cfg["enabled_skills"] = enabled

    try:
        with cfg_path.open("w") as fh:
            yaml_parser.dump(data, fh)
    except (OSError, TypeError) as e:
        print(f"  ERROR writing {profile} config: {e}")
        return False
    return True


def _skill_exists_on_disk(skill_name: str) -> bool:
    """Check if a skill exists in the shared skills library."""
    skills_dir = DEFAULT_HOME / "skills"
    if not skills_dir.exists():
        return False
    try:
        # Check direct path and nested paths
        if (skills_dir / skill_name / "SKILL.md").exists():
            return True
        # Check for nested skill (e.g. devops/kanban-ops)
        for skill_md in skills_dir.rglob(f"{skill_name}/SKILL.md"):
            return True
    except (OSError, PermissionError) as e:
        print(f"  ERROR checking skill on disk '{skill_name}': {e}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Denji auto-promotion — promote frequently-borrowed skills")
    ap.add_argument("--days", type=int, default=30, help="Observation window in days")
    ap.add_argument("--threshold", type=int, default=3, help="Borrow count threshold for auto-promotion")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-skill decisions")
    args = ap.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True

    counts = _borrow_counts(args.days)
    if not counts:
        print("No borrow events found in the observation window.")
        return 0

    promotions: list[tuple[str, str, int]] = []  # (profile, skill, count)
    for profile in sorted(counts):
        already = _already_enabled(profile, yaml) if args.apply else set()
        for skill, count in sorted(counts[profile].items()):
            if skill in NEVER_AUTO_PROMOTE:
                if args.verbose:
                    print(f"  NEVER_AUTO_PROMOTE {profile}/{skill}: {count} borrows")
                continue
            if count >= args.threshold:
                if not args.apply or skill not in already:
                    # Only auto-promote skills that actually exist on disk.
                    if not _skill_exists_on_disk(skill):
                        if args.verbose:
                            print(f"  SKIP {profile}/{skill}: not on disk ({count} borrows, possibly orphaned)")
                        continue
                    promotions.append((profile, skill, count))
                elif args.verbose:
                    print(f"  SKIP {profile}/{skill}: already enabled ({count} borrows)")

    if not promotions:
        print("No skills meet the auto-promotion threshold.")
        return 0

    print(f"{'DRY RUN' if not args.apply else 'APPLYING'} · window={args.days}d · threshold={args.threshold}")
    print(f"Proposed promotions: {len(promotions)} across {len(set(p[0] for p in promotions))} profiles\n")

    for profile, skill, count in promotions:
        print(f"  {profile:20s} → {skill:40s} ({count} borrows)")

    if not args.apply:
        print("\nRun with --apply to write changes.")
        return 0

    written = 0
    recorded = 0
    for profile, skill, count in promotions:
        added = _add_to_enabled(profile, skill, yaml, verbose=args.verbose)
        if added:
            written += 1
            _record_promotion(profile, skill, count, args.days)
            recorded += 1
            print(f"  PROMOTED {profile}/{skill} ({count} borrows)")

    print(f"\nDone: {written} enabled_skills updated, {recorded} ledger events recorded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"❌ denji-auto-promote · unexpected error: {e}")
        sys.exit(1)
