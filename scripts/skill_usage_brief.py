#!/usr/bin/env python3
"""Build a local maintenance brief from Hermes skill usage telemetry.

The script intentionally reports metadata only: skill name, path, description,
and usage sidecar fields. It does not print skill bodies or mutate skills.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SILENT = "[SILENT]"


@dataclass(frozen=True)
class SkillMeta:
    name: str
    path: Path
    relative_path: str
    description: str
    created_by: str | None = None


@dataclass(frozen=True)
class BriefItem:
    name: str
    reason: str
    path: str
    description: str
    use_count: int
    view_count: int
    last_used_at: str | None
    last_viewed_at: str | None
    action: str


@dataclass(frozen=True)
class Brief:
    skills_dir: Path
    usage_file: Path
    generated_at: datetime
    stale_days: int
    unused_grace_days: int
    include_missing_usage: bool
    items: list[BriefItem]

    @property
    def summary(self) -> dict[str, int]:
        counts = {"total_items": len(self.items), "stale": 0, "never_used": 0, "missing_usage": 0}
        for item in self.items:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def default_skills_dir() -> Path:
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home).expanduser() / "skills"
    return Path.home() / ".hermes" / "skills"


def load_usage(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"usage file must contain a JSON object: {path}")
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse simple YAML-ish frontmatter without importing PyYAML.

    We only need scalar metadata fields and stop at the closing delimiter, so a
    tiny conservative parser keeps this script dependency-free.
    """
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if first.strip() != "---":
            return result
        for line in handle:
            stripped = line.strip()
            if stripped == "---":
                break
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            value = raw_value.strip().strip('"\'')
            if value:
                result[key.strip()] = value
    return result


def discover_skills(skills_dir: Path) -> dict[str, SkillMeta]:
    discovered: dict[str, SkillMeta] = {}
    if not skills_dir.exists():
        return discovered
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        if any(part.startswith(".") for part in skill_file.relative_to(skills_dir).parts[:-1]):
            continue
        metadata = _parse_frontmatter(skill_file)
        fallback_name = skill_file.parent.name
        name = metadata.get("name", fallback_name)
        try:
            relative_path = skill_file.relative_to(skills_dir).as_posix()
        except ValueError:
            relative_path = skill_file.as_posix()
        discovered[name] = SkillMeta(
            name=name,
            path=skill_file,
            relative_path=relative_path,
            description=metadata.get("description", ""),
            created_by=metadata.get("created_by"),
        )
    return discovered


def _days_since(now: datetime, value: str | None) -> int | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return max(0, (now - parsed).days)


def _reason_and_action(
    *,
    meta: SkillMeta,
    usage: dict[str, Any] | None,
    now: datetime,
    stale_days: int,
    unused_grace_days: int,
    include_missing_usage: bool,
) -> tuple[str, str] | None:
    if usage is None:
        if not include_missing_usage:
            return None
        return "missing_usage", "Run or view once if still useful; otherwise consider curator review."

    state = str(usage.get("state") or "active").lower()
    if state in {"archived", "deleted", "disabled"}:
        return None
    if bool(usage.get("pinned")):
        return None

    use_count = int(usage.get("use_count") or 0)
    last_used_at = usage.get("last_used_at")
    created_at = usage.get("created_at")

    if use_count <= 0 and last_used_at is None:
        age_days = _days_since(now, created_at)
        if age_days is None or age_days >= unused_grace_days:
            return "never_used", "Decide: try it this week, pin it, or let curator archive it."
        return None

    stale_for_days = _days_since(now, str(last_used_at) if last_used_at else None)
    if stale_for_days is not None and stale_for_days >= stale_days:
        return "stale", "Review whether to refresh, pin, merge, or archive via hermes curator."

    return None


def build_brief(
    *,
    skills_dir: Path,
    usage_file: Path | None = None,
    now: datetime | None = None,
    stale_days: int = 30,
    unused_grace_days: int = 14,
    include_missing_usage: bool = False,
) -> Brief:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    usage_file = usage_file or (skills_dir / ".usage.json")
    usage = load_usage(usage_file)
    skills = discover_skills(skills_dir)
    items: list[BriefItem] = []

    for name, meta in sorted(skills.items()):
        entry = usage.get(name)
        classification = _reason_and_action(
            meta=meta,
            usage=entry,
            now=now,
            stale_days=stale_days,
            unused_grace_days=unused_grace_days,
            include_missing_usage=include_missing_usage,
        )
        if classification is None:
            continue
        reason, action = classification
        entry = entry or {}
        items.append(
            BriefItem(
                name=name,
                reason=reason,
                path=meta.relative_path,
                description=meta.description,
                use_count=int(entry.get("use_count") or 0),
                view_count=int(entry.get("view_count") or 0),
                last_used_at=entry.get("last_used_at"),
                last_viewed_at=entry.get("last_viewed_at"),
                action=action,
            )
        )

    return Brief(
        skills_dir=skills_dir,
        usage_file=usage_file,
        generated_at=now,
        stale_days=stale_days,
        unused_grace_days=unused_grace_days,
        include_missing_usage=include_missing_usage,
        items=items,
    )


def render_json(brief: Brief) -> str:
    payload = {
        "generated_at": brief.generated_at.isoformat(),
        "skills_dir": str(brief.skills_dir),
        "usage_file": str(brief.usage_file),
        "thresholds": {"stale_days": brief.stale_days, "unused_grace_days": brief.unused_grace_days},
        "include_missing_usage": brief.include_missing_usage,
        "summary": brief.summary,
        "items": [item.__dict__ for item in brief.items],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_markdown(brief: Brief) -> str:
    if not brief.items:
        return "## Skill usage brief\n\nNo skill-maintenance items found."

    summary = brief.summary
    lines = [
        "## Skill usage brief",
        "",
        f"- Generated: `{brief.generated_at.isoformat()}`",
        f"- Skills dir: `{brief.skills_dir}`",
        f"- Attention items: {summary['total_items']} "
        f"(stale: {summary.get('stale', 0)}, never used: {summary.get('never_used', 0)}, "
        f"missing telemetry: {summary.get('missing_usage', 0)})",
        "",
        "### Items",
    ]
    for item in brief.items:
        description = f" — {item.description}" if item.description else ""
        last_used = item.last_used_at or "never"
        lines.extend(
            [
                f"- **{item.name}** `{item.reason}`{description}",
                f"  - Path: `{item.path}`",
                f"  - Usage: use_count={item.use_count}, view_count={item.view_count}, last_used={last_used}",
                f"  - Next action: {item.action}",
            ]
        )
    lines.extend(
        [
            "",
            "### Suggested command",
            "",
            "```bash",
            "hermes curator status",
            "hermes curator run",
            "```",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local Hermes skill usage maintenance brief.")
    parser.add_argument("--skills-dir", type=Path, default=default_skills_dir())
    parser.add_argument("--usage-file", type=Path, default=None)
    parser.add_argument("--now", help="ISO timestamp for deterministic tests/scheduled runs")
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument("--unused-grace-days", type=int, default=14)
    parser.add_argument(
        "--include-missing-usage",
        action="store_true",
        help="Also report installed skills with no .usage.json entry (can be noisy on fresh installs)",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--silent-if-empty", action="store_true", help=f"Print exactly {SILENT!r} when no items exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = parse_datetime(args.now) if args.now else datetime.now(timezone.utc)
    assert now is not None
    brief = build_brief(
        skills_dir=args.skills_dir.expanduser(),
        usage_file=args.usage_file.expanduser() if args.usage_file else None,
        now=now,
        stale_days=args.stale_days,
        unused_grace_days=args.unused_grace_days,
        include_missing_usage=args.include_missing_usage,
    )
    if args.silent_if_empty and not brief.items:
        print(SILENT)
    elif args.format == "json":
        print(render_json(brief))
    else:
        print(render_markdown(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
