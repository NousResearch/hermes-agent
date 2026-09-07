"""Build the structured evidence payload for the weekly agent-led review.

Only real invocation evidence from the local Wisdom ledger is used. Bundled
and hub-installed skills, managed org mirrors and archived skills are
excluded before the agent ever sees them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_skills_dir
from tools.skill_usage import is_bundled, is_hub_installed

from .. import qualification as _qualification
from ..store import WisdomStore
from .history import SuggestionHistory
from .policy import AgentLedPolicy

logger = logging.getLogger(__name__)

EXCLUDED_ROOTS = frozenset({"_org", "_wisdom", ".archive", ".hub"})


@dataclass
class SkillEvidence:
    skill_name: str
    path: str
    description: str
    content_hash: str
    window_days: int
    invocation_count: int
    days_used: int
    last_used_day: str | None
    usage_by_day: dict[str, int]
    required_environment_variables: list[str] = field(default_factory=list)
    required_commands: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    supporting_signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("path", None)
        # Arbitrary frontmatter metadata can contain deployment configuration.
        value.pop("frontmatter", None)
        return value


@dataclass
class EvidencePayload:
    window_days: int
    window_start: str
    window_end: str
    min_aggregate_count: int
    organization: dict[str, Any]
    candidates: list[SkillEvidence]
    excluded: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "min_aggregate_count": self.min_aggregate_count,
            "organization": self.organization,
            "candidates": [c.as_dict() for c in self.candidates],
            "excluded": self.excluded,
        }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def read_frontmatter(skill_path: Path) -> dict[str, Any]:
    try:
        from ..package import _frontmatter

        value = _frontmatter(skill_path)
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def dependency_hints(frontmatter: dict[str, Any]) -> tuple[list[str], list[str]]:
    hermes_meta = (frontmatter.get("metadata") or {}).get("hermes") or {}
    env = _string_list(frontmatter.get("required_environment_variables")) or _string_list(
        hermes_meta.get("required_environment_variables")
    )
    commands = _string_list(frontmatter.get("required_commands")) or _string_list(
        hermes_meta.get("required_commands")
    )
    return env, commands


def _listing(root: Path, sub: str) -> list[str]:
    folder = root / sub
    if not folder.is_dir():
        return []
    return sorted(str(p.relative_to(root)) for p in folder.rglob("*") if p.is_file())[:50]


def is_excluded_path(path: Path, *, skills_root: Path | None = None) -> str | None:
    """Return an exclusion reason for paths outside the user's own skill tree."""
    root = (skills_root or get_skills_dir()).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return "outside_skills_dir"
    if relative.parts and relative.parts[0] in EXCLUDED_ROOTS:
        return "managed_or_archived"
    return None


def provenance_exclusion(skill_name: str, path: Path, *, skills_root: Path | None = None) -> str | None:
    if is_bundled(skill_name):
        return "bundled"
    if is_hub_installed(skill_name):
        return "hub_installed"
    return is_excluded_path(path, skills_root=skills_root)


def windowed_usage(store: WisdomStore, *, since_day: str, timezone_name: str) -> dict[str, dict[str, Any]]:
    """Per-skill day->count map for local skills used on/after ``since_day``."""
    rows: dict[str, dict[str, Any]] = {}
    with store.transaction() as db:
        cursor = db.execute(
            "SELECT s.id, s.canonical_path, u.day_local, u.use_count "
            "FROM usage_day u JOIN local_skill s ON s.id=u.skill_id "
            "WHERE s.deleted_at IS NULL AND s.source_kind='local' "
            "AND u.timezone_name=? AND u.day_local>=? ORDER BY s.id, u.day_local",
            (timezone_name, since_day),
        )
        for skill_id, canonical_path, day, count in cursor.fetchall():
            entry = rows.setdefault(str(skill_id), {"path": str(canonical_path), "days": {}})
            entry["days"][str(day)] = int(count)
    return rows


def build_evidence(
    *,
    store: WisdomStore,
    policy: AgentLedPolicy,
    history: SuggestionHistory,
    organization: dict[str, Any] | None = None,
    at: datetime | None = None,
    skills_root: Path | None = None,
    supporting_signals: dict[str, list[str]] | None = None,
) -> EvidencePayload:
    now = (at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    profile_tz, timezone_name = _qualification._profile_timezone()
    today = now.astimezone(profile_tz).date()
    start_day = today - timedelta(days=policy.window_days - 1)
    usage = windowed_usage(store, since_day=start_day.isoformat(), timezone_name=timezone_name)

    excluded: dict[str, list[str]] = {
        "bundled": [],
        "hub_installed": [],
        "managed_or_archived": [],
        "outside_skills_dir": [],
        "missing": [],
        "below_min_count": [],
        "dismissed": [],
        "previously_handled": [],
        "recently_suggested": [],
    }
    candidates: list[SkillEvidence] = []
    signals = supporting_signals or {}

    for skill_id, entry in usage.items():
        path = Path(entry["path"])
        skill_name = path.name
        if not path.is_dir():
            excluded["missing"].append(skill_name)
            continue
        reason = provenance_exclusion(skill_name, path, skills_root=skills_root)
        if reason:
            excluded[reason].append(skill_name)
            continue
        days: dict[str, int] = entry["days"]
        total = sum(days.values())
        if total < policy.min_aggregate_count:
            excluded["below_min_count"].append(skill_name)
            continue
        content_hash, _tree = _qualification.snapshot_tree(path)
        if history.is_suppressed(skill_name, content_hash, at=now):
            excluded["dismissed"].append(skill_name)
            continue
        if history.previously_handled(skill_name, content_hash):
            excluded["previously_handled"].append(skill_name)
            continue
        if history.recently_suggested(
            skill_name, content_hash, cooldown_days=policy.resuggest_cooldown_days, at=now
        ):
            excluded["recently_suggested"].append(skill_name)
            continue
        frontmatter = read_frontmatter(path)
        env, commands = dependency_hints(frontmatter)
        candidates.append(
            SkillEvidence(
                skill_name=skill_name,
                path=str(path),
                description=str(frontmatter.get("description") or "").strip(),
                content_hash=content_hash,
                window_days=policy.window_days,
                invocation_count=total,
                days_used=len(days),
                last_used_day=max(days) if days else None,
                usage_by_day=dict(sorted(days.items())),
                required_environment_variables=env,
                required_commands=commands,
                scripts=_listing(path, "scripts"),
                references=_listing(path, "references"),
                frontmatter={
                    k: v
                    for k, v in frontmatter.items()
                    if k in {"name", "description", "version", "platforms", "metadata"}
                },
                supporting_signals=list(signals.get(skill_name, [])),
            )
        )

    candidates.sort(key=lambda c: (-c.invocation_count, -c.days_used, c.skill_name))
    return EvidencePayload(
        window_days=policy.window_days,
        window_start=start_day.isoformat(),
        window_end=today.isoformat(),
        min_aggregate_count=policy.min_aggregate_count,
        organization=dict(organization or {}),
        candidates=candidates,
        excluded={k: sorted(v) for k, v in excluded.items()},
    )
