"""Offline write-side metrics for agent-created skills.

This is Arm A of issue #96704.  It deliberately consumes only the local skill
store, ``.usage.json``, and the curator ledger; it never starts a model or
writes telemetry while measuring it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


_METADATA_DIRS = {".archive", ".hub", ".locks", ".git", "__pycache__"}
_FRONTMATTER_RE = re.compile(r"^---\s*$")
_KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):\s*(?P<value>.*)$")


@dataclass(frozen=True)
class Skill:
    name: str
    path: str
    description: str
    class_key: str
    usage: dict[str, Any]


@dataclass(frozen=True)
class Metrics:
    generated_at: str
    as_of: str
    window_days: int
    skills_root: str
    usage_path: str
    ledger_path: str
    agent_created_skills: int
    mature_skills: int
    observed_triggered_skills: int
    trigger_precision: float | None
    duplicate_class_skills: int
    duplicate_class_groups: int
    duplicate_class_rate: float | None
    creation_events: int
    observed_sessions: int
    creation_rate_per_session: float | None
    session_count_source: str
    warnings: tuple[str, ...]
    skills: tuple[dict[str, Any], ...]


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\\\"'":
        return value[1:-1]
    return value


def read_frontmatter(path: Path) -> dict[str, str]:
    """Read the small scalar subset needed by the metrics without new dependencies."""
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:400]
    except OSError:
        return {}
    if not lines or not _FRONTMATTER_RE.match(lines[0]):
        return {}
    values: dict[str, str] = {}
    for line in lines[1:]:
        if _FRONTMATTER_RE.match(line):
            break
        match = _KEY_RE.match(line.strip())
        if match:
            values[match.group("key")] = _parse_scalar(match.group("value"))
    return values


def _read_usage(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(name): value for name, value in data.items() if isinstance(value, dict)} if isinstance(data, dict) else {}


def _read_bundled_names(skills_root: Path) -> set[str]:
    manifest = skills_root / ".bundled_manifest"
    try:
        lines = manifest.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return set()
    return {line.split(":", 1)[0].strip() for line in lines if line.strip() and ":" in line}


def _iter_skill_files(skills_root: Path) -> Iterable[Path]:
    if not skills_root.is_dir():
        return
    for path in sorted(skills_root.rglob("SKILL.md")):
        if any(part in _METADATA_DIRS for part in path.relative_to(skills_root).parts):
            continue
        yield path


def _class_key(path: Path, skills_root: Path, frontmatter: dict[str, str]) -> str:
    """Use an authored category when present, otherwise the on-disk category.

    This is intentionally a structural duplicate proxy, not an LLM semantic
    judgment.  The report names it as such so callers do not mistake category
    collisions for proof that two skills are duplicates.
    """
    category = frontmatter.get("category") or frontmatter.get("metadata.hermes.category")
    if category:
        return category.lower().strip()
    relative = path.parent.relative_to(skills_root)
    return relative.parts[0].lower() if relative.parts else "<root>"


def discover_skills(skills_root: Path, usage: dict[str, dict[str, Any]]) -> list[Skill]:
    bundled = _read_bundled_names(skills_root)
    skills: list[Skill] = []
    for path in _iter_skill_files(skills_root):
        fm = read_frontmatter(path)
        name = fm.get("name") or path.parent.name
        record = usage.get(name)
        if name in bundled or not isinstance(record, dict):
            continue
        if not (record.get("created_by") == "agent" or record.get("agent_created") is True):
            continue
        skills.append(Skill(name=name, path=str(path), description=fm.get("description", ""),
                            class_key=_class_key(path, skills_root, fm), usage=record))
    return skills


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _creation_events(ledger_path: Path) -> tuple[int, set[str], bool]:
    events = 0
    sessions: set[str] = set()
    malformed = False
    try:
        lines = ledger_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return 0, set(), False
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed = True
            continue
        if not isinstance(row, dict) or row.get("action") != "create":
            continue
        events += 1
        evidence = row.get("evidence")
        if isinstance(evidence, dict) and evidence.get("session_id"):
            sessions.add(str(evidence["session_id"]))
    return events, sessions, malformed


def calculate_metrics(skills_root: Path, *, usage_path: Path | None = None,
                      ledger_path: Path | None = None, window_days: int = 30,
                      as_of: datetime | None = None) -> Metrics:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    as_of = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
    usage_path = usage_path or skills_root / ".usage.json"
    ledger_path = ledger_path or skills_root / ".curator_ledger.jsonl"
    usage = _read_usage(usage_path)
    skills = discover_skills(skills_root, usage)
    cutoff = as_of - timedelta(days=window_days)
    mature = []
    triggered = []
    skill_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for skill in skills:
        created = _timestamp(skill.usage.get("created_at"))
        last_used = _timestamp(skill.usage.get("last_used_at"))
        is_mature = created is not None and created <= cutoff
        # .usage.json stores counters plus the latest timestamp, not an event
        # history.  A hit is therefore reported only when the latest observed
        # use itself is inside the window; this is conservative and explicit.
        hit = bool(is_mature and skill.usage.get("use_count", 0) and created and last_used
                   and created <= last_used <= created + timedelta(days=window_days))
        if is_mature:
            mature.append(skill)
            if hit:
                triggered.append(skill)
        if created is None:
            warnings.append(f"{skill.name}: missing or invalid created_at")
        if skill.usage.get("use_count", 0) and last_used is None:
            warnings.append(f"{skill.name}: use_count is non-zero but last_used_at is missing")
        skill_rows.append({
            "name": skill.name,
            "path": skill.path,
            "class_key": skill.class_key,
            "created_at": skill.usage.get("created_at"),
            "use_count": skill.usage.get("use_count", 0),
            "last_used_at": skill.usage.get("last_used_at"),
            "mature": is_mature,
            "observed_trigger_within_window": hit,
        })

    classes: dict[str, list[Skill]] = defaultdict(list)
    for skill in skills:
        classes[skill.class_key].append(skill)
    duplicate_groups = {key: group for key, group in classes.items() if len(group) > 1}
    duplicate_skills = sum(len(group) for group in duplicate_groups.values())
    creation_events, sessions, malformed = _creation_events(ledger_path)
    if malformed:
        warnings.append("curator ledger contains malformed rows; valid rows only were counted")
    if creation_events and not sessions:
        warnings.append("creation events have no session_id; creation rate per session is unavailable")
    if not ledger_path.exists():
        warnings.append("curator ledger is absent; creation rate per session is unavailable")
    if skills and not mature:
        warnings.append("no agent-created skill has aged through the measurement window")
    warnings.append("trigger precision uses latest-use telemetry; .usage.json has no per-use event history")
    rate = creation_events / len(sessions) if sessions else None
    return Metrics(
        generated_at=datetime.now(timezone.utc).isoformat(), as_of=as_of.isoformat(), window_days=window_days,
        skills_root=str(skills_root), usage_path=str(usage_path), ledger_path=str(ledger_path),
        agent_created_skills=len(skills), mature_skills=len(mature), observed_triggered_skills=len(triggered),
        trigger_precision=(len(triggered) / len(mature)) if mature else None,
        duplicate_class_skills=duplicate_skills, duplicate_class_groups=len(duplicate_groups),
        duplicate_class_rate=(duplicate_skills / len(skills)) if skills else None,
        creation_events=creation_events, observed_sessions=len(sessions),
        creation_rate_per_session=rate, session_count_source="curator ledger evidence.session_id" if sessions else "unavailable",
        warnings=tuple(dict.fromkeys(warnings)), skills=tuple(skill_rows),
    )


def _jsonable(metrics: Metrics) -> dict[str, Any]:
    data = asdict(metrics)
    data["warnings"] = list(metrics.warnings)
    data["skills"] = list(metrics.skills)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure write-side skill metrics without model/API runs.")
    default_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    parser.add_argument("--home", type=Path, default=default_home, help="Hermes home containing skills/")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--as-of", help="UTC ISO timestamp for reproducible reports")
    parser.add_argument("--output", type=Path, help="Write JSON report here instead of stdout")
    args = parser.parse_args(argv)
    try:
        as_of = _timestamp(args.as_of) if args.as_of else None
        metrics = calculate_metrics(args.home / "skills", window_days=args.window_days, as_of=as_of)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2) + "\\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

