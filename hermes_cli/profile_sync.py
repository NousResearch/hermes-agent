"""Safe profile sync inspection and allowlisted skill sync for Hermes profiles.

This module is a seat belt around the common desire to "sync profiles" without
accidentally copying credentials, sessions, cron state, auth stores, or other
live profile state. Memory comparison is read-only until merge rules exist.
The only supported write path is skills-only ``--apply --with-backup``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent.skill_utils import is_excluded_skill_path
from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists


_MEMORY_FILES = ("memories/MEMORY.md", "memories/USER.md")
_EXCLUDED_SKILL_FILE_NAMES = {".usage.json"}
_EXCLUDED_SKILL_DIR_NAMES = {"__pycache__", ".git"}
_EXCLUDED_STATE = [
    ".env",
    "auth.json",
    "config.yaml",
    "state.db",
    "sessions/",
    "cron/",
    "logs/",
    "gateway pid/state files",
    "memories/ (apply disabled until merge rules exist)",
]


@dataclass(frozen=True)
class FileFingerprint:
    exists: bool
    sha256: str = ""
    bytes: int = 0
    entry_count: int | None = None


@dataclass(frozen=True)
class SkillFingerprint:
    relative_dir: str
    sha256: str
    file_count: int
    bytes: int


def build_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "profile-sync",
        help="Profile memory/skill comparison and skills-only sync",
        description=(
            "Compare profile memories and skills without copying risky profile data. "
            "The only write mode is `skills --apply --with-backup`; memory sync is "
            "intentionally read-only until merge rules exist. This command never "
            "touches .env, auth.json, sessions, cron, config.yaml, logs, gateways, "
            "or live runtime state."
        ),
    )
    sub = parser.add_subparsers(dest="profile_sync_action")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", required=True, help="Source profile name, e.g. engineer")
        p.add_argument("--target", required=True, help="Target profile name, e.g. default")
        p.add_argument(
            "--json",
            action="store_true",
            help="Emit machine-readable JSON instead of a text report",
        )

    diff = sub.add_parser("diff", help="Dry-run memory + skill comparison")
    add_common(diff)
    diff.set_defaults(func=_cmd_profile_sync_exit)

    skills = sub.add_parser("skills", help="Skill-library comparison or skills-only sync")
    add_common(skills)
    skills.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview only. This is the default unless --apply is supplied.",
    )
    skills.add_argument(
        "--apply",
        action="store_true",
        help="Copy missing/changed skills from source to target. Requires --with-backup.",
    )
    skills.add_argument(
        "--with-backup",
        action="store_true",
        help="Create a timestamped backup of the target skills directory before applying.",
    )
    skills.set_defaults(func=_cmd_profile_sync_exit)

    memories = sub.add_parser("memories", help="Dry-run MEMORY.md / USER.md comparison")
    add_common(memories)
    memories.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Read-only mode (currently the only supported mode)",
    )
    memories.add_argument(
        "--apply",
        action="store_true",
        help="Reserved; memory apply is intentionally disabled until merge rules exist.",
    )
    memories.set_defaults(func=_cmd_profile_sync_exit)

    parser.set_defaults(func=_cmd_profile_sync_exit)
    return parser


def _resolve_profile(name: str) -> tuple[str, Path]:
    canon = normalize_profile_name(name)
    if not profile_exists(canon):
        raise ValueError(f"profile '{name}' does not exist")
    return canon, get_profile_dir(canon)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint_file(path: Path, *, count_entries: bool = False) -> FileFingerprint:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return FileFingerprint(False)
    entry_count = None
    if count_entries:
        try:
            text = data.decode("utf-8", errors="ignore")
            entry_count = len([chunk for chunk in text.split("§") if chunk.strip()])
        except Exception:
            entry_count = None
    return FileFingerprint(True, _hash_bytes(data), len(data), entry_count)


def _iter_skill_files(skill_dir: Path) -> Iterable[Path]:
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(skill_dir).parts
        if any(part in _EXCLUDED_SKILL_DIR_NAMES for part in rel_parts):
            continue
        if path.name in _EXCLUDED_SKILL_FILE_NAMES:
            continue
        yield path


def _fingerprint_skill(skill_dir: Path, relative_dir: str) -> SkillFingerprint:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in _iter_skill_files(skill_dir):
        rel = path.relative_to(skill_dir).as_posix()
        data = path.read_bytes()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        file_count += 1
        total_bytes += len(data)
    return SkillFingerprint(relative_dir, digest.hexdigest(), file_count, total_bytes)


def _load_skills(profile_dir: Path) -> dict[str, SkillFingerprint]:
    root = profile_dir / "skills"
    if not root.is_dir():
        return {}
    out: dict[str, SkillFingerprint] = {}
    for skill_md in sorted(root.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        try:
            if is_excluded_skill_path(skill_dir):
                continue
        except Exception:
            # Exclusion helper is defensive; if it cannot classify, include the
            # skill in the report rather than silently hiding it.
            pass
        rel = skill_dir.relative_to(root).as_posix()
        out[rel] = _fingerprint_skill(skill_dir, rel)
    return out


def _compare_skills(source_dir: Path, target_dir: Path) -> dict[str, object]:
    source = _load_skills(source_dir)
    target = _load_skills(target_dir)
    source_keys = set(source)
    target_keys = set(target)
    common = source_keys & target_keys
    changed = sorted(k for k in common if source[k].sha256 != target[k].sha256)
    missing = sorted(source_keys - target_keys)
    extra = sorted(target_keys - source_keys)
    return {
        "source_count": len(source),
        "target_count": len(target),
        "missing_in_target": missing,
        "extra_in_target": extra,
        "changed": changed,
        "would_copy": sorted(missing + changed),
        "would_delete": [],
    }


def _compare_memories(source_dir: Path, target_dir: Path) -> dict[str, object]:
    files = []
    for rel in _MEMORY_FILES:
        src = _fingerprint_file(source_dir / rel, count_entries=True)
        dst = _fingerprint_file(target_dir / rel, count_entries=True)
        status = "same"
        if src.exists and not dst.exists:
            status = "missing_in_target"
        elif dst.exists and not src.exists:
            status = "extra_in_target"
        elif src.sha256 != dst.sha256:
            status = "changed"
        files.append(
            {
                "path": rel,
                "status": status,
                "source_exists": src.exists,
                "target_exists": dst.exists,
                "source_bytes": src.bytes,
                "target_bytes": dst.bytes,
                "source_entries": src.entry_count,
                "target_entries": dst.entry_count,
            }
        )
    return {"files": files, "apply_supported": False}


def _backup_target_skills(target_dir: Path) -> str:
    skills_dir = target_dir / "skills"
    backup_root = target_dir / "backups" / "profile-sync"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive_base = backup_root / f"skills-{stamp}"
    if skills_dir.exists():
        return shutil.make_archive(str(archive_base), "zip", root_dir=str(skills_dir))
    empty_marker = backup_root / f"skills-{stamp}.empty.txt"
    empty_marker.write_text("Target skills directory did not exist before profile-sync apply.\n", encoding="utf-8")
    return str(empty_marker)


def _copy_skill_dir(source_root: Path, target_root: Path, rel: str) -> None:
    source_skill = source_root / rel
    target_skill = target_root / rel
    if not (source_skill / "SKILL.md").is_file():
        raise ValueError(f"refusing to copy non-skill directory: {source_skill}")
    target_skill.parent.mkdir(parents=True, exist_ok=True)
    tmp_skill = target_skill.parent / f".{target_skill.name}.profile-sync-tmp"
    if tmp_skill.exists():
        shutil.rmtree(tmp_skill)
    shutil.copytree(source_skill, tmp_skill, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git"))
    if target_skill.exists():
        shutil.rmtree(target_skill)
    tmp_skill.replace(target_skill)


def _apply_skills(source_dir: Path, target_dir: Path, skills_report: dict[str, object], *, with_backup: bool) -> dict[str, object]:
    if not with_backup:
        raise ValueError("skills --apply requires --with-backup")
    source_root = source_dir / "skills"
    target_root = target_dir / "skills"
    if not source_root.is_dir():
        raise ValueError(f"source skills directory does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    to_copy = list(skills_report.get("would_copy", []))
    backup_path = _backup_target_skills(target_dir)
    copied: list[str] = []
    for rel in to_copy:
        if not isinstance(rel, str):
            continue
        _copy_skill_dir(source_root, target_root, rel)
        copied.append(rel)
    return {
        "applied": True,
        "backup_path": backup_path,
        "copied": copied,
        "deleted": [],
        "note": "skills-only sync; extra target skills kept; memories/config/auth/sessions/cron/logs untouched",
    }


def _build_report(args) -> dict[str, object]:
    source_name, source_dir = _resolve_profile(args.source)
    target_name, target_dir = _resolve_profile(args.target)
    action = getattr(args, "profile_sync_action", None) or "diff"
    apply_requested = bool(getattr(args, "apply", False))
    if action == "diff" and apply_requested:
        raise ValueError("profile-sync diff is read-only; use `profile-sync skills --apply --with-backup`")
    if action == "memories" and apply_requested:
        raise ValueError("memory apply is disabled until merge rules exist")

    report: dict[str, object] = {
        "mode": "apply" if apply_requested else "dry-run",
        "action": action,
        "source": {"name": source_name, "path": str(source_dir)},
        "target": {"name": target_name, "path": str(target_dir)},
        "safety": {
            "read_only": not apply_requested,
            "write_scope": "skills-only" if apply_requested else "none",
            "excluded_state": list(_EXCLUDED_STATE),
        },
    }
    if action in {"diff", "skills"}:
        skills = _compare_skills(source_dir, target_dir)
        report["skills"] = skills
        if action == "skills" and apply_requested:
            report["apply"] = _apply_skills(
                source_dir,
                target_dir,
                skills,
                with_backup=bool(getattr(args, "with_backup", False)),
            )
            # Re-read after apply so callers can verify convergence.
            report["post_apply_skills"] = _compare_skills(source_dir, target_dir)
    if action in {"diff", "memories"}:
        report["memories"] = _compare_memories(source_dir, target_dir)
    return report


def _print_list(label: str, items: list[str], *, limit: int = 50) -> None:
    print(f"  {label}: {len(items)}")
    for item in items[:limit]:
        print(f"    - {item}")
    if len(items) > limit:
        print(f"    ... {len(items) - limit} more")


def _print_text_report(report: dict[str, object]) -> None:
    source = report["source"]  # type: ignore[index]
    target = report["target"]  # type: ignore[index]
    mode = report.get("mode", "dry-run")
    title = "Profile sync apply" if mode == "apply" else "Profile sync dry-run"
    print(title)
    print(f"Source: {source['name']} ({source['path']})")  # type: ignore[index]
    print(f"Target: {target['name']} ({target['path']})")  # type: ignore[index]
    if mode == "apply":
        print("Safety: skills-only apply; backup required; memories/config/auth/session/cron/log/runtime state excluded")
    else:
        print("Safety: read-only; excluded .env/auth/config/session/cron/log/runtime state")

    skills = report.get("skills")
    if isinstance(skills, dict):
        print("\nSkills")
        print(f"  source_count: {skills['source_count']}")
        print(f"  target_count: {skills['target_count']}")
        _print_list("missing_in_target", list(skills["missing_in_target"]))
        _print_list("extra_in_target", list(skills["extra_in_target"]))
        _print_list("changed", list(skills["changed"]))
        _print_list("would_copy", list(skills["would_copy"]))
        _print_list("would_delete", list(skills["would_delete"]))

    apply_result = report.get("apply")
    if isinstance(apply_result, dict):
        print("\nApplied")
        print(f"  backup_path: {apply_result['backup_path']}")
        _print_list("copied", list(apply_result["copied"]))
        _print_list("deleted", list(apply_result["deleted"]))
        print(f"  note: {apply_result['note']}")
        post = report.get("post_apply_skills")
        if isinstance(post, dict):
            print("\nPost-apply skills check")
            _print_list("missing_in_target", list(post["missing_in_target"]))
            _print_list("changed", list(post["changed"]))

    memories = report.get("memories")
    if isinstance(memories, dict):
        print("\nMemories")
        for item in memories["files"]:  # type: ignore[index]
            print(
                "  {path}: {status} "
                "(source entries={source_entries}, target entries={target_entries}, "
                "source bytes={source_bytes}, target bytes={target_bytes})".format(**item)
            )
        print("  apply_supported: false")

    if mode == "apply":
        print("\nNo memory/config/auth/session/cron/log/runtime files were modified.")
    else:
        print("\nNo files were modified.")


def _cmd_profile_sync_exit(args) -> None:
    raise SystemExit(cmd_profile_sync(args))


def cmd_profile_sync(args) -> int:
    action = getattr(args, "profile_sync_action", None)
    if action is None:
        print("profile-sync requires a subcommand: diff, skills, or memories", file=sys.stderr)
        return 2
    try:
        report = _build_report(args)
    except Exception as exc:
        print(f"profile-sync: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    return 0
