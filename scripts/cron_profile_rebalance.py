#!/usr/bin/env python3
"""Plan and safely stage Hermes cron jobs into profile-local stores.

The default mode is dry-run. Apply modes write only profile-local cron stores and
scripts after creating backups. The script never starts gateways or executes
jobs; it reports the profile-scoped commands operators should run next.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cron_profile_assignment_audit import assignment_for_job, load_job_store

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

MIGRATION_KEY = "profile_migration"
SOURCE_PROFILE = "default"
MIGRATION_TARGET_PROFILES = ("cpa-tax-researcher", "rva-firm-ops", "rva-leads", "rva-profit-pulse")


@dataclass
class MovePlan:
    job_id: str
    name: str
    source_profile: str
    target_profile: str
    action: str
    execution_type: str
    shadow_required: bool
    gateway_required: bool
    default_action: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    script_copy: dict[str, str] | None = None
    skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    already_staged: bool = False


@dataclass
class Manifest:
    generated_at: str
    source_profile: str
    source_home: str
    moves: list[MovePlan]
    gateway_profiles: list[str]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        moves: list[dict[str, Any]] = []
        for move in self.moves:
            row = asdict(move)
            if move.script_copy:
                relative = move.script_copy["relative"]
                row["script_copy"] = {
                    "relative": relative,
                    "source": f"<{self.source_profile}:scripts/{relative}>",
                    "target": f"<{move.target_profile}:scripts/{relative}>",
                }
            moves.append(row)
        return {
            "generated_at": self.generated_at,
            "source_profile": self.source_profile,
            "source_home": f"<profile:{self.source_profile}>",
            "gateway_profiles": self.gateway_profiles,
            "summary": self.summary,
            "moves": moves,
        }


def default_source_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def profile_home(default_home: Path, profile: str) -> Path:
    if profile == "default":
        return default_home
    return default_home / "profiles" / profile


def jobs_file_for(home: Path) -> Path:
    return home / "cron" / "jobs.json"


@contextlib.contextmanager
def jobs_store_lock(home: Path):
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    lock_file = cron_dir / ".jobs.lock"
    handle = lock_file.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        elif msvcrt is not None:
            getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
            elif msvcrt is not None:
                getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
        finally:
            handle.close()


def load_jobs_for_home(home: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs_file = jobs_file_for(home)
    if not jobs_file.exists():
        return [], {"shape": "wrapped"}
    return load_job_store(jobs_file)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_jobs_for_home(
    home: Path,
    jobs: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    *,
    lock: bool = True,
) -> None:
    if lock:
        with jobs_store_lock(home):
            save_jobs_for_home(home, jobs, metadata, lock=False)
        return
    metadata = dict(metadata or {})
    shape = metadata.pop("shape", "wrapped")
    jobs_file = jobs_file_for(home)
    if shape == "list":
        _atomic_write_json(jobs_file, jobs)
        return
    payload = {"jobs": jobs, **metadata, "updated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_write_json(jobs_file, payload)


def backup_jobs_file(home: Path) -> Path | None:
    jobs_file = jobs_file_for(home)
    if not jobs_file.exists():
        return None
    backup_dir = home / "cron" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"jobs-{_now_stamp()}.json"
    shutil.copy2(jobs_file, backup_path)
    return backup_path


def backup_existing_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak-{_now_stamp()}")
    shutil.copy2(path, backup_path)
    return backup_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_source_script(source_home: Path, script: str) -> tuple[Path | None, str | None]:
    """Resolve and sandbox a source script path under source_home/scripts."""
    scripts_dir = (source_home / "scripts").resolve()
    raw = Path(script).expanduser()
    candidate = raw if raw.is_absolute() else scripts_dir / raw
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None, "source_script_unresolvable"
    if not _is_relative_to(resolved, scripts_dir):
        return None, "source_script_outside_profile_scripts"
    return resolved, None


def script_copy_plan(source_home: Path, target_home: Path, script: str) -> tuple[dict[str, str] | None, str | None]:
    source_script, error = resolve_source_script(source_home, script)
    if error:
        return None, error
    assert source_script is not None
    if not source_script.exists():
        return None, "missing_source_script"
    rel = source_script.relative_to((source_home / "scripts").resolve())
    target_script = target_home / "scripts" / rel
    return {"source": str(source_script), "target": str(target_script), "relative": str(rel)}, None


def job_skills(job: dict[str, Any]) -> list[str]:
    raw = job.get("skills")
    if raw is None:
        raw = [job.get("skill")] if job.get("skill") else []
    elif isinstance(raw, str):
        raw = [raw]
    result: list[str] = []
    for value in raw:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def available_skills(profile: Path) -> set[str]:
    skills_dir = profile / "skills"
    found: set[str] = set()
    if not skills_dir.exists():
        return found
    for child in skills_dir.iterdir():
        if child.is_dir():
            found.add(child.name)
    for skill_md in skills_dir.rglob("SKILL.md"):
        found.add(skill_md.parent.name)
        try:
            lines = skill_md.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        if lines[:1] != ["---"]:
            continue
        for line in lines[1:40]:
            if line.strip() == "---":
                break
            if line.startswith("name:"):
                found.add(line.split(":", 1)[1].strip().strip('"\''))
                break
    return found


def is_same_staged_job(existing: dict[str, Any], source_job: dict[str, Any]) -> bool:
    meta = existing.get(MIGRATION_KEY)
    return isinstance(meta, dict) and meta.get("source_job_id") == source_job.get("id") and meta.get("source_profile") == SOURCE_PROFILE


def make_target_job(job: dict[str, Any], target_profile: str, *, shadow_required: bool) -> dict[str, Any]:
    copied = dict(job)
    now = datetime.now(timezone.utc).isoformat()
    copied["enabled"] = False
    copied["state"] = "paused"
    copied["paused_reason"] = "profile_migration_staged_shadow" if shadow_required else "profile_migration_staged"
    copied["paused_at"] = now
    copied["next_run_at"] = None
    copied[MIGRATION_KEY] = {
        "source_profile": SOURCE_PROFILE,
        "source_job_id": job.get("id"),
        "target_profile": target_profile,
        "source_enabled": bool(job.get("enabled", True)),
        "shadow_required": shadow_required,
        "staged_at": now,
    }
    return copied


def build_manifest(
    source_home: Path,
    *,
    default_home: Path | None = None,
    target_profiles: set[str] | None = None,
) -> Manifest:
    source_home = source_home.resolve()
    default_home = (default_home or source_home).resolve()
    source_jobs, _source_metadata = load_jobs_for_home(source_home)
    target_cache: dict[str, tuple[Path, list[dict[str, Any]], set[str]]] = {}
    moves: list[MovePlan] = []

    for job in source_jobs:
        if not bool(job.get("enabled", True)):
            continue
        assignment = assignment_for_job(job)
        target_profile = assignment.target_profile
        if target_profiles is not None and target_profile not in target_profiles:
            continue
        execution_type = assignment.execution_type
        shadow_required = not bool(job.get("no_agent"))
        job_id = str(job.get("id") or "")
        if not job_id:
            moves.append(
                MovePlan(
                    job_id="",
                    name=assignment.name,
                    source_profile=SOURCE_PROFILE,
                    target_profile=target_profile,
                    action="blocked",
                    execution_type=execution_type,
                    shadow_required=shadow_required,
                    gateway_required=target_profile != SOURCE_PROFILE,
                    default_action="leave_active_blocked",
                    blockers=["missing_job_id"],
                )
            )
            continue
        if target_profile == SOURCE_PROFILE:
            moves.append(
                MovePlan(
                    job_id=job_id,
                    name=assignment.name,
                    source_profile=SOURCE_PROFILE,
                    target_profile=target_profile,
                    action="keep",
                    execution_type=execution_type,
                    shadow_required=False,
                    gateway_required=False,
                    default_action="keep_active",
                )
            )
            continue

        target_home = profile_home(default_home, target_profile).resolve()
        if target_profile not in target_cache:
            target_jobs, _target_metadata = load_jobs_for_home(target_home)
            target_cache[target_profile] = (target_home, target_jobs, available_skills(target_home))
        _home, target_jobs, skill_names = target_cache[target_profile]
        blockers: list[str] = []
        warnings: list[str] = []
        script_plan = None
        already_staged = False

        existing = next((candidate for candidate in target_jobs if candidate.get("id") == job.get("id")), None)
        if existing:
            if is_same_staged_job(existing, job):
                already_staged = True
            else:
                blockers.append("id_collision")
        output_dir = target_home / "cron" / "output" / job_id
        if output_dir.exists() and not already_staged:
            blockers.append("output_history_collision")

        if job.get("script"):
            script_plan, error = script_copy_plan(source_home, target_home, str(job["script"]))
            if error:
                blockers.append(error)

        skills = job_skills(job)
        missing = [skill for skill in skills if skill not in skill_names]
        if missing:
            blockers.append("missing_target_skills")
            warnings.append("missing skills: " + ", ".join(missing))

        action = "blocked" if blockers else ("already_staged" if already_staged else "stage")
        default_action = "leave_active_blocked" if blockers else ("leave_active_shadow" if shadow_required else "disable_after_verification")
        moves.append(
            MovePlan(
                job_id=job_id,
                name=assignment.name,
                source_profile=SOURCE_PROFILE,
                target_profile=target_profile,
                action=action,
                execution_type=execution_type,
                shadow_required=shadow_required,
                gateway_required=True,
                default_action=default_action,
                blockers=blockers,
                warnings=warnings,
                script_copy=script_plan,
                skills=skills,
                missing_skills=missing,
                already_staged=already_staged,
            )
        )

    gateway_profiles = sorted({move.target_profile for move in moves if move.gateway_required and not move.blockers})
    summary = {
        "total_active": len(moves),
        "kept": sum(1 for move in moves if move.action == "keep"),
        "to_stage": sum(1 for move in moves if move.action == "stage"),
        "blocked": sum(1 for move in moves if move.action == "blocked"),
        "already_staged": sum(1 for move in moves if move.action == "already_staged"),
        "shadow_required": sum(1 for move in moves if move.shadow_required and move.action in {"stage", "already_staged"}),
    }
    return Manifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_profile=SOURCE_PROFILE,
        source_home=str(source_home),
        moves=moves,
        gateway_profiles=gateway_profiles,
        summary=summary,
    )


def _index_jobs(jobs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job_id = str(job.get("id") or "")
        if job_id:
            indexed[job_id] = job
    return indexed


def apply_stage(manifest: Manifest, *, default_home: Path) -> dict[str, Any]:
    source_home = Path(manifest.source_home)
    backups: dict[str, str | None] = {}
    script_backups: dict[str, str] = {}
    copied_scripts: list[str] = []
    staged_jobs: list[str] = []
    skipped: dict[str, str] = {}

    with jobs_store_lock(source_home):
        source_jobs, _source_metadata = load_jobs_for_home(source_home)
    source_by_id = _index_jobs(source_jobs)

    for move in manifest.moves:
        if move.action != "stage":
            continue
        source_job = source_by_id.get(move.job_id)
        if source_job is None:
            skipped[move.job_id] = "source_job_missing"
            continue
        target_home = profile_home(default_home, move.target_profile)
        target_home.joinpath("cron").mkdir(parents=True, exist_ok=True)
        target_home.joinpath("scripts").mkdir(parents=True, exist_ok=True)
        if move.script_copy:
            source = Path(move.script_copy["source"])
            target = Path(move.script_copy["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or source.read_bytes() != target.read_bytes():
                script_backup = backup_existing_file(target)
                if script_backup is not None:
                    script_backups[str(target)] = str(script_backup)
                shutil.copy2(source, target)
                copied_scripts.append(str(target))
        with jobs_store_lock(target_home):
            if move.target_profile not in backups:
                backup = backup_jobs_file(target_home)
                backups[move.target_profile] = str(backup) if backup else None
            target_jobs, metadata = load_jobs_for_home(target_home)
            if any(is_same_staged_job(job, source_job) for job in target_jobs):
                continue
            target_jobs.append(make_target_job(source_job, move.target_profile, shadow_required=move.shadow_required))
            save_jobs_for_home(target_home, target_jobs, metadata, lock=False)
        staged_jobs.append(move.job_id)

    return {
        "backups": backups,
        "script_backups": script_backups,
        "copied_scripts": copied_scripts,
        "staged_jobs": staged_jobs,
        "skipped": skipped,
    }


def apply_cutover(
    manifest: Manifest,
    *,
    default_home: Path,
    verified_job_ids: set[str],
    accepted_agent_job_ids: set[str] | None = None,
    remove_source: bool = False,
) -> dict[str, Any]:
    accepted_agent_job_ids = accepted_agent_job_ids or set()
    source_home = Path(manifest.source_home)
    changed_profiles: set[str] = set()
    cutover: list[str] = []
    skipped: dict[str, str] = {}
    backups: dict[str, str | None] = {}
    source_backup: Path | None = None

    with jobs_store_lock(source_home):
        source_jobs, source_metadata = load_jobs_for_home(source_home)
        source_by_id = _index_jobs(source_jobs)

        for move in manifest.moves:
            if move.action not in {"stage", "already_staged"}:
                continue
            if move.job_id not in verified_job_ids:
                skipped[move.job_id] = "not_verified"
                continue
            source_job = source_by_id.get(move.job_id)
            if source_job is None:
                skipped[move.job_id] = "source_job_missing"
                continue
            if move.shadow_required and move.job_id not in accepted_agent_job_ids:
                skipped[move.job_id] = "agent_shadow_not_accepted"
                continue
            target_home = profile_home(default_home, move.target_profile)
            with jobs_store_lock(target_home):
                target_jobs, target_metadata = load_jobs_for_home(target_home)
                if move.target_profile not in backups:
                    backup = backup_jobs_file(target_home)
                    backups[move.target_profile] = str(backup) if backup else None
                updated = False
                for target_job in target_jobs:
                    if is_same_staged_job(target_job, source_job):
                        now = datetime.now(timezone.utc).isoformat()
                        target_job["enabled"] = True
                        target_job["state"] = "scheduled"
                        target_job["next_run_at"] = None
                        target_job.pop("paused_reason", None)
                        target_job.pop("paused_at", None)
                        meta = dict(target_job.get(MIGRATION_KEY) or {})
                        meta["cutover_at"] = now
                        target_job[MIGRATION_KEY] = meta
                        updated = True
                        break
                if not updated:
                    skipped[move.job_id] = "target_job_not_staged"
                    continue
                if source_backup is None:
                    source_backup = backup_jobs_file(source_home)
                save_jobs_for_home(target_home, target_jobs, target_metadata, lock=False)
            changed_profiles.add(move.target_profile)

            if remove_source:
                source_jobs = [job for job in source_jobs if str(job.get("id") or "") != move.job_id]
            else:
                source_job["enabled"] = False
                source_job["state"] = "paused"
                source_job["paused_reason"] = "profile_migration_cutover"
                source_job["paused_at"] = datetime.now(timezone.utc).isoformat()
            cutover.append(move.job_id)

        if cutover:
            backups[SOURCE_PROFILE] = str(source_backup) if source_backup else None
            if remove_source:
                save_jobs_for_home(source_home, source_jobs, source_metadata, lock=False)
            else:
                save_jobs_for_home(source_home, list(source_by_id.values()), source_metadata, lock=False)
    return {
        "cutover": cutover,
        "skipped": skipped,
        "changed_profiles": sorted(changed_profiles),
        "backups": backups if cutover else {},
    }


def manifest_as_markdown(manifest: Manifest) -> str:
    data = manifest.to_dict()
    lines = ["# Cron Profile Rebalance Dry Run", ""]
    lines.append("## Summary")
    for key, value in data["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Gateway Profiles")
    if data["gateway_profiles"]:
        for profile in data["gateway_profiles"]:
            lines.append(f"- `{profile}`: run `hermes -p {profile} gateway start`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Moves")
    lines.append("")
    lines.append("| Job | Target | Action | Default action | Blockers |")
    lines.append("|---|---|---|---|---|")
    for move in manifest.moves:
        blockers = ", ".join(move.blockers) if move.blockers else ""
        lines.append(f"| `{move.name}` | `{move.target_profile}` | {move.action} | {move.default_action} | {blockers} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-home", type=Path, default=default_source_home())
    parser.add_argument("--default-home", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--apply-stage", action="store_true")
    parser.add_argument("--cutover-verified", nargs="*", default=[])
    parser.add_argument("--accepted-agent", nargs="*", default=[])
    parser.add_argument("--remove-source", action="store_true")
    parser.add_argument("--target-profile", action="append", choices=MIGRATION_TARGET_PROFILES)
    args = parser.parse_args(argv)

    if args.apply_stage and args.cutover_verified:
        parser.error("run --apply-stage and --cutover-verified as separate invocations after verification")
    if args.accepted_agent and not args.cutover_verified:
        parser.error("--accepted-agent requires --cutover-verified")

    default_home = (args.default_home or args.source_home).resolve()
    target_profiles = set(args.target_profile) if args.target_profile else None
    manifest = build_manifest(args.source_home, default_home=default_home, target_profiles=target_profiles)
    result: dict[str, Any] | None = None
    if args.apply_stage:
        result = apply_stage(manifest, default_home=default_home)
    if args.cutover_verified:
        result = apply_cutover(
            manifest,
            default_home=default_home,
            verified_job_ids=set(args.cutover_verified),
            accepted_agent_job_ids=set(args.accepted_agent),
            remove_source=args.remove_source,
        )

    if args.format == "json":
        payload = manifest.to_dict()
        if result is not None:
            payload["apply_result"] = result
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(manifest_as_markdown(manifest), end="")
        if result is not None:
            print("\n## Apply Result")
            print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
