"""Export cron job definitions to a directory (optionally a git repo) and import them back.

One YAML file per job (``jobs/<id>.yaml``), keys sorted, holding only the authored fields of
``job_definition.JOB_DEFINITION_FIELDS`` plus the repeat budget. Scheduler state (next run, last
status, run counters, pause markers) never lands in a file, so re-exporting an unchanged store is a
byte-identical no-op and a git history shows only edits a person made. ``origin`` is dropped: it is
the chat (platform, chat id, user) the job was created from, not part of what the job does.
Credential-shaped strings in any field are masked, so the directory is safe to push; the masked
fields are listed under ``redacted_fields`` and import leaves those fields untouched on a job that
already exists, so a round trip never overwrites a real value with its mask.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hermes_yaml as yaml
from agent.redact import redact_credential_shapes
from cron.job_definition import JOB_DEFINITION_FIELDS, import_job_definitions
from cron.jobs import _jobs_lock, load_jobs
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_platform.resolver import locate_command

__all__ = ["ExportResult", "export_jobs", "import_jobs", "read_exported_jobs"]

JOBS_SUBDIR = "jobs"
_EXPORT_FIELDS = JOB_DEFINITION_FIELDS - {"origin", "skill"}  # ``skill`` mirrors skills[0]
_GIT_TIMEOUT_S = 120
_FALLBACK_IDENTITY = ("-c", "user.name=Hermes cron export", "-c", "user.email=hermes@localhost")


@dataclass(frozen=True)
class ExportResult:
    directory: Path
    written: tuple[str, ...]
    removed: tuple[str, ...]
    redacted: tuple[str, ...]  # "<job id>.<field>" whose value had a credential masked
    commit: Optional[str]  # new commit SHA; None when not committing or nothing changed


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return redact_credential_shapes(value, force=True, redact_url_credentials=True)
    if isinstance(value, list):
        return [_mask(item) for item in value]
    if isinstance(value, dict):
        return {key: _mask(item) for key, item in value.items()}
    return value


def _definition(job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {"id": job["id"]}
    redacted: list[str] = []
    for key in sorted(_EXPORT_FIELDS):
        value = job.get(key)
        if value is None or value == [] or value == {}:
            continue
        masked = _mask(value)
        if masked != value:
            redacted.append(f"{job['id']}.{key}")
        record[key] = masked
    times = (job.get("repeat") or {}).get("times")
    if times is not None:
        record["repeat"] = {"times": times}
    if redacted:
        record["redacted_fields"] = [name.split(".", 1)[1] for name in redacted]
    return record, redacted


def _file_name(job_id: Any) -> str:
    name = f"{job_id}.yaml"
    if Path(name).name != name or str(job_id).startswith("."):
        raise ValueError(f"cron job id {job_id!r} is not a safe file name")
    return name


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    git = locate_command("git").command
    if not git:
        raise RuntimeError("git was not found on PATH; export without --git or install git")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(
            [*git, "-C", str(directory), *args], capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, stdin=subprocess.DEVNULL, env=env, timeout=_GIT_TIMEOUT_S,
            creationflags=windows_hide_flags())
    except subprocess.TimeoutExpired as exc:  # a commit hook that never returns
        raise RuntimeError(f"git {args[0]} did not finish in {_GIT_TIMEOUT_S}s") from exc


def _commit(directory: Path, message: str) -> Optional[str]:
    if not (directory / ".git").exists():
        init = _git(directory, "init", "-q")
        if init.returncode != 0:
            raise RuntimeError(f"git init failed: {init.stderr.strip()}")
    _git(directory, "add", "-A", "--", JOBS_SUBDIR)
    if _git(directory, "diff", "--cached", "--quiet", "--", JOBS_SUBDIR).returncode == 0:
        return None
    # Only the files this export owns are committed; anything else staged stays staged.
    identity = () if _git(directory, "config", "user.email").stdout.strip() else _FALLBACK_IDENTITY
    done = _git(directory, *identity, "commit", "-q", "-m", message, "--", JOBS_SUBDIR)
    if done.returncode != 0:
        raise RuntimeError(f"git commit failed: {(done.stderr or done.stdout).strip()}")
    return _git(directory, "rev-parse", "HEAD").stdout.strip()


def export_jobs(directory: Path, *, git: bool = False, message: Optional[str] = None) -> ExportResult:
    """Mirror the active profile's cron jobs into ``directory/jobs/`` (stale files removed)."""
    directory = Path(directory).expanduser()
    jobs_dir = directory / JOBS_SUBDIR
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted((job for job in load_jobs() if isinstance(job, dict) and job.get("id")),
                  key=lambda job: str(job["id"]))
    written, redacted, keep = [], [], set()
    for job in jobs:
        record, masked = _definition(job)
        redacted.extend(masked)
        path = jobs_dir / _file_name(job["id"])
        keep.add(path.name)
        text = yaml.safe_dump(record, sort_keys=True, width=100)
        if not path.exists() or path.read_text(encoding="utf-8-sig") != text:
            path.write_text(text, encoding="utf-8")
            written.append(path.name)
    removed = sorted(p.name for p in jobs_dir.glob("*.yaml") if p.name not in keep)
    for name in removed:
        (jobs_dir / name).unlink()
    commit = _commit(directory, message or f"cron export: {len(jobs)} job(s)") if git else None
    return ExportResult(directory, tuple(written), tuple(removed), tuple(redacted), commit)


def read_exported_jobs(directory: Path) -> dict[str, dict[str, Any]]:
    """Job id -> authored record for every ``jobs/*.yaml`` under ``directory``."""
    jobs_dir = Path(directory).expanduser() / JOBS_SUBDIR
    if not jobs_dir.is_dir():
        raise ValueError(f"{jobs_dir} does not exist; point at a directory made by `hermes cron export`")
    shipped: dict[str, dict[str, Any]] = {}
    for path in sorted(jobs_dir.glob("*.yaml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(record, dict) or not record.get("id"):
            raise ValueError(f"{path.name}: expected a mapping with an 'id'")
        shipped[str(record["id"])] = {
            k: v for k, v in record.items() if k in _EXPORT_FIELDS or k in ("repeat", "redacted_fields")}
    return shipped


def import_jobs(directory: Path) -> list[str]:
    """Merge exported definitions into the active store; returns the imported job ids.

    Known ids get their definition refreshed and keep their scheduler state; new ids arrive
    paused for review (the same merge profile distributions use). A field the export masked keeps
    the store's real value on a known job; a new job carries the masked text for the user to fix."""
    shipped = read_exported_jobs(directory)
    with _jobs_lock():  # re-entrant: the merge below takes it again
        local = {job.get("id"): job for job in load_jobs() if isinstance(job, dict)}
        for job_id, record in shipped.items():
            for key in record.pop("redacted_fields", None) or ():
                if job_id in local and key in local[job_id]:
                    record[key] = local[job_id][key]
        import_job_definitions(
            shipped, paused_reason=f"Imported from {directory}; review it, then resume.")
    return sorted(shipped)
