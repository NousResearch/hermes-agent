"""The reproducible installer: deploy the package, register the two jobs.

This is the one place that turns "the code exists in this worktree" into
"the pipeline actually runs on schedule." Before this module existed, that
step was a set of manual shell commands run once by hand and never written
down anywhere a reviewer or a future session could see, re-run, or check.
Running this twice is always safe:

* The deploy step always copies the current package fresh — safe because
  ``$HERMES_HOME/scripts/claude_selfimprove/`` holds only files this
  installer itself writes; nothing user-authored ever lives there.
* The two cron jobs are created only when a job with that exact name does
  not already exist, checked by reading ``$HERMES_HOME/cron/jobs.json``
  directly rather than assuming a fresh profile. A second run against an
  already-installed profile deploys the latest code and touches no cron
  state at all.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import paths

PACKAGE_DIR = Path(__file__).resolve().parent
CRON_SCRIPTS_DIR = PACKAGE_DIR / "cron_scripts"
SCAN_SCRIPT_NAME = "claude_selfimprove_scan.py"
CONSOLIDATE_SCRIPT_NAME = "claude_selfimprove_consolidate.py"

NIGHTLY_JOB_NAME = "claude-selfimprove-nightly-scan"
WEEKLY_JOB_NAME = "claude-selfimprove-weekly-consolidate"
NIGHTLY_SCHEDULE = "0 3 * * *"
WEEKLY_SCHEDULE = "0 4 * * 0"

JobCreatorFn = Callable[..., tuple[bool, str]]


@dataclass
class InstallResult:
    deployed_package: bool = False
    deployed_scan_script: bool = False
    deployed_consolidate_script: bool = False
    nightly_job_created: bool = False
    nightly_job_already_existed: bool = False
    weekly_job_created: bool = False
    weekly_job_already_existed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _deploy_package(scripts_dir: Path, *, dry_run: bool) -> bool:
    dest = scripts_dir / "claude_selfimprove"
    if dry_run:
        return True
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        PACKAGE_DIR, dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return True


def _deploy_entry_script(scripts_dir: Path, name: str, *, dry_run: bool) -> bool:
    if dry_run:
        return True
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CRON_SCRIPTS_DIR / name, scripts_dir / name)
    return True


def _existing_job_names(hermes_home: Path) -> set[str]:
    """Job names already registered in this profile's cron store.

    Reads ``cron/jobs.json`` directly rather than shelling out to
    ``hermes cron list`` — this needs to run before we know a working
    ``hermes`` invocation exists (the installer's own job-creation step is
    what proves that), and the jobs file is the same data
    ``cron/jobs.py``/``hermes cron list`` themselves read from.
    """
    jobs_file = hermes_home / "cron" / "jobs.json"
    if not jobs_file.exists():
        return set()
    try:
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    if isinstance(jobs, dict):
        jobs = list(jobs.values())
    if not isinstance(jobs, list):
        return set()
    return {str(j["name"]) for j in jobs if isinstance(j, dict) and j.get("name")}


def _create_job_via_cli(
    name: str, schedule: str, script: str, *, hermes_bin: str = "hermes", timeout: int = 30
) -> tuple[bool, str]:
    cmd = [
        hermes_bin, "cron", "create", schedule,
        "--name", name, "--script", script, "--no-agent", "--deliver", "local",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return False, f"hermes CLI not found: {exc}"
    except subprocess.TimeoutExpired:
        return False, f"hermes cron create timed out after {timeout}s"
    except OSError as exc:
        return False, f"hermes cron create failed to start: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"hermes cron create exited {result.returncode}: {detail[:400]}"
    return True, result.stdout or ""


def run(
    *,
    dry_run: bool = False,
    job_creator: JobCreatorFn | None = None,
    hermes_bin: str = "hermes",
) -> InstallResult:
    """Deploy the package and idempotently register both cron jobs."""
    result = InstallResult()
    create = job_creator or _create_job_via_cli
    hermes_home = paths.hermes_home()
    scripts_dir = hermes_home / "scripts"

    try:
        result.deployed_package = _deploy_package(scripts_dir, dry_run=dry_run)
        result.deployed_scan_script = _deploy_entry_script(scripts_dir, SCAN_SCRIPT_NAME, dry_run=dry_run)
        result.deployed_consolidate_script = _deploy_entry_script(
            scripts_dir, CONSOLIDATE_SCRIPT_NAME, dry_run=dry_run
        )
    except OSError as exc:
        result.errors.append(f"deploy failed: {exc}")
        return result

    existing = _existing_job_names(hermes_home)

    if NIGHTLY_JOB_NAME in existing:
        result.nightly_job_already_existed = True
    elif dry_run:
        result.nightly_job_created = True  # "would create"
    else:
        ok, msg = create(NIGHTLY_JOB_NAME, NIGHTLY_SCHEDULE, SCAN_SCRIPT_NAME, hermes_bin=hermes_bin)
        if ok:
            result.nightly_job_created = True
        else:
            result.errors.append(f"nightly job: {msg}")

    if WEEKLY_JOB_NAME in existing:
        result.weekly_job_already_existed = True
    elif dry_run:
        result.weekly_job_created = True
    else:
        ok, msg = create(WEEKLY_JOB_NAME, WEEKLY_SCHEDULE, CONSOLIDATE_SCRIPT_NAME, hermes_bin=hermes_bin)
        if ok:
            result.weekly_job_created = True
        else:
            result.errors.append(f"weekly job: {msg}")

    return result
