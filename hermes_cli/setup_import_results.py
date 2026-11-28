"""Keep first-time setup's import results across its screen-clearing menus."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
from pathlib import Path
import shlex

logger = logging.getLogger("hermes_cli.setup")
_IMPORT_JOBS: ContextVar[dict[Path, set[str]] | None] = ContextVar("setup_import_jobs", default=None)


@contextmanager
def setup_import_results_scope():
    """Keep hints local to this wizard invocation, including its quick/blank-slate runners."""
    token = _IMPORT_JOBS.set({})
    try:
        yield
    finally:
        _IMPORT_JOBS.reset(token)


def import_job_snapshot(hermes_home: Path) -> dict[str, dict] | None:
    """Read the owning home's job store; an unavailable store cannot establish new job IDs."""
    from cron import jobs

    if not (hermes_home / "cron" / "jobs.json").exists():
        return {}
    try:
        with jobs.use_cron_store(hermes_home):
            return {job["id"]: job for job in jobs.load_jobs()}
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not read paused routine hints after import: %s", exc)
        return None


def remember_import_jobs(hermes_home: Path, before: dict[str, dict] | None) -> None:
    """Track only paused jobs created while this invocation ran an importer, never old jobs."""
    from cron.jobs import effective_job_state

    state = _IMPORT_JOBS.get()
    if state is None or before is None:
        return
    after = import_job_snapshot(hermes_home)
    if after is not None:
        state.setdefault(hermes_home.resolve(), set()).update(
            job_id for job_id, job in after.items()
            if job_id not in before and effective_job_state(job) == "paused"
        )


def print_import_job_hints(hermes_home: Path) -> None:
    """Repeat still-paused new jobs at the common completion summary, without parsing child stdout."""
    from cron.jobs import effective_job_state
    from hermes_constants import profile_name_for_home
    from hermes_cli.setup import _info

    home = Path(hermes_home)
    ids = (_IMPORT_JOBS.get() or {}).get(home.resolve(), set())
    if not ids:
        return
    current = import_job_snapshot(home)
    paused = [job for job_id, job in (current or {}).items()
              if job_id in ids and effective_job_state(job) == "paused"]
    if not paused:
        return
    profile = profile_name_for_home(home) or "default"
    _info(None, "New paused routines", "  Before resuming, turn off the original routine in your previous agent.")
    for job in paused:
        command = shlex.join(["hermes", "-p", profile, "cron", "resume", job["id"]])
        _info(f"  {command}    # {job['name']}")
    _info("  Results are saved in cron output; choose a delivery platform with:",
          f"  {shlex.join(['hermes', '-p', profile, 'cron', 'edit'])} <id> --deliver <platform>", None)
