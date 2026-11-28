"""Observe early Codex quota recovery without waiting for HTTP in the cron tick."""

from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import Any

from cron import jobs, quota_hold
from cron.constants import is_recurring
from hermes_constants import get_hermes_home, hermes_home_key
from utils import file_signature

logger = logging.getLogger("cron.scheduler")
_PROBE_INTERVAL_SECONDS = 300
_MAX_ACTIVE_PROFILES = 4


@dataclass(frozen=True)
class _Recovery:
    revisions: tuple
    held_jobs: dict[str, dict[str, Any]]


@dataclass
class _Attempt:
    expires_at: float
    future: Future[_Recovery | None] | None = None


_attempts: dict[str, _Attempt] = {}
_attempts_lock = threading.Lock()


def _eligible(job: dict, now: datetime) -> bool:
    return (bool(job.get("enabled")) and job.get("state") == "scheduled"
            and is_recurring(job) and not job.get("no_agent") and not job.get("base_url")
            and job.get("next_run_at") == job.get(quota_hold.STATE_KEY)
            and quota_hold.hold_active(job, now)
            and not any(job.get(key) for key in ("manual_run_at", "fire_claim", "run_claim", "pending_slot")))


def _revision(path: Path) -> tuple[int, int, int, int] | None:
    try:
        return file_signature(path.stat())
    except FileNotFoundError:
        return None


def _route_revisions() -> tuple:
    from hermes_cli.auth import _auth_file_path, _global_auth_file_path
    from hermes_cli.config import get_config_path

    paths = (_auth_file_path(), _global_auth_file_path(), get_config_path(), get_hermes_home() / ".env")
    return tuple((path, _revision(path)) for path in paths if path is not None)


def _codex_jobs(held_jobs: dict[str, dict]) -> dict[str, dict]:
    from agent.secret_scope import get_secret_str
    from hermes_cli.config import is_provider_enabled, load_config_readonly, split_model_config_default

    config = load_config_readonly()
    model = config.get("model")
    model = model if isinstance(model, dict) else {}
    if str(model.get("openai_runtime") or "").strip().lower() == "codex_app_server":
        return {}  # The backend owns its login; Hermes's OAuth store isn't evidence about it.
    providers = config.get("providers")
    codex = providers.get("openai-codex") if isinstance(providers, dict) else None
    if isinstance(codex, dict) and not is_provider_enabled(codex):
        return {}
    configured = model.get("provider")
    if not configured and isinstance(model.get("default"), dict):
        _, configured = split_model_config_default(model["default"])
    configured = configured or get_secret_str("HERMES_INFERENCE_PROVIDER", "")
    cron_config = config.get("cron")
    cron_provider = cron_config.get("model_provider") if isinstance(cron_config, dict) else None
    return {job_id: job for job_id, job in held_jobs.items()
            if str(job.get("provider") or cron_provider or configured).strip().lower() == "openai-codex"}


def _codex_reopened() -> bool:
    from agent.credential_pool import STATUS_DEAD
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error, read_credential_pool
    from hermes_cli.auth_codex import (
        _codex_pool_rate_limit_status,
        _codex_pool_route_base_url,
        _probe_codex_quota_restored,
        _stripped,
        resolve_codex_runtime_credentials,
    )

    try:
        creds = resolve_codex_runtime_credentials(read_only=True)
        token, base_url = creds["api_key"], creds["base_url"]
    except AuthError as exc:
        if not is_rate_limited_auth_error(exc) or not (entry := _codex_pool_rate_limit_status()):
            return False
        token, base_url = entry["access_token"], _codex_pool_route_base_url(entry.get("base_url"))
    # The execution resolver is pool-first, but the read-only auth resolver is singleton-first.
    # Without a single route, opening one login's window cannot attest to the held route.
    rows = read_credential_pool("openai-codex")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        if row.get("last_status") == STATUS_DEAD:
            continue
        if (_stripped(row.get("access_token")) != token
                or _codex_pool_route_base_url(row.get("base_url")) != base_url):
            return False
    # The same cache may contain a successful probe from BEFORE this job's last 429.
    return _probe_codex_quota_restored(token, base_url=base_url, min_interval_seconds=0) is True


def _observe(held_jobs: dict[str, dict]) -> _Recovery | None:
    from cron.scheduler import _fire_secret_scope

    revisions = _route_revisions()
    with _fire_secret_scope():
        targets = _codex_jobs(held_jobs)
        if targets and _codex_reopened():
            return _Recovery(revisions, targets)
    return None


def _apply(recovery: _Recovery) -> None:
    with jobs._jobs_lock():
        if recovery.revisions != _route_revisions():
            return
        stored = jobs.load_jobs()
        now = jobs._hermes_now()
        changed = False
        for job in stored:
            job_id = job.get("id")
            if (isinstance(job_id, str) and recovery.held_jobs.get(job_id) == job and _eligible(job, now)
                    and not jobs._job_running_in_this_process(job_id)):
                changed |= quota_hold.resume_on_schedule(job, now)
        if changed:
            jobs.save_jobs(stored)


def _run_probe(future: Future[_Recovery | None], held_jobs: dict[str, dict]) -> None:
    try:
        future.set_result(_observe(held_jobs))
    except Exception:
        logger.debug("Cron quota observation failed; keeping holds", exc_info=True)
        future.set_result(None)


def _at_capacity() -> bool:
    return sum(1 for item in _attempts.values()
               if item.future is not None and not item.future.done()) >= _MAX_ACTIVE_PROFILES


def _collect_or_start(home: Path) -> _Recovery | None:
    key = hermes_home_key(home)
    now = time.monotonic()
    with _attempts_lock:
        for expired in [key for key, item in _attempts.items() if item.expires_at <= now
                        and (item.future is None or item.future.done())]:
            del _attempts[expired]
        if attempt := _attempts.get(key):
            if attempt.future is None or not attempt.future.done():
                return None
            future, attempt.future = attempt.future, None
            return future.result()
        if _at_capacity():
            return None

    wall_now = jobs._hermes_now()
    held = {job["id"]: job for job in jobs.load_jobs() if _eligible(job, wall_now)}
    with _attempts_lock:
        if key in _attempts or (held and _at_capacity()):
            return None
        attempt = _Attempt(now + _PROBE_INTERVAL_SECONDS)
        if held:
            # The ticker already selected the owning home/store. Preserve that context across the
            # thread hop and install that home's credential scope inside the worker, not here.
            future = Future()
            threading.Thread(target=copy_context().run, args=(_run_probe, future, held),
                             name="cron-quota-probe", daemon=True).start()
            attempt.future = future
        _attempts[key] = attempt
    return None


def reconcile() -> None:
    """On an admitted tick, consume a completed observation or start one without waiting."""
    try:
        if recovery := _collect_or_start(get_hermes_home()):
            _apply(recovery)
    except Exception:
        logger.debug("Cron quota recovery could not reconcile holds", exc_info=True)
