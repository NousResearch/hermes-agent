"""Cron calls for plugins: schedule one named job, and read where cron delivers a job's output.

Published as a whole module (``hermes_cli/data/plugin_api.json``), so a dashboard route or a
standalone process without a ``ctx`` can use it too. Each function uses the scheduler's own job
and delivery logic. Guide: plugins/index.md § Schedule a job once and read where cron
delivers it.
"""
from __future__ import annotations

import contextlib as _contextlib
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING, Any as _Any, Optional as _Optional

from cron.jobs import AmbiguousJobReference

if _TYPE_CHECKING:
    from cron.scheduler import CronSchedulerRegistrationError

__all__ = [
    "AmbiguousJobReference",
    "CronSchedulerRegistrationError",
    "delivery_platforms",
    "ensure_named_job",
]


def __getattr__(name: str) -> _Any:
    if name == "CronSchedulerRegistrationError":
        from cron.scheduler import CronSchedulerRegistrationError
        return CronSchedulerRegistrationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@_contextlib.contextmanager
def _home_scope(home: _Optional[str | _Path]):
    """Run the block for profile ``home``, or for the active profile when ``home`` is ``None``.

    ``home`` is bound whole (home, secrets and terminal policy, as core binds a profile) with its
    cron store. A profile other than this process's own also runs under multiplex semantics, as a
    cron fire for it does, so a process-wide setting (``HERMES_TIMEZONE``) never speaks for it."""
    if home is None:
        yield
        return
    from agent.runtime_scope import bind_profile_scope
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from cron.jobs import use_cron_store
    from cron.scheduler_provider import routed_profile_fire

    with bind_profile_scope(home), use_cron_store(home), _contextlib.ExitStack() as stack:
        if routed_profile_fire(home):
            stack.callback(reset_multiplex_context, set_multiplex_context(True))
        yield


def _expanded(home: _Optional[str | _Path]) -> _Optional[_Path]:
    """*home* read the way Hermes reads a home path (``~`` and environment variables expanded), so
    the existence check, the profile bind and the cron store all name one directory."""
    if home is None:
        return None
    from hermes_constants import _expand_hermes_home
    return _expand_hermes_home(str(home))


def ensure_named_job(
    name: str,
    *,
    prompt: str,
    schedule: str,
    repeat: _Optional[int] = None,
    deliver: _Optional[str] = None,
    enabled_toolsets: _Optional[list[str]] = None,
    home: _Optional[str | _Path] = None,
) -> dict[str, _Any]:
    """Return the stored job named exactly ``name``, creating it when none exists.

    Every job counts, disabled ones included. Two or more with that name raise
    ``AmbiguousJobReference`` and create nothing. An existing enabled job has its trigger registered
    with the active scheduler provider again and is returned as stored, in the shape a new job has:
    its prompt and schedule are not updated. A new job is created and registered the way ``hermes cron add`` does it. A
    registration that fails, for a new job or an existing one, raises that command's
    ``CronSchedulerRegistrationError`` (a ``RuntimeError`` with the saved job as ``.job``).

    ``name`` must be a non-empty ``str`` with no leading or trailing whitespace, since cron could
    not find any other name again; anything else raises ``ValueError`` before anything is written.

    ``home`` runs the call against that profile's home, cron store, secrets and settings (its
    time zone for a schedule), all restored afterwards; ``~`` and environment variables are expanded
    once. Its secrets are read from its own files, also when it is the profile already active.
    A ``home`` that is not an existing directory raises ``FileNotFoundError`` and is not created.
    Inside a hosted activity, which cannot act for a profile, binding one is refused. The lookup
    and create are not atomic across processes: a caller that can race with itself serializes its own calls.
    """
    from cron.jobs import _normalize_job_record, load_jobs
    from cron.scheduler import CronSchedulerRegistrationError, create_job_with_scheduler_registration
    from cron.scheduler_provider import resolve_cron_scheduler

    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(
            f"job name must be non-empty with no leading or trailing whitespace, got {name!r}")
    home = _expanded(home)
    if home is not None and not home.is_dir():
        raise FileNotFoundError(f"Hermes home does not exist: {home}")
    with _home_scope(home):
        matches = [job for job in map(_normalize_job_record, load_jobs()) if job.get("name") == name]
        if len(matches) > 1:
            raise AmbiguousJobReference(name, matches)
        if matches:
            job = matches[0]
            if job.get("enabled", True):
                try:
                    resolve_cron_scheduler().register_job(job)
                except Exception as exc:
                    raise CronSchedulerRegistrationError(job, exc) from exc
            return job
        return create_job_with_scheduler_registration(
            prompt=prompt, schedule=schedule, name=name, repeat=repeat, deliver=deliver,
            enabled_toolsets=enabled_toolsets)


def delivery_platforms(
    job_id: _Optional[str] = None, *, home: _Optional[str | _Path] = None,
) -> _Optional[tuple[str, ...]]:
    """The platforms cron would deliver a job's output to right now, lower-cased, in order.

    ``job_id=None`` answers for the cron run bound in this process's calling context using that
    firing run's own delivery settings — edits or deletions saved to the job store mid-run do not
    change the answer — and returns ``None`` outside one (or in a delegated child). A hook in a
    plugin host has no cron run bound in its own process and calls ``ctx.cron_delivery_platforms()``
    instead so the lookup runs in the gateway. Passing an explicit ``job_id`` previews the saved job
    in the store (returning ``None`` when no stored job has that id). ``()`` means the job delivers
    nowhere. It reads the normal delivery lane; a ``failure_deliver`` override is not consulted. The
    targets come from cron's own resolver, so ``all``, the ``origin`` fallback, home channels and
    Bot Chat tokens follow cron's rules. Only platform names are returned, never chat or thread ids.
    A non-``None`` ``home`` requires an explicit ``job_id``; otherwise raises ``ValueError``.
    A preview with ``home`` uses that profile's home, cron store and secrets, or returns ``None``
    if the directory does not exist. Firing-run lookups use the profile already bound to the run.
    """
    from cron.execution_identity import _current_cron_execution_job
    from cron.jobs import get_job
    from cron.scheduler_delivery import _resolve_delivery_targets

    if home is not None and job_id is None:
        raise ValueError("job_id is required when home is specified")
    home = _expanded(home)
    if home is not None and not home.is_dir():
        return None
    firing_job = _current_cron_execution_job() if job_id is None else None
    if job_id is None and firing_job is None:
        return None
    with _home_scope(home):
        job = firing_job if job_id is None else get_job(job_id)
        if job is None:
            return None
        platforms = [str(target["platform"]).lower() for target in _resolve_delivery_targets(job)]
    return tuple(dict.fromkeys(platforms))
