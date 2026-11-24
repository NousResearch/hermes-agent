"""Cron calls for plugins: schedule one named job, and read where cron delivers a job's output.

Published as a whole module (``hermes_cli/data/plugin_api.json``), so a dashboard route or a
standalone process without a ``ctx`` can use it too. Each function forwards to the code cron itself
runs; none adds a rule of its own. Guide: plugins/index.md § Schedule a job once and read where cron
delivers it.
"""
from __future__ import annotations

import contextlib as _contextlib
from pathlib import Path as _Path
from typing import Any as _Any, Optional as _Optional

from cron.jobs import AmbiguousJobReference

__all__ = ["AmbiguousJobReference", "delivery_platforms", "ensure_named_job"]


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
    return None if home is None else _Path(home).expanduser()


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
    with the active scheduler provider again and is returned as stored: its prompt and schedule are
    not updated. A new job is created and registered the way ``hermes cron add`` does it; errors from
    either step propagate unchanged.

    ``name`` must be a non-empty ``str`` with no leading or trailing whitespace, since cron could
    not find any other name again; anything else raises ``ValueError`` before anything is written.

    ``home`` runs the call against that profile's home, cron store, secrets and settings (its
    time zone for a schedule), all restored afterwards; ``~`` is expanded. Its secrets are read from
    its own files, also when it is the profile already active. A ``home`` that is not an existing
    directory raises ``FileNotFoundError`` and is not created. Inside a hosted activity, which
    cannot act for a profile, binding one is refused. The lookup and the create are not atomic
    across processes: a caller that can race with itself serializes its own calls.
    """
    from cron.jobs import list_jobs
    from cron.scheduler import create_job_with_scheduler_registration
    from cron.scheduler_provider import resolve_cron_scheduler

    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(
            f"job name must be non-empty with no leading or trailing whitespace, got {name!r}")
    home = _expanded(home)
    if home is not None and not home.is_dir():
        raise FileNotFoundError(f"Hermes home does not exist: {home}")
    with _home_scope(home):
        matches = [job for job in list_jobs(include_disabled=True) if job.get("name") == name]
        if len(matches) > 1:
            raise AmbiguousJobReference(name, matches)
        if matches:
            job = matches[0]
            if job.get("enabled", True):
                resolve_cron_scheduler().register_job(job)
            return job
        return create_job_with_scheduler_registration(
            prompt=prompt, schedule=schedule, name=name, repeat=repeat, deliver=deliver,
            enabled_toolsets=enabled_toolsets)


def delivery_platforms(
    job_id: _Optional[str] = None, *, home: _Optional[str | _Path] = None,
) -> _Optional[tuple[str, ...]]:
    """The platforms cron would deliver a job's output to right now, lower-cased, in order.

    ``job_id=None`` answers for the cron run executing in this context, and returns ``None`` outside
    one. ``None`` also when no stored job has that id. ``()`` for a job that delivers nowhere. The
    targets come from cron's own resolver, so ``all``, the ``origin`` fallback, home channels and
    Bot Chat tokens follow cron's rules. Only platform names are returned, never chat or thread ids.
    ``home`` runs the call against that profile's home, cron store and secrets; ``None`` when it
    is not an existing directory, which is not created.
    """
    from cron.execution_identity import current_cron_execution
    from cron.jobs import get_job
    from cron.scheduler_delivery import _resolve_delivery_targets

    home = _expanded(home)
    if home is not None and not home.is_dir():
        return None
    if job_id is None:
        run = current_cron_execution()
        if run is None:
            return None
        job_id = run.job_id
    with _home_scope(home):
        job = get_job(job_id)
        if job is None:
            return None
        platforms = [str(target["platform"]).lower() for target in _resolve_delivery_targets(job)]
    return tuple(dict.fromkeys(platforms))
