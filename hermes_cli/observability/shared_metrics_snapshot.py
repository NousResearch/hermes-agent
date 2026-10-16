"""Collect the once-a-day install configuration snapshot for shared metrics.

Only counts, closed enums (terminal backend, bundled memory provider, shipped locale), the main
provider id and the profile's age are read out; server names, plugin names, skill names, job
prompts and profile names stay local (the contract buckets every count).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from itertools import islice
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import shared_metrics_contract as contract

# The largest bucket starts at 251: counting past it only costs I/O.
_COUNT_CAP = 251


def _sub(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else None


def _mcp_server_count(config: dict[str, Any]) -> int:
    servers = _sub(config, "mcp_servers")
    if not isinstance(servers, dict):
        return 0
    return sum(1 for spec in servers.values() if not (isinstance(spec, dict) and spec.get("enabled") is False))


def _plugin_count(config: dict[str, Any]) -> int:
    enabled = _sub(_sub(config, "plugins"), "enabled")
    return len(enabled) if isinstance(enabled, list) else 0


def _cron_job_count() -> int:
    try:
        # utf-8-sig: same dialect as cron/jobs.load_jobs (Windows editors may leave a BOM).
        with open(get_hermes_home() / "cron" / "jobs.json", encoding="utf-8-sig") as handle:
            jobs = json.load(handle).get("jobs", [])
    except (OSError, ValueError, AttributeError):
        return 0
    return sum(1 for job in jobs if isinstance(job, dict) and job.get("enabled", True))


def _skill_count() -> int:
    from agent.skill_utils import iter_skill_index_files

    skills_dir = get_hermes_home() / "skills"
    if not skills_dir.is_dir():
        return 0
    return sum(1 for _ in islice(iter_skill_index_files(skills_dir, "SKILL.md"), _COUNT_CAP))


def _profile_count() -> int:
    from hermes_cli.profiles import list_profile_names

    return len(list_profile_names())


def _messaging_platform_count() -> int:
    from gateway.config import load_gateway_config

    return len(load_gateway_config().get_connected_platforms())


def _display_language() -> str:
    from agent.i18n import get_language

    return get_language()


def _safe_count(reader) -> int:
    try:
        return reader()
    except Exception:
        return 0


# The profile's first session never moves once it exists, so it is read from state.db once.
_first_session_at: dict[str, float] = {}


def _first_session_started_at(home: Path | None = None) -> float | None:
    """Epoch of the profile's first session; ``now`` when it has none yet (a brand-new install).

    Raises ``sqlite3.Error`` when state.db exists but cannot be read.
    """
    home = home or get_hermes_home()
    if str(home) in _first_session_at:
        return _first_session_at[str(home)]
    database = home / "state.db"
    if not database.exists():
        return time.time()
    from hermes_cli.sqlite_safe_read import connect_tracked

    with contextlib.closing(
        connect_tracked(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    ) as connection:
        row = connection.execute("SELECT MIN(started_at) FROM sessions").fetchone()
    if not row or row[0] is None:
        return time.time()
    _first_session_at[str(home)] = float(row[0])
    return _first_session_at[str(home)]


def install_age_bucket(home: Path | None = None) -> str:
    """How long ago the profile's first-ever session started, bucketed; ``unknown`` if unreadable.
    ``home`` defaults to the bound profile; callers off the turn thread must pass theirs."""
    from .shared_metrics_fields import install_age_bucket as bucket

    try:
        first = _first_session_started_at(home)
    except sqlite3.Error:
        return "unknown"
    return bucket(time.time() - first)


def collect_install_snapshot(config: dict[str, Any]) -> dict[str, str]:
    """Bounded snapshot fields for the active profile's ``config``."""
    model = _sub(config, "model")
    language = "other"
    with contextlib.suppress(Exception):
        language = _display_language()
    from .shared_metrics_install import install_v4_snapshot_fields

    return {**contract.install_snapshot_fields(
        memory_provider=_sub(_sub(config, "memory"), "provider"),
        mcp_servers=_mcp_server_count(config),
        plugins=_plugin_count(config),
        skills=_skill_count(),
        cron_jobs=_cron_job_count(),
        profiles=_profile_count(),
        messaging_platforms=_safe_count(_messaging_platform_count),
        install_age_bucket=install_age_bucket(),
        main_provider=_sub(model, "provider") if isinstance(model, dict) else None,
        terminal_backend=_sub(_sub(config, "terminal"), "backend"),
        display_language=language,
    ), **install_v4_snapshot_fields(config)}
