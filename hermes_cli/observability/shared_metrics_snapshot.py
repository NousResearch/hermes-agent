"""Collect the once-a-day install configuration snapshot for shared metrics.

Only counts and a bundled memory-provider name are read out; server names, plugin names, skill
names, job prompts and profile names stay local (the contract buckets every count).
"""

from __future__ import annotations

import json
from itertools import islice
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


def collect_install_snapshot(config: dict[str, Any]) -> dict[str, str]:
    """Bounded snapshot fields for the active profile's ``config``."""
    return contract.install_snapshot_fields(
        memory_provider=_sub(_sub(config, "memory"), "provider"),
        mcp_servers=_mcp_server_count(config),
        plugins=_plugin_count(config),
        skills=_skill_count(),
        cron_jobs=_cron_job_count(),
        profiles=_profile_count(),
    )
