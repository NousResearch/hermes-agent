#!/usr/bin/env python3
"""Profile-store routing utility for agent cron jobs.

Maps job names to their target profile cron store path
(``$HERMES_HOME/profiles/<profile>/cron/jobs.json``). This is a simple lookup
table, not a complex routing system — it exists so profile-route agent jobs
are explicitly bound to the gateway that owns them, instead of relying on
whatever HERMES_HOME the ticker happens to hold at call time.

Design intent (#4707): a profile's cron jobs both LIVE in that profile's
HERMES_HOME and EXECUTE under it. The lookup here mirrors the per-profile
storage contract pinned in ``tests/cron/test_cron_profile_isolation.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

# Job name -> owning profile. Source of truth for the 3 agent jobs whose
# prompts were rewritten to be profile-portable in p13/profile-route.
JOB_TO_PROFILE: dict[str, str] = {
    "content-engine-personal-llm": "ceecee",
    "MrHermagi Daily Lesson": "mrhermagi",
    "wesker-ops-daily": "wesker",
}


def get_profile_for_job(job_name: str) -> str:
    """Return the owning profile for ``job_name`` or raise KeyError.

    KeyError (not ValueError) keeps this a pure lookup — callers that want
    a soft-fall should catch it. An unknown job has no implicit profile.
    """
    try:
        return JOB_TO_PROFILE[job_name]
    except KeyError:
        raise KeyError(f"no profile routing for job: {job_name!r}")


def get_profile_cron_store(job_name: str, *, hermes_home: Path | None = None) -> Path:
    """Resolve the profile cron store path for ``job_name``.

    ``hermes_home`` overrides the ``HERMES_HOME`` env var (useful for tests).
    When ``None`` the env var must be set — the cron store is per-profile, so
    a missing HERMES_HOME is an ambiguous-routing error, not a default.
    """
    if hermes_home is None:
        env = os.environ.get("HERMES_HOME")
        if not env:
            raise ValueError("HERMES_HOME is not set; cannot resolve profile cron store")
        hermes_home = Path(env)
    profile = get_profile_for_job(job_name)
    return Path(hermes_home) / "profiles" / profile / "cron" / "jobs.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_name", help="job name to resolve a profile cron store for")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=None,
        help="override HERMES_HOME (defaults to the env var)",
    )
    args = parser.parse_args(argv)
    store = get_profile_cron_store(args.job_name, hermes_home=args.hermes_home)
    print(str(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
