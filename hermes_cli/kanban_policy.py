"""Kanban policy knobs (claim TTL, crash grace, rate-limit cooldown).

Extracted verbatim from ``hermes_cli/kanban_db.py`` (wave-1 godfile
decomposition, shard s1, cluster c2 — ``kanban_policy.py``). Each resolver
reads its ``HERMES_KANBAN_*`` env override and falls back to a built-in
default; the defaults stay defined in ``hermes_cli.kanban_db`` (they are
referenced by tests and by staying dispatcher code) and are imported at the
bottom of this module.
"""

from __future__ import annotations

import os
from typing import Optional

def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Return the effective claim TTL, honoring the kanban env override.

    Explicit call-site values win. Otherwise a positive integer from
    ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` overrides the built-in default.
    Invalid or non-positive env values fall back silently so existing
    installs keep working.
    """
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    raw = os.environ.get("HERMES_KANBAN_CLAIM_TTL_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return DEFAULT_CLAIM_TTL_SECONDS
def _resolve_crash_grace_seconds() -> int:
    """Return the crash-detection grace period in seconds.

    Reads ``HERMES_KANBAN_CRASH_GRACE_SECONDS`` from the environment;
    falls back to ``DEFAULT_CRASH_GRACE_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 restores immediate-reclaim
    behaviour (useful for tests).
    """
    raw = os.environ.get("HERMES_KANBAN_CRASH_GRACE_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_CRASH_GRACE_SECONDS
def _resolve_rate_limit_cooldown_seconds() -> int:
    """Return the rate-limit requeue cooldown in seconds.

    Reads ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` from the environment;
    falls back to ``DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 disables the cooldown (re-spawn on
    the next tick) — useful for tests that want to assert the task becomes
    spawnable again immediately.
    """
    raw = os.environ.get(
        "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", ""
    ).strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


# Imported at the bottom to break the import cycle with kanban_db.py (which
# re-imports the moved functions from this module). Function bodies resolve
# these names at call time, so bottom placement is safe.
from hermes_cli.kanban_db import (  # noqa: E402
    DEFAULT_CLAIM_TTL_SECONDS,
    DEFAULT_CRASH_GRACE_SECONDS,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
)
