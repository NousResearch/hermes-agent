"""Cron imports remain usable when a daemon spans an on-disk upgrade.

A long-running scheduler already has ``hermes_cli.sqlite_util`` and ``cron.jobs`` cached from
BEFORE the upgrade; the first lazy import of a cron store afterwards must not need names those
stale modules lack (``scheduler_prompt._build_job_prompt`` imports ``cron.notepad`` unguarded, so
an ``ImportError`` there fails every job tick until restart).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SKEW_SCRIPT = """
import sys, types
import hermes_cli.sqlite_util as sqlite_util
import cron.jobs as jobs

# The pre-upgrade sqlite_util only had add_column_if_missing / write_txn.
for name in ("open_db", "transaction"):
    delattr(sqlite_util, name)
sys.modules.pop("cron.{store}", None)

import cron.{store}
"""

_OCCURRENCES_SKEW_SCRIPT = """
from datetime import datetime, timedelta, timezone
import cron.jobs as jobs

# Model a daemon that loaded cron.jobs before these constants existed, then
# lazy-loads the newer occurrences module from disk during a due scan.
jobs.__dict__.pop("FIRE_CLAIM_SKEW_SECONDS", None)
jobs.__dict__.pop("FIRE_CLAIM_TTL_SECONDS", None)

from cron.occurrences import completed_occurrence, unclaimed_pending_slot

assert not completed_occurrence({"id": "job"}, "2026-01-01T00:00:00+00:00")

# A slot stamped by another owner whose lease has lapsed is restored: the TTL comparison runs.
now = datetime.now(timezone.utc)
stale = (now - timedelta(hours=1)).isoformat()
job = {"id": "job", "schedule": {"kind": "interval"},
       "pending_slot": {"scheduled_at": stale, "at": stale, "by": "other-machine"}}
assert unclaimed_pending_slot(job, now) == stale
"""


@pytest.mark.parametrize("store", ["notepad", "incidents", "executions", "delivery_queue"])
def test_lazy_cron_stores_import_against_pre_upgrade_sqlite_util(store):
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _SKEW_SCRIPT.format(store=store)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_occurrences_uses_skew_constant_without_cached_jobs_export():
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _OCCURRENCES_SKEW_SCRIPT],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
