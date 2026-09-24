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
for name in ("FIRE_CLAIM_SKEW_SECONDS", "FIRE_CLAIM_TTL_SECONDS"):
    delattr(jobs, name)

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


def test_occurrences_resolve_fire_claim_constants_without_cached_jobs_exports():
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _OCCURRENCES_SKEW_SCRIPT],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# A daemon that crossed the 2026-09-23 upgrade mid-run kept ``hermes_time`` cached from the revision
# that predated ``safe_strftime`` (it had no such name), then lazy-imported a consumer from the newly
# checked-out tree: ``ImportError: cannot import name 'safe_strftime' from 'hermes_time'
# (/var/lib/hermes/src/hermes_time.py)`` marked two cron runs failed (jobs b20a472b7e80 /
# f98d49b4524e) even though the workers kept running to completion.
_SAFE_STRFTIME_SKEW_SCRIPT = """
import sys

import hermes_time

# The pre-upgrade hermes_time had no formatting helpers at all.
for name in ("safe_strftime", "_repair_surrogates"):
    if hasattr(hermes_time, name):
        delattr(hermes_time, name)
assert not hasattr(hermes_time, "safe_strftime")

consumers = [
    "cron.scheduler",
    "cron.quota_hold",
    "agent.learning_graph_render",
    "agent.insights",
    "agent.billing_usage",
    "agent.auxiliary_unavailable",
    "agent.account_usage",
    "agent.system_prompt",
    "hermes_cli.status_auth",
    "hermes_cli.goals",
    "gateway.message_timestamps",
    "tools.session_search_tool",
]
for module in consumers:
    sys.modules.pop(module, None)
    try:
        __import__(module)
    except ImportError as exc:
        # An unrelated missing optional dep must not mask the regression under test.
        if "safe_strftime" in str(exc):
            raise
        print("skipped", module, exc)

from hermes_time_format import _repair_surrogates, safe_strftime

assert safe_strftime.__module__ == "hermes_time_format"
assert _repair_surrogates is not None
print("ok")
"""


def test_safe_strftime_consumers_import_against_pre_upgrade_hermes_time():
    """The helper lives in its own leaf module, so a consumer imported after the upgrade loads it
    from disk even though the cached ``hermes_time`` predates it."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _SAFE_STRFTIME_SKEW_SCRIPT],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pre_upgrade_hermes_time_still_fails_when_a_consumer_imports_it_directly():
    """Guard the guard: the test above is only meaningful while importing the name from the cached
    module really does raise, i.e. the failure signature being reproduced is the real one."""
    script = """
import hermes_time
delattr(hermes_time, "safe_strftime")
try:
    from hermes_time import safe_strftime  # noqa: F401
except ImportError as exc:
    assert "cannot import name 'safe_strftime'" in str(exc), exc
    print("reproduced")
else:
    raise AssertionError("stale hermes_time should not export safe_strftime")
"""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
