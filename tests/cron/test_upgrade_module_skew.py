"""Cron lazy stores remain usable when a daemon spans the SQLite-owner move.

A long-running scheduler may still have the pre-upgrade ``hermes_cli.sqlite_util`` and
``cron.jobs`` objects cached when newer cron store modules are loaded from disk. The new stores
must resolve ``storage.sqlite_util`` independently at call time; otherwise one stale module graph
can break every job tick until the daemon restarts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SQLITE_OWNER_SKEW_SCRIPT = """
import importlib
import os
import sys
import tempfile
import types

with tempfile.TemporaryDirectory() as home:
    os.environ["HERMES_HOME"] = home

    import hermes_cli
    import cron.jobs as jobs

    # Model the pre-upgrade module object retained by a long-running daemon.
    # It intentionally lacks the newer open_db / transaction exports.
    stale_sqlite_util = types.ModuleType("hermes_cli.sqlite_util")
    stale_sqlite_util.add_column_if_missing = lambda *args, **kwargs: None
    stale_sqlite_util.write_txn = lambda *args, **kwargs: None
    sys.modules["hermes_cli.sqlite_util"] = stale_sqlite_util
    hermes_cli.sqlite_util = stale_sqlite_util

    # Load the post-upgrade store from disk, then force its late SQLite import
    # to resolve from the new owner rather than anything retained above.
    sys.modules.pop("cron.{store}", None)
    store = importlib.import_module("cron.{store}")

    sys.modules.pop("storage.sqlite_util", None)
    import storage
    if hasattr(storage, "sqlite_util"):
        delattr(storage, "sqlite_util")

    conn = store._connect()
    conn.close()

    fresh_sqlite_util = sys.modules["storage.sqlite_util"]
    assert fresh_sqlite_util is not stale_sqlite_util
    assert hasattr(fresh_sqlite_util, "open_db")
    assert hasattr(fresh_sqlite_util, "transaction")
    assert sys.modules["hermes_cli.sqlite_util"] is stale_sqlite_util
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
def test_lazy_cron_stores_resolve_new_sqlite_owner_across_upgrade_skew(store):
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _SQLITE_OWNER_SKEW_SCRIPT.format(store=store)],
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
