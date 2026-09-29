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
