"""Cron imports remain usable when a daemon spans an on-disk upgrade."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_lazy_cron_stores_do_not_require_new_symbols_on_cached_executions_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import sys
import cron.executions as executions

for name in ("ledger_transaction", "open_ledger", "prepare_ledger"):
    delattr(executions, name)
sys.modules.pop("cron.incidents", None)
sys.modules.pop("cron.notepad", None)

import cron.incidents
import cron.notepad
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
