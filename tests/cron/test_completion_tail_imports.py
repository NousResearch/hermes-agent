"""Every lazily-imported module the cron completion tail reaches is loaded before the run starts.

Why: ``mark_job_run`` / ``finish_execution`` run minutes after a job began. A restart-safe
worker (or a desktop-standalone gateway) that spans an in-place ``hermes update`` then resolves a
first-time ``from cron import quota_hold`` against the NEW file on disk while its cached
``hermes_time`` is the OLD module -> ``ImportError: cannot import name 'safe_strftime'``; the job's
output is written but never delivered, ``last_status`` is never updated and no incident opens.
A weekly report vanished that way. Loading the tail's modules at package import pins the whole
completion path to one code generation.
"""

import ast
import sys
from pathlib import Path

import pytest

_CRON = Path(__file__).resolve().parents[2] / "cron"
# The functions that run after the job body: bookkeeping, ledger, incident, failure copy.
_TAIL_FILES = ("jobs.py", "executions.py", "incidents.py", "scheduler_failure_copy.py",
               "quota_hold.py", "unreachable_retry.py", "occurrences.py")


def _function_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mod = node.module
                if mod.startswith(("cron.", "hermes_", "agent.", "gateway.", "tools.")) or mod == "cron":
                    found.add(mod)
                    if mod == "cron":
                        found.update(f"cron.{a.name}" for a in node.names)
    return found


@pytest.mark.parametrize("filename", _TAIL_FILES)
def test_completion_tail_modules_are_loaded_with_the_cron_package(filename):
    import cron  # noqa: F401  (the package import is what pins the generation)

    wanted = {m for m in _function_level_imports(_CRON / filename) if m.startswith("cron.")}
    missing = sorted(m for m in wanted if m not in sys.modules)
    assert not missing, f"{filename}: cron modules first imported inside the completion tail: {missing}"
