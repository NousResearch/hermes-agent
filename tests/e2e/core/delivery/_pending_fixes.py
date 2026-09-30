"""Merge-order-safe expected failures for live gaps whose fix is an open PR.

A plain ``xfail(strict=True)`` turns main red the moment its fix merges (XPASS), and a
non-strict one guards nothing. Instead, each gap here has a PROBE: a few lines that reproduce
the defect's mechanism on the tree under test, in a throwaway interpreter with its own
``HOME``/``HERMES_HOME`` (no state leaks into the suite process). A cell asks ``gap_open`` and
tolerates the gap only while the probe still reproduces the defect; once the fix is in the tree
the cell runs as a plain test, so it must pass. Whichever lands first, suite or fix, main stays
green, and a probe that disagrees with the end-to-end cell still fails loudly instead of hiding.

Probes exercise behaviour only (never read source text). When a fix has landed, delete its
entry and every ``gap_open`` call naming it.

A gap with no fix PR yet has no probe: ``known_failure`` is a run-time xfail keyed on the gap's
own assertion message, so the cell XFAILs only while it fails exactly that way, fails loudly on
any other failure, and simply passes once someone fixes the gap.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

_PRELUDE = "import json, os, sys\nsys.path.insert(0, os.getcwd())\n"

# PR -> (extra env, script). A script prints ``open`` while the defect reproduces, ``fixed`` once
# it no longer does; anything else (including a crash) fails the cell that asked.
PROBES: Dict[int, tuple] = {
    # No timezone configured: the next cron occurrence kept the base time's fixed UTC offset, so a
    # 09:00 job in a DST process zone fired at 10:00 local the day after spring-forward.
    119970: ({"TZ": "America/New_York"}, r'''
from datetime import datetime
from cron import jobs
# what an unconfigured clock returns: the process zone's offset of the moment, as a fixed offset
jobs._hermes_now = lambda: datetime.fromisoformat("2026-03-07T09:00:30-05:00")
nxt = jobs.compute_next_run({"kind": "cron", "expr": "0 9 * * *"})
print("fixed" if nxt == "2026-03-08T09:00:00-04:00" else "open")
'''),
}


@functools.lru_cache(maxsize=None)
def gap_open(pr: int) -> bool:
    """True while the defect PR ``pr`` fixes still reproduces on this tree."""
    extra_env, script = PROBES[pr]
    with tempfile.TemporaryDirectory(prefix=f"gap-{pr}-") as tmp:
        home = Path(tmp) / "home"
        (home / ".hermes").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTEST_", "HERMES_")) and not k.endswith("_API_KEY")}
        env.update({"HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
                    "PYTHONPATH": str(REPO_ROOT), **extra_env})
        proc = subprocess.run([sys.executable, "-c", _PRELUDE + script], cwd=str(REPO_ROOT), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    verdict = proc.stdout.strip().splitlines()[-1:] if proc.returncode == 0 else []
    if verdict not in (["open"], ["fixed"]):
        raise AssertionError(f"probe for #{pr} is broken (rc={proc.returncode}): "
                             f"{json.dumps(proc.stdout[-800:])} {proc.stderr[-2000:]}")
    return verdict == ["open"]


@contextlib.contextmanager
def known_failure(pattern: str, reason: str,
                  on_xfail: Optional[Callable[[], None]] = None) -> Iterator[None]:
    """Run-time xfail for a live gap without a fix PR: an ``AssertionError`` raised inside the
    block whose message matches ``pattern`` XFAILs the cell; any other failure propagates, and a
    clean pass stays a pass. Wrap only the final assertions, after every wait has settled, so a
    lost reply, a failed restart or a timeout can never be mistaken for the gap."""
    try:
        yield
    except AssertionError as exc:
        if not re.search(pattern, str(exc)):
            raise
        if on_xfail is not None:
            on_xfail()
        pytest.xfail(f"{reason} [observed: {str(exc).splitlines()[0][:240]}]")
