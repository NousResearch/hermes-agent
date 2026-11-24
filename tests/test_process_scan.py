"""A child that exits while a test scans its parent's children is skipped, not raised on.

``tests/process_scan.py`` is what tests use to find a worker/supervisor child by command line.
"""

from __future__ import annotations

import os
import subprocess

import psutil
import pytest

from tests.process_scan import children_matching
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")


def test_an_exited_unreaped_child_is_skipped_and_the_live_one_is_found():
    exited = subprocess.Popen(["true"])
    keeper = subprocess.Popen(["sleep", "30"])
    try:
        wait_until(lambda: psutil.Process(exited.pid).status() == psutil.STATUS_ZOMBIE, what="zombie child")
        found = children_matching(os.getpid(), lambda p: p.cmdline()[:1] == ["sleep"])
        assert [p.pid for p in found] == [keeper.pid]
    finally:
        keeper.kill()
        keeper.wait()
        exited.wait()
