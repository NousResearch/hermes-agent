"""``prefer_as_oom_victim`` raises a child's OOM score relative to its spawner, capped."""

import subprocess
import sys

import pytest

@pytest.mark.platforms("linux")
def test_child_score_is_raised_relative_to_the_spawner_and_capped():
    from hermes_platform.oom import prefer_as_oom_victim

    own = int(open("/proc/self/oom_score_adj", encoding="utf-8").read())
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        prefer_as_oom_victim(proc.pid)
        assert int(open(f"/proc/{proc.pid}/oom_score_adj", encoding="utf-8").read()) == min(1000, own + 500)
        prefer_as_oom_victim(proc.pid)
        assert int(open(f"/proc/{proc.pid}/oom_score_adj", encoding="utf-8").read()) == min(1000, own + 1000)
    finally:
        proc.kill()
        proc.wait()
