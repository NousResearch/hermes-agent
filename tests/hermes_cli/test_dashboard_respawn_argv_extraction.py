"""Exact argv capture, not decoding ps display text, makes inline commands replayable."""
import subprocess
import sys
import time
from unittest.mock import patch

import psutil
import pytest

from hermes_cli import main_dashboard


@pytest.mark.platforms("posix")
def test_capture_and_replay_preserve_actual_inline_source_and_spaced_args():
    source = "import time\nvalue = 'literal 012 and \\012'\ntime.sleep(30)\n"
    args = [sys.executable, "-I", "-c", source, "dashboard", "arg with spaces", "9012"]
    child = subprocess.Popen(args, stdin=subprocess.DEVNULL)
    replay = None
    try:
        time.sleep(0.2)  # health: allow HX013 -- let the real child enter its bootstrap before capture
        assert child.poll() is None
        captured = main_dashboard._dashboard_cmdline_for_pid(child.pid)
        assert captured is not None
        assert captured[1:] == args[1:]
        rebuilt = main_dashboard._respawnable_command_for_current_install(captured)
        assert rebuilt == captured
        replay = subprocess.Popen(rebuilt, stdin=subprocess.DEVNULL)
        time.sleep(1.1)  # health: allow HX013 -- exercise the respawn liveness grace, not just successful spawn
        assert replay.poll() is None
    finally:
        for process in [child, replay]:
            if process is not None:
                process.terminate()
                process.wait(timeout=5)


@pytest.mark.platforms("posix")
def test_psutil_preserves_exact_source_and_ps_fallback_refuses_lossy_inline_source():
    source = "import time\nvalue = '012'\ntime.sleep(30)\n"
    exact = [sys.executable, "-I", "-c", source, "dashboard", "arg with spaces"]
    with patch.object(main_dashboard.os.path, "exists", return_value=False), \
         patch("psutil.Process") as process, \
         patch.object(main_dashboard, "_run_probe") as probe:
        process.return_value.cmdline.return_value = exact
        assert main_dashboard._dashboard_cmdline_for_pid(888) == exact
        probe.assert_not_called()
        process.side_effect = psutil.AccessDenied(888)
        probe.return_value = subprocess.CompletedProcess([], 0, "python3 -I -c import time\\012 dashboard")
        assert main_dashboard._dashboard_cmdline_for_pid(888) is None
