"""Service launcher children must survive desktop startup's orphan sweep."""
import signal
import subprocess
import sys
from unittest.mock import patch

import psutil
import pytest

from hermes_cli import gateway


@pytest.mark.platforms("macos")
def test_service_child_survives_reaper_but_unmanaged_process_is_selected(monkeypatch):
    # Real process ancestry, but interception at the signal boundary: no gateway
    # is launched or signalled, and no owner profile is read.
    wrapper = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE); "
         "print(p.pid,flush=True); sys.stdin.read(); p.terminate(); p.wait()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert wrapper.stdout is not None
        child = int(wrapper.stdout.readline())
        assert psutil.Process(child).ppid() == wrapper.pid
        unmanaged = 99999999
        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway, "_get_service_pids", lambda all_profiles=False: {wrapper.pid})
        monkeypatch.setattr("gateway.status._read_pid_record", lambda: None)
        monkeypatch.setattr("gateway.status._read_gateway_lock_record", lambda: None)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda **kw: None)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda exclude_pids=None: [
            pid for pid in (wrapper.pid, child, unmanaged) if pid not in (exclude_pids or set())])
        monkeypatch.setattr("gateway.status.write_planned_stop_marker", lambda pid: None)
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        with patch.object(gateway.os, "kill") as kill:
            assert gateway._reap_unsupervised_gateway_orphans() is True
        assert kill.call_args_list == [((unmanaged, signal.SIGTERM),)]
        assert psutil.Process(child).is_running()
    finally:
        wrapper.communicate(timeout=10)
        assert wrapper.returncode == 0
