"""A broken host transport is not evidence that its worker process has exited."""
import subprocess
import sys

import pytest

from hermes_cli.plugin_host import PluginHost
from hermes_cli.plugins import PluginManager


@pytest.mark.platforms("any")
def test_replacement_reaps_previous_live_process_before_starting_new_host(tmp_path):
    host = PluginHost(PluginManager(scope_key=str(tmp_path)))
    previous = subprocess.Popen([sys.executable, "-c", "from threading import Event; Event().wait()"])
    # Inject transport loss, but keep a real live process: replacing its handle
    # without waiting for exit would let old and new background workers overlap.
    host._proc = previous
    host._channel = None
    try:
        host.ensure_started()
        assert previous.poll() is not None
        assert host.alive
        assert host._proc is not previous
    finally:
        host.shutdown()
        if previous.poll() is None:
            previous.kill()
        previous.wait(timeout=5)


@pytest.mark.platforms("any")
def test_failed_reap_stays_fenced_and_later_start_can_recover(tmp_path, monkeypatch):
    host = PluginHost(PluginManager(scope_key=str(tmp_path)))
    previous = subprocess.Popen([sys.executable, "-c", "from threading import Event; Event().wait()"])
    host._proc = previous
    real_wait = previous.wait
    attempts = 0

    def fail_first_wait(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.TimeoutExpired(previous.args, 3)
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(previous, "wait", fail_first_wait)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            host.ensure_started()
        assert host._proc is previous
        assert not host.alive
        assert host._stopping
        host.ensure_started()
        assert host.alive
        assert not host._stopping
        assert previous.poll() is not None
    finally:
        host.shutdown()
        if previous.poll() is None:
            previous.kill()
        real_wait(timeout=5)
