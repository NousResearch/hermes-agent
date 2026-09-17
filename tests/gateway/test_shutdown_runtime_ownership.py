"""Exit-state persistence is not the end of canonical writer lifetime."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize('timed_out', [False, True])
def test_exit_state_keeps_runtime_reserved_until_final_cleanup(tmp_path, monkeypatch, timed_out):
    import gateway.run as run
    from gateway.run_shutdown import GatewayShutdownMixin
    from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock, write_pid_file, remove_pid_file

    home = tmp_path / 'runtime'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    states = []
    runner = SimpleNamespace(
        _restart_requested=False, _exit_reason=None,
        _update_runtime_status=lambda *args: states.append(args),
    )
    ctx = SimpleNamespace(timed_out=timed_out, active_agents={}, elapsed=lambda: 0)
    probe = '''
from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
claimed = acquire_gateway_runtime_lock()
print('CLAIMED', claimed)
if claimed:
    release_gateway_runtime_lock()
'''

    def contender():
        result = subprocess.run(
            [sys.executable, '-c', probe], cwd=os.getcwd(), env=os.environ.copy(),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    assert acquire_gateway_runtime_lock()
    write_pid_file()
    try:
        assert contender() == 'CLAIMED False'
        GatewayShutdownMixin._stop_persist_exit_state(runner, ctx)
        assert contender() == 'CLAIMED False', 'exit-state phase released ownership before writer drain'
        assert (home / 'gateway.pid').exists()
        assert (home / '.clean_shutdown').exists() is not timed_out
        assert states == [('stopped', None)]
    finally:
        remove_pid_file()
        release_gateway_runtime_lock()
    assert contender() == 'CLAIMED True'
