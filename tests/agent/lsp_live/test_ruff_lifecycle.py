"""Exercise retained LSP ownership with the installed real Ruff language server."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import psutil
import pytest


@pytest.fixture
def ruff_service(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home, workspace = tmp_path / 'home', tmp_path / 'workspace'
    home.mkdir()
    workspace.mkdir()
    subprocess.run(['git', 'init', str(workspace)], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[3]
    file = workspace / 'hermes_time.py'
    shutil.copyfile(root / 'hermes_time.py', file)
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    token = set_hermes_home_override(home)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(home / 'config.yaml', {'lsp': {
        'enabled': True, 'idle_timeout': 0, 'wait_timeout': 10, 'install_strategy': 'off',
        'trusted_workspaces': [str(workspace)],
        'servers': {'repair-ruff': {'command': [sys.executable, '-m', 'ruff', 'server'],
                                   'extensions': ['.py'], 'language_id': 'python', 'root_markers': ['.git']}},
    }})
    from agent.lsp.manager import LSPService

    service = LSPService.create_from_config()
    assert service is not None and service.is_active()
    try:
        yield service, workspace, file
    finally:
        assert service.shutdown() is True, service._get_shutdown_error()
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


def owned_processes(service):
    with service._state_lock:
        parents = [psutil.Process(entry.client._proc.pid) for entry in service._clients.values()]
    return [process for parent in parents for process in [parent, *parent.children(recursive=True)]]


def test_real_ruff_write_check_and_service_shutdown(ruff_service):
    service, workspace, file = ruff_service
    body = file.read_text(encoding='utf-8')
    future = service.start_edit_check(str(file), pre_text=None, post_text=body, owner=uuid4().hex)
    assert future is not None
    result = future.result(timeout=60)
    assert result.diagnostics is not None
    actual = subprocess.run([sys.executable, '-m', 'ruff', 'check', '--output-format', 'json', str(file)],
                            cwd=workspace, check=False, capture_output=True, text=True, encoding='utf-8')
    assert actual.returncode in (0, 1), actual.stderr
    expected = {(item['code'], item['location']['row'] - 1) for item in json.loads(actual.stdout)}
    observed = {(str(item['code']), item['range']['start']['line']) for item in result.diagnostics}
    assert observed == expected
    processes = owned_processes(service)
    assert processes
    assert service.shutdown() is True
    _gone, alive = psutil.wait_procs(processes, timeout=10)
    assert not alive


def test_real_workspace_release_waits_for_its_active_lease(ruff_service):
    service, workspace, file = ruff_service
    lease = service._loop.run(service._acquire_client(str(file)), timeout=30)
    assert lease is not None and lease.client.is_running
    processes = owned_processes(service)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            released = pool.submit(service.release_workspace, str(workspace))
            deadline = time.monotonic() + 10
            while True:
                with service._state_lock:
                    retiring = any(entry.retiring for entry in service._clients.values())
                if retiring:
                    break
                assert time.monotonic() < deadline, 'Native retirement did not begin'
                time.sleep(0.01)
            assert not released.done()
            service._loop._loop.call_soon_threadsafe(lease.release)
            assert released.result(timeout=30) == 1
        _gone, alive = psutil.wait_procs(processes, timeout=10)
        assert not alive
    finally:
        if service._loop._loop is not None:
            service._loop._loop.call_soon_threadsafe(lease.release)
