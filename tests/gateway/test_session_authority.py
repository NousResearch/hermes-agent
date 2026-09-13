"""Real messaging ingress and authenticated WS must share one warm agent."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_messaging_then_authenticated_ws_reuses_live_agent(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    # A fresh interpreter keeps import-time profile caches and background workers
    # away from both the user's state and pytest's process-global fixtures.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
               PYTHONPATH=str(repo), PYTHONUNBUFFERED="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "fixtures" / "shared_authority_peer.py")],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    receipt = json.loads((state / "receipt.json").read_text())
    print(json.dumps(receipt, indent=2))
    assert receipt["messaging_completed"], receipt
    assert receipt["unauthenticated_rejected"], receipt
    assert receipt["ws_completed"], receipt
    assert receipt["same_agent"], receipt


@pytest.mark.linux_only
def test_native_queue_survives_owner_kill_without_replaying_unknown(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
               PYTHONPATH=str(repo), PYTHONUNBUFFERED='1', AUTHORITY_PROBE_MODE='crash')
    command = [sys.executable, str(Path(__file__).parent / 'fixtures' / 'shared_authority_peer.py')]
    with (state / 'owner.log').open('w+') as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while not (state / 'crash-ready.json').exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    pytest.fail(log.read())
                time.sleep(0.05)
            child.kill()
            assert child.wait(timeout=10) < 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    env['AUTHORITY_PROBE_MODE'] = 'recover'
    for expected_calls in (1, 0):
        result = subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=70)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
        receipt = json.loads((state / 'recovery-receipt.json').read_text())
        assert receipt['model_calls'] == expected_calls, receipt
        print(json.dumps(receipt, indent=2))
