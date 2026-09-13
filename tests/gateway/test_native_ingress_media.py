"""Native attachments are committed bytes, not mutable adapter paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def _peer(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
               PYTHONPATH=str(repo), PYTHONUNBUFFERED='1')
    return [sys.executable, str(Path(__file__).parent / 'fixtures' / 'native_ingress_media_peer.py'), mode], env, repo, state


@pytest.mark.linux_only
def test_accepted_media_survives_mutation_cleanup_and_owner_kill(tmp_path):
    command, env, repo, state = _peer(tmp_path, 'capture')
    with (state / 'owner.log').open('w+') as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while not (state / 'accepted.json').exists():
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
    for calls in (1, 0):
        command[-1] = 'recover'
        result = subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads((state / 'recovered.json').read_text())
        assert receipt['model_calls'] == calls, receipt
        print(json.dumps(receipt))


def test_native_context_never_upgrades_metadata_or_transport_trust(tmp_path):
    command, env, repo, state = _peer(tmp_path, 'guards')
    result = subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    print((state / 'guards.json').read_text())
