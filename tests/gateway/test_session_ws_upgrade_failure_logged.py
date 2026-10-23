"""An owner that fails a session WebSocket upgrade names the cause in gateway.log (#139922).

The reporter's owner could not import its session WebSocket handler, so every cron agent turn and
managed-worker dial got ``InvalidStatus: ... HTTP 500`` while gateway.log stayed empty: uvicorn's own
error logger is not a gateway logger. This boots a real owner (``python -m gateway.run``) in a
throwaway home with that handler broken, then dials it the way an external cron worker and a managed
worker do.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest

from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until

# The owner, with ``tui_gateway.ws.handle_ws`` failing the way the unimportable onboarding module did.
_BROKEN_OWNER = textwrap.dedent('''
    import runpy, sys
    import tui_gateway.ws as ws
    del ws.handle_ws
    def __getattr__(name):
        if name == 'handle_ws':
            raise NameError("name 'Path' is not defined")
        raise AttributeError(name)
    ws.__getattr__ = __getattr__
    sys.argv = ['gateway.run']
    runpy.run_module('gateway.run', run_name='__main__', alter_sys=True)
''')

# An external cron worker's agent turn (no owner in-process), then a managed worker's persistence call.
_DIALS = textwrap.dedent('''
    import json
    from cron.scheduler_authority import run_canonical_job
    from agent.runtime_session_store import WorkerRPC
    from hermes_constants import get_hermes_home
    success, _output, _answer, error = run_canonical_job({'id': 'cron4job', 'name': 'cron-4'},
                                                         execution_id='cron4-exec')
    try:
        WorkerRPC(get_hermes_home())('worker.heartbeat')
        worker = None
    except Exception as exc:
        worker = f'{type(exc).__name__}: {exc}'
    print(json.dumps({'cron': [success, error], 'worker': worker}))
''')


@pytest.mark.platforms("linux")
def test_a_failed_session_upgrade_is_logged_with_its_cause(tmp_path):
    home = tmp_path / 'state'
    home.mkdir(mode=0o700)
    user = tmp_path / 'user'
    user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': 'http://127.0.0.1:9/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1', OPENAI_API_KEY='loopback-only')
    with (tmp_path / 'owner.out').open('w+') as out:
        owner = subprocess.Popen([sys.executable, '-c', _BROKEN_OWNER], cwd=root, env=env,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        try:
            descriptor = {}

            def ready_or_exited():
                if owner.poll() is not None:
                    return 'exited'  # fail now, not after the deadline
                try:
                    descriptor.update(control(home, 'identify'))
                except (OSError, ValueError):
                    return None
                return descriptor.get('state') == 'ready' and 'ready'

            try:
                outcome = wait_until(ready_or_exited, timeout=35, what='the owner to be ready')
            except AssertionError as exc:
                outcome = str(exc)
            out.flush()
            out.seek(0)
            assert outcome == 'ready', (outcome, descriptor, out.read())
            dialed = subprocess.run([sys.executable, '-c', _DIALS], cwd=root, env=env, capture_output=True,
                                    text=True, timeout=60, check=False)
            assert dialed.returncode == 0, dialed.stderr
            result = json.loads(dialed.stdout.strip().splitlines()[-1])
            # The failure itself is unchanged: the owner still refuses the upgrade with HTTP 500.
            assert result['cron'][0] is False and 'HTTP 500' in result['cron'][1], result
            assert result['worker'] is not None and 'HTTP 500' in result['worker'], result
        finally:
            if owner.poll() is None:
                owner.send_signal(signal.SIGINT)
                try:
                    owner.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    owner.kill()
                    owner.wait(timeout=5)
    log = (home / 'logs' / 'gateway.log').read_text(encoding='utf-8')
    refused = [line for line in log.splitlines() if 'session WebSocket upgrade' in line]
    # One line per refused dial (the cron turn's and the worker's), each naming the cause.
    assert len(refused) == 2, log
    assert all("NameError: name 'Path' is not defined" in line for line in refused), refused


@pytest.mark.asyncio
@pytest.mark.parametrize('refused_first', [False, True])
async def test_the_log_line_names_the_status_the_dialer_got(caplog, refused_first):
    """Uvicorn answers 500 only when the handler raised before sending anything; a handler that had
    already closed the upgrade got its own refusal (403) out, so the line must not claim a 500."""
    from gateway.run_api import GatewayRuntimeAPI

    api = GatewayRuntimeAPI(None, None, None)

    async def fail(scope, receive, send):
        if refused_first:
            await send({'type': 'websocket.close', 'code': 1008})
        raise RuntimeError('late failure')

    api._serve = fail
    sent = []

    async def send(message):
        sent.append(message)

    scope = {'type': 'websocket', 'path': '/api/ws', 'client': ('127.0.0.1', 5)}
    with caplog.at_level('ERROR', logger='gateway.run_api'), pytest.raises(RuntimeError, match='late failure'):
        await api(scope, None, send)
    [line] = [record.getMessage() for record in caplog.records if 'session WebSocket upgrade' in record.getMessage()]
    assert 'RuntimeError: late failure' in line
    assert ('HTTP 500' in line) is not refused_first, line
