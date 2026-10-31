#!/usr/bin/env python3
"""Real isolated API admission with an unavailable subscription; no model or mocks.

Run with a prepared checkout interpreter. Linux's timeout owns a hard bound even
if this foreground driver is interrupted. No installed profile or token is used.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid


def request(origin, key, method, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(origin + path, data=payload, method=method,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Do not inherit host proxies or follow a listener's redirect to another host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=5) as response:
        raw = response.read(262145)
        assert len(raw) <= 262144, 'API response exceeded the fixture bound'
        return json.loads(raw)


async def observe_completion(session=None, admission=None):
    # Existing authenticated native-client transport, never implicit owner start.
    observed = {'status': 'failed', 'phase': 'imports'}
    try:
        from hermes_cli.gateway_client import connect_gateway
        from hermes_cli.gateway_runtime import discover_gateway_endpoint
        observed['phase'] = 'discovery'
        home = await asyncio.to_thread(Path(os.environ['HERMES_HOME']).resolve)
        found = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=2)
        observed['discovery_state'] = found.state if found.state in {
            'ready', 'starting', 'absent', 'inaccessible', 'incompatible',
            'draining', 'conflict'} else 'unclassified'
        observed['discovery_reason'] = found.reason_code if found.reason_code in {
            'runtime_protocol', 'profile_mismatch', 'control_timeout',
            'session_authority_unavailable', 'invalid_runtime_identity',
            'invalid_api_origin', 'invalid_runtime_state', 'invalid_supervisor',
            'unsafe_control_path', 'control_unavailable'} else 'unclassified'
        assert found.state == 'ready' and found.endpoint is not None
        observed['profile_matches'] = await asyncio.to_thread(Path(found.endpoint.profile_id).resolve) == home
        assert observed['profile_matches']
        if session is None:
            observed['status'] = 'passed'
            return observed
        observed['phase'] = 'connection'
        async with connect_gateway(endpoint=found.endpoint, max_size=262144) as client:
            observed['phase'] = 'replay'
            replay = await client.rpc('session.events.since', session_id=session,
                                      last_sequence=0, _timeout=5)
            # An unbound cursor deliberately answers snapshot_required with its
            # epoch; read that same bounded ring again using the returned epoch.
            replay = await client.rpc('session.events.since', session_id=session,
                last_sequence=0, replay_epoch=replay['replay_epoch'], _timeout=5)
            observed['phase'] = 'completion-selection'
            matches = [frame['payload'] for frame in replay['events']
                       if frame['type'] == 'message.complete'
                       and frame['payload'].get('admission_id') == admission]
            observed['completion_count'] = len(matches)
            assert len(matches) == 1
            observed['phase'] = 'projection'
            assert matches[0]['error_surface']['failure_context'] == {
                'stage': 'runtime_resolution', 'source': 'typed_exception_chain',
                'code': 'codex_auth_missing'}
            observed['status'] = 'passed'
    except Exception as exc:  # health: allow BLE001 -- Return only fixed diagnostic stages and allowlisted reasons, never client text or credentials.
        name = type(exc).__name__
        observed['error_type'] = name if name in {
            'AssertionError', 'KeyError', 'ImportError', 'ModuleNotFoundError',
            'GatewayRPCError', 'GatewayClientError', 'TimeoutError', 'OSError',
            'InvalidStatus', 'InvalidHandshake', 'ValueError', 'TypeError',
            'PermissionError', 'ConnectionRefusedError'} else 'unclassified'
        reason = str(exc)
        observed['reason'] = reason if reason in {
            'permission_denied', 'not_found', 'profile_mismatch', 'invalid_params',
            'request_failed', 'runtime_draining'} else 'unclassified'
    return observed


def run_observer(home, env, log, *arguments):
    with tempfile.TemporaryFile(dir=home.parent) as output:
        subprocess.run([sys.executable, '-m', 'scripts.e2e_resolution_failure', *arguments],
            env=env, cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
            stdout=output, stderr=log, timeout=20, check=False)
        output.seek(0)
        raw = output.read(8193)
        assert len(raw) <= 8192, 'Native observer response exceeded bound'
        return json.loads(raw)


def journey(home, env, log, receipt):
    checkout_env = Path(__file__).resolve().parents[1] / ".env"
    assert not checkout_env.exists() and not checkout_env.is_symlink(), "Checkout dotenv refuses isolation"
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    key = secrets.token_hex(32)
    config = {'model': {'provider': 'openai-codex'},
              'auth': {'adopt_external_logins': False},
              'auxiliary': {'title_generation': {'enabled': False}},
              'gateway': {'multiplex_profiles': False, 'api_server': {
                  'enabled': True, 'host': '127.0.0.1', 'port': port, 'key': key}}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    os.chmod(home / 'config.yaml', 0o600)
    argv = [shutil.which('timeout'), '--signal=TERM', '--kill-after=15s', '210s',
            sys.executable, '-m', 'gateway.run']
    process = subprocess.Popen(argv, env=env, cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    checks = receipt['checks']
    receipt['phase'] = 'readiness'
    try:
        origin = f'http://127.0.0.1:{port}'
        deadline = time.monotonic() + 90
        tick = threading.Event()
        while True:
            assert process.poll() is None, 'Fixture gateway exited before readiness'
            try:
                request(origin, key, 'GET', '/health')
                break
            except (OSError, urllib.error.URLError):
                assert time.monotonic() < deadline, 'Fixture gateway readiness deadline'
                tick.wait(0.2)
        checks['isolated_api_ready'] = True
        receipt['phase'] = 'canonical-readiness'
        deadline = time.monotonic() + 60
        while True:
            receipt['native_readiness'] = run_observer(home, env, log, '--probe')
            if receipt['native_readiness']['status'] == 'passed':
                break
            assert receipt['native_readiness'].get('discovery_state') in {'starting', 'absent'}, 'Canonical fixture discovery refused'
            assert process.poll() is None and time.monotonic() < deadline, 'Canonical fixture readiness deadline'
            tick.wait(0.5)
        checks['canonical_native_ready'] = True
        receipt['phase'] = 'admission'
        session = 'resolution-e2e-' + uuid.uuid4().hex
        request(origin, key, 'POST', '/api/sessions', {'id': session})
        admitted = request(origin, key, 'POST', '/v1/runs', {
            'session_id': session, 'provider': 'openai-codex',
            'input': 'Bounded unavailable-subscription diagnostic. Do not run tools or fall back.'})
        run_id = admitted['run_id']
        deadline = time.monotonic() + 30
        while True:
            state = request(origin, key, 'GET', '/v1/runs/' + run_id)
            if state['status'] in {'completed', 'failed', 'cancelled'}:
                break
            assert time.monotonic() < deadline, 'Fixture admission terminal deadline'
            tick.wait(0.2)
        assert state['status'] == 'failed', 'Unavailable subscription must fail honestly'
        checks['unavailable_subscription_failed'] = True
        receipt['phase'] = 'durable-result'
        # Observe the real HTTP admission's durable effect, never fabricate a row
        # or invoke a result helper with a synthetic exception.
        with sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('SELECT admission_id,status,outcome FROM session_admissions '
                "WHERE principal_id='api' AND request_id=? AND target_session_id=?",
                (run_id, session)).fetchall()
            assert len(rows) == 1 and rows[0][1:] == ('terminal', 'failed')
            encoded = db.execute('SELECT value FROM state_meta WHERE key=?',
                ('gateway.admission.result.v1.' + rows[0][0],)).fetchone()
            assert encoded and len(encoded[0].encode()) <= 262144
            result = json.loads(encoded[0])['result']
        assert result.get('api_calls') == 0, 'Refusal must precede model calls'
        assert result.get('failure_context') == {
            'stage': 'runtime_resolution', 'source': 'typed_exception_chain',
            'code': 'codex_auth_missing'}, 'Typed creation-time context must survive settlement'
        checks['typed_context_retained'] = True
        checks['no_credentials_or_model_spend'] = True
        receipt['phase'] = 'canonical-completion'
        receipt['observer'] = run_observer(home, env, log, '--completion', session, rows[0][0])
        assert receipt['observer']['status'] == 'passed', 'Canonical completion context must reach native client'
        checks['canonical_completion_context'] = True
        return checks
    finally:
        # SIGINT is native foreground Ctrl+C. Signal the exact owned timeout
        # handle; its forwarding never enters a host service-manager stop path.
        receipt['phase_before_cleanup'] = receipt['phase']
        receipt['phase'] = 'cleanup'
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=240)
        receipt['fixture_exit_code'] = process.returncode
        assert process.returncode == 0, 'Owned foreground fixture cleanup failed'
        receipt['cleanup'] = 'passed'


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--probe':
        observed = asyncio.run(observe_completion())
        print(json.dumps(observed, sort_keys=True))
        return 0 if observed['status'] == 'passed' else 1
    if len(sys.argv) == 4 and sys.argv[1] == '--completion':
        observed = asyncio.run(observe_completion(sys.argv[2], sys.argv[3]))
        print(json.dumps(observed, sort_keys=True))
        return 0 if observed['status'] == 'passed' else 1
    if sys.platform != 'linux' or shutil.which('timeout') is None:
        raise SystemExit('This real service journey requires Linux timeout; other hosts unverified.')
    receipt = {'journey': 'native-resolution-failure', 'checks': {}, 'model_calls_expected': 0}
    try:
        with tempfile.TemporaryDirectory(prefix='hermes-resolution-e2e-') as directory:
            root = Path(directory)
            home = root / 'profile'
            home.mkdir(mode=0o700)
            managed = root / 'managed'
            managed.mkdir(mode=0o700)
            env = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'SYSTEMROOT') if key in os.environ}
            env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(root / 'tools'),
                       HERMES_GATEWAY_LOCK_DIR=str(root / 'gateway-locks'),
                       HERMES_MANAGED_DIR=str(managed),
                       PYTHONUNBUFFERED='1')
            with (root / 'gateway.log').open('wb') as log:
                os.chmod(root / 'gateway.log', 0o600)
                receipt['checks'] = journey(home, env, log, receipt)
        receipt.update(status='passed', cleanup='passed')
    except Exception as exc:  # health: allow BLE001 -- The public receipt must never include a private listener key, body or log traceback.
        # Never copy listener keys, bodies, native output or exception text.
        receipt.update(status='failed', error_type=type(exc).__name__)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
