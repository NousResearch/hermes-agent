#!/usr/bin/env python3
"""Real opt-in native s6 launch journey. Zero model turns; no internal doubles.

Run using an existing compatible Hermes interpreter, with --fixture under a private
short path (the native AF_UNIX socket must fit). Never use an owner's home.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time


def wait_for(predicate, timeout=90):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError("private_native_boundary")


def journey(root, checks):
    root.mkdir(mode=0o700)  # Refuse a preexisting fixture; never adopt or clear it.
    source = Path(__file__).resolve().parents[1]
    home = root / 'p'
    home.mkdir(mode=0o700)
    scan = root / 's'
    scan.mkdir(mode=0o700)
    for key in list(os.environ):
        if key not in {'PATH', 'LANG', 'LC_ALL'}:
            del os.environ[key]
    os.environ['PATH'] = '/command:' + os.environ.get('PATH', '/usr/bin:/bin')
    os.environ.update(S6_KEEP_ENV='1', HOME=str(home), HERMES_HOME=str(home), PYTHONPATH=str(source),
        HERMES_MANAGED_DIR=str(root / 'managed'), HERMES_RUNTIME_DIR=str(root / 'runtime'),
        HERMES_GATEWAY_LOCK_DIR=str(root / 'locks'), XDG_CACHE_HOME=str(root / 'cache'),
        XDG_STATE_HOME=str(root / 'state'), TMPDIR=str(root), PYTHONDONTWRITEBYTECODE='1')
    Path(os.environ['HERMES_MANAGED_DIR']).mkdir(mode=0o700)
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        port = reserved.getsockname()[1]
    (home / 'config.yaml').write_text(json.dumps({'auth': {'adopt_external_logins': False},
        'model': {'provider': 'openai-codex'}, 'auxiliary': {'title_generation': {'enabled': False}},
        'gateway': {'multiplex_profiles': False, 'api_server': {'enabled': True,
            'host': '127.0.0.1', 'port': port, 'key': secrets.token_hex(32)}}}), encoding='utf-8')
    (home / 'SOUL.md').write_text('Private disposable service fixture.\n', encoding='utf-8')
    (home / 'USER.md').write_text('Private disposable owner fixture.\n', encoding='utf-8')
    from hermes_cli.service_manager import S6ServiceManager, _s6_run
    from gateway.control_socket import identify_gateway

    manager = S6ServiceManager(scan)
    scan_process = None
    try:
        with (root / 'service.private.log').open('wb') as output:
            scan_process = subprocess.Popen(['/command/s6-svscan', str(scan)], env=os.environ.copy(),
                stdin=subprocess.DEVNULL, stdout=output, stderr=output)
            wait_for(lambda: (scan / '.s6-svscan/control').exists(), 5)
            try:
                manager.register_profile_gateway('default', launch_admission=True)
            except ValueError:
                checks['opt_in_refuses_implicit_start_before_enrollment'] = True
            else:
                raise AssertionError('unsafe registration accepted')
            manager.register_profile_gateway('default', start_now=False, launch_admission=True,
                extra_env={'HOME': str(home), 'HERMES_HOME': str(home), 'PYTHONPATH': str(source)})
            slot = scan / 'gateway-default'
            wait_for(lambda: _s6_run('s6-svok', str(slot)).returncode == 0, 5)
            checks['native_registration_enrolls_persistently_down_slot'] = not manager.is_running('gateway-default')
            manager.start('gateway-default')
            original = wait_for(lambda: identify_gateway(home))
            bound = original.get('launch_admission')
            assert isinstance(bound, dict) and bound['launch']['pid'] == original['pid']
            assert original['launch_supervisor'] == 's6' and bound['launch_profile'] == 'default'
            assert bound['runtime_profile_binding_verified'] is False and bound['behavior_accepted'] is False
            checks['real_gateway_reports_exact_current_s6_launch_after_drop'] = True
            # A separate foreground gateway cannot acquire this original native lock. No replacement.
            result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'gateway', 'run', '--no-supervise'],
                cwd=source, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, timeout=30, check=False)
            assert result.returncode in {0, 1}
            assert identify_gateway(home)['pid'] == original['pid']
            checks['ordinary_foreground_contender_leaves_original_owner_unchanged'] = True
            manager.stop('gateway-default')
            wait_for(lambda: not manager.is_running('gateway-default'), 30)
            assert not identify_gateway(home)
            # External edit of this private real enrollment must not authorize a successor.
            enrollment = slot / 'launch-admission.json'
            prior = enrollment.read_bytes()
            changed = json.loads(prior)
            changed['profile'] = 'different'
            enrollment.write_text(json.dumps(changed), encoding='utf-8')
            before = (slot / 'run').read_bytes()
            refusal_count = (root / 'service.private.log').read_bytes().count(b'native_launch_admission_refused')
            manager.start('gateway-default')
            wait_for(lambda: (root / 'service.private.log').read_bytes().count(
                b'native_launch_admission_refused') > refusal_count, 10)
            wait_for(lambda: not manager.is_running('gateway-default'), 10)
            assert not identify_gateway(home) and (slot / 'run').read_bytes() == before
            assert enrollment.read_text(encoding='utf-8-sig') == json.dumps(changed)
            checks['changed_enrollment_refuses_successor_without_rewrite'] = True
    finally:
        if scan_process is not None:
            _s6_run('s6-svc', '-d', str(scan / 'gateway-default'))
            wait_for(lambda: not manager.is_running('gateway-default'), 30)
            _s6_run('s6-svscanctl', '-t', str(scan))
            scan_process.wait(timeout=15)
        checks['owned_supervisor_and_gateway_cleanup'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    status = 'FAIL'
    error = None
    try:
        journey(args.fixture.resolve(), checks)
        assert all(checks.values())
        status = 'PASS'
    except (OSError, ValueError, AssertionError, RuntimeError, KeyError, TypeError,
            ImportError, subprocess.SubprocessError) as exc:
        import traceback
        last = traceback.extract_tb(exc.__traceback__)[-1]
        error = f'{type(exc).__name__}:{Path(last.filename).name}:{last.lineno}'
    print(json.dumps({'status': status, 'checks': checks, 'error_type': error,
        'model_turns_requested': 0, 'production_accepted': False}))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
