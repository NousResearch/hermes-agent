#!/usr/bin/env python3
"""Private real named-profile authority binding; no model turns or shared targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from e2e_s6_launch_admission import wait_for


def journey(root, checks, observations, *, exercise=None, prepare_enrollment=None):
    root.mkdir(mode=0o700)
    source = Path(__file__).resolve().parents[1]
    default = root / 'p'
    named = default / 'profiles/n'
    other = default / 'profiles/other'
    for home in [default, named]:
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / 'config.yaml').write_text(json.dumps({'auth': {'adopt_external_logins': False},
            'model': {'provider': 'openai-codex'}, 'gateway': {'multiplex_profiles': True},
            'auxiliary': {'title_generation': {'enabled': False}}}), encoding='utf-8')
        (home / 'SOUL.md').write_text('Private disposable identity.\n', encoding='utf-8')
        (home / 'USER.md').write_text('Private disposable owner.\n', encoding='utf-8')
    other.mkdir(mode=0o700)  # A real markerless directory is not an eligible native profile.
    scan = root / 's'
    scan.mkdir(mode=0o700)
    (root / 'r').mkdir(mode=0o700)
    # Linux open-directory aliases keep native AF_UNIX paths short. Every physical
    # socket/log/state target still resolves into this private leaf, not shared /tmp.
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    alias = Path(f'/proc/{os.getpid()}/fd/{root_fd}/r')
    for key in list(os.environ):
        if key not in {'PATH', 'LANG', 'LC_ALL'}:
            del os.environ[key]
    os.environ['PATH'] = '/command:' + os.environ.get('PATH', '/usr/bin:/bin')
    os.environ.update(S6_KEEP_ENV='1', HOME=str(root), HERMES_HOME=str(default),
        PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root),
        XDG_RUNTIME_DIR=str(alias), HERMES_RUNTIME_DIR=str(root / 'runtime'),
        HERMES_MANAGED_DIR=str(root / 'managed'), HERMES_GATEWAY_LOCK_DIR=str(root / 'locks'),
        XDG_CACHE_HOME=str(root / 'cache'), XDG_STATE_HOME=str(root / 'state'))
    (root / 'managed').mkdir(mode=0o700)
    from gateway.control_socket import identify_gateway, resolve_server_socket_path
    from hermes_cli.service_manager import S6ServiceManager, _s6_run

    manager = S6ServiceManager(scan)
    slot = scan / 'gateway-default'
    socket_path, _ = resolve_server_socket_path(default)
    logger = default / 'logs/gateways/default'
    assert alias.resolve() == root / 'r'
    assert len(str(socket_path).encode()) <= 100
    assert all(path.resolve().is_relative_to(root) for path in [slot, logger, socket_path])
    assert scan != Path('/run/service') and slot != Path('/run/service/gateway-default')
    assert os.environ['S6_KEEP_ENV'] == '1'
    assert Path(sys.executable).resolve().is_file()
    assert Path('/command/s6-svscan').resolve().is_file()
    assert all((home / 'config.yaml').resolve().is_relative_to(root) for home in [default, named])
    observations['preflight'] = {'service_target_private': True, 'logger_target_private': True,
        'socket_physical_target_private': True, 'container_environment_import_disabled': True,
        'global_service_cli_not_used': True}
    other_before = list(other.iterdir())
    process = None
    try:
        with (root / 'service.private.log').open('wb') as output:
            process = subprocess.Popen(['/command/s6-svscan', str(scan)],
                env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=output, stderr=output)
            wait_for(lambda: (scan / '.s6-svscan/control').exists(), 5)
            guard = prepare_enrollment(default) if prepare_enrollment is not None else None
            manager.register_profile_gateway('default', start_now=False, launch_admission=True,
                launch_guard=guard,
                extra_env={'HOME': str(root), 'HERMES_HOME': str(default), 'PYTHONPATH': str(source)})
            wait_for(lambda: _s6_run('s6-svok', str(slot)).returncode == 0, 5)
            _s6_run('s6-svc', '-u', str(slot), check=True)
            def ready():
                current = identify_gateway(default)
                return current if current and current.get('state') == 'ready' else None

            first = wait_for(ready, 90)
            enrolled = first.get('launch_admission') or {}
            assert first['profile'] == enrolled['launch_profile'] == 'default'
            assert Path(first['hermes_home']).resolve() == default
            assert first['pid'] == enrolled['launch']['pid']
            checks['original_multiplexer_current_launch_matches_exact_home'] = True
            observations['runtime_binding_observed'] = enrolled.get('runtime_profile_binding_verified')
            assert enrolled.get('runtime_profile_binding_verified') is True
            binding = enrolled['runtime_profile_binding']
            assert binding['scope'] == 'observed_served_profiles'
            assert binding['instance_id'] == first['instance_id']
            rows = {row['profile']: row for row in binding['profiles']}
            assert set(rows) == {'default', 'n'}
            assert rows['n']['home_identity'] == [named.stat().st_dev, named.stat().st_ino]
            assert rows['default']['home_identity'] == [default.stat().st_dev, default.stat().st_ino]
            assert rows['default']['authority_epoch'] == first['authority_epoch']
            assert type(rows['n']['authority_epoch']) is int and rows['n']['authority_epoch'] > 0
            assert enrolled['behavior_accepted'] is False and first['launch_supervisor'] == 's6'
            checks['live_original_authority_epoch_and_runtime_instance_bound'] = True
            assert identify_gateway(other) is None and list(other.iterdir()) == other_before
            checks['markerless_home_not_promoted_or_mutated'] = True
            again = identify_gateway(default)
            assert (again['pid'], again['start_time'], again['instance_id']) == (
                first['pid'], first['start_time'], first['instance_id'])
            assert again['launch_admission']['runtime_profile_binding'] == binding
            checks['repeat_original_read_preserves_run_and_exact_binding'] = True
            if exercise is not None:
                exercise(default, slot, manager, first, checks, observations)
    finally:
        try:
            if process is not None:
                _s6_run('s6-svc', '-d', str(slot))
                wait_for(lambda: not manager.is_running('gateway-default'), 30)
                _s6_run('s6-svscanctl', '-t', str(scan))
                process.wait(timeout=15)
                checks['owned_gateway_and_supervisor_reaped'] = True
        finally:
            os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    checks, observations = {}, {}
    error = None
    try:
        journey(args.fixture.resolve(), checks, observations)
        assert checks and all(checks.values())
    except (OSError, ValueError, AssertionError, RuntimeError, KeyError, TypeError,
            ImportError, subprocess.SubprocessError) as exc:
        last = traceback.extract_tb(exc.__traceback__)[-1]
        error = f'{type(exc).__name__}:{Path(last.filename).name}:{last.lineno}'
    print(json.dumps({'status': 'PASS' if error is None else 'FAIL', 'checks': checks,
        'observations': observations, 'error': error, 'model_turns_requested': 0,
        'behavior_accepted': False}))
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
