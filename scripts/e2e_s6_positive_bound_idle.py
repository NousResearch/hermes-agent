#!/usr/bin/env python3
"""Private exact-binding idle pause, s6 once-at-most hold and explicit recovery.

This is disposable process recovery, not independently installed host recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time
import traceback

from e2e_s6_dynamic_authority import binding, run_identity
from e2e_s6_launch_admission import wait_for
from e2e_s6_served_profile import journey


def preserved_files(home, slot):
    paths = [p / name for p in [home, home / 'profiles/n']
             for name in ['SOUL.md', 'USER.md', 'config.yaml']]
    paths += [slot / name for name in ['run', 'finish', 'launch-admission.json', 'down']]
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), stat.S_IMODE(p.stat().st_mode))
            for p in paths}


def supervisor_state(slot):
    from hermes_cli.service_manager import _s6_run

    result = _s6_run('s6-svstat', '-o', 'up,wantedup,pid', str(slot), check=True)
    fields = result.stdout.split()
    assert len(fields) == 3 and fields[0] in {'true', 'false'} and fields[1] in {'true', 'false'}
    return fields


def original_gone(launch):
    from hermes_cli.service_manager_admission import _process

    try:
        return _process(launch['pid']) != launch
    except (OSError, ValueError):
        return True


def exercise(home, slot, manager, original, checks, observations):
    from gateway.control_socket import identify_gateway, query_gateway_control
    from hermes_cli.service_manager import _s6_run

    assert original['launch_binding_idle_protocol'] == 1
    prior_files = preserved_files(home, slot)
    # The existing persistent down file remains untouched. -O holds respawn but
    # neither stops the current child nor rewrites its enrolled launcher.
    _s6_run('s6-svc', '-O', str(slot), check=True)
    wait_for(lambda: supervisor_state(slot) == ['true', 'false', str(original['pid'])], 5)
    held = identify_gateway(home)
    assert run_identity(held) == run_identity(original) and binding(held) == binding(original)
    assert held['launch_admission'] == original['launch_admission']
    checks['private_once_at_most_retains_exact_original_and_binding'] = True
    answer = query_gateway_control(home, 'pause-if-idle-bound', params={
        'expected_binding': held['launch_admission'], 'target': 'private-positive-e2e'}, timeout=8)
    assert answer and answer.get('pid') == original['pid']
    assert answer.get('binding_protocol') == 1 and answer.get('binding_status') == 'matched'
    assert answer.get('pausing') is True and answer.get('already_stopping') is False
    checks['exact_current_binding_admitted_by_original_native_idle_decision'] = True
    wait_for(lambda: supervisor_state(slot)[:2] == ['false', 'false'], 30)
    assert original_gone(original['launch_admission']['launch'])
    assert identify_gateway(home) is None
    # Observe a finite stability interval, not a fabricated durable-hold claim.
    until = time.monotonic() + 2
    while time.monotonic() < until:
        assert supervisor_state(slot)[:2] == ['false', 'false']
        time.sleep(0.1)
    checks['original_exited_and_private_supervisor_kept_slot_down'] = True
    assert preserved_files(home, slot) == prior_files
    checks['pause_preserved_profile_launcher_enrollment_bytes_and_modes'] = True
    _s6_run('s6-svc', '-u', str(slot), check=True)

    def recovered():
        current = identify_gateway(home)
        return current if current and current.get('state') == 'ready' else None

    successor = wait_for(recovered, 90)
    after = successor['launch_admission']
    before = original['launch_admission']
    assert successor['instance_id'] != original['instance_id'] and after['launch'] != before['launch']
    assert after['supervisor'] == before['supervisor'] and after['slot_identity'] == before['slot_identity']
    previous = {r['profile']: r for r in binding(original)['profiles']}
    current = {r['profile']: r for r in binding(successor)['profiles']}
    assert set(current) == set(previous) == {'default', 'n'}
    assert all(current[k]['home_identity'] == previous[k]['home_identity']
               and current[k]['authority_epoch'] > previous[k]['authority_epoch'] for k in current)
    assert preserved_files(home, slot) == prior_files
    checks['private_recovery_new_run_new_epochs_preserved_identity_and_enrollment'] = True
    observations['matching_pause_requests'] = 1
    observations['hold_scope'] = 'same private s6 supervisor, observed down for two seconds'
    observations['recovery_scope'] = 'explicit private slot up; NOT independent host recovery or method activation'
    observations['unqualified'] = ['matched-busy', 'matched-stopping', 'restart-wait-ticket',
        'supervisor-loss-hold', 'installed-independent-recovery', 'shared-activation']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    checks, observations = {}, {}
    error = None
    try:
        journey(args.fixture.resolve(), checks, observations, exercise=exercise)
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
