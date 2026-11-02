#!/usr/bin/env python3
"""Real private native profile lifecycle and successor observation, without model turns.

Observations invalidate stale evidence; they are not atomic admission or stop grants.
"""
import argparse
import json
from pathlib import Path
import subprocess
import traceback

from e2e_s6_launch_admission import wait_for
from e2e_s6_served_profile import journey


def binding(current):
    receipt = current['launch_admission']
    assert current['state'] == 'ready'
    assert receipt['runtime_profile_binding_verified'] is True
    assert receipt['behavior_accepted'] is False
    result = receipt['runtime_profile_binding']
    assert result['instance_id'] == current['instance_id']
    return result


def run_identity(current):
    receipt = current['launch_admission']
    return (receipt['launch'], receipt['supervisor'], current['instance_id'])


def exercise(home, slot, manager, original, checks, observations):
    from gateway.control_socket import (
        identify_gateway, request_serve_profile_hot, request_unserve_profile,
    )
    from hermes_cli.service_manager import _s6_run

    original_binding = binding(original)
    original_rows = {row['profile']: row for row in original_binding['profiles']}
    # Each operation is issued once. Pending responses are reconciled by read, never replayed.
    removed = request_unserve_profile(home, 'n')
    assert removed and ('unserved' in removed or removed.get('pending') is True)

    def removed_observation():
        current = identify_gateway(home)
        if not current or current.get('state') != 'ready':
            return None
        receipt = current.get('launch_admission') or {}
        observed = receipt.get('runtime_profile_binding') or {}
        return current if {r['profile'] for r in observed.get('profiles', [])} == {'default'} else None

    after_remove = wait_for(removed_observation, 15)
    assert run_identity(after_remove) == run_identity(original)
    assert binding(after_remove) != original_binding
    assert binding(after_remove)['profiles'] == [original_rows['default']]
    checks['real_unserve_excludes_previous_authority_same_owner'] = True
    served = request_serve_profile_hot(home, 'n')
    assert served and ('served' in served or served.get('pending') is True)

    def complete_observation():
        current = identify_gateway(home)
        if not current or current.get('state') != 'ready':
            return None
        receipt = current.get('launch_admission') or {}
        observed = receipt.get('runtime_profile_binding') or {}
        return current if {r['profile'] for r in observed.get('profiles', [])} == {'default', 'n'} else None

    after_add = wait_for(complete_observation, 30)
    new_rows = {row['profile']: row for row in binding(after_add)['profiles']}
    assert run_identity(after_add) == run_identity(original)
    assert new_rows['default'] == original_rows['default']
    assert new_rows['n']['home_identity'] == original_rows['n']['home_identity']
    assert new_rows['n']['authority_epoch'] > original_rows['n']['authority_epoch']
    checks['real_reserve_same_home_new_epoch_invalidates_old_projection'] = True
    # Only the proven private slot is stopped; no gateway CLI or replace path is used.
    _s6_run('s6-svc', '-d', str(slot), check=True)
    wait_for(lambda: not manager.is_running('gateway-default'), 30)
    assert identify_gateway(home) is None
    checks['stopped_original_no_longer_answers'] = True
    _s6_run('s6-svc', '-u', str(slot), check=True)
    successor = wait_for(complete_observation, 90)
    current_binding = binding(successor)
    successor_receipt = successor['launch_admission']
    assert successor['instance_id'] != original['instance_id']
    assert successor_receipt['launch'] != original['launch_admission']['launch']
    assert successor_receipt['supervisor'] == original['launch_admission']['supervisor']
    assert successor_receipt['slot_identity'] == original['launch_admission']['slot_identity']
    successor_rows = {row['profile']: row for row in current_binding['profiles']}
    assert all(successor_rows[name]['home_identity'] == new_rows[name]['home_identity']
               and successor_rows[name]['authority_epoch'] > new_rows[name]['authority_epoch']
               for name in new_rows)
    checks['same_enrollment_supervisor_successor_has_new_run_and_all_epochs'] = True
    observations['qualified_edges'] = ['unserve', 're-serve', 'private-stop', 'private-successor']
    observations['unqualified_edges'] = ['retiring-writer', 'draining-window', 'overflow',
        'atomic-idle-admission', 'root-to-user', 'shared-activation']
    observations['stale_projection_is_not_a_grant'] = True


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
