#!/usr/bin/env python3
"""Private real bound-idle refusals; no matching pause, shared target or model turn."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import traceback

from e2e_s6_dynamic_authority import binding, run_identity
from e2e_s6_launch_admission import wait_for
from e2e_s6_served_profile import journey


def refused(home, expected, original):
    from gateway.control_socket import identify_gateway, query_gateway_control

    result = query_gateway_control(home, 'pause-if-idle-bound',
        params={'expected_binding': expected}, timeout=8)
    assert result and result.get('pausing') is False
    assert result.get('binding_refused') is True
    assert result.get('binding_status') == 'mismatch' and result.get('binding_protocol') == 1
    current = identify_gateway(home)
    assert run_identity(current) == run_identity(original)
    assert binding(current) == binding(original)


def exercise(home, slot, manager, original, checks, observations):
    from gateway.control_socket import identify_gateway, request_unserve_profile, request_serve_profile_hot
    from hermes_cli.service_manager import _s6_run

    assert original['launch_binding_idle_protocol'] == 1
    expected = original['launch_admission']
    refused(home, None, original)
    checks['missing_evidence_refused_original_remains_ready'] = True
    wrong = copy.deepcopy(expected)
    wrong['runtime_profile_binding']['profiles'] = [
        r for r in wrong['runtime_profile_binding']['profiles'] if r['profile'] == 'default']
    refused(home, wrong, original)
    checks['partial_served_set_refused_original_remains_ready'] = True
    wrong = {**expected, 'launch_profile': 'other'}
    refused(home, wrong, original)
    checks['wrong_profile_refused_original_remains_ready'] = True
    wrong = copy.deepcopy(expected)
    wrong['runtime_profile_binding']['profiles'][0]['authority_epoch'] = True
    refused(home, wrong, original)
    checks['boolean_epoch_not_accepted_as_integer'] = True
    answer = request_unserve_profile(home, 'n')
    assert answer and (answer.get('unserved') == 'n' or answer.get('pending') is True)

    def ready_names(names):
        current = identify_gateway(home)
        if not current or current.get('state') != 'ready':
            return None
        receipt = current.get('launch_admission') or {}
        value = receipt.get('runtime_profile_binding') or {}
        return current if {r['profile'] for r in value.get('profiles', [])} == names else None

    removed = wait_for(lambda: ready_names({'default'}), 15)
    refused(home, expected, removed)
    checks['original_set_after_real_unserve_refused'] = True
    answer = request_serve_profile_hot(home, 'n')
    assert answer and (answer.get('served') == 'n' or answer.get('pending') is True)
    added = wait_for(lambda: ready_names({'default', 'n'}), 30)
    assert binding(added) != binding(original)
    refused(home, expected, added)
    checks['old_epoch_after_real_reserve_refused'] = True
    _s6_run('s6-svc', '-d', str(slot), check=True)
    wait_for(lambda: not manager.is_running('gateway-default'), 30)
    _s6_run('s6-svc', '-u', str(slot), check=True)
    successor = wait_for(lambda: ready_names({'default', 'n'}), 90)
    assert successor['instance_id'] != original['instance_id']
    refused(home, expected, successor)
    checks['old_original_receipt_refused_by_real_successor'] = True
    observations['positive_matching_pause'] = 'NOT REQUESTED; recovery integration remains unqualified'
    observations['refusal_scope'] = 'Original native bound-idle decision, seven negative requests, no replay'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
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
