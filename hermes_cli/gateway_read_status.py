"""Forward authenticated dashboard read-status writes to the exact live owner."""
import json
import os
import time

from hermes_state_runtime import RuntimeStoreError


def _exchange(home, request, timeout):
    from hermes_cli.gateway_runtime_discovery import connect_private, _identify_response
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import query_runtime_control
        return _identify_response(query_runtime_control(home, request, timeout))
    deadline = time.monotonic() + timeout
    with connect_private(home, timeout) as peer:
        peer.sendall(request)
        data = bytearray()
        while b'\n' not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            peer.settimeout(remaining)
            chunk = peer.recv(4096)
            if not chunk or len(data) + len(chunk) > 65536:
                raise ValueError('invalid_control_response')
            data.extend(chunk)
    return _identify_response(bytes(data))


def forward_read_status(profile, actor, params):
    """None means no owner exists; every uncertain owner/result fails closed.

    Discover only: a read acknowledgement must never start or restart a gateway.
    After dispatch no error permits an offline write or automatic replay.
    """
    from gateway.session_read_status import FORWARD_TIMEOUT_SECONDS
    from hermes_cli.gateway_runtime import control_home_for, discover_gateway_endpoint
    from hermes_cli.web_server_cron import _cron_profile_home

    home = _cron_profile_home(profile)[1]
    discovery = discover_gateway_endpoint(home)
    if discovery.state == 'absent':
        return None
    if discovery.state != 'ready' or discovery.endpoint is None:
        raise RuntimeStoreError('runtime_coordination_required')
    endpoint = discovery.endpoint
    envelope = {'instance_id': endpoint.instance_id, 'profile_id': endpoint.profile_id,
                'principal_id': actor.subject, 'params': params}
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': 'session-read-status', 'params': envelope}).encode() + b'\n'
    try:
        reply = _exchange(control_home_for(home, endpoint), request, FORWARD_TIMEOUT_SECONDS + 2)
    except (OSError, ValueError):
        raise RuntimeStoreError('mutation_outcome_unknown') from None
    if isinstance(reply.get('error'), str):
        raise RuntimeStoreError(reply['error'])
    result = reply.get('result')
    if not isinstance(result, dict):
        raise RuntimeStoreError('mutation_outcome_unknown')
    return result
