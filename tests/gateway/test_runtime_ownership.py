"""Real process exclusion before writable gateway construction."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.linux_only
def test_losing_start_never_constructs_writable_runner(tmp_path):
    from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
    assert acquire_gateway_runtime_lock()
    code = '''
import asyncio
from pathlib import Path
import os
import gateway.run as run
from gateway.config import GatewayConfig
class Witness:
    def __init__(self, config):
        Path(os.environ['WITNESS']).write_text('writable init reached')
        raise RuntimeError('initializer reached')
run.GatewayRunner = Witness
try:
    result = asyncio.run(run.start_gateway(GatewayConfig(), verbosity=None))
except RuntimeError:
    result = True
print('CLAIMED', result)
'''
    witness = tmp_path / 'witness'
    try:
        result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'WITNESS': str(witness)},
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert not witness.exists(), result.stdout + result.stderr
        assert 'CLAIMED False' in result.stdout
    finally:
        release_gateway_runtime_lock()


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_reserved_home_is_eligible_for_same_user_bootstrap(tmp_path):
    import asyncio
    import json

    from gateway.control_socket import GatewayControlServer
    from gateway.runtime_bootstrap import TicketStore
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    home = tmp_path / 'new-private-home'
    owner = ProfileOwnership()
    old_umask = os.umask(0o022)
    try:
        owner.reserve([home])
    finally:
        os.umask(old_umask)
    server = GatewayControlServer(home)
    server.ticket_store = TicketStore('fixture-owner', frozenset({str(home)}))
    try:
        assert await server.start()
        pointer = home / 'gateway.sock.path'
        socket_path = pointer.read_text().strip() if pointer.exists() else str(home / 'gateway.sock')
        reader, writer = await asyncio.open_unix_connection(socket_path)
        try:
            writer.write(json.dumps({'protocol': 1, 'verb': 'session-ticket', 'id': 1,
                'params': {'profile_id': str(home), 'instance_id': 'fixture-owner',
                           'purpose': 'interactive'}}).encode() + b'\n')
            await writer.drain()
            reply = json.loads(await asyncio.wait_for(reader.readline(), 2))
            assert reply.get('ok') is True, reply.get('error')
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()
        owner.close()

    # Pre-existing readable homes are refused, not silently chmodded.
    existing = tmp_path / 'existing-readable-home'
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    owner.reserve([existing])
    try:
        observed = await asyncio.to_thread(discover_gateway_endpoint, existing)
        assert (observed.state, observed.reason_code) == ('inaccessible', 'unsafe_control_permissions')
        assert existing.stat().st_mode & 0o777 == 0o755
    finally:
        owner.close()


@pytest.mark.linux_only
def test_profile_reservations_unwind_without_releasing_another_owner(tmp_path):
    from gateway import runtime_ownership
    homes = [tmp_path / 'a', tmp_path / 'b']
    for home in homes:
        home.mkdir()
    blocker = runtime_ownership.ProfileOwnership()
    blocker.reserve([homes[1]])
    contender = runtime_ownership.ProfileOwnership()
    with pytest.raises(runtime_ownership.OwnershipConflict):
        contender.reserve(reversed(homes))
    free = runtime_ownership.ProfileOwnership()
    free.reserve([homes[0]])
    contender.close()
    with pytest.raises(runtime_ownership.OwnershipConflict):
        contender.reserve([homes[1]])
    blocker.close()
    contender.reserve([homes[1]])
    blocker.close()  # late cleanup must not affect replacement
    with pytest.raises(runtime_ownership.OwnershipConflict):
        blocker.reserve([homes[1]])
    free.close()
    contender.close()
