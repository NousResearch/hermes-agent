"""Real process exclusion before writable gateway construction."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.platforms("linux")
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
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stderr
        assert not witness.exists(), result.stdout + result.stderr
        assert 'CLAIMED False' in result.stdout
    finally:
        release_gateway_runtime_lock()


@pytest.mark.platforms("linux")
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

    # Pre-existing homes writable by others are refused, not silently chmodded (a readable
    # operator-chosen mode is fine: only write lets another user swap the socket).
    existing = tmp_path / 'existing-writable-home'
    existing.mkdir(mode=0o757)
    existing.chmod(0o757)
    owner.reserve([existing])
    try:
        observed = await asyncio.to_thread(discover_gateway_endpoint, existing)
        assert (observed.state, observed.reason_code) == ('inaccessible', 'unsafe_control_permissions')
        assert existing.stat().st_mode & 0o777 == 0o757
    finally:
        owner.close()


@pytest.mark.platforms("linux")
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


@pytest.mark.platforms("linux")
@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason="root opens a mode-000 file")
@pytest.mark.parametrize("mode", [0o000, 0o400])
def test_stale_unopenable_gateway_lock_is_replaced_on_boot(tmp_path, monkeypatch, mode):
    """Main's #42685: a lock this user cannot open (left by a root-run gateway) and that no process
    holds is unlinked and recreated, so boot does not exit "already held" forever."""
    import json
    from gateway.status import acquire_gateway_runtime_lock, release_gateway_runtime_lock
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    monkeypatch.setenv('HERMES_HOME', str(home))
    lock = home / 'gateway.lock'
    lock.write_text('{"pid": 1}', encoding='utf-8')
    lock.chmod(mode)
    try:
        assert acquire_gateway_runtime_lock()
        assert lock.stat().st_mode & 0o777 == 0o600  # a fresh inode (the stale one is not ours to chmod)
        assert json.loads(lock.read_text(encoding='utf-8-sig'))['pid'] == os.getpid()
    finally:
        release_gateway_runtime_lock()


@pytest.mark.platforms("linux")
@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason="root opens a mode-000 file")
def test_unusable_gateway_lock_held_or_linked_is_never_unlinked(tmp_path):
    """The stale-lock recovery never removes an inode a live process holds, and never follows a
    symlink or touches a hardlinked lock."""
    from gateway import runtime_ownership
    homes = {kind: tmp_path / kind for kind in ('held', 'held_readable', 'hardlink', 'symlink')}
    for home in homes.values():
        home.mkdir(mode=0o700)
    held = [homes[kind] / 'gateway.lock' for kind in ('held', 'held_readable')]
    for path in held:
        path.write_text('', encoding='utf-8')
    holder = subprocess.Popen(  # a live owner (flock on both inodes) from another process
        [sys.executable, '-c', 'import fcntl,sys;fs=[open(p) for p in sys.argv[1:]];'
         '[fcntl.flock(f,fcntl.LOCK_EX) for f in fs];print(1,flush=True);sys.stdin.read()',
         *map(str, held)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == '1'
        held[0].chmod(0)  # unopenable: only /proc/locks can show the holder
        held[1].chmod(0o400)  # readable: the flock probe sees the holder
        (tmp_path / 'other').write_text('', encoding='utf-8')
        os.link(tmp_path / 'other', homes['hardlink'] / 'gateway.lock')
        (homes['hardlink'] / 'gateway.lock').chmod(0)
        (homes['symlink'] / 'gateway.lock').symlink_to(tmp_path / 'target')
        before = {kind: os.lstat(home / 'gateway.lock').st_ino for kind, home in homes.items()}
        for kind, home in homes.items():
            with pytest.raises(OSError):
                runtime_ownership.ProfileOwnership().reserve([home])
            assert os.lstat(home / 'gateway.lock').st_ino == before[kind], kind
        assert not (tmp_path / 'target').exists()
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
