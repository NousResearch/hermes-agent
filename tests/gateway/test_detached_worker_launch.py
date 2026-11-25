"""A detachable worker of a systemd-supervised gateway is launched with what its scope probe validated.

A system-level gateway unit (``User=`` + linger) inherits no login-session variables. The
``systemd-run --user --scope`` probe reaches the user manager through the derived bus env; the
worker launch must carry that same env into the wrapper, and preparing the launch (which may run
that probe, up to its timeout) must not hold the event loop.
"""
import asyncio
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import session_managed_worker as managed
from tests.waiting import wait_until
from tools import process_registry

DETACHED = SimpleNamespace(config=lambda: {'gateway': {'detached_workers': True}})

# Stand-in for systemd-run: without a user bus it fails like the real one, before the command runs;
# otherwise it records the bus it was given and execs the command in place (same pid, same pipes).
FAKE_SYSTEMD_RUN = """#!/bin/sh
[ -n "$DBUS_SESSION_BUS_ADDRESS" ] || { echo bus-missing; exit 72; }
printf %s "$DBUS_SESSION_BUS_ADDRESS" > "$(dirname "$0")/bus-seen"
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""


@pytest.fixture
def short_dir():
    path = Path(tempfile.mkdtemp(prefix='hdw-', dir='/tmp'))  # AF_UNIX: tmp_path may be too long
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def system_service_gateway(tmp_path, short_dir, monkeypatch):
    """A supervised gateway whose own environ has no session bus, while ``/run/user/<uid>/bus`` exists."""
    runtime = short_dir / 'run'
    runtime.mkdir(mode=0o700)
    bus = socket.socket(socket.AF_UNIX)
    bus.bind(str(runtime / 'bus'))
    tools = tmp_path / 'bin'
    tools.mkdir()
    fake = tools / 'systemd-run'
    fake.write_text(FAKE_SYSTEMD_RUN)
    fake.chmod(0o755)
    monkeypatch.setenv('PATH', f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    monkeypatch.delenv('DBUS_SESSION_BUS_ADDRESS', raising=False)
    monkeypatch.setenv('INVOCATION_ID', 'system-gateway-unit')
    monkeypatch.setattr(process_registry, '_is_supervised_gateway_process', lambda: True)
    monkeypatch.setattr(process_registry, '_default_user_runtime_dir', lambda: runtime)
    monkeypatch.setattr(process_registry, '_SYSTEMD_SCOPE_AVAILABLE', None)
    try:
        yield SimpleNamespace(bus=f"unix:path={runtime / 'bus'}", runtime=str(runtime), seen=tools / 'bus-seen')
    finally:
        bus.close()


def _owner():
    return SimpleNamespace(profile_id='profile', pending_results={}, pending_stops={})


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('worker_env', ['inherited', 'profile_runtime_dir_without_bus'])
async def test_scoped_worker_starts_with_the_probed_user_bus(system_service_gateway, short_dir, monkeypatch,
                                                             worker_env):
    """The probe succeeds through the bus derived from the gateway's environ; the launched wrapper
    must get that bus too, so the real worker execs in place and introduces itself instead of the
    wrapper failing first. A routed profile's own env may name another private runtime dir with no
    bus: the probed bus still wins there."""
    if worker_env != 'inherited':
        other = short_dir / 'profile-run'
        other.mkdir(mode=0o700)
        monkeypatch.setattr(managed, '_worker_env', lambda authority: {**os.environ, 'XDG_RUNTIME_DIR': str(other)})
    spawned = await managed._spawn_registered_worker(_owner(), SimpleNamespace(session_id='s'),
                                                     {'admission_id': 'a' * 32, 'generation': 1}, DETACHED)
    assert spawned is not None
    worker, process = spawned[:2]
    try:
        hello = await worker.next_frame(30, ack=0)
        assert hello['type'] == 'hello' and hello['pid'] == process.pid
        assert system_service_gateway.seen.read_text() == system_service_gateway.bus
        assert spawned[2:] == (True,)
    finally:
        await asyncio.to_thread(worker.close)


@pytest.mark.platforms('linux')
def test_scoped_worker_env_adds_only_the_bus_to_the_profile_env(system_service_gateway, monkeypatch):
    """A routed profile's scrubbed env stays the base: the scope adds the probed bus, nothing else
    from the launch env."""
    profile_env = {'HERMES_HOME': '/profiles/b', 'PATH': os.environ['PATH'], 'XDG_RUNTIME_DIR': '/profiles/b/run'}
    monkeypatch.setattr(managed, '_worker_env', lambda authority: dict(profile_env))
    monkeypatch.setenv('LAUNCH_ONLY_SETTING', 'launch')
    argv, env, detachable = managed._worker_launch(_owner(), {'admission_id': 'a' * 32}, DETACHED)
    assert detachable
    assert argv[0].endswith('systemd-run') and argv[-2:] == ['-m', 'agent.managed_worker']
    assert env == profile_env | {'DBUS_SESSION_BUS_ADDRESS': system_service_gateway.bus,
                                 'XDG_RUNTIME_DIR': system_service_gateway.runtime}


@pytest.mark.platforms('linux')
def test_unscoped_worker_keeps_its_env(monkeypatch):
    """Without opt-in, or outside a supervised gateway (detachable by its own session), the env is ``_worker_env``'s."""
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    assert managed._worker_launch(_owner(), {'admission_id': 'a' * 32}, None) == (
        [managed.sys.executable, '-m', 'agent.managed_worker'], None, False)
    monkeypatch.delenv('INVOCATION_ID', raising=False)
    assert managed._worker_launch(_owner(), {'admission_id': 'a' * 32}, DETACHED) == (
        [managed.sys.executable, '-m', 'agent.managed_worker'], None, True)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_scope_probe_does_not_hold_the_event_loop(monkeypatch):
    """The scope probe can block on D-Bus up to its timeout; an unrelated loop callback must run meanwhile."""
    loop = asyncio.get_running_loop()
    probing, ticked = threading.Event(), threading.Event()
    seen = {}

    def probe():
        probing.set()
        seen['loop_ran'] = ticked.wait(10)
        return False

    async def unrelated():
        await asyncio.to_thread(probing.wait, 10)
        loop.call_soon(ticked.set)

    monkeypatch.setenv('INVOCATION_ID', 'system-gateway-unit')
    monkeypatch.setattr(process_registry, '_is_supervised_gateway_process', lambda: True)
    monkeypatch.setattr(process_registry, '_systemd_run_user_scope_available', probe)
    monkeypatch.setattr(managed.subprocess, 'Popen', _refuse_spawn)
    other = asyncio.create_task(unrelated())
    with pytest.raises(OSError, match='fixture: no spawn'):
        await managed._spawn_registered_worker(_owner(), SimpleNamespace(session_id='s'),
                                               {'admission_id': 'a' * 32, 'generation': 1}, DETACHED)
    await other
    assert seen == {'loop_ran': True}


def _refuse_spawn(*_args, **_kwargs):
    raise OSError('fixture: no spawn')


# Introduces itself, records the bootstrap it was sent (and whether its listener's directory
# exists), then exits.
BOOTSTRAP_RECORDER = """import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
frame = json.loads(sys.stdin.buffer.readline())
listener = frame.get('detach_v1', {}).get('socket')
open(sys.argv[1], 'w').write(json.dumps({'keys': sorted(frame), 'listener': listener,
                                         'listener_dir': bool(listener) and os.path.isdir(os.path.dirname(listener))}))
"""


def _run_managed_turn(tmp_path, monkeypatch, profile='owned'):
    """One real managed turn through ``execute_managed`` (real SQLite, real child): what the child received."""
    import contextlib
    import json
    import subprocess
    import sys
    from gateway.session_authority import LiveSession, SessionAuthority, admission_row
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_policy import build_policy
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session
    from tests.gateway.fixtures.storage_handles import storage_handle

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    received = tmp_path / 'bootstrap.json'

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', BOOTSTRAP_RECORDER, str(received)], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 's', 'test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id=profile, instance_id='i',
                                 epoch=db.runtime.begin_epoch(instance_id='i'), persistence=storage_handle(tmp_path))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = SessionActor('human', profile, frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef(profile, 's')
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {'gateway': {'detached_workers': True}},
                          profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = admission_row(db.admissions.claim(epoch=authority.epoch, session_id='s'))
        with contextlib.suppress(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(60):
                await managed.execute_managed(authority, ref, row, policy)

    with db:
        asyncio.run(scenario())
    return json.loads(received.read_text())


@pytest.mark.platforms('linux')
@pytest.mark.usefixtures('authority_cwd_in_tmp')
@pytest.mark.parametrize('launch', ['degraded', 'own_session'])
def test_worker_a_restart_would_kill_is_not_detached(tmp_path, monkeypatch, launch):
    """Under a supervised gateway with no user scope, a stop that hands the worker off instead of
    stopping it would only leave it to the service's cgroup kill (``unknown`` instead of
    ``interrupted``): such a worker gets no detach listener. Outside systemd it does."""
    if launch == 'degraded':
        monkeypatch.setenv('INVOCATION_ID', 'system-gateway-unit')
        monkeypatch.setattr(process_registry, '_is_supervised_gateway_process', lambda: True)
        monkeypatch.setattr(process_registry, '_systemd_run_user_scope_available', lambda: False)
    else:
        monkeypatch.delenv('INVOCATION_ID', raising=False)
    received = _run_managed_turn(tmp_path, monkeypatch)
    assert ('detach_v1' in received['keys']) is (launch == 'own_session')


@pytest.mark.platforms('linux')
@pytest.mark.usefixtures('authority_cwd_in_tmp')
def test_a_turn_that_keeps_its_owner_leaves_no_listener_directory(tmp_path, short_dir, monkeypatch):
    """A profile home too long for the outbox listener gets a fallback path under the runtime dir;
    the owner creates nothing there, so a turn that never changes owner leaves nothing behind (the
    worker makes that directory only while it waits for a new owner)."""
    monkeypatch.delenv('INVOCATION_ID', raising=False)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(short_dir))
    received = _run_managed_turn(tmp_path, monkeypatch, profile='p' * 30)
    assert received['listener'].startswith(str(short_dir) + os.sep) and not received['listener_dir']
    assert os.listdir(short_dir) == []


@pytest.mark.platforms('linux')
def test_degraded_scope_warning_names_the_managed_worker_consequence(monkeypatch, caplog):
    """The shared helper's default warning is cron's (``cron.require_restart_safe_scope``), which
    does not govern this caller: the operator is told what happens to managed workers instead."""
    monkeypatch.setenv('INVOCATION_ID', 'system-gateway-unit')
    monkeypatch.setattr(process_registry, '_is_supervised_gateway_process', lambda: True)
    monkeypatch.setattr(process_registry, '_systemd_run_user_scope_available', lambda: False)
    monkeypatch.setattr(process_registry, '_scope_degraded_warned', False)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    with caplog.at_level(logging.WARNING, logger=process_registry.logger.name):
        assert managed._worker_launch(_owner(), {'admission_id': 'a' * 32}, DETACHED)[2] is False
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 1 and 'gateway.detached_workers is off' in warned[0] and 'cron' not in warned[0]


# Reads nothing until told to (a worker that stopped reading its controls), then everything to EOF.
STALLED_READER = """import os, sys, time
while not os.path.exists(sys.argv[1]):
    time.sleep(0.05)
data = sys.stdin.buffer.read()
open(sys.argv[2], 'w').write(str(len(data)))
"""


@pytest.mark.platforms('linux')
def test_gateway_stop_hands_off_without_waiting_for_a_blocked_control_write(tmp_path):
    """A detachable worker that stopped reading leaves the control writer blocked in a pipe write,
    holding the write lock. The stop (on the gateway loop) must still return at once; the hang-up
    then follows as soon as that write completes, so the worker reads EOF, and a control still
    queued behind it is not written to a worker this owner gave away. Releasing the handed-off
    worker does not wait for that write either."""
    from agent.managed_worker import encode_frame
    from gateway.run_runtime import stop_authority_turns
    go, received = tmp_path / 'go', tmp_path / 'received'
    child = subprocess.Popen([sys.executable, '-c', STALLED_READER, str(go), str(received)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
    worker = managed.ManagedWorker(child)
    try:
        worker.detachable = True
        worker.writer.start()
        blocked = {'type': 'fixture', 'pad': 'x' * (1 << 20)}  # far past the pipe buffer
        worker.control(blocked)
        wait_until(worker.write_lock.locked, timeout=10, what='the control writer to block mid-write')
        worker.control({'type': 'queued_behind_it'})
        live = SimpleNamespace(event_stream=SimpleNamespace(execution={'execution_generation': 3}))
        authority = SimpleNamespace(_managed_workers={'s': worker}, sessions={'s': live}, _managed_spawning={})
        stopping = threading.Thread(target=stop_authority_turns, args=(authority,), daemon=True)
        stopping.start()
        stopping.join(5)
        assert not stopping.is_alive(), 'the stop waited on a blocked control write'
        assert worker.handed_off and live.handed_off_generation == 3
        # The turn then lets go of the worker (execute_managed's finally): bounded too.
        releasing = threading.Thread(target=worker.release, daemon=True)
        releasing.start()
        releasing.join(15)
        assert not releasing.is_alive(), 'release waited on the blocked control write'
        go.touch()
        wait_until(received.exists, timeout=30, what='the worker to read its controls to EOF')
        assert int(received.read_text()) == len(encode_frame(blocked))
    finally:
        go.touch()
        worker.closed.set()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        child.stdout.close()
