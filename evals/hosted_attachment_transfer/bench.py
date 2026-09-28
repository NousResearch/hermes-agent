"""Hosted-room attachment transfer cost over the real private control socket.

Stands up two served profiles (alpha = room source owner, beta = task target) on
one multiplexed runner with a real ``GatewayControlServer``, commits one
attachment of N bytes, then measures for (a) ``produce('submit')`` through
``HostedRoomOwnerRPC.submit`` and (b) ``check_remote_hosted_admission``:

* ``round_trips``  — ``owner_request`` exchanges over the socket
* ``wall_s``       — wall time of the operation
* ``source_read``  — bytes the source read from its blob store
* ``source_hashed``— bytes the source SHA-256 hashed (store verify + handler)

Usage (repo root, worktree venv):
    .venv/bin/python evals/hosted_attachment_transfer/bench.py [--mb 5,15]
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in list(os.environ):
    if key.startswith('HERMES_') or key.endswith(('_API_KEY', '_TOKEN')):
        os.environ.pop(key, None)
TMP = Path(tempfile.mkdtemp(prefix='hosted-attachment-bench-'))
os.environ['HOME'] = str(TMP)
os.environ['HERMES_HOME'] = str(TMP / '.hermes')

from pytest import MonkeyPatch  # noqa: E402


class SourceMeter:
    """Counts source-side blob reads and hashes without touching the wire."""

    def __init__(self, monkeypatch):
        import hashlib
        from gateway import session_hosted_transport as transport
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore
        self.round_trips = self.read = self.hashed = 0
        self.local = threading.local()
        meter = self

        real_request = transport.owner_request

        def owner_request(*args, **kwargs):
            meter.round_trips += 1
            return real_request(*args, **kwargs)
        monkeypatch.setattr(transport, 'owner_request', owner_request)

        real_chunk = transport.source_attachment_chunk

        def source_attachment_chunk(*args, **kwargs):
            meter.local.in_source = True
            try:
                return real_chunk(*args, **kwargs)
            finally:
                meter.local.in_source = False
        monkeypatch.setattr(transport, 'source_attachment_chunk', source_attachment_chunk)

        class CountingHashlib:
            def sha256(self, data=b''):
                if getattr(meter.local, 'in_source', False):
                    meter.hashed += len(data)
                return hashlib.sha256(data)

            def __getattr__(self, name):
                return getattr(hashlib, name)
        monkeypatch.setattr(transport, 'hashlib', CountingHashlib())

        real_read_blob = HostedRoomAttachmentStore._read_blob

        def _read_blob(store, **kwargs):
            data = real_read_blob(store, **kwargs)
            meter.read += len(data)
            meter.hashed += len(data)
            return data
        monkeypatch.setattr(HostedRoomAttachmentStore, '_read_blob', _read_blob)
        if hasattr(HostedRoomAttachmentStore, 'read_range'):
            real_range = HostedRoomAttachmentStore.read_range

            def read_range(store, **kwargs):
                saved = real_range(store, **kwargs)
                meter.read += len(saved.data)
                return saved
            monkeypatch.setattr(HostedRoomAttachmentStore, 'read_range', read_range)

    def snapshot(self):
        return dict(round_trips=self.round_trips, source_read=self.read, source_hashed=self.hashed)

    def measure(self, label, fn):
        before, started = self.snapshot(), time.perf_counter()
        result = fn()
        wall = time.perf_counter() - started
        after = self.snapshot()
        return result, {'op': label, 'wall_s': round(wall, 3),
                        **{k: after[k] - before[k] for k in before}}


def _mux(monkeypatch):
    from gateway.control_socket import GatewayControlServer
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
    case_root = Path(tempfile.mkdtemp(prefix='case-', dir=TMP))
    root, homes = _reserve_homes(case_root, monkeypatch)
    for name, home in homes:
        config = {'model': {'default': 'fixture-' + name}, 'platform_toolsets': {'cli': []},
                  'hosted_rooms': {'profiles': {n: str(h) for n, h in homes if h != home}}}
        (home / 'config.yaml').write_text(json.dumps(config))
    process_ownership.reserve([home for _, home in homes])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = _runner(root, homes)

    def call(coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(60)
    call(initialize_gateway_runtime(runner))
    for authority in runner.session_authorities:
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    descriptor = runner.session_runtime_descriptor
    descriptor.update(state='ready', capabilities=['session-authority-v1'],
                      api_origin='http://127.0.0.1:1', supervisor='none')
    server = runner.session_control_server = GatewayControlServer(
        root, verb_handlers={'identify': lambda: descriptor})
    assert call(server.start())
    return runner, dict(homes), loop, call


def run_case(size_bytes, monkeypatch):
    from gateway import hosted_rooms
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.session_authorities import owner_scope
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.session_hosted_transport import (
        HostedRoomOwnerRPC, check_remote_hosted_admission, install_hosted_transport)
    from hermes_state_runtime import list_session_admissions

    runner, homes, loop, call = _mux(monkeypatch)
    meter = SourceMeter(monkeypatch)
    for authority in runner.session_authorities:
        with owner_scope(authority):
            service = CanonicalHostedRoomService(authority, loop)
            authority.hosted_room_service = service
            install_hosted_transport(runner.session_control_server, authority, loop, attest=service.attest)
    source = runner.session_authorities.require(homes['alpha'])
    target = runner.session_authorities.require(homes['beta'])
    with owner_scope(source):
        source.hosted_room_service.authorize_room('alice', 'room', create=True)
        hosted_rooms.create_room(source.db.db_path, room_id='room', name='Room',
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
            members=[{'member_id': 'helper', 'profile': 'beta', 'handle': 'helper'}])
    data = os.urandom(size_bytes)
    store = HostedRoomAttachmentStore(source.db.db_path)
    saved = store.put(room_id='room', upload_id='upload', kind='file', name='blob.bin',
                      mime='application/octet-stream', data=data)
    manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['helper'])
    manifest[0]['event_id'] = 'event'
    identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
    payload = {'target_profile': 'beta', 'target_member_id': 'helper', 'source_event_seq': 1,
               'prompt': 'input', 'attachments': manifest}
    tasks.admit_task(source.db.db_path, identity, payload=payload, clock=time.time)
    lease = tasks.acquire_lease(source.db.db_path, room_id='room',
        gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1,
        process_generation='bench', ttl_seconds=300, clock=time.time)
    tasks.start_task(source.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)

    rpc = HostedRoomOwnerRPC(home=homes['beta'], source_home=homes['alpha'],
                            room_id='room', member_id='helper', profile='beta')
    sid = rpc.create(profile='beta', source='bot_room', title='Group: room')['session_id']
    rows = []
    _, submit = meter.measure('produce(submit)', lambda: rpc.submit(
        profile='beta', source='bot_room', session_id=sid, prompt='input', task=identity,
        execution_generation=1, attachments=manifest, on_terminal=lambda row: None))
    rows.extend(list_session_admissions(target.db, session_id=sid, pending_only=False))
    assert len(rows) == 1 and rows[0]['status'] == 'queued'
    # Stop the driver-side history poller so it does not count against the preflight.
    with rpc._lock:
        rpc.callbacks.clear()
    if rpc._monitor is not None:
        rpc._monitor.join(5)
    ok, preflight = meter.measure('check_remote_hosted_admission',
                                  lambda: check_remote_hosted_admission(target, rpc.ref, rows[0]))
    assert ok is True
    for authority in runner.session_authorities:
        service = getattr(authority, 'hosted_room_service', None)
        if service is not None:
            service.stop(timeout=5)
        authority.db.close()
    call(runner.session_control_server.stop())
    loop.call_soon_threadsafe(loop.stop)
    return [submit, preflight]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mb', default='5,15', help='comma-separated attachment sizes in MB (decimal)')
    args = parser.parse_args()
    from gateway.session_hosted_transport import _CHUNK_BYTES
    report = {'chunk_bytes': _CHUNK_BYTES, 'cases': []}
    for mb in (int(v) for v in args.mb.split(',')):
        size = mb * 1_000_000
        monkeypatch = MonkeyPatch()
        try:
            results = run_case(size, monkeypatch)
        finally:
            monkeypatch.undo()
        report['cases'].append({'size_bytes': size, 'results': results})
        for result in results:
            print(f"{mb:>3} MB  {result['op']:<32} rt={result['round_trips']:<5} wall={result['wall_s']:>8.3f}s  "
                  f"src_read={result['source_read']/1e6:>9.1f}MB  src_hashed={result['source_hashed']/1e6:>9.1f}MB",
                  flush=True)
    (ROOT / 'evals' / 'hosted_attachment_transfer' / 'last_run.json').write_text(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
