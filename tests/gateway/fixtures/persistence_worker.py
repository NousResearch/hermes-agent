"""Owned inert tool witness. Commands arrive through a private test pipe."""
import json
import os
from pathlib import Path
import sys

import psutil


def writable_fds(home):
    result = []
    for path in Path('/proc/self/fd').iterdir():
        try:
            target = path.resolve()
            flags = next(line.split(':')[1].strip() for line in
                         (Path('/proc/self/fdinfo') / path.name).read_text().splitlines()
                         if line.startswith('flags:'))
            if target.name in ('state.db', 'state.db-wal', 'state.db-shm') and int(flags, 8) & 3:
                result.append(str(target))
        except FileNotFoundError:
            continue
    return result


def reason(fn):
    try:
        fn()
    except Exception as exc:
        return str(exc)
    raise AssertionError('expected refusal')


canonical_opens = []
def audit(event, args):
    if event == 'sqlite3.connect' and 'state.db' in str(args[0]):
        canonical_opens.append(str(args[0]))
sys.addaudithook(audit)

for line in sys.stdin:
    try:
        command = json.loads(line)
        if command['op'] == 'start':
            from agent import runtime_session_store as module
            home = Path(command['home'])
            transport = module.WorkerRPC(home)
            scope = {'profile_id': str(home), 'session_id': command['session_id'],
                     'execution_id': 'owned-worker', 'generation': 0,
                     'pid': os.getpid(), 'birth': psutil.Process().create_time(), 'secret': 'private-worker-secret'}
            registration = transport('worker.register', **scope, kind='compute')
            assert reason(lambda: transport('session.create', request_id='forbidden-worker-create', source='cli')) == 'permission_denied'
            scope['epoch'] = registration['owner_epoch']
            store = module.RuntimeSessionStore(transport, scope, home / 'worker-outboxes' / 'owned-worker', max_bytes=20000)
            sid = scope['session_id']
            store.try_acquire_session_turn_lease(sid, 'owned-worker-lease')
            (home / 'tool-marker').write_text('once')
            messages = [{'role': 'user', 'content': 'start'},
                        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call', 'type': 'function',
                         'function': {'name': 'inert', 'arguments': '{}'}}]},
                        {'role': 'tool', 'content': 'tool-marker', 'tool_call_id': 'call', 'tool_name': 'inert'}]
            original = store.rpc
            lost = []
            def lose_ack(method, **params):
                result = original(method, **params)
                lost.append(params)
                raise TimeoutError('lost_ack')
            store.rpc = lose_ack
            assert reason(lambda: store.append_messages_batch(sid, messages, turn_lease_holder='owned-worker-lease')) == 'lost_ack'
            store.rpc = original
            recovered = store.retry_pending()
            assert recovered[0]['count'] == 3
            saved = lost[0]
            assert original('worker.persist', **saved) == recovered[0]
            store.queue_token_counts(sid, input_tokens=11, model='inert-model', api_call_count=1)
            assert store.flush_token_counts()
            output = {'registered': True, 'rows': 3, 'lost_ack_replayed': True,
                'epoch': scope['epoch'], 'pid': os.getpid(),
                'conflict': reason(lambda: original('worker.persist', **(saved | {'payload': {'messages': []}}))),
                'wrong_session': reason(lambda: original('worker.persist', **(saved | {'session_id': 'foreign'}))),
                'wrong_profile': reason(lambda: original('worker.persist', **(saved | {'profile_id': '/foreign'}))),
                'wrong_generation': reason(lambda: original('worker.persist', **(saved | {'generation': 99}))),
                'outbox_full': reason(lambda: store.append_messages_batch(sid, [{'role': 'user', 'content': 'x' * 30000}])),
                'writable_canonical_fds': writable_fds(home), 'canonical_opens': list(canonical_opens)}
        elif command['op'] == 'outage':
            store.retry_pending()  # clear the deliberate full-entry rejection
            failure = reason(lambda: store.record_auxiliary_usage(sid, 'compression', input_tokens=5, model='aux'))
            assert store.failure and store.journal['pending']
            output = {'pending': len(store.journal['pending']), 'failure': failure,
                      'canonical_opens': list(canonical_opens)}
        else:
            stale = reason(lambda: transport('worker.persist', **saved))
            adopted = transport('worker.adopt', **{k: v for k, v in scope.items() if k != 'epoch'})
            bad = dict(scope, birth=scope['birth'] + 1)
            assert reason(lambda: transport('worker.adopt', **{k: v for k, v in bad.items() if k != 'epoch'})) == 'permission_denied'
            store.adopt(adopted['owner_epoch'])
            assert len(store.retry_pending()) == 1
            store.refresh_session_turn_lease(sid, 'owned-worker-lease')
            store.release_session_turn_lease(sid, 'owned-worker-lease')
            assert store.finish()['status'] == 'terminal'
            output = {'stale': stale, 'epoch': adopted['owner_epoch'], 'pid': os.getpid(),
                      'writable_canonical_fds': writable_fds(home), 'canonical_opens': list(canonical_opens)}
        sys.stdout.write(json.dumps(output) + '\n'); sys.stdout.flush()
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({'fatal': repr(exc)}) + '\n'); sys.stdout.flush()
