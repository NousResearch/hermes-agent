"""Only in the disposable two-daemon probe: hold real EOF handling and the finish call.

No request fields, authority guards, RPC replies or attachment results are replaced.
The driver opens the native successor socket before releasing the held call: this
probe is about epoch ordering, not latency of the cold WebSocket handler.
"""
import json
import os
from pathlib import Path
import runpy
import sys

from tests.waiting import wait_until

HOME = Path(os.environ['HERMES_HOME'])


def publish(name, fields):
    target = HOME / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(fields))
    temp.replace(target)


def wait_for(name):
    return wait_until((HOME / name).exists, timeout=105, what=name)


def worker():
    from agent.managed_worker_detach.channel import DetachableChannel
    from agent.runtime_session_store import WorkerPersistenceError, WorkerRPC

    await_owner = DetachableChannel.await_owner
    rpc = WorkerRPC.__call__

    def hold_eof(self, old):
        (HOME / 'eof-held').touch()
        wait_for('release-eof')
        return await_owner(self, old)

    def hold_finish(self, method, **params):
        if method != 'worker.persist' or params.get('operation') != 'execution.finish':
            return rpc(self, method, **params)
        if not (HOME / 'finish-held.json').exists():
            publish('finish-held.json', {k: params[k] for k in ('pid', 'epoch', 'sequence', 'execution_id')})
            wait_for('successor-observed')
        try:
            return rpc(self, method, **params)
        except WorkerPersistenceError as exc:
            publish('client-refusal.json', {'error': str(exc), 'epoch': params['epoch']})
            raise
        # The test driver releases the native attachment after observing the rejection.

    DetachableChannel.await_owner = hold_eof
    WorkerRPC.__call__ = hold_finish
    sys.argv[1:] = []
    runpy.run_module('agent.managed_worker', run_name='__main__')


def owner():
    from gateway import session_detached_workers, session_managed_worker, session_worker
    if not (HOME / 'finish-held.json').exists():
        launch = session_managed_worker._worker_launch

        def wrap(*args, **kwargs):
            argv, env, detachable = launch(*args, **kwargs)
            assert argv[-2:] == ['-m', 'agent.managed_worker']
            return [*argv[:-2], str(Path(__file__)), 'worker'], env, detachable

        session_managed_worker._worker_launch = wrap
    else:
        request = session_worker.worker_request
        from persistence.core.errors import AuthorityRefusal
        handshake = session_detached_workers._handshake

        def hold_handshake(record):
            wait_for('release-eof')
            return handshake(record)

        async def observe(connection, ref, params, *, operation):
            if operation != 'persist' or params.get('operation') != 'execution.finish':
                return await request(connection, ref, params, operation=operation)
            identity = {k: params[k] for k in ('pid', 'epoch', 'sequence', 'execution_id')}
            try:
                result = await request(connection, ref, params, operation=operation)
            except AuthorityRefusal as exc:
                publish('server-refusal.json', identity | {'error': str(exc)})
                raise
            publish('server-applied.json', identity)
            return result

        session_detached_workers._handshake = hold_handshake
        session_worker.worker_request = observe
    runpy.run_module('gateway.run', run_name='__main__')


if __name__ == '__main__':
    (worker if sys.argv[1:] == ['worker'] else owner)()
