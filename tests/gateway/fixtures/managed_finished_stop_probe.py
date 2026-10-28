"""Hold only the real managed worker's finished frame at its native parent."""
import os
from pathlib import Path
import runpy

from gateway.session_managed_worker import ManagedWorker
from tests.waiting import async_wait_until
from tools.process_registry import process_registry
from utils import atomic_json_write

home = Path(os.environ['HERMES_HOME'])
next_frame, kill = ManagedWorker.next_frame, process_registry.kill_process


async def held_finished(self, *args, **kwargs):
    frame = await next_frame(self, *args, **kwargs)
    if frame == {'type': 'finished'} and not (home / 'parent-held-finished.json').exists():
        atomic_json_write(home / 'parent-held-finished.json', {'worker_pid': self.process.pid})
        await async_wait_until(lambda: (home / 'parent-release-finished').exists(), timeout=90,
                               what='test releases native finished frame')
    return frame


def refuse_cleanup(session_id, **kwargs):
    if (home / 'refuse-cleanup').exists() and kwargs.get('source') == 'session.interrupt':
        return {'status': 'error', 'error': 'fixture native termination refusal'}
    return kill(session_id, **kwargs)


ManagedWorker.next_frame = held_finished
process_registry.kill_process = refuse_cleanup
runpy.run_module('gateway.run', run_name='__main__')
