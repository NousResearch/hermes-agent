"""Launch the actual managed worker with a finite native cleanup uncertainty fixture."""
from pathlib import Path
import os
import shlex
import runpy
import subprocess
import sys

worker = str(Path(__file__).with_name('managed_result_cleanup_worker.py'))
original = subprocess.Popen
home = Path(os.environ['HERMES_HOME'])
from gateway.session_managed_worker import ManagedWorker
send = ManagedWorker.send


def bind_foreign_job(self, frame):
    if 'scope' in frame and 'route' in frame and not (home / 'foreign-job.json').exists():
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.process_registry import process_registry
        from utils import atomic_json_write
        tokens = set_session_vars(session_id='foreign-parent', session_key=frame['route'])
        try:
            foreign = process_registry.spawn_local('exec ' + shlex.join([sys.executable, '-c',
                'import threading; threading.Event().wait(90)']), cwd=str(home),
                task_id=frame['scope']['session_id'], owner_task_id=frame['scope']['session_id'],
                session_key=frame['route'], persist_on_release=True)
        finally:
            clear_session_vars(tokens)
        atomic_json_write(home / 'foreign-job.json', dict(id=foreign.id, pid=foreign.pid,
            host_start_time=foreign.host_start_time, parent_session_id=foreign.parent_session_id,
            owner_task_id=foreign.owner_task_id, session_key=foreign.session_key))
    return send(self, frame)


ManagedWorker.send = bind_foreign_job


class Child(original):
    def __init__(self, args, *positional, **kwargs):
        if args == [sys.executable, '-m', 'agent.managed_worker']:
            args = [sys.executable, worker]
        super().__init__(args, *positional, **kwargs)


subprocess.Popen = Child
runpy.run_module('gateway.run', run_name='__main__')
