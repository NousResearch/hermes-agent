"""Complete a real job and model turn, then withhold only its original ownership record."""
import json
import os
from pathlib import Path
import shlex
import sys

from agent import managed_worker as worker
from gateway import session_kanban
from gateway.session_context import clear_session_vars, set_session_vars
from tests.waiting import wait_until
from utils import atomic_json_write

home = Path(os.environ['HERMES_HOME'])
original_init = worker.WorkerControls.__init__
controls = []


def remember_controls(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    controls.append(self)


original_stop = worker.WorkerControls.stop_processes


def record_real_refusal(self):
    try:
        return original_stop(self)
    except RuntimeError as exc:
        if str(exc) == 'process_stop_unknown':
            (home / 'cleanup-refused').touch()
        raise


original_run = session_kanban.run_worker_turns


def run(agent, *args, **kwargs):
    if (home / 'known-result.json').exists():
        return original_run(agent, *args, **kwargs)
    from tools.process_registry import process_registry
    route = controls[0].route
    tokens = set_session_vars(session_id=agent.session_id, session_key=route)
    try:
        job = process_registry.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'print("physical original output",flush=True); raise SystemExit(7)']), cwd=str(home),
            task_id=agent.session_id, owner_task_id=agent.session_id, session_key=route)
    finally:
        clear_session_vars(tokens)
    assert job._completion_event.wait(10)
    result_path = home / 'logs' / 'process-results' / (job.id + '.json')
    (home / 'genuine-result.bin').write_bytes(result_path.read_bytes())
    atomic_json_write(home / 'jobs.json', dict(owned=dict(id=job.id, pid=job.pid,
        host_start_time=job.host_start_time, parent_session_id=job.parent_session_id,
        owner_task_id=job.owner_task_id, session_key=job.session_key),
        foreign=json.loads((home / 'foreign-job.json').read_text())))
    result = original_run(agent, *args, **kwargs)
    result_path.write_text('{unreadable native retained ownership')
    atomic_json_write(home / 'known-result.json', result)
    wait_until(lambda: controls[0].stop_processes_requested, timeout=20)
    return result


worker.WorkerControls.__init__ = remember_controls
worker.WorkerControls.stop_processes = record_real_refusal
session_kanban.run_worker_turns = run
raise SystemExit(worker.main())
