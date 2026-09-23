"""Real AIAgent in a separate interpreter, using only the worker transport."""
import json
import os
from pathlib import Path
import sys

import psutil

from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC


def writable_fds():
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


def main():
    command = json.loads(sys.stdin.readline())
    home = Path(command['home'])
    opens = []
    def audit(event, args):
        if event == 'sqlite3.connect' and 'state.db' in str(args[0]):
            import traceback
            opens.append({'path': str(args[0]), 'stack': traceback.format_stack(limit=12)})
    sys.addaudithook(audit)
    rpc = WorkerRPC(home)
    scope = dict(profile_id=str(home), session_id=command['session_id'],
                 execution_id='agent-worker', generation=0, pid=os.getpid(),
                 birth=psutil.Process().create_time(), secret='private-agent-worker')
    registered = rpc('worker.register', **scope, kind='compute')
    scope['epoch'] = registered['owner_epoch']
    store = RuntimeSessionStore(rpc, scope, home / 'worker-outboxes' / 'agent-worker')
    from run_agent import AIAgent
    agent = AIAgent(model='worker-model', provider='custom', base_url=command['url'], api_key='loopback-only',
                    session_db=store, session_id=command['session_id'], enabled_toolsets=[],
                    skip_context_files=True, skip_memory=True, skip_background_review=True, quiet_mode=True)
    try:
        refusals = {}
        if command.get('probe_constructors'):
            from concurrent.futures import ThreadPoolExecutor
            from hermes_state_registry import release_or_close
            from tools.delegate_tool import _open_child_session_db
            from tools.async_delegation import _connect
            from cron.scheduler import _open_cron_session_db, run_job
            operations = {'child': lambda: _open_child_session_db(agent),
                          'cron': lambda: _open_cron_session_db({'id': 'worker-probe'}),
                          'ledger': _connect}
            def attempt(name, fn):
                try:
                    opened = fn()
                    if opened is not None:
                        release_or_close(opened)
                    return name, 'unexpected_success'
                except Exception as exc:
                    return name, str(exc)
            with ThreadPoolExecutor(max_workers=1) as pool:
                refusals = dict(pool.submit(attempt, name, fn).result() for name, fn in operations.items())
            scripts = home / 'scripts'
            scripts.mkdir(exist_ok=True)
            (scripts / 'worker-probe.py').write_text("print('SCRIPT_ONLY_RESULT')\n")
            script_result = run_job({'id': 'script-probe', 'name': 'script-probe', 'no_agent': True,
                                     'script': 'worker-probe.py', 'prompt': '', 'deliver': 'local'})
            refusals['script_result'] = script_result
        result = agent.run_conversation('WORKER_INFERENCE')
        context = store.get_session(command['session_id'])
        store.flush_token_counts()
        finished = store.finish()
        receipt = {'result': result['final_response'], 'failed': result.get('failed'),
                   'context': context, 'finished': finished, 'opens': opens, 'fds': writable_fds(), 'refusals': refusals}
        Path(command['receipt']).write_text(json.dumps(receipt, default=str))
    finally:
        agent._end_session_on_close = False  # session lifecycle belongs to the authority
        agent.close()
        store.close()


if __name__ == '__main__':
    main()
