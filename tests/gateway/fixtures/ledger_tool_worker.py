"""Real registered worker with the production terminal tool enabled."""
import json
import os
from pathlib import Path
import sys

import psutil

from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
from tests.gateway.fixtures.agent_persistence_worker import writable_fds


def main():
    command = json.loads(sys.stdin.readline())
    home = Path(command['home'])
    opens = []
    def audit(event, args):
        if event == 'sqlite3.connect' and 'state.db' in str(args[0]):
            opens.append(str(args[0]))
    sys.addaudithook(audit)
    rpc = WorkerRPC(home)
    scope = dict(profile_id=str(home), session_id=command['session_id'], execution_id='tool-worker', generation=0,
                 pid=os.getpid(), birth=psutil.Process().create_time(), secret='private-tool-worker')
    registered = rpc('worker.register', **scope, kind='compute')
    scope['epoch'] = registered['owner_epoch']
    store = RuntimeSessionStore(rpc, scope, home / 'worker-outboxes' / 'tool-worker')
    from tools.async_delegation_worker import worker_ledger
    assert worker_ledger().store is store
    from run_agent import AIAgent
    agent = AIAgent(model='worker-model', provider='custom', base_url=command['url'], api_key='loopback-only',
        session_db=store, session_id=scope['session_id'], enabled_toolsets=['terminal'],
        skip_context_files=True, skip_memory=True, skip_background_review=True, quiet_mode=True)
    try:
        result = agent.run_conversation('Run the inert terminal marker, then report it.')
        finished = store.finish()
        Path(command['receipt']).write_text(json.dumps({'result': result['final_response'],
            'failed': result.get('failed'), 'opens': opens, 'fds': writable_fds(), 'finished': finished}))
    finally:
        agent._end_session_on_close = False
        agent.close()
        store.close()


if __name__ == '__main__':
    main()
