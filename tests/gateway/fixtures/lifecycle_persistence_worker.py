"""Lifecycle extension of the existing separate-interpreter worker fixture."""
import json
import os
from pathlib import Path
import sys
import time

import psutil

from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
from tests.gateway.fixtures.agent_persistence_worker import writable_fds


def main():
    command = json.loads(sys.stdin.readline())
    home, sid = Path(command['home']), command['session_id']
    opens = []
    sys.addaudithook(lambda event, args: opens.append(str(args[0]))
                    if event == 'sqlite3.connect' and 'state.db' in str(args[0]) else None)
    rpc = WorkerRPC(home)
    scope = dict(profile_id=str(home), session_id=sid, execution_id='lifecycle-worker', generation=0,
                 pid=os.getpid(), birth=psutil.Process().create_time(), secret='private-lifecycle-worker')
    scope['epoch'] = rpc('worker.register', **scope, kind='compute')['owner_epoch']
    store = RuntimeSessionStore(rpc, scope, home / 'worker-outboxes' / 'lifecycle-worker')
    from run_agent import AIAgent
    agent = AIAgent(model='worker-model', provider='custom', base_url=command['url'], api_key='loopback-only',
                    session_db=store, session_id=sid, enabled_toolsets=[], skip_context_files=True,
                    skip_memory=True, skip_background_review=True, quiet_mode=True)
    try:
        result = agent.run_conversation('WORKER_INFERENCE')
        assert agent._session_db_created, 'constructor did not persist'
        assert not result.get('failed'), result
        context = store.get_session(sid)
        assert store.set_session_title(sid, 'worker-title')
        assert not store.set_auto_title(sid, 'late-generated', source='llm')
        assert store.get_session_title_source(sid) == 'user'
        store.touch_session_activity(sid, time.time() + 1, description='worker-active')
        store.clear_session_activity_labels(sid)
        assert store.set_latest_user_api_content(sid, 'WORKER_INFERENCE', 'WIRE_WORKER_INFERENCE') == 1
        store.update_session_billing_route(sid, provider='fixture-switched', base_url=command['url'])
        lifecycle = store.session_lifecycle_statuses([sid])
        store.end_session(sid, 'agent_close')
        finished = store.finish()
        foreign = {}
        for operation, payload in [('session.end', {'end_reason': 'late'}),
                                   ('session.title', {'title': 'foreign-title', 'source': 'user'}),
                                   ('session.activity_clear', {})]:
            try:
                rpc('worker.persist', **dict(scope, session_id=command['foreign']),
                    sequence=store.journal['next_sequence'], operation=operation, payload=payload)
            except Exception as exc:
                foreign[operation] = str(exc)
        ready = dict(result=result['final_response'], context=context, finished=finished,
                     lifecycle=lifecycle, foreign=foreign, opens=opens, fds=writable_fds())
        Path(command['ready']).write_text(json.dumps(ready, default=str))
        # Parent starts a genuine later owner turn before releasing this callback.
        assert sys.stdin.readline().strip() == 'successor-settled'
        late = {}
        for operation, payload in [('session.end', {'end_reason': 'late'}),
                                   ('session.title', {'title': 'late-title', 'source': 'user'}),
                                   ('session.activity_clear', {}),
                                   ('session.billing_route', {'provider': 'late', 'base_url': 'late', 'billing_mode': None})]:
            try:
                rpc('worker.persist', **scope, sequence=store.journal['next_sequence'],
                    operation=operation, payload=payload)
            except Exception as exc:
                late[operation] = str(exc)
        ready.update(late=late, opens=opens, fds=writable_fds())
        Path(command['receipt']).write_text(json.dumps(ready, default=str))
    finally:
        agent._end_session_on_close = False
        agent.close()
        store.close()


if __name__ == '__main__':
    main()
