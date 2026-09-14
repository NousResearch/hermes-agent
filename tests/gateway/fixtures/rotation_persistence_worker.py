"""Registered worker invokes real AIAgent compression over an ordinary daemon."""
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
    import traceback
    sys.addaudithook(lambda event, args: opens.append({'path': str(args[0]), 'stack': traceback.format_stack(limit=18)})
                     if event == 'sqlite3.connect' and 'state.db' in str(args[0]) else None)
    rpc = WorkerRPC(home)
    scope = dict(profile_id=str(home), session_id=command['session_id'], execution_id='compression-worker',
                 generation=0, pid=os.getpid(), birth=psutil.Process().create_time(), secret='private-compression-worker')
    registered = rpc('worker.register', **scope, kind='compute')
    scope['epoch'] = registered['owner_epoch']
    store = RuntimeSessionStore(rpc, scope, home / 'worker-outboxes' / 'compression-worker')
    from run_agent import AIAgent
    agent = AIAgent(model='worker-model', provider='custom', base_url=command['url'], api_key='loopback-only',
        session_db=store, session_id=scope['session_id'], enabled_toolsets=[], skip_context_files=True,
        skip_memory=True, skip_background_review=True, quiet_mode=True)
    agent.compression_in_place = False
    frozen_tools = json.loads(json.dumps(agent.tools))
    try:
        # Real compression implementation; only inference is supplied by the loopback peer.
        messages = [{'role': 'system', 'content': 'BYTE_STABLE_PREFIX'}]
        for i in range(12):
            messages.extend([{'role': 'user', 'content': f'USER_{i} ' + 'remember this fact ' * 180},
                             {'role': 'assistant', 'content': f'ANSWER_{i} ' + 'acknowledged fact ' * 180}])
        store.update_system_prompt(scope['session_id'], 'BYTE_STABLE_PREFIX')
        agent._session_db_created = True
        before = agent.session_id
        compressed, prompt = agent._compress_context(messages, 'BYTE_STABLE_PREFIX', force=True, approx_tokens=30000)
        assert len(compressed) < len(messages), (len(compressed), len(messages), [
            dict(operation=e['operation'], keys=[list(m) for m in e['payload'].get('messages', [])])
            for e in store.journal['pending']])
        history = store.get_messages_as_conversation(agent.session_id, include_row_ids=True)
        assert history, 'compaction did not persist history'
        assert before != agent.session_id, 'actual AIAgent rotation did not publish'
        assert store.scope['session_id'] == agent.session_id
        assert agent.tools == frozen_tools
        restart_proof = {}
        if command.get('restart'):
            Path(command['ready']).write_text(json.dumps(dict(session_id=agent.session_id, epoch=store.scope['epoch'])))
            assert sys.stdin.readline().strip() == 'outage'
            try:
                store.touch_session_activity(agent.session_id, description='surviving rotated worker')
                raise AssertionError('owner outage accepted a mutation')
            except Exception:
                assert len(store.journal['pending']) == 1
            Path(command['outage']).write_text('pending')
            assert sys.stdin.readline().strip() == 'adopt'
            try:
                store.retry_pending()
                raise AssertionError('new owner accepted old epoch')
            except Exception as exc:
                assert 'stale_epoch' in str(exc), exc
            assigned = {k: v for k, v in store.scope.items() if k != 'epoch'}
            adopted = rpc('worker.adopt', **assigned)
            restart_proof = {'before_epoch': store.scope['epoch'], 'after_epoch': adopted['owner_epoch'],
                             'pid': os.getpid(), 'pending_sequence': store.journal['pending'][0]['sequence']}
            store.adopt(adopted['owner_epoch'])
            assert len(store.retry_pending()) == 1
        result = agent.run_conversation('AFTER_FULL_ROTATION', conversation_history=compressed)
        assert not result.get('failed'), result
        history = store.get_messages_as_conversation(agent.session_id, include_row_ids=True)
        store.finish()
        Path(command['receipt']).write_text(json.dumps(dict(before=before, after=agent.session_id,
            history=history, compressed_count=len(compressed), input_count=len(messages), opens=opens,
            fds=writable_fds(), negatives={}, result=result['final_response'], restart=restart_proof)))
    finally:
        agent._end_session_on_close = False
        agent.close()
        store.close()


if __name__ == '__main__':
    main()
