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
    sys.addaudithook(lambda event, args: opens.append(str(args[0]))
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
        negatives = {}
        for label, altered in [('epoch', dict(store.scope, epoch=scope['epoch'] - 1)),
                               ('foreign', dict(store.scope, session_id=command['foreign']))]:
            try:
                rpc('worker.persist', **altered, sequence=store.journal['next_sequence'],
                    operation='compression.watermark', payload={})
                raise AssertionError(label + ' accepted')
            except Exception as exc:
                negatives[label] = str(exc)
        assert 'stale_epoch' in negatives['epoch'] and 'permission_denied' in negatives['foreign'], negatives
        assert store.try_acquire_compression_lock(agent.session_id, 'live-holder')
        assert not store.refresh_compression_lock(agent.session_id, 'stale-holder')
        store.release_compression_lock(agent.session_id, 'stale-holder')
        assert store.get_compression_lock_holder(agent.session_id) == 'live-holder'
        store.release_compression_lock(agent.session_id, 'live-holder')
        parent = agent.session_id
        assert store.try_acquire_compression_lock(parent, 'rotation-holder')
        watermark = store.get_active_message_watermark(parent)
        store.append_messages_batch(parent, [{'role': 'user', 'content': 'CONCURRENT_TAIL',
            'api_content': ' exact wire tail ', 'display_metadata': {'_accepted_input_id': 'tail-delivery'}}])
        def lost_ack(method, **params):
            rpc(method, **params)
            raise TimeoutError('lost-ack')
        store.rpc = lost_ack
        try:
            store.publish_compression_child(parent_session_id=parent, child_session_id='worker-rotated-child',
                source='cli', messages=[{'role': 'assistant', 'content': 'ROTATED_SUMMARY'}],
                system_prompt=prompt, compression_lock_holder='rotation-holder', watermark=watermark)
            raise AssertionError('expected lost ack')
        except TimeoutError:
            pass
        store.rpc = rpc
        store.retry_pending()
        assert store.scope['session_id'] == 'worker-rotated-child'
        history = store.get_messages_as_conversation('worker-rotated-child', include_row_ids=True)
        assert [m['content'] for m in history] == ['ROTATED_SUMMARY', 'CONCURRENT_TAIL']
        assert history[-1]['api_content'] == ' exact wire tail '
        assert history[-1]['display_metadata'] == {'_accepted_input_id': 'tail-delivery'}
        assert store.get_compression_lineage('worker-rotated-child')[0] == before
        finished = store.finish()
        proof = dict(before=before, after=store.scope['session_id'], prompt=prompt, history=history,
                     compressed_count=len(compressed), input_count=len(messages), opens=opens,
                     fds=writable_fds(), negatives=negatives, finished=finished)
        Path(command['receipt']).write_text(json.dumps(proof))
    finally:
        agent._end_session_on_close = False
        agent.close()
        store.close()


if __name__ == '__main__':
    main()
