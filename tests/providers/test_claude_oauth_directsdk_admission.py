"""One upstream admission and first-response authority over native recovery."""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'plugins/model-providers/claude-oauth-directsdk'))
import directsdk

NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame=json.loads(line)
    if frame.get('type') != 'user':
        continue
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}),flush=True)
        continue
    break
url=os.environ['ANTHROPIC_BASE_URL']+'/v1/messages'
for _ in range(2):
    try:
        urllib.request.urlopen(urllib.request.Request(url,data=b'{}',headers={'Content-Type':'application/json'}),timeout=5).read()
    except urllib.error.HTTPError:
        break
print(json.dumps({'type':'assistant','message':{'id':'first','role':'assistant','content':[{'type':'text','text':'FIRST'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''


@pytest.mark.parametrize('stop,partial_tool', [('end_turn', False), ('max_tokens', False), ('model_context_window_exceeded', False), ('max_tokens', True)])
def test_first_response_owns_usage_and_stops_recovery(tmp_path, stop, partial_tool):
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'FIRST'}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':stop},'usage':usage},
                {'type':'message_stop'},
            ]
            if partial_tool:
                events[4:4] = [
                    {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': 'cut', 'name': 'mcp__hermes__read_file', 'input': {}}},
                    {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': '{"path":"'}},
                    {'type': 'content_block_stop', 'index': 1},
                ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}], tools=[{'type': 'function', 'function': {'name': 'read_file', 'description': 'Fixture', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}])
        assert len(calls)==1
        assert result.choices[0].message.content=='FIRST'
        assert result.choices[0].finish_reason==('stop' if stop=='end_turn' else ('model_context_window_exceeded' if stop=='model_context_window_exceeded' else 'length'))
        assert result.usage.prompt_tokens==0
        if partial_tool:
            assert result.choices[0].message.tool_calls[0].function.arguments == '{"path":"'
            assert not result.choices[0].message.reasoning_details
        else:
            assert result.choices[0].message.reasoning_details[0]['messages'][0]['stop_reason']==stop
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_cancel_closes_the_active_upstream_socket(tmp_path):
    entered, disconnected = threading.Event(), threading.Event()
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            self.rfile.read(1)
            disconnected.set()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)], env={'PATH':os.defpath, 'HOME':str(tmp_path), 'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.create, model='sonnet', messages=[{'role':'user', 'content':'fixture'}])
            try:
                assert entered.wait(5)
            finally:
                client.cancel()
            with pytest.raises(RuntimeError, match='cancelled'):
                result.result(timeout=3)
            assert disconnected.wait(2)
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


@pytest.mark.parametrize('response_kind,continuation', [
    (kind, action) for kind in ('text', 'tool', 'context')
    for action in ('allow', 'budget_denied', 'cancelled')
] + [('context', 'redirect')])
def test_ordinary_hermes_loop_owns_continuation(tmp_path, monkeypatch, continuation, response_kind):
    """Real host loop + real relay; only the native process and HTTPS peer are fixtures."""
    from unittest.mock import patch
    from run_agent import AIAgent
    from agent.iteration_budget import IterationBudget

    partial_tool = response_kind == 'tool'
    context_overflow = response_kind == 'context'
    calls, order, receipts, requests = [], [], [], []
    compressions, saved, refunds = [], [], []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            calls.append(self.path)
            order.append('upstream')
            part = 'Part 1 ' if len(calls) == 1 else 'Part 2'
            stop = 'max_tokens' if len(calls) == 1 else 'end_turn'
            if context_overflow and len(calls) == 1:
                stop = 'model_context_window_exceeded'
            usage = {'input_tokens': 10, 'output_tokens': 3}
            events = [
                {'type': 'message_start', 'message': {'id': f'msg_{len(calls)}', 'role': 'assistant', 'model': 'claude-sonnet-4-6', 'content': [], 'usage': usage}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': part}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': stop}, 'usage': usage},
                {'type': 'message_stop'},
            ]
            if partial_tool and len(calls) == 1:
                events[4:4] = [
                    {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': 'cut', 'name': 'mcp__hermes__fixture_read', 'input': {}}},
                    {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': '{"path":"'}},
                    {'type': 'content_block_stop', 'index': 1},
                ]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    native_env = {
        'PATH': os.defpath, 'HOME': str(tmp_path),
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}',
    }
    clients = []

    class RecordingClient(directsdk.Client):
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            result = super().create(**kwargs)
            if not kwargs.get('stream'):
                receipts.append(result.usage.model_dump()['native_admission'])
                return result

            def stream():
                try:
                    for chunk in result:
                        if hasattr(chunk, '_response'):
                            receipts.append(chunk._response.usage.model_dump()['native_admission'])
                        yield chunk
                finally:
                    result.close()
            return stream()

    def fixture_client(**_):
        client = RecordingClient(command=[sys.executable, str(native)], env=native_env)
        clients.append(client)
        return client

    from providers import get_provider_profile
    monkeypatch.setattr(get_provider_profile('claude-oauth-directsdk'), 'create_client', fixture_client)
    try:
        definitions = [{'type': 'function', 'function': {'name': 'fixture_read', 'description': 'Fixture only', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}] if partial_tool else []
        with patch('model_tools.get_tool_definitions', return_value=definitions), patch('model_tools.check_toolset_requirements', return_value={}):
            agent = AIAgent(provider='claude-oauth-directsdk', model='claude-sonnet-4-6',
                            api_key='external-process', base_url='process://claude-oauth-directsdk',
                            quiet_mode=True, skip_context_files=True, skip_memory=True,
                            save_trajectories=False, max_iterations=4)
        agent._cached_system_prompt = 'You are helpful.'
        agent.compression_enabled = False
        # Exercise the host's distinct truncated-tool path; streaming recovery
        # may discard the malformed call and select plain-text continuation.
        agent._disable_streaming = partial_tool
        history = None
        if context_overflow:
            agent.compression_enabled = True
            history = [{'role': 'user', 'content': 'old fixture ' * 1000},
                       {'role': 'assistant', 'content': 'old answer'}]

            def compress(messages, *args, **kwargs):
                compressions.append(copy.deepcopy(messages))
                order.append('compress')
                # Verify routing, not the already-existing summarizer algorithm.
                return copy.deepcopy(messages[2:]), 'compressed fixture system'

            monkeypatch.setattr(agent, '_compress_context', compress)
            monkeypatch.setattr(agent, '_persist_session', lambda messages, *_: saved.append(copy.deepcopy(messages)))
        agent.step_callback = lambda *_: order.append('step')
        consume = IterationBudget.consume
        refund = IterationBudget.refund

        def refunded(budget):
            refunds.append(budget.used)
            return refund(budget)

        monkeypatch.setattr(IterationBudget, 'refund', refunded)

        def admitted(budget):
            if continuation == 'budget_denied' and budget.used == 1:
                # Race-shaped hard denial at the atomic consume boundary, not the
                # separate, intentionally admitted budget-exhaustion grace call.
                while consume(budget):
                    pass
            result = consume(budget)
            order.append('budget' if result else 'denied')
            return result

        monkeypatch.setattr(IterationBudget, 'consume', admitted)

        def hook(name, **_):
            if name not in ('pre_api_request', 'post_api_request'):
                return []
            order.append(name)
            if name == 'post_api_request' and len(calls) == 1:
                if continuation == 'cancelled':
                    agent.interrupt()
                elif continuation == 'redirect':
                    # A redirect admitted while the model was active can remain
                    # queued when its completed response reaches this hook.
                    agent._pending_redirect = 'Use the corrected fixture instruction.'
                    agent._interrupt_requested = True

        monkeypatch.setattr('hermes_cli.lifecycle.has_hook', lambda name: name in ('pre_api_request', 'post_api_request'))
        monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', hook)
        with patch.object(agent, '_cleanup_task_resources'):
            result = agent.run_conversation('Complete the fixture response.', conversation_history=history)

        expected = 2 if continuation in ('allow', 'redirect') else 1
        if context_overflow:
            assert len(compressions) == (0 if continuation in ('cancelled', 'redirect') else 1)
            assert any(any(m.get('content') == 'Part 1 ' for m in rows) for rows in saved)
        assert refunds == []
        assert len(calls) == len(requests) == len(receipts) == expected, (
            order, result.get('error'), (result.get('final_response') or '')[:300]
        )
        assert all(row['upstream_requests'] == 1 and row['blocked_requests'] == 1 for row in receipts)
        expected_order = ['budget', 'step', 'pre_api_request', 'upstream', 'post_api_request'] * expected
        if context_overflow and continuation not in ('cancelled', 'redirect'):
            expected_order.insert(5, 'compress')
        if continuation == 'budget_denied':
            expected_order.append('denied')
        assert order == expected_order, order
        assert all(request['model'] == 'claude-sonnet-4-6' for request in requests)
        assert agent.session_api_calls == expected
        assert agent.session_output_tokens == expected * 3
        assert not any(m.get('tool_calls') or m.get('role') == 'tool' for m in result['messages'])
        if continuation in ('allow', 'redirect'):
            assert result['completed'] is True
            assert result['final_response'] == ('Part 2' if partial_tool or context_overflow else 'Part 1 Part 2')
            assert result['api_calls'] == 2
            if partial_tool or context_overflow:
                assert sum(m.get('content') == 'Part 1 ' for m in result['messages']) == 1
                assert not any(m.get('tool_calls') for m in requests[1]['messages'])
                if context_overflow and continuation != 'redirect':
                    assert not any('old fixture' in (m.get('content') or '') for m in requests[1]['messages'])
                    assert not any('truncated by the output length limit' in (m.get('content') or '') for m in requests[1]['messages'])
            else:
                assert requests[1]['messages'][-1]['role'] == 'user'
                assert 'truncated by the output length limit' in requests[1]['messages'][-1]['content']
        elif continuation == 'cancelled':
            assert result['interrupted'] is True
        else:
            assert result['completed'] is False
            # Hermes may append its existing iteration-limit explanation; the
            # partial answer must survive once, still explicitly incomplete.
            assert result['final_response'].startswith('Part 1')
            assert sum((m.get('content') or '').startswith('Part 1') for m in result['messages']) == 1
            assert not any(m.get('_length_continuation_nudge') for m in result['messages'])
    finally:
        for client in clients:
            client.close()
        peer.shutdown()
        thread.join()
        peer.server_close()
