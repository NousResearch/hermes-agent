"""Messaging ``/refine`` on a session whose cached agent is gone.

The gateway drops a session's cached agent after an idle hour, under memory pressure, around a
``/moa`` turn, and on restart; the conversation lives on in the session's stored transcript. A
messaging ``/refine`` read only the cached agent and answered "Nothing to refine yet — send a message
first." It now refines the stored conversation, sends the summary into the same chat, and its review
request carries what a resident agent's carries. Both run as the user's own request (``explicit``), so a
memory removal is applied, not staged as an unattended proposal, as on the CLI, TUI and local sessions.

Real ``python -m gateway.run``, a real user-plugin messaging platform (the chaos fake), a loopback
model that answers the review with a ``memory`` tool call. A gateway restart drops the cached agent
here because the idle sweep's first pass is a minute in.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.e2e.core.chaos._gateway_harness import GatewayProc

REVIEW_MARK = 'Review the conversation above'


def _is_review(body):
    return any(REVIEW_MARK in str(m.get('content') or '') for m in body.get('messages', []))


class _Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        messages = body.get('messages', [])
        if _is_review(body) and messages[-1].get('role') != 'tool':
            # A user-asked review may delete memory (an unattended one only stages the removal).
            stale = 'COLD_FACT' if 'cold pass' in json.dumps(messages) else 'WARM_FACT'
            call = {'id': f'call_{len(self.server.requests)}', 'type': 'function', 'function': {
                'name': 'memory', 'arguments': json.dumps({'action': 'remove', 'target': 'memory',
                                                           'old_text': stale})}}
            message, finish = {'role': 'assistant', 'content': None, 'tool_calls': [call]}, 'tool_calls'
        else:
            text = next((m.get('content') or '' for m in reversed(messages) if m['role'] == 'user'), '')
            message, finish = {'role': 'assistant', 'content': 'ACK_' + str(text)[:40]}, 'stop'
        if body.get('stream'):
            delta = dict(message)
            if message.get('tool_calls'):
                delta['tool_calls'] = [dict(c, index=0) for c in message['tool_calls']]
            chunk = {'id': 'l', 'object': 'chat.completion.chunk', 'model': body.get('model'),
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            payload, kind = f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode(), 'text/event-stream'
        else:
            payload, kind = json.dumps({'id': 'l', 'object': 'chat.completion', 'model': body.get('model'),
                                        'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1,
                                                  'total_tokens': 2}}).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _gateway(root, url, **overrides):
    gw = GatewayProc(root=root, base_url=url, tag=f'refine-cold-{root.name}')
    config = {
        '_config_version': 52,
        'model': {'provider': 'openrouter', 'default': 'openai/gpt-5', 'base_url': url,
                  'api_key': 'loopback-only', 'api_mode': 'chat_completions'},
        'provider_routing': {'only': ['probe-route'], 'data_collection': 'deny'},
        'prefill_messages_file': 'prefill.json',
        'agent': {'api_max_retries': 0},
        'auxiliary': {'title_generation': {'enabled': False}},
        'compression': {'enabled': False},
        'memory': {'memory_enabled': True, 'user_profile_enabled': False},
        'updates': {'check': False},
        'plugins': {'enabled': ['chaos-fake']},
        'platforms': {'chaosfake': {'enabled': True}},
        'platform_toolsets': {'chaosfake': ['terminal', 'memory']},
        'terminal': {'backend': 'local'},
        **overrides,
    }
    (gw.hermes_home / 'config.yaml').write_text(json.dumps(config))
    (gw.hermes_home / 'prefill.json').write_text(json.dumps([
        {'role': 'user', 'content': 'PREFILL_EXAMPLE'}, {'role': 'assistant', 'content': 'Noted.'}]))
    (gw.hermes_home / '.env').write_text('OPENROUTER_API_KEY=loopback-only\n')
    memories = gw.hermes_home / 'memories'
    if not memories.exists():
        memories.mkdir()
        (memories / 'MEMORY.md').write_text('WARM_FACT\n§\nCOLD_FACT\n§\nKEEP_FACT')
    return gw


@pytest.mark.platforms("linux")  # the chaos harness reaps by POSIX signals
def test_messaging_refine_reviews_the_stored_conversation_once_the_cached_agent_is_gone(tmp_path):
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _Model)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    root = Path(tmp_path) / 'gw'
    root.mkdir()

    def refine(gw, focus):
        mark, before = gw.mark(), len(peer.requests)
        gw.send_user('/refine ' + focus)
        summary = gw.wait_for(lambda e: e.ev == 'send' and '💾' in e.data.get('text', ''), 60,
                              f'{focus} summary', since=mark)
        review = next((r for r in peer.requests[before:] if _is_review(r)), None)
        assert summary is not None and review is not None, (focus, gw.sends_since(mark), gw.log_tail())
        text = json.dumps(review['messages'])
        return {'chat': summary.data['chat'], 'model': review.get('model'), 'provider': review.get('provider'),
                'session_id': review.get('session_id'), 'prefill': 'PREFILL_EXAMPLE' in text,
                'saw_conversation': 'OPENED_PR_42' in text,
                'tools': sorted(t['function']['name'] for t in review.get('tools') or [])}

    out = {}
    try:
        warm = _gateway(root, url)
        out['memory_file'] = warm.hermes_home / 'memories' / 'MEMORY.md'
        try:
            warm.start()
            mark = warm.mark()
            warm.send_user('OPENED_PR_42')
            assert warm.wait_for(lambda e: e.ev == 'send' and 'ACK_OPENED_PR_42' in e.data.get('text', ''),
                                 60, 'turn reply', since=mark), warm.log_tail()
            out['warm'] = refine(warm, 'resident pass')
        finally:
            warm.stop()
        cold = _gateway(root, url)  # same home and state.db; the restarted gateway holds no agent
        try:
            cold.start()
            out['cold'] = refine(cold, 'cold pass')
        finally:
            cold.stop()
    finally:
        peer.shutdown()
        peer.server_close()

    assert out['cold']['saw_conversation'] and out['cold']['chat'] == 'chaos-chat', out
    # Each /refine applied its removal as the user's own request (both surfaces pass explicit=True).
    memory = out['memory_file'].read_text()
    assert 'WARM_FACT' not in memory and 'COLD_FACT' not in memory and 'KEEP_FACT' in memory, memory
    assert out['warm']['provider'] == {'only': ['probe-route'], 'data_collection': 'deny'}, out
    assert out['cold'] == out['warm'], ('a cold /refine must send what a resident one sends', out)



@pytest.mark.platforms("linux")  # the chaos harness reaps by POSIX signals
def test_bare_refine_runs_while_automatic_reviews_are_disabled(tmp_path):
    """``auxiliary.background_review.enabled: false`` stops only the automatic post-turn fork (#87250):
    a bare ``/refine`` (no focus) on a resident agent and on a restarted gateway's stored conversation
    still reviews and reports (#100762)."""
    import time
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _Model)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    root = Path(tmp_path) / 'gw'
    root.mkdir()
    # nudge_interval 1 makes every turn due an automatic memory review, so the switch is what stops it.
    overrides = {'auxiliary': {'title_generation': {'enabled': False}, 'background_review': {'enabled': False}},
                 'memory': {'memory_enabled': True, 'user_profile_enabled': False, 'nudge_interval': 1}}
    seen = {}

    def bare_refine(gw, label):
        mark, before = gw.mark(), len(peer.requests)
        gw.send_user('/refine')
        summary = gw.wait_for(lambda e: e.ev == 'send' and '💾' in e.data.get('text', ''), 60,
                              f'{label} bare /refine summary', since=mark)
        seen[label] = (summary is not None, sum(map(_is_review, peer.requests[before:])), gw.sends_since(mark))

    try:
        warm = _gateway(root, url, **overrides)
        try:
            warm.start()
            mark = warm.mark()
            warm.send_user('OPENED_PR_42')
            assert warm.wait_for(lambda e: e.ev == 'send' and 'ACK_OPENED_PR_42' in e.data.get('text', ''),
                                 60, 'turn reply', since=mark), warm.log_tail()
            # health: allow HX013 -- an absence window: a review that never spawns sets no event to wait on
            time.sleep(5)  # with enabled: true the same turn's automatic review reaches the model well inside this
            seen['automatic'] = sum(map(_is_review, peer.requests))
            bare_refine(warm, 'warm')
        finally:
            warm.stop()
        # The loopback review removes WARM_FACT again; put it back so the cold removal has a target.
        (warm.hermes_home / 'memories' / 'MEMORY.md').write_text('WARM_FACT\n§\nKEEP_FACT')
        cold = _gateway(root, url, **overrides)  # restarted: no cached agent, the stored conversation
        try:
            cold.start()
            bare_refine(cold, 'cold')
        finally:
            cold.stop()
    finally:
        peer.shutdown()
        peer.server_close()

    assert seen['automatic'] == 0, ('enabled: false must stop the automatic review', seen)
    assert all(seen[leg][0] and seen[leg][1] for leg in ('warm', 'cold')), seen
