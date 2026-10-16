"""Ordinary managed runs register the session's shell hooks and outbound webhooks in the worker,
where tools and model calls actually run (unsupportedpastels §2.4); bypass sessions register none.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from tests.gateway.test_managed_worker_construct import run_pair


class Receiver(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.server.deliveries.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        self.send_response(204)
        self.end_headers()


@pytest.mark.platforms("linux")
def test_managed_turn_fires_shell_hooks_and_webhooks_in_the_worker(tmp_path):
    receiver = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    receiver.deliveries = []
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    fired = tmp_path / 'state' / 'hook-fired.log'

    def extra(home):
        # No auto-accept: consent comes from the session's own `--accept-hooks` launch flag.
        return {'hooks': {'pre_llm_call': [{'command': f'sh -c "echo $PPID >> {fired}"'}],
                          'outbound': [{'url': f'http://127.0.0.1:{receiver.server_port}/h', 'events': ['pre_llm_call']}]}}

    async def turn(ws, sid, name, settle, servers):
        before_fires = fired.read_text().split() if fired.exists() else []
        before_hooks = len(receiver.deliveries)
        assert (await settle(ws, sid, name + '-plain', 'PLAIN_TURN'))['final_response'] == 'PRIMARY_OK'
        after = fired.read_text().split() if fired.exists() else []
        return {'shell_pids': after[len(before_fires):],
                'webhooks': [d['hook_event_name'] for d in receiver.deliveries[before_hooks:]]}
    try:
        results = run_pair(tmp_path, extra, turn, accept_hooks=True)
    finally:
        receiver.shutdown()
        receiver.server_close()
    for name in ('inproc', 'managed'):
        assert len(results[name]['shell_pids']) == 1, results
        assert results[name]['webhooks'] and set(results[name]['webhooks']) == {'pre_llm_call'}, results
    # The managed turn's hook ran in its own worker interpreter, not the owner.
    assert results['managed']['shell_pids'] != results['inproc']['shell_pids'], results


def test_worker_hook_registration_follows_the_session_policy(tmp_path, monkeypatch):
    from gateway.session_local import _bypass_policy
    from gateway.session_policy import build_policy, register_worker_hooks
    calls = []
    monkeypatch.setattr('agent.shell_hooks.register_from_config',
                        lambda cfg, accept_hooks=False: calls.append(('shell', cfg['hooks'], accept_hooks)))
    monkeypatch.setattr('agent.outbound_webhooks.register_from_config',
                        lambda cfg: calls.append(('outbound', cfg['hooks'])))
    hooks = {'pre_llm_call': [{'command': 'true'}]}
    launch = {'cwd': str(tmp_path), 'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}
    register_worker_hooks(build_policy(launch, {'hooks': hooks}))
    register_worker_hooks(build_policy({**launch, 'accept_hooks': True}, {'hooks': hooks}))
    kanban = build_policy(launch, {'hooks': hooks})
    from dataclasses import replace
    register_worker_hooks(replace(kanban, kanban_json=json.dumps({'accept_hooks': True})))
    assert calls == [('shell', hooks, False), ('outbound', hooks), ('shell', hooks, True), ('outbound', hooks),
                     ('shell', hooks, True), ('outbound', hooks)]
    calls.clear()
    for bypass in ({'ignore_user_config': True}, {'safe_mode': True}):
        register_worker_hooks(_bypass_policy({**launch, **bypass, 'accept_hooks': True}, private_secrets={}))
    assert calls == []
