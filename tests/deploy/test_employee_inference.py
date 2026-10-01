import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
from deploy.railway import codex_inference
from deploy.railway.hindsight.start import environment
from deploy.railway.hindsight.bank_reconciler import load_policy


def test_codex_endpoint_refreshes_once_and_keeps_structured_output(monkeypatch):
    calls, refreshes = [], []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append((self.headers['Authorization'], json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            if len(calls) == 1:
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            event = {'type': 'response.completed', 'response': {'id': 'test', 'output': [{'type':'message','content':[{'type':'output_text','text':'{"answer":"done"}'}]}]}}
            self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def credentials(force_refresh=False):
        refreshes.append(force_refresh)
        return {'api_key': 'fresh' if force_refresh else 'expired', 'base_url':f'http://127.0.0.1:{server.server_port}'}
    monkeypatch.setattr(codex_inference, 'resolve_codex_runtime_credentials', credentials)
    monkeypatch.setenv('HINDSIGHT_INFERENCE_KEY', 'private-test-key')
    try:
        with TestClient(codex_inference.app) as client:
            body = {'model':'gpt-5.6-luna', 'input':[{'role':'system','content':'Extract facts'},{'role':'user','content':'A decision'}],
                    'text':{'format':{'type':'json_schema','name':'answer','schema':{'type':'object'}}}}
            assert client.post('/v1/responses', json=body).status_code == 401
            result = client.post('/v1/responses', json=body, headers={'Authorization':'Bearer private-test-key'})
            assert result.status_code == 200
            assert result.json()['output'][0]['content'][0]['text'] == '{"answer":"done"}'
            assert refreshes == [False, True]
            sent = calls[-1][1]
            assert sent['instructions'].strip() == 'Extract facts'
            assert sent['input'] == body['input'][1:]
            assert sent['text'] == body['text']
            assert sent['store'] is False and sent['stream'] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hindsight_preserves_snapshot_except_declared_substitutions():
    snapshot = json.loads((Path(__file__).resolve().parents[2] / 'docs/specs/reference/hindsight-config.json').read_text(encoding='utf-8-sig'))
    result = environment(snapshot, {'DATABASE_URL':'postgresql://private/db','HINDSIGHT_API_KEY':'tenant',
                                   'HINDSIGHT_INFERENCE_KEY':'inference', 'OPENROUTER_API_KEY':'router'})
    substitutions = {'HINDSIGHT_API_LLM_BASE_URL', 'HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL', 'HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL'}
    for key, value in snapshot['environment'].items():
        if key not in substitutions:
            assert result[key] == (json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)
    policy = load_policy(result)
    assert policy.bank == snapshot['environment']['HINDSIGHT_API_DEFAULT_BANK_TEMPLATE']['bank']
    assert result['HINDSIGHT_API_LLM_API_KEY'] != result['HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY']


def test_bootstrap_hindsight_secret_is_available_only_to_its_profile(tmp_path, monkeypatch):
    from deploy.railway.bootstrap import seed_service_secrets
    from gateway.run import _profile_runtime_scope
    from agent.secret_scope import set_multiplex_active
    from plugins.memory.hindsight.employee import config
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir(); second.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(first))
    monkeypatch.setenv('HINDSIGHT_API_KEY', 'deployment-owned-test-key')
    seed_service_secrets()
    set_multiplex_active(True)
    try:
        for home, expected in ((first, 'deployment-owned-test-key'), (second, ''), (first, 'deployment-owned-test-key')):
            with _profile_runtime_scope(home):
                assert config()['api_key'] == expected
    finally:
        set_multiplex_active(False)


def test_hindsight_supervisor_applies_changes_once_and_keeps_child_during_outage(monkeypatch, tmp_path):
    """Drive service polls without launching Hindsight or touching a real account."""
    import io
    import signal
    import urllib.request
    from deploy.railway.hindsight import start
    snapshot = Path(__file__).resolve().parents[2] / 'docs/specs/reference/hindsight-config.json'
    (tmp_path / 'hindsight-config.json').write_bytes(snapshot.read_bytes())
    monkeypatch.setattr(start, '__file__', str(tmp_path / 'start.py'))
    children, acks, handlers = [], [], {}
    polls = 0
    stopped = False
    monkeypatch.setenv('DATABASE_URL', 'postgresql://test/db')
    monkeypatch.setenv('HINDSIGHT_API_KEY', 'tenant')
    monkeypatch.setenv('HINDSIGHT_INFERENCE_KEY', 'inference')

    class Child:
        def __init__(self, args, env):
            self.args, self.env, self.terminated = args, env, False
            children.append(self)
        def poll(self):
            return 0 if self.terminated or self.args[0] != 'hindsight-api' else None
        def terminate(self):
            self.terminated = True
        def wait(self, **kwargs):
            return 0

    class Response(io.BytesIO):
        status = 200

    def request(req, **kwargs):
        nonlocal polls
        url = req if isinstance(req, str) else req.full_url
        if url.endswith('/health'):
            assert url == 'http://[::1]:8888/health'
            return Response(b'{}')
        assert req.headers['Authorization'] == 'Bearer inference'
        if url.endswith('/settings/status'):
            acks.append(json.loads(req.data))
            return Response(b'{}')
        polls += 1
        if polls == 2:
            assert len(children) == 2 and not children[0].terminated
            raise OSError('temporary control outage')
        if polls == 4:
            assert len(children) == 2  # unchanged revision does not restart
        revision = 'first' if polls < 4 else 'second'
        return Response(json.dumps({'revision': revision, 'llm_model': 'gpt-5.6-luna',
            'llm_reasoning_effort': 'low' if revision == 'first' else 'high',
            'reflect_llm_reasoning_effort': 'medium', 'openrouter_api_key': revision}).encode())

    def sleep(_):
        nonlocal stopped
        if polls >= 4 and not stopped:
            stopped = True
            handlers[signal.SIGTERM]()

    monkeypatch.setattr(start.subprocess, 'Popen', Child)
    monkeypatch.setattr(urllib.request, 'urlopen', request)
    monkeypatch.setattr(signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr('time.sleep', sleep)
    start.main()
    assert len(children) == 4
    assert all(child.poll() == 0 for child in children)
    assert children[2].env['HINDSIGHT_API_LLM_REASONING_EFFORT'] == 'high'
    assert children[2].env['HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY'] == 'second'
    assert children[2].env['HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY'] == 'second'
    assert [ack['revision'] for ack in acks] == ['first', 'first', 'second']


def test_bootstrap_openrouter_seed_preserves_dashboard_rotation(tmp_path, monkeypatch):
    from deploy.railway.bootstrap import seed_service_secrets
    from deploy.railway.hindsight_settings import current
    from hermes_cli.config import save_env_value
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'deployment-key')
    seed_service_secrets()
    assert current()['openrouter_api_key'] == 'deployment-key'
    save_env_value('OPENROUTER_API_KEY', 'dashboard-key')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'deployment-key')
    seed_service_secrets()
    assert current()['openrouter_api_key'] == 'dashboard-key'
