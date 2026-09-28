import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from plugins.memory.hindsight import HindsightMemoryProvider
from plugins.memory.hindsight.employee import _CLIENT_POLICY


def test_real_client_retain_recall_and_reflect_protocol():
    calls = []
    class API(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path,body))
            if self.path.endswith('/reflect'):
                result = {'text':'The current decision is confirmed.', 'based_on':{'memories':[], 'mental_models':[], 'directives':[]}}
            elif self.path.endswith('/recall'):
                result = {'results':[]}
            else:
                result = {'success':True,'operation_id':'op-test','bank_id':'employee','items_count':1,'async':True}
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    server = ThreadingHTTPServer(('127.0.0.1',0),API)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    provider = HindsightMemoryProvider(config={**_CLIENT_POLICY,'api_url':f'http://127.0.0.1:{server.server_port}', 'api_key':'test','bank_id':'employee'})
    try:
        provider.initialize('conversation')
        provider._retain_batch(provider._build_retain_kwargs('Confirmed decision'),bank_id='employee',retain_async=True)
        assert provider._recall('What is decided?') == []
        from agent.memory_manager import MemoryManager
        manager = MemoryManager()
        manager.add_provider(provider)
        assert manager.has_tool('recall')
        assert any(schema['name'] == 'recall' for schema in manager.get_all_tool_schemas())
        result = json.loads(manager.handle_tool_call('recall', {'query': 'What is decided?'}))
        assert 'confirmed' in result['result']
        assert [s['name'] for s in provider.get_tool_schemas()] == ['recall']
        assert calls[1][1]['prefer_observations'] is True
        assert calls[1][1]['max_tokens'] == _CLIENT_POLICY['recall_max_tokens']
    finally:
        if provider._client is not None:
            provider._client.close()
        server.shutdown();server.server_close();thread.join()


def test_manager_retention_preserves_buffered_turn_authors(monkeypatch):
    from agent.memory_manager import MemoryManager
    provider = HindsightMemoryProvider(config={**_CLIENT_POLICY, 'api_url': 'http://127.0.0.1:1',
                                             'api_key': 'test', 'bank_id': 'employee', 'retain_every_n_turns': 2})
    provider.initialize('shared', user_id='first', user_name='First', chat_type='group')
    queued, retained = [], []
    monkeypatch.setattr(provider, '_enqueue_retain', queued.append)
    monkeypatch.setattr(provider, '_retain_batch', lambda item, **kwargs: retained.append(item))
    manager = MemoryManager()
    manager.add_provider(provider)
    monkeypatch.setattr(manager, '_submit_background', lambda fn, **kwargs: fn())
    author = {'id': 'first', 'name': 'First', 'is_bot': False}
    manager.sync_all('I own sales.', 'Saved.', session_id='shared', turn_author=author)
    author.update(id='second', name='Second')
    manager.sync_all('I own support.', 'Saved.', session_id='shared', turn_author=author)
    author['name'] = 'Changed after buffering'
    queued.pop()()
    turns = json.loads(retained[0]['content'])
    assert [turn[0]['author']['id'] for turn in turns] == ['first', 'second']
    assert 'First' in turns[0][0]['content'] and 'Second' in turns[1][0]['content']
    assert 'user_id' not in retained[0]['metadata'] and 'user_name' not in retained[0]['metadata']
    manager.sync_all('Still support.', 'Saved.', session_id='shared', turn_author={'id': 'second', 'name': 'Second'})
    provider._enqueue_pending_turns(reason='switch')
    queued.pop()()
    assert 'user_id' not in retained[-1]['metadata']
    provider.on_session_switch('new')
    for _ in range(2):
        manager.sync_all('Still support.', 'Saved.', session_id='new', turn_author={'id': 'second', 'name': 'Second'})
    queued.pop()()
    assert retained[-1]['metadata']['user_id'] == 'second'
    assert retained[-1]['metadata']['user_name'] == 'Second'


def test_native_setup_and_dashboard_share_profile_configuration(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import load_config_readonly
    from hermes_cli.memory_setup import _post_setup_hook
    from hermes_cli.web_server_memory import _read_memory_provider_existing_values
    from agent.secret_scope import set_multiplex_active, is_multiplex_active
    from plugins.memory.hindsight import _load_config
    from plugins.memory.hindsight import setup
    from plugins.memory.hindsight.config_schema import CONFIG_SCHEMA
    from hermes_cli.web_routers.memory_providers import _update_memory_provider_config, _declared_provider_payload
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'launch'))
    provider = HindsightMemoryProvider()
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for index, name in enumerate(('a', 'b', 'a')):
            home = tmp_path / name
            url = f'http://127.0.0.1:{9000 + index}'
            inputs = iter((url, name))
            monkeypatch.setattr('builtins.input', lambda prompt: next(inputs))
            monkeypatch.setattr(setup, '_secret_prompt', lambda prompt: 'test-key-' + name)
            with _profile_runtime_scope(home):
                assert _post_setup_hook(provider, load_config_readonly())
            with _profile_runtime_scope(home):
                runtime = _load_config()
                assert runtime['api_url'] == url and runtime['bank_id'] == name
                assert runtime['api_key'] == 'test-key-' + name
                assert runtime['mode'] == 'local_external'
                assert _read_memory_provider_existing_values('hindsight')['url'] == url
                provider.save_config({'url': url + '/v2', 'bank_id': name}, str(home))
                assert _load_config()['api_url'] == url + '/v2'
                _update_memory_provider_config(CONFIG_SCHEMA, {'url': url + '/dashboard'})
                assert _load_config()['api_url'] == url + '/dashboard'
                fields = {field['key']: field for field in _declared_provider_payload(CONFIG_SCHEMA)['fields']}
                assert fields['url']['value'] == url + '/dashboard'
            assert 'test-key-' not in (home / 'config.yaml').read_text()
            assert not (home / 'hindsight/config.json').exists()
        with _profile_runtime_scope(tmp_path / 'b'):
            assert _load_config()['api_url'] == 'http://127.0.0.1:9001/dashboard'
            assert _load_config()['api_key'] == 'test-key-b'
        assert {field['key'] for field in provider.get_config_schema()} == {'url', 'bank_id', 'api_key'}
    finally:
        set_multiplex_active(previous)


def test_failed_retain_retries_real_http_in_order_across_session_switch():
    calls = []
    accepted = []
    recovered = threading.Event()
    class API(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"version":"0.5.0"}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            status = 503 if len(calls) == 1 else 200
            if status == 200:
                accepted.append(body)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'bank_id': 'employee', 'items_count': 1, 'async': True}).encode())
            if len(accepted) == 2:
                recovered.set()
    server = ThreadingHTTPServer(('127.0.0.1', 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = HindsightMemoryProvider(config={**_CLIENT_POLICY, 'api_url': f'http://127.0.0.1:{server.server_port}',
        'api_key': 'test', 'bank_id': 'employee', 'retain_every_n_turns': 1})
    try:
        provider.initialize('before')
        provider.sync_turn('First decision', 'Confirmed', session_id='before')
        provider.on_session_switch('after')
        provider.sync_turn('Second decision', 'Confirmed', session_id='after')
        assert recovered.wait(15)
        provider.shutdown()
        assert calls[0] == calls[1]
        assert len(accepted) == 2
        assert 'First decision' in accepted[0]['items'][0]['content']
        assert 'Second decision' in accepted[1]['items'][0]['content']
        assert accepted[0]['items'][0]['document_id'] != accepted[1]['items'][0]['document_id']
    finally:
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join()


def test_slow_prefetch_cannot_publish_after_switch_or_newer_worker(monkeypatch):
    provider = HindsightMemoryProvider(config={**_CLIENT_POLICY, 'api_url': 'http://localhost:1', 'bank_id': 'employee'})
    provider._recall_sync = False
    provider._prefetch_waits_for_retain = False
    entered = threading.Event()
    release = threading.Event()
    def recall(query):
        if query == 'old':
            entered.set()
            assert release.wait(10)
        return query, 1
    monkeypatch.setattr(provider, '_do_recall', recall)
    # Deterministically exercise the bounded-join expiry without a clock race.
    monkeypatch.setattr(provider, '_join_prefetch', lambda *args, **kwargs: None)
    for switch in (False, True):
        entered.clear()
        release.clear()
        provider.queue_prefetch('old')
        old = provider._prefetch_thread
        assert entered.wait(5)
        if switch:
            provider.on_session_switch('new-session')
        provider.queue_prefetch('new')
        provider._prefetch_thread.join(5)
        release.set()
        old.join(5)
        assert provider.prefetch('current') == provider._finish_prefetch('new', 1)


def test_shutdown_releases_retired_provider_and_transcript():
    import gc
    import weakref
    provider = HindsightMemoryProvider(config={**_CLIENT_POLICY, 'api_url': 'http://localhost:1', 'bank_id': 'employee'})
    provider._register_atexit()
    reference = weakref.ref(provider)
    provider.shutdown()
    del provider
    gc.collect()
    assert reference() is None


def test_recall_timing_current_vs_next_turn_uses_real_client():
    from agent.memory_manager import MemoryManager
    calls = []
    class API(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body['query'])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'results': [{'id': 'fact', 'text': body['query'], 'type': 'world'}]}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), API)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for current_turn in (False, True):
            provider = HindsightMemoryProvider(config={**_CLIENT_POLICY,
                'api_url': f'http://127.0.0.1:{server.server_port}', 'bank_id': 'employee',
                'recall_sync': current_turn, 'prefetch_waits_for_retain': False})
            provider.initialize('conversation')
            manager = MemoryManager()
            manager.add_provider(provider)
            try:
                first = manager.prefetch_all('First topic')
                assert ('First topic' in first) if current_turn else first == ''
                manager.queue_prefetch_all('First topic')
                assert manager.flush_pending(timeout=10)
                if provider._prefetch_thread:
                    provider._prefetch_thread.join(10)
                second = manager.prefetch_all('Second topic')
                assert ('Second topic' if current_turn else 'First topic') in second
            finally:
                manager.shutdown_all()
        assert calls == ['First topic', 'First topic', 'Second topic']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
