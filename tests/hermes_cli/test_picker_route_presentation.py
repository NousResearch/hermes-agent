"""Real terminal picker presentation, with immutable route selection IDs."""
import json
import time
from unittest.mock import Mock

from cli import HermesCLI
from hermes_cli.cli_model_switch_mixin import _show_model_picker


def test_picker_displays_backing_and_named_role_not_raw_alias(tmp_path, monkeypatch):
    import threading
    ready = threading.Event()
    metadata = {
        'active:main': {'role': 'main', 'backing_model': 'Qwen3.8-27b', 'residency': 'ready'},
        'active:aux': {'role': 'aux', 'backing_model': 'Qwen3.8-27b', 'residency': 'idle', 'mode': 'shared-main'},
        'auto': {'role': 'auto', 'backing_model': 'Qwen3.8-27b', 'residency': 'ready'},
    }
    for details in metadata.values():
        details.update(observed_at=time.time(), freshness={'age_s': 0, 'max_age_s': 15, 'stale': False})
    monkeypatch.setattr('hermes_cli.picker_presentation._fetch_metadata', lambda entry: metadata)
    entry = {'name': 'Turbofit', 'base_url': 'http://127.0.0.1:1/v1', 'picker_metadata': True,
             'models': {i: {'picker_label': label} for i, label in [
                 ('active:main', 'Turbofit:Main'), ('active:aux', 'Turbofit:Aux'), ('auto', 'Turbofit:Auto')]}}
    from types import SimpleNamespace
    ctx = SimpleNamespace(custom_providers=[entry], user_providers={})
    row = {'slug': 'custom:turbofit', 'name': 'Turbofit', 'base_url': entry['base_url'],
           'is_current': True, 'models': list(entry['models'])}
    monkeypatch.setattr('hermes_cli.inventory.build_models_payload', lambda *a, **k: {'providers': [row]})
    cli = HermesCLI.__new__(HermesCLI)
    cli.model = 'active:aux'
    cli.provider = 'custom'
    cli._capture_modal_input_snapshot = Mock()
    cli._invalidate = Mock(side_effect=lambda **kw: ready.set() if threading.current_thread().name == 'picker-metadata' else None)
    _show_model_picker(cli, ctx, False)
    assert ready.wait(2)
    rendered = ''.join(text for _, text in cli._get_model_picker_display_fragments())
    assert 'Turbofit:Aux' in rendered
    assert 'Qwen3.8-27b' in rendered
    assert 'active:aux' not in rendered
    cli._handle_model_picker_selection()
    rendered = ''.join(text for _, text in cli._get_model_picker_display_fragments())
    assert 'Turbofit:Main' in rendered
    assert 'Turbofit:Auto' in rendered
    assert 'Shared main' in rendered
    assert 'Idle /' in rendered and 'unloaded' in rendered
    assert 'active:main' not in rendered
    assert cli._model_picker_state['_filtered_pairs'] == list(enumerate(row['models']))


def test_real_inventory_picker_and_global_persistence_without_network(tmp_path, monkeypatch, capsys):
    import socket
    import yaml
    from hermes_constants import get_hermes_home
    from hermes_cli.inventory import load_picker_context
    home = get_hermes_home()
    assert str(home).startswith(str(tmp_path.parent)) or '/sandbox/' in str(home)
    home.mkdir(parents=True, exist_ok=True)
    entry = {'name': 'Turbofit', 'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'fixture-only',
             'discover_models': False, 'models': {
                 'auto': {'picker_label': 'Turbofit:Auto'},
                 'active:main': {'picker_label': 'Turbofit:Main'},
                 'active:aux': {'picker_label': 'Turbofit:Aux'}}}
    config = {'model': {'provider': 'custom:turbofit', 'default': 'active:main', 'base_url': entry['base_url']},
              'custom_providers': [entry], 'providers': {}}
    (home / 'config.yaml').write_text(yaml.safe_dump(config))
    calls = []
    def no_connect(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Picker attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', no_connect)
    cli = HermesCLI.__new__(HermesCLI)
    cli.model, cli.provider, cli.base_url, cli.api_key = 'active:main', 'custom:turbofit', entry['base_url'], 'fixture-only'
    cli.agent, cli.api_mode = None, 'chat_completions'
    cli._capture_modal_input_snapshot = Mock()
    cli._restore_modal_input_snapshot = Mock()
    cli._invalidate = Mock()
    cli._app = None
    cli.session_id = None
    cli._session_db = None
    ctx = load_picker_context()
    _show_model_picker(cli, ctx, False)
    rows = cli._model_picker_state['providers']
    tf = [r for r in rows if r['slug'] == 'custom:turbofit']
    assert len(tf) == 1
    assert set(tf[0]['models']) == {'active:main', 'auto', 'active:aux'}
    assert tf[0].get('picker_presentation'), (tf[0], ctx.custom_providers)
    assert calls == []
    cli._model_picker_state['selected'] = rows.index(tf[0])
    cli._handle_model_picker_selection()
    for model, query in [('active:aux', 'Turbofit:Aux'), ('auto', 'Turbofit:Auto'), ('active:main', 'Turbofit:Main')]:
        if cli._model_picker_state is None:
            before_open = len(calls)
            _show_model_picker(cli, load_picker_context(), False)
            assert len(calls) == before_open
            rows = cli._model_picker_state['providers']
            cli._model_picker_state['selected'] = next(i for i,r in enumerate(rows) if r['slug'] == 'custom:turbofit')
            cli._handle_model_picker_selection()
        cli._model_picker_state['filter'] = query
        rendered = ''.join(t for _,t in cli._get_model_picker_display_fragments())
        assert 'Unknown model' in rendered and '? Unknown' in rendered
        assert [m for _,m in cli._model_picker_state['_filtered_pairs']] == [model]
        cli._model_picker_state['selected'] = 0
        cli._handle_model_picker_selection(persist_global=True)
        saved = yaml.safe_load((home / 'config.yaml').read_text())
        assert saved['model']['default'] == model
        assert saved['model']['provider'] == 'custom:turbofit'
        assert saved['model']['base_url'] == entry['base_url']
        assert cli.model == model
    # Selection deliberately runs the existing endpoint/context validation; opening does not.
    assert calls
    summary = capsys.readouterr().out
    assert '✓ Model switched: active:' not in summary
    for label in ('Turbofit:Main', 'Turbofit:Aux', 'Turbofit:Auto'):
        assert 'Provider: ' + label in summary


def test_stale_and_unsupported_metadata_are_unknown(monkeypatch):
    from hermes_cli.picker_presentation import route_fields
    row = {'picker_presentation': {'labels': {'active:aux': 'Turbofit:Aux'},
           'models': {'active:aux': {'backing_model': 'old-backing', 'residency': 'ready'}},
           'observed_at': time.monotonic() - 60}}
    assert route_fields(row, 'active:aux') == ('Turbofit:Aux', 'old-backing', '? Unknown')
    row['picker_presentation']['observed_at'] = time.monotonic()
    row['picker_presentation']['models']['active:aux']['residency'] = 'down'
    assert route_fields(row, 'active:aux') == ('Turbofit:Aux', 'old-backing', '? Unknown')


def test_real_catalogue_metadata_open_is_get_only(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import yaml
    from hermes_constants import get_hermes_home
    from hermes_cli.inventory import load_picker_context
    from hermes_cli.picker_presentation import route_fields
    requests = []
    metadata = {'role': 'aux', 'backing_model': 'actual-shared-main', 'residency': 'unknown'}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(('GET', self.path))
            body = json.dumps({'data': [{'id': 'active:aux', 'metadata': metadata}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            requests.append(('POST', self.path))
            self.send_error(405)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        url = f'http://127.0.0.1:{server.server_port}/v1'
        entry = {'name': 'Turbofit', 'base_url': url, 'api_key': 'fixture',
                 'discover_models': False, 'picker_metadata': True,
                 'models': {'active:aux': {'picker_label': 'Turbofit:Aux'}}}
        (home / 'config.yaml').write_text(yaml.safe_dump({'custom_providers': [entry],
            'model': {'provider': 'custom:turbofit', 'default': 'active:aux', 'base_url': url}}))
        cli = HermesCLI.__new__(HermesCLI)
        cli.model, cli.provider = 'active:aux', 'custom:turbofit'
        cli._capture_modal_input_snapshot = Mock()
        complete = threading.Event()
        def invalidate(**kwargs):
            if threading.current_thread().name == 'picker-metadata':
                complete.set()
        cli._invalidate = invalidate
        _show_model_picker(cli, load_picker_context(), False)
        assert complete.wait(3)
        row = next(r for r in cli._model_picker_state['providers'] if r['slug'] == 'custom:turbofit')
        assert route_fields(row, 'active:aux') == ('Turbofit:Aux', 'actual-shared-main', '? Unknown')
        assert requests == [('GET', '/v1/models')]
        rendered = ''.join(t for _, t in cli._get_model_picker_display_fragments())
        assert 'actual-shared-main' in rendered and 'active:aux' not in rendered
        old_row = row
        metadata['backing_model'] = 'new-shared-main'
        complete.clear()
        _show_model_picker(cli, load_picker_context(), False)
        assert complete.wait(3)
        row = next(r for r in cli._model_picker_state['providers'] if r['slug'] == 'custom:turbofit')
        assert route_fields(row, 'active:aux')[1] == 'new-shared-main'
        assert route_fields(old_row, 'active:aux')[1] == 'actual-shared-main'
        assert requests == [('GET', '/v1/models'), ('GET', '/v1/models')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_narrow_picker_wraps_long_backing_identity(monkeypatch):
    from prompt_toolkit.utils import get_cwidth
    monkeypatch.setenv('COLUMNS', '40')
    row = {'slug': 'custom:turbofit', 'name': 'Turbofit', 'models': ['active:main'],
        'picker_presentation': {'labels': {'active:main': 'Turbofit:Main'}, 'observed_at': time.monotonic(),
            'models': {'active:main': {'backing_model': 'Qwen3.8-27B-Unleashed-UD-Q3_K_XL-Long-Identity', 'residency':'unknown'}}}}
    cli = HermesCLI.__new__(HermesCLI)
    cli._model_picker_state = {'stage':'model', 'provider_data':row,'model_list':row['models'],'selected':0}
    text = ''.join(t for _,t in cli._get_model_picker_display_fragments())
    assert max(get_cwidth(line) for line in text.splitlines()) <= 40
    assert len(text.splitlines()) <= 18
    assert 'Turbofit:Main' in text
