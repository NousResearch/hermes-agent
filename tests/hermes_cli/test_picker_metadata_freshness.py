"""Producer age is never renewed by catalogue receipt or wall-clock changes."""
import copy

import pytest

from hermes_cli.picker_presentation import route_fields


def row_for(**changes):
    metadata = {'role': 'main', 'backing_model': 'Exact/Main:Q4', 'residency': 'ready',
                'mode': 'local', 'observed_at': 1700000000,
                'freshness': {'age_s': 10, 'max_age_s': 15, 'stale': False}}
    metadata.update(changes)
    return {'name': 'Turbofit', 'picker_presentation': {
        'models': {'active:main': metadata}, 'observed_at': 100}}


@pytest.mark.parametrize('state,label', [('ready', '● Ready'), ('loading', '◐ Loading'),
    ('idle', '○ Idle / unloaded'), ('error', '! Error'), ('unknown', '? Unknown')])
def test_states_with_fresh_observation(monkeypatch, state, label):
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 102)
    assert route_fields(row_for(residency=state), 'active:main') == (
        'Turbofit:Main', 'Exact/Main:Q4', label)


def test_producer_age_plus_monotonic_age_expires_identity_survives(monkeypatch):
    row = row_for()
    before = copy.deepcopy(row)
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 105.01)
    assert route_fields(row, 'active:main') == ('Turbofit:Main', 'Exact/Main:Q4', '? Unknown')
    assert row == before


@pytest.mark.parametrize('freshness', [None, {}, [], {'age_s': 0, 'max_age_s': 15, 'stale': True},
    {'age_s': 16, 'max_age_s': 15, 'stale': False},
    *[{'age_s': value, 'max_age_s': 15, 'stale': False} for value in (-1, True, '0', float('nan'), float('inf'))],
    *[{'age_s': 0, 'max_age_s': value, 'stale': False} for value in (0, -1, True, '15', float('nan'), float('inf'))],
    {'age_s': 0, 'max_age_s': 15, 'stale': 'false'}])
def test_invalid_or_stale_freshness_is_unknown_on_new_receipt(monkeypatch, freshness):
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 100)
    assert route_fields(row_for(freshness=freshness), 'active:main')[1:] == ('Exact/Main:Q4', '? Unknown')


@pytest.mark.parametrize('observed', [None, -1, True, '1700000000', float('nan'), float('inf')])
def test_invalid_producer_timestamp_is_unknown(monkeypatch, observed):
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 100)
    assert route_fields(row_for(observed_at=observed), 'active:main')[2] == '? Unknown'


def test_old_metadata_is_identity_only(monkeypatch):
    row = row_for()
    details = row['picker_presentation']['models']['active:main']
    del details['freshness']
    del details['observed_at']
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 100)
    assert route_fields(row, 'active:main')[1:] == ('Exact/Main:Q4', '? Unknown')


def test_wall_clock_is_not_used_and_shared_mode_survives_expiry(monkeypatch):
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 116)
    monkeypatch.setattr('hermes_cli.picker_presentation.time.time', lambda: pytest.fail('wall clock used'))
    assert route_fields(row_for(mode='shared-main'), 'active:main')[1:] == (
        'Exact/Main:Q4', 'Shared main · ? Unknown')


@pytest.mark.parametrize('receipt', [None, True, '100', -1, float('nan'), float('inf'), 10**1000, 101])
def test_invalid_or_future_receipt_fails_closed(monkeypatch, receipt):
    row = row_for()
    row['picker_presentation']['observed_at'] = receipt
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: 100)
    assert route_fields(row, 'active:main')[2] == '? Unknown'


@pytest.mark.parametrize('maximum,now,expected', [(2, 102, '● Ready'), (2, 102.01, '? Unknown'),
    (60, 115, '● Ready'), (60, 115.01, '? Unknown')])
def test_producer_limit_and_client_cap(monkeypatch, maximum, now, expected):
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: now)
    row = row_for(freshness={'age_s': 0, 'max_age_s': maximum, 'stale': False})
    assert route_fields(row, 'active:main')[2] == expected


@pytest.mark.parametrize('state,label', [('ready', '● Ready'), ('loading', '◐ Loading'),
    ('idle', '○ Idle / unloaded'), ('error', '! Error'), ('unknown', '? Unknown')])
@pytest.mark.parametrize('width', [40, 60])
def test_actual_renderer_states_and_expiry(monkeypatch, state, label, width):
    from cli import HermesCLI
    from prompt_toolkit.utils import get_cwidth
    now = [100]
    monkeypatch.setattr('hermes_cli.picker_presentation.time.monotonic', lambda: now[0])
    monkeypatch.setenv('COLUMNS', str(width))
    row = row_for(residency=state)
    row.update(slug='custom:turbofit', models=['active:main'])
    cli = HermesCLI.__new__(HermesCLI)
    cli._model_picker_state = {'stage': 'model', 'provider_data': row,
                              'model_list': row['models'], 'selected': 0}
    def rendered():
        text = ''.join(t for _, t in cli._get_model_picker_display_fragments())
        assert max(get_cwidth(line) for line in text.splitlines()) <= width
        assert 'Exact/Main:Q4' in text and 'Turbofit:Main' in text
        assert 'active:main' not in text
        return ' '.join(text.replace('│', ' ').split())
    assert label in rendered()
    now[0] = 106
    assert '? Unknown' in rendered()



def test_reopening_http_catalogue_does_not_renew_stale_observation():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from hermes_cli.picker_presentation import attach_picker_presentation
    requests = []
    metadata = row_for()['picker_presentation']['models']['active:main']
    metadata['freshness'] = {'age_s': 60, 'max_age_s': 15, 'stale': False}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(('GET', self.path))
            body = json.dumps({'data': [{'id': 'active:main', 'metadata': metadata}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/v1'
        entry = {'name': 'Turbofit', 'base_url': url, 'picker_metadata': True}
        for _ in range(2):
            row = {'name': 'Turbofit', 'slug': 'custom:turbofit', 'base_url': url}
            workers = attach_picker_presentation([row], custom_providers=[entry])
            assert len(workers) == 1
            workers[0].join(3)
            assert not workers[0].is_alive()
            assert route_fields(row, 'active:main') == ('Turbofit:Main', 'Exact/Main:Q4', '? Unknown')
        assert requests == [('GET', '/v1/models'), ('GET', '/v1/models')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
