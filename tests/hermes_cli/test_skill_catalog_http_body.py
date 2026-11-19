"""Oversized hub bodies bypass optional routing without losing stock input."""
import asyncio
import json
from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.responses import Response

pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']

CAP = 256 * 1024


def _padded(plan, size):
    raw = json.dumps(dict(identifier=plan['identifier'], action='install')).encode()
    return raw + b' ' * (size - len(raw))


@pytest.mark.parametrize('query', ['', '?profile=default'])
def test_oversized_live_provider_request_keeps_pinned_stock_response(client, catalog, monkeypatch, query):
    from hermes_cli.skill_catalog import http
    entry, plan = catalog
    calls = []
    original = http._hint

    def hint(raw):
        calls.append('parse')
        return original(raw)

    monkeypatch.setattr(http, '_hint', hint)
    response = client.post('/api/skills/hub/review' + query, content=_padded(plan, CAP + 1))
    frozen = json.loads((Path(__file__).parents[1] / 'fixtures/skill_catalog/stock_extensions.json').read_text())
    case = next(row for row in frozen['requests'] if row['method'] == 'POST'
                and row['path'] == '/api/skills/hub/review')
    assert response.status_code == case['status'], response.text
    assert response.content == case['body'].encode()
    assert dict(response.headers) == case['headers']
    assert calls == []
    assert entry.provider.effects == 0


def test_exact_body_limit_still_routes_to_provider(client, catalog):
    _entry, plan = catalog
    response = client.post('/api/skills/hub/review', content=_padded(plan, CAP))
    assert response.status_code == 200, response.text
    assert response.json() == plan


def test_providerless_oversized_body_matches_pinned_stock(client):
    response = client.post('/api/skills/hub/review', content=b'x' * (CAP + 1))
    assert response.status_code == 405
    assert response.content == b'{"detail":"Method Not Allowed"}'
    assert response.headers['allow'] == 'GET'


@pytest.mark.parametrize('chunks', [
    [b'x' * CAP, b'y', b'z' * CAP],
    [b'x' * (CAP * 2), b'z'],
])
def test_oversized_stream_stops_before_tail_then_replays_exact_stock_bytes(monkeypatch, chunks):
    from hermes_cli.skill_catalog import http
    received, stock_bodies = [], []

    async def run():
        messages = iter(dict(type='http.request', body=chunk, more_body=index < len(chunks) - 1)
                        for index, chunk in enumerate(chunks))

        async def receive():
            message = next(messages)
            received.append(message['body'])
            return message

        request = Request(dict(type='http', method='POST', path='/api/skills/hub/install',
            scheme='http', server=('test', 80), query_string=b'profile=default', headers=[]), receive)

        async def stock(request):
            # ASGI receive is chunk-granular; the final transport-owned chunk may
            # exceed the cap, but no following chunk is read before delegation.
            assert len(received) == (2 if len(chunks) == 3 else 1)
            stock_bodies.append(await request.body())
            return Response(b'stock')

        def forbidden(*args, **kwargs):
            pytest.fail('oversized extension parsed or inferred a provider scope')

        monkeypatch.setattr(http, '_hint', forbidden)
        monkeypatch.setattr(http, '_service', forbidden)
        result = await http.delegate(request, stock)
        assert result.body == b'stock'

    asyncio.run(run())
    assert stock_bodies == [b''.join(chunks)]
    assert received == chunks
