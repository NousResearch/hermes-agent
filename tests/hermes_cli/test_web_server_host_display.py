import pytest


@pytest.fixture
def client(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip('fastapi/starlette not installed')

    import hermes_cli.web_server as ws

    c = TestClient(ws.app)
    c.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return ws, c


def test_host_display_returns_explicit_vnc_url(client, monkeypatch):
    ws, c = client
    monkeypatch.setattr(
        ws,
        'load_config',
        lambda: {'desktop': {'host_vnc_url': 'http://127.0.0.1:6080/vnc.html?autoconnect=true'}},
    )

    response = c.get('/api/host-display')

    assert response.status_code == 200
    assert response.json() == {
        'available': True,
        'reason': None,
        'url': 'http://127.0.0.1:6080/vnc.html?autoconnect=true',
    }


@pytest.mark.parametrize(
    'url',
    [
        'javascript:alert(1)',
        'file:///etc/passwd',
        'http://user:password@127.0.0.1:6080/vnc.html',
        'https://host.example/vnc.html?password=secret',
        'https://host.example/vnc.html?api_key=secret',
        'https://host.example/vnc.html#token=secret',
        'http:///vnc.html',
    ],
)
def test_host_display_rejects_unsafe_urls(client, monkeypatch, url):
    ws, c = client
    monkeypatch.setattr(ws, 'load_config', lambda: {'desktop': {'host_vnc_url': url}})

    response = c.get('/api/host-display')

    assert response.status_code == 200
    assert response.json() == {
        'available': False,
        'reason': 'Host VNC URL is invalid',
        'url': None,
    }


def test_host_display_reports_unconfigured(client, monkeypatch):
    ws, c = client
    monkeypatch.setattr(ws, 'load_config', lambda: {'desktop': {}})

    response = c.get('/api/host-display')

    assert response.status_code == 200
    assert response.json() == {
        'available': False,
        'reason': 'Set desktop.host_vnc_url to the host noVNC page',
        'url': None,
    }
