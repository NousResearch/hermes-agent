"""HTTP lookup cannot turn another process's proven-live catalog into stock."""
import multiprocessing
import signal

import pytest

from hermes_constants import get_hermes_home
from tests.hermes_cli.test_skill_catalog_projection import entry, projection, record

pytest_plugins = ['tests.hermes_cli.test_skill_catalog_http']


def _remote_owner(home, connection):
    owner = entry(home)
    store = projection(home)
    handle = store.start(owner)
    try:
        store.publish(owner, [record(owner, 'remote')])
        connection.send(owner.source_id)
        connection.recv()
    finally:
        owner.stopped = True
        handle.close()
        connection.close()


@pytest.fixture
def remote(request, monkeypatch):
    from hermes_cli import profiles
    from hermes_cli.config import atomic_config_write
    home = get_hermes_home()
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: get_hermes_home())
    if getattr(request, 'param', None) == 'other':
        home = home / 'profiles/other'
        home.mkdir(parents=True)
        atomic_config_write(home / 'config.yaml', {})
    target = home / 'skills/remote'
    target.mkdir(parents=True)
    (target / 'SKILL.md').write_text('---\nname: remote\ndescription: remote\n---\nRemote')
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_remote_owner, args=(home, child))
    process.start()
    child.close()
    assert parent.poll(10), 'remote owner never published its actual OS lease'
    source = parent.recv()
    try:
        yield source, process
    finally:
        if process.is_alive():
            parent.send('close')
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        assert process.exitcode in (0, -signal.SIGTERM)


@pytest.mark.parametrize('route', ['official', 'sources', 'search', 'review', 'confirm', 'install'])
def test_remote_live_owner_is_source_unavailable_not_stock(client, remote, monkeypatch, route):
    from hermes_cli.web_routers import skills
    source, process = remote
    calls = []

    def spawn(*args, **kwargs):
        calls.append('spawn')
        return dict(ok=True, pid=123)

    monkeypatch.setattr(skills, 'spawn_profile_action', spawn)
    path = '/api/skills/hub/' + route
    if route in ('review', 'confirm', 'install'):
        result = client.post(path, json=dict(identifier=source + ':item', action='install'))
    else:
        result = client.get(path, params=dict(source=source, q='remote'))
    assert process.is_alive()
    assert result.status_code == 503, result.text
    assert result.json()['error']['code'] == 'provider_unavailable'
    assert calls == []


def test_oversized_remote_live_provider_refuses_before_stock(client, remote, monkeypatch):
    from hermes_cli.web_routers import skills
    source, process = remote
    spawned = []
    monkeypatch.setattr(skills, 'spawn_profile_action', lambda *a, **kw: spawned.append(kw) or {'ok': True})
    import json
    raw = json.dumps(dict(identifier=source + ':item')).encode()
    result = client.post('/api/skills/hub/install', content=raw + b' ' * (262145 - len(raw)),
                         headers={'content-type': 'application/json'})
    assert process.is_alive()
    assert result.status_code == 413, result.text
    assert result.json()['error']['code'] == 'request_too_large'
    assert spawned == []


@pytest.mark.parametrize('remote', ['other'], indirect=True)
def test_oversized_body_selected_remote_profile_refuses_until_lease_dies(client, remote, monkeypatch):
    import json
    from hermes_cli.skill_catalog.registry import snapshot
    from hermes_cli.web_routers import skills
    source, process = remote
    assert snapshot(scope=str(get_hermes_home())) == ()
    spawned = []
    monkeypatch.setattr(skills, 'spawn_profile_action', lambda *a, **kw: spawned.append(kw) or {'ok': True})
    raw = json.dumps(dict(profile='other', identifier=source + ':item')).encode()
    small = client.post('/api/skills/hub/install', content=raw, headers={'content-type': 'application/json'})
    assert small.status_code == 503, small.text
    assert small.json()['error']['code'] == 'provider_unavailable'
    padded = raw + b' ' * (262145 - len(raw))
    large = client.post('/api/skills/hub/install', content=padded, headers={'content-type': 'application/json'})
    assert large.status_code == 413, large.text
    assert large.json()['error']['code'] == 'request_too_large'
    assert spawned == []
    process.terminate()
    process.join(5)
    assert not process.is_alive()
    stock = client.post('/api/skills/hub/install', content=padded, headers={'content-type': 'application/json'})
    assert stock.status_code == 200, stock.text
    assert len(spawned) == 1


@pytest.mark.platforms('posix')
def test_dead_remote_owner_does_not_retain_http_catalog_authority(client, remote):
    source, process = remote
    process.terminate()
    process.join(5)
    assert not process.is_alive()
    result = client.post('/api/skills/hub/review', json=dict(identifier=source + ':item', action='install'))
    assert result.status_code == 405, result.text
    # The stale document remains, but kernel lease release restores stock.
