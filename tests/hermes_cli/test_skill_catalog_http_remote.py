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
def remote():
    home = get_hermes_home()
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


@pytest.mark.platforms('posix')
def test_dead_remote_owner_does_not_retain_http_catalog_authority(client, remote):
    source, process = remote
    process.terminate()
    process.join(5)
    assert not process.is_alive()
    result = client.post('/api/skills/hub/review', json=dict(identifier=source + ':item', action='install'))
    assert result.status_code == 405, result.text
    # The stale document remains, but kernel lease release restores stock.
