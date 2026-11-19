"""Installed HTTP rows require one coherent live owner/claim snapshot."""
import asyncio
import copy
import threading

import pytest

from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_installed import _provider, _skill


def _rows():
    from hermes_cli.web_routers.skills import get_skills
    return asyncio.run(get_skills())


@pytest.mark.parametrize('drift', ['new_target', 'identifier', 'revision', 'state', 'missing'])
def test_installed_read_cannot_invent_or_replace_live_claim(_isolate_hermes_home, monkeypatch, drift):
    from hermes_cli.skill_catalog.registry import snapshot
    home = get_hermes_home()
    _skill(home, 'nested/folder', 'shown-name')
    _skill(home, 'local', 'local')
    manager, record = _provider(home)
    entry, = snapshot()
    changed = copy.deepcopy(record) | {
        'new_target': {'target_relative_path': 'local'},
        'identifier': {'identifier': entry.source_id + ':different-item'},
        'revision': {'installation_revision': 'install-2'},
        'state': {'managed_state': 'pending'},
        'missing': {},
    }[drift]
    records = [] if drift == 'missing' else [changed]
    monkeypatch.setattr(entry.provider, 'installed', lambda request: dict(
        ok=True, data=dict(items=records, next_cursor=None)))
    before = entry.mutations.projection.read()
    try:
        with pytest.raises(CatalogError) as caught:
            _rows()
        assert caught.value.code == 'ownership_state_unavailable'
        assert entry.mutations.projection.read() == before
        assert (home / 'skills/local/SKILL.md').read_text().endswith('Approved\n')
    finally:
        manager.unload()


@pytest.mark.parametrize('change', ['disable', 'reregister', 'new_revision', 'unrelated_revision'])
def test_installed_read_barrier_rejects_changed_authority(_isolate_hermes_home, monkeypatch, change):
    from agent.memory_provider import spawn_context_thread
    from hermes_cli.skill_catalog.registry import snapshot
    home = get_hermes_home()
    _skill(home, 'nested/folder', 'shown-name')
    manager, record = _provider(home)
    entry, = snapshot()
    entered, release = threading.Event(), threading.Event()
    output = []
    original = entry.provider.installed

    def read(request):
        result = original(request)
        entered.set()
        assert release.wait(5), 'test did not release installed read'
        return result

    def request():
        try:
            output.append(_rows())
        except BaseException as exc:
            output.append(exc)

    monkeypatch.setattr(entry.provider, 'installed', read)
    thread = spawn_context_thread(request, name='catalog-installed-reader')
    thread.start()
    replacement = None
    try:
        assert entered.wait(3), 'provider installed read not reached'
        if change in ('disable', 'reregister'):
            manager.unload()
            if change == 'reregister':
                replacement, _ = _provider(home)
        elif change == 'new_revision':
            entry.mutations.projection.publish(entry, [record | {'installation_revision': 'install-2'}])
        else:
            # Unrelated path activity is not grounds for dropping a valid row.
            with entry.mutations.projection.transaction() as doc:
                doc['path_revisions']['unrelated'] = 1
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert len(output) == 1
        if change == 'unrelated_revision':
            assert output[0][0]['provenance'] == 'provider'
        else:
            assert isinstance(output[0], CatalogError), output
            assert output[0].code == 'ownership_state_unavailable'
    finally:
        release.set()
        thread.join(5)
        if replacement is not None:
            replacement.unload()
        manager.unload()
