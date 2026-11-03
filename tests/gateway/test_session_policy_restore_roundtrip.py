"""Stored policy round-trip regressions for hermes-agent issues #148473 and #148540."""

from dataclasses import asdict

import pytest

from gateway.session_policy import build_policy, restore_policy
from hermes_state_runtime import RuntimeStoreError


def _receipt(tmp_path, source='cli'):
    return asdict(build_policy({'source': source, 'cwd': str(tmp_path),
                                'model': 'fixture-model'}, {}))


@pytest.mark.parametrize('source, platform', [
    ('cli', 'cli'), ('tui', 'tui'), ('gui', 'desktop'), ('acp', 'acp'),
])
def test_peer_refreshed_local_source_restores(tmp_path, source, platform):
    stored = _receipt(tmp_path, source)
    assert stored['platform'] == platform
    stored['source'] = 'local'  # peer refresh retains the platform but relabels the row
    if source == 'acp':
        stored['edit_approval_policy'] = 'workspace_session'
    restored = restore_policy(stored)
    assert (restored.source, restored.platform) == ('local', platform)
    assert restored.edit_approval_policy == stored['edit_approval_policy']


@pytest.mark.parametrize('model', [None, ''])
def test_writer_empty_model_sentinels_restore(tmp_path, model):
    for source in ('cli', 'local'):
        stored = _receipt(tmp_path)
        stored.update(source=source, model=model)
        assert restore_policy(stored).model == model


@pytest.mark.parametrize('corruption', [
    {'source': 'bogus'},
    {'source': 'cli', 'platform': 'desktop'},
    {'source': 'local', 'platform': 'bot_room'},
    {'source': 'local', 'model': '  '},
    {'source': 'local', 'model': 123},
    {'source': 'local', 'edit_approval_policy': 'session'},
])
def test_restore_still_rejects_corruption(tmp_path, corruption):
    stored = _receipt(tmp_path)
    stored.update(corruption)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_policy(stored)


def test_stored_local_label_is_not_a_creating_surface(tmp_path):
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'local', 'cwd': str(tmp_path), 'model': 'fixture-model'}, {})
