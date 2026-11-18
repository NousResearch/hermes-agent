"""Receipts our own writers persist restore (#148473, #148540).

``gateway/session_local.py`` stores ``model=None`` when no model resolves (``... or None``) and
``gateway/session_kanban.py`` stores the raw ``''``; the launch path resolves an empty policy model
from the profile's config. A peer refresh relabels an interactive receipt's ``source`` to ``local``
and keeps its platform. ``restore_policy`` refused both shapes, so the session answered
``storage_unavailable`` forever. Real corruption still refuses.
"""

from dataclasses import asdict

import pytest

from gateway.session_policy import build_policy, restore_policy
from persistence.core.errors import AuthorityRefusal


def _receipt(tmp_path, source='cli'):
    return asdict(build_policy({'source': source, 'cwd': str(tmp_path),
                                'model': 'fixture-model'}, {}))


@pytest.mark.parametrize('source, platform, model', [
    ('cli', 'cli', None), ('cli', 'cli', ''), ('tui', 'tui', None),
    ('gui', 'desktop', ''), ('acp', 'acp', 'fixture-model'),
])
def test_writer_shaped_receipts_restore(tmp_path, source, platform, model):
    stored = _receipt(tmp_path, source)
    assert stored['platform'] == platform
    stored['model'] = model
    for label in (source, 'local'):  # as created, and after a peer refresh relabelled it
        relabelled = {**stored, 'source': label}
        if source == 'acp':
            relabelled['edit_approval_policy'] = 'workspace_session'
        restored = restore_policy(relabelled)
        assert (restored.source, restored.platform, restored.model) == (label, platform, model)
        assert restored.edit_approval_policy == relabelled['edit_approval_policy']


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
    with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
        restore_policy(stored)
