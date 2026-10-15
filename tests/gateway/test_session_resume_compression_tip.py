"""Desktop lists compression tips; session.resume must remap tip -> owner receipt."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection


def _attach_snapshot(owner_ref):
    return SimpleNamespace(
        subscription_id='sub',
        history=(),
        pending=(),
        prompts=(),
        replay_epoch='e',
        last_sequence=0,
        handle=SimpleNamespace(
            ref=owner_ref,
            execution_state='idle',
            authority_epoch=1,
            revision=1,
            execution_generation=0,
        ),
    )


@pytest.mark.asyncio
async def test_session_resume_remaps_local_source_compression_tip(monkeypatch):
    """Session lists project the tip id after compression. Cold resume must remap
    ``source=local`` tips to the owner via resolve_local_target — not only
    LEGACY_LOCAL_SOURCES — or restore_local_session raises storage_unavailable
    (no policy receipt on the child id)."""
    owner_id = 'local-owner-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    tip_id = '20261015_221307_d93723'
    owner_ref = SessionRef('fixture', owner_id)
    resolved = []

    monkeypatch.setattr(
        'gateway.session_local_migration.resolve_local_target',
        lambda authority, actor, sid: resolved.append(sid) or owner_ref)

    row = {
        'id': tip_id,
        'source': 'local',
        'chat_id': owner_id,
        'parent_session_id': owner_id,
    }
    authority = SimpleNamespace(
        profile_id='fixture',
        instance_id='one',
        events={},
        db=SimpleNamespace(
            get_session=lambda sid: row if sid == tip_id else None,
            get_compression_tip=lambda sid: tip_id if sid in (tip_id, owner_id) else None,
        ),
        sessions={owner_id: SimpleNamespace(source=SimpleNamespace(platform=None))},
        authorize=lambda *a, **k: None,
        attach=AsyncMock(return_value=_attach_snapshot(owner_ref)),
    )
    connection = AuthorityConnection(
        authority, object(),
        {'user_id': 'owner', 'capabilities': [
            'session:create', 'session:read', 'session:control']},
    )
    result = await connection.resume(SessionRef('fixture', tip_id), {})
    assert resolved == [tip_id]
    assert result['session_id'] == owner_id
    assert result['stored_session_id'] == owner_id


@pytest.mark.asyncio
async def test_session_resume_still_remaps_legacy_cli_source(monkeypatch):
    """Pre-authority cli/tui/gui rows keep the previous remap path."""
    owner_id = 'legacy-cli-owner'
    tip_id = 'legacy-cli-tip'
    owner_ref = SessionRef('fixture', owner_id)
    resolved = []

    monkeypatch.setattr(
        'gateway.session_local_migration.resolve_local_target',
        lambda authority, actor, sid: resolved.append(sid) or owner_ref)

    authority = SimpleNamespace(
        profile_id='fixture',
        instance_id='one',
        events={},
        db=SimpleNamespace(
            get_session=lambda sid: {
                'id': tip_id, 'source': 'cli', 'chat_id': owner_id,
            } if sid == tip_id else None,
            get_compression_tip=lambda sid: tip_id,
        ),
        sessions={owner_id: SimpleNamespace(source=SimpleNamespace(platform=None))},
        authorize=lambda *a, **k: None,
        attach=AsyncMock(return_value=_attach_snapshot(owner_ref)),
    )
    connection = AuthorityConnection(
        authority, object(),
        {'user_id': 'owner', 'capabilities': [
            'session:create', 'session:read', 'session:control']},
    )
    result = await connection.resume(SessionRef('fixture', tip_id), {})
    assert resolved == [tip_id]
    assert result['session_id'] == owner_id
