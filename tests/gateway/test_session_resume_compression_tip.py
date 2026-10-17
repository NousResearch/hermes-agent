"""Fast unit test of session.resume's legacy-source branch: a pre-authority cli/tui/gui tip is handed
to resolve_local_target (monkeypatched here). Canonical ``source=local`` segments are covered end to
end, with a real authority and SQLite, by test_local_resume_compression_child.py (LC-12)."""
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
