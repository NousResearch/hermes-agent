"""An admitted completion keeps main's post-compression guarantee: the executed event carries the
ownership proof, so one transient owner read in the turn's resolver never drops it (N77)."""
import asyncio

import pytest

from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner


@pytest.mark.asyncio
async def test_admitted_completion_survives_a_transient_owner_read_after_compression(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import admission_row
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_seed import seed_messages
    from tests.gateway.test_completion_admission import pending

    runner, _, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='human', source=source, message_id='human'))
    key = runner.session_store._generate_session_key(source)
    sid = receipt.ref.session_id
    event = dict(pending(key, 'ci-exec'), parent_session_id=sid)
    assert await runner._deliver_async_delegation_group([event]) is True
    row = admission_row(authority.persistence.admissions.list(session_id=sid, pending_only=False)[-1])
    db = authority.persistence
    seed_messages(db, sid, {'role': 'user', 'content': 'human'})
    child = sid + '-child'  # the owner compresses after admission
    db.lineage.publish_compression_child(sid, CompressionChild(session_id=child, source='discord'),
                                         [NewMessage.from_live({'role': 'user', 'content': 'summary'})],
                                         require_compression_lease=False)
    assert runner.session_store.advance_compression_session(key, sid, child) is not None
    resolved = []

    async def turn(ev):
        resolved.append(await runner._hmwa_resolve_session(ev, ev.source))
        return 'ok'

    async def transient(*a, **k):
        return 'retry', a[0], None
    monkeypatch.setattr(runner, '_handle_message', turn)
    monkeypatch.setattr(runner, '_lookup_completion_owner', transient)
    await execute_admission(authority, SessionRef(authority.profile_id, sid), row)
    assert resolved and resolved[0] is not None, 'admitted completion dropped before its turn'
    assert resolved[0][1].session_id == child
    await asyncio.sleep(0)
