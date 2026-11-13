"""A persisted messaging wait resumes through the same canonical owner as human input."""
from types import SimpleNamespace

import pytest

from tests.gateway.test_automation_compression_media import _discord_owner, _admit_human


@pytest.mark.asyncio
async def test_messaging_timer_commits_before_claim_and_deduplicates(tmp_path, monkeypatch):
    from agent import auxiliary_client
    from gateway.platforms.event import MessageEvent
    from gateway.run_goal_waits import poll_goal_waits
    from gateway.session_results import execution_result
    from hermes_cli import goals
    from hermes_state_runtime import claim_session_input, settle_session_input, list_session_admissions
    runner, _adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    turns, judges = [], []
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        receipt = await _admit_human(runner, authority, MessageEvent(text='work', source=source, message_id='human'))
        ref = receipt.ref
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                             generation=row['generation'], outcome='completed')
        mgr = goals.GoalManager(ref.session_id)
        mgr.set('finish deferred messaging work')
        mgr.wait_for_seconds(1)
        mgr.state.waiting_until = 1.0
        mgr._save()
        before = authority.db.get_meta('goal:' + ref.session_id)
        async def turn(event, source, key, generation):
            turns.append(event.text)
            execution_result.get()['result'] = {'final_response': 'finished', 'messages': [], 'failed': False}
            return 'finished'
        def judge(**kwargs):
            judges.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict":"done"}'))])
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        monkeypatch.setattr(auxiliary_client, 'call_llm', judge)
        await poll_goal_waits(runner)
        await poll_goal_waits(runner)  # lost ACK before a claim
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert len(rows) == 2 and rows[-1]['status'] == 'queued'
        assert authority.db.get_meta('goal:' + ref.session_id) == before and turns == judges == []
        await authority._drain(ref)
        assert turns == [mgr.next_continuation_prompt()] and len(judges) == 1
        assert goals.GoalManager(ref.session_id).state.status == 'done'
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert [(r['status'], r['outcome']) for r in rows] == [('terminal', 'completed')] * 2
        await poll_goal_waits(runner)
        assert len(turns) == 1
    finally:
        authority.db.close()
