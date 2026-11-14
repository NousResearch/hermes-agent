"""A bad legacy goal row must not disable the unattended gateway wake driver."""

import pytest

from tests.gateway.test_session_end_delegations import _admissions, _local


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_raw', [
    pytest.param('{invalid json', id='invalid-json'),
    pytest.param('null', id='null-shape'),
    pytest.param('[]', id='array-shape'),
    pytest.param('{"goal":"legacy","turns_used":"invalid"}', id='invalid-number'),
    pytest.param('{"goal":"legacy","turns_used":[1]}', id='array-number'),
    pytest.param('{"goal":"legacy","turns_used":1e999}', id='overflow-number'),
    # Exceeds both Python recursion limits and Python 3.14's separate JSON C-stack budget.
    pytest.param('[' * 100_000 + '0' + ']' * 100_000, id='recursive-json'),
])
async def test_bad_legacy_goal_row_does_not_starve_healthy_parked_goal(tmp_path, monkeypatch, bad_raw):
    from gateway.run_goal_waits import poll_goal_waits
    from hermes_cli import goals

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(goals, '_get_session_db', lambda: authority.db)
        live = authority.sessions[ref.session_id]
        assert not live.subscribers
        target = runner.session_store.lookup_by_session_key(live.route).session_id

        manager = goals.GoalManager(target)
        manager.set('Complete deferred work without a viewer')
        manager.wait_for_seconds(1)
        manager.state.waiting_until = 1.0
        manager._save()

        # A migrated or damaged unrelated goal must not abort the entire scan.
        authority.db.set_meta('goal:invalid-legacy', bad_raw)
        assert _admissions(authority, ref) == []

        await poll_goal_waits(runner)

        queued = _admissions(authority, ref)
        assert len(queued) == 1, 'ready goal must be admitted despite an unreadable other row'
        assert queued[0]['status'] == 'queued'
        assert authority.db.get_meta('goal:invalid-legacy') == bad_raw
    finally:
        authority.db.close()
