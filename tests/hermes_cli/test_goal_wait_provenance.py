"""Goal evaluation must not overwrite a control issued during the auxiliary call."""
import pytest


@pytest.mark.parametrize('control', ['pause', 'clear', 'replace', 'manual-wait', 'legacy-metadata'])
def test_control_during_judge_wins(tmp_path, monkeypatch, control):
    from hermes_cli import goals
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)
    manager = goals.GoalManager('parent')
    manager.set('original')
    saved = []
    def legacy_metadata():
        import json
        raw = json.loads(db.get_meta('goal:parent'))
        raw['unknown_legacy_control'] = True
        db.set_meta('goal:parent', json.dumps(raw, indent=2))
    def judge(*args, **kwargs):
        current = goals.GoalManager('parent')
        {'pause': current.pause, 'clear': current.clear, 'replace': lambda: current.set('replacement'),
         'legacy-metadata': legacy_metadata,
         'manual-wait': lambda: current.wait_for_seconds(900, 'manual cooldown')}[control]()
        saved.append(db.get_meta('goal:parent'))
        return 'done', 'old evaluation', False, None, False
    monkeypatch.setattr(goals, 'judge_goal', judge)
    try:
        decision = manager.evaluate_after_turn('result')
        assert db.get_meta('goal:parent') == saved[0], 'judge resurrected a superseded goal decision'
        assert not decision['should_continue'] and decision['verdict'] != 'done'
    finally:
        db.close()


@pytest.mark.parametrize('flag', ['true', 'false', '1', '"true"', 'null'])
def test_worker_wait_discriminator_is_an_exact_bool(flag):
    from hermes_cli.goals import _parse_judge_response
    verdict, _reason, failed, directive = _parse_judge_response(
        '{"verdict":"wait","wait_for_seconds":600,"wait_on_delegations":' + flag + '}')
    if flag in {'true', 'false'}:
        assert not failed and directive == {'seconds': 600, 'delegations': flag == 'true'}
    else:
        assert failed and verdict == 'continue' and directive is None


@pytest.mark.parametrize('workers,dependency,complete_during', [(0, True, False), (1, True, False),
                                                             (1, False, False), (1, True, True)])
def test_judge_worker_dependency_uses_owned_durable_evidence(tmp_path, monkeypatch, workers, dependency, complete_during):
    from hermes_cli import goals
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)
    db.create_session('parent', source='cli')
    if workers:
        db._execute_write(lambda conn: conn.execute("INSERT INTO async_delegations "
            "(delegation_id, origin_session, parent_session_id, state, dispatched_at, updated_at) "
            "VALUES('owned-worker','route','parent','running',1,1)"))
    manager = goals.GoalManager('parent')
    manager.set('finish work')
    def judge(*args, **kwargs):
        if complete_during:
            db._execute_write(lambda conn: conn.execute("UPDATE async_delegations SET state='completed'"))
        return ('wait', 'waiting for work' if dependency else 'rate limit', False,
                {'seconds': 600, 'delegations': dependency}, False)
    monkeypatch.setattr(goals, 'judge_goal', judge)
    try:
        result = manager.evaluate_after_turn('waiting', active_delegations=workers)
        if dependency and (not workers or complete_during):
            assert result['should_continue'] and manager.state.waiting_until == 0
        else:
            assert result['verdict'] == 'wait'
            if not dependency:
                db._execute_write(lambda conn: conn.execute("UPDATE async_delegations SET state='completed'"))
                assert goals.GoalManager('parent').is_waiting(), 'worker completion cleared a cooldown'
    finally:
        db.close()


def test_same_clock_replacement_wait_has_a_new_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import goals
    from hermes_cli.goals_waits import state_token
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)
    monkeypatch.setattr(goals, 'time', SimpleNamespace(time=lambda: 100.0))
    try:
        mgr = goals.GoalManager('parent')
        mgr.set('same goal')
        mgr.wait_for_seconds(60)
        old = state_token(mgr.state)
        mgr.wait_for_seconds(60)
        assert state_token(mgr.state) != old, 'manual wait recreated an already admitted token'
    finally:
        db.close()


def test_legacy_worker_count_is_not_completion_evidence(tmp_path, monkeypatch):
    from hermes_cli import goals
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)
    monkeypatch.setattr(goals, 'count_active_delegations', lambda *a: 0)
    try:
        mgr = goals.GoalManager('parent')
        mgr.set('finish worker work')
        mgr.wait_for_seconds(600, on_delegations=4)
        assert goals.GoalManager('parent').is_waiting(), 'an empty restart roster is not worker completion evidence'
    finally:
        db.close()
