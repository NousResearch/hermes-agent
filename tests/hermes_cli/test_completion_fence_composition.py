
from hermes_cli import kanban_db_connect
"""Isolated checks of operator completion fences + reviewed lifecycle union."""
import json
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb

@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'board.db'))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    with kanban_db_connect.connect(tmp_path / 'board.db') as c:
        assert Path(c.execute('PRAGMA database_list').fetchone()[2]) == tmp_path / 'board.db'
        yield c

def observed(c, t):
    return c.execute('SELECT status FROM tasks WHERE id=?', (t,)).fetchone()[0], c.execute('SELECT COALESCE(MAX(id),0) FROM task_events WHERE task_id=?', (t,)).fetchone()[0]

@pytest.mark.parametrize('change', ['status','event'])
def test_stale_observation_does_not_stage(conn,monkeypatch,change):
    t=kb.create_task(conn,title='fence',assignee='integrator')
    status,event=observed(conn,t)
    if change=='status':
        with kanban_db_connect.write_txn(conn):conn.execute("UPDATE tasks SET status='blocked' WHERE id=?",(t,))
    else: kb.add_comment(conn,t,'operator','new observation')
    def forbidden(*a,**kw):raise AssertionError('must refuse before staging')
    monkeypatch.setattr(kb,'_merge_completion_prose_artifacts',forbidden)
    before=observed(conn,t)
    assert not kb.complete_task(conn,t,expected_status=status,expected_event_id=event,fire_lifecycle_hook=False)
    assert observed(conn,t)==before

def test_staging_race_rechecked(conn,monkeypatch):
    t=kb.create_task(conn,title='race',assignee='integrator')
    status,event=observed(conn,t)
    original=kb._merge_completion_prose_artifacts
    def stage(*a,**kw):
        value=original(*a,**kw)
        kb.add_comment(conn,t,'operator','edit during staging')
        return value
    monkeypatch.setattr(kb,'_merge_completion_prose_artifacts',stage)
    assert not kb.complete_task(conn,t,expected_status=status,expected_event_id=event,fire_lifecycle_hook=False)
    assert observed(conn,t)[0]==status

def test_matching_fence_keeps_review_requirement(conn):
    t=kb.create_task(conn,title='needs independent review',assignee='integrator',review_requirement={'required':True,'owner':'pr-reviewer'})
    status,event=observed(conn,t)
    assert kb.complete_task(conn,t,expected_status=status,expected_event_id=event,summary='implementation only',fire_lifecycle_hook=False)
    assert observed(conn,t)[0]!='done'
    kinds=[r[0] for r in conn.execute('SELECT kind FROM task_events WHERE task_id=?',(t,))]
    assert 'review_handoff_required' in kinds
    assert 'delivery_accepted' not in kinds
