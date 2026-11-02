"""Custom DB ownership requires a matching explicit DB/board pair."""
from pathlib import Path
import sqlite3
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_artifacts as kba


@pytest.mark.parametrize('board', ['default', 'research'])
def test_completion_with_explicit_custom_db_board(tmp_path, monkeypatch, board):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_KANBAN_DB', str(tmp_path / 'custom.sqlite'))
    monkeypatch.setenv('HERMES_KANBAN_BOARD', board)
    kb.init_db()
    ws = tmp_path / 'workspace'
    ws.mkdir()
    artifact = ws / 'report.txt'
    artifact.write_bytes(b'custom db report')
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title='custom db', workspace_kind='dir', workspace_path=str(ws))
        assert kb.complete_task(conn, task_id, summary='done', metadata={'artifacts': [str(artifact)]}, fire_lifecycle_hook=False)
        event = [e for e in kb.list_events(conn, task_id) if e.kind == 'completed'][-1]
        stored = Path(event.payload['artifacts'][0])
        root = home / 'kanban' if board == 'default' else home / 'kanban' / 'boards' / board
        assert stored == root / 'attachments' / task_id / 'report.txt'
        assert stored.read_bytes() == b'custom db report'
        assert kb.get_task(conn, task_id).status == 'done'
    assert artifact.read_bytes() == b'custom db report'


@pytest.mark.parametrize('board,pin_matches', [('', True), ('../escape', True), ('default', False)])
def test_custom_db_rejects_missing_invalid_or_mismatched_pin(tmp_path, monkeypatch, board, pin_matches):
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path / 'home'))
    db = tmp_path / 'custom.sqlite'
    monkeypatch.setenv('HERMES_KANBAN_DB', str(db if pin_matches else tmp_path / 'other.sqlite'))
    monkeypatch.setenv('HERMES_KANBAN_BOARD', board)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        with pytest.raises(kb.ArtifactPreservationError, match='ownership'):
            kba._connection_board(conn)


@pytest.mark.parametrize('board', ['default', 'research'])
def test_canonical_connection_beats_conflicting_explicit_pin(tmp_path, monkeypatch, board):
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    db = tmp_path / 'kanban.db' if board == 'default' else tmp_path / 'kanban' / 'boards' / board / 'kanban.db'
    db.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HERMES_KANBAN_DB', str(db))
    monkeypatch.setenv('HERMES_KANBAN_BOARD', 'wrong-board')
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        assert kba._connection_board(conn) == board
