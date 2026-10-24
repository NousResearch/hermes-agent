from contextlib import contextmanager
import sqlite3

import pytest
from fastapi import HTTPException

from plugins.kanban.dashboard import plugin_api as api


@pytest.fixture
def fake_board(monkeypatch):
    conn = sqlite3.connect(':memory:')

    @contextmanager
    def board(_board):
        yield 'default', conn

    monkeypatch.setattr(api, '_board_conn', board)
    return conn


def test_mirror_list_only_serializes_safe_metadata(fake_board, monkeypatch):
    monkeypatch.setattr(api.kbsm, 'list_mirrors', lambda *_a, **_k: [{
        'id': 8, 'profile': 'coder', 'platform': 'telegram', 'chat_id': 'chat',
        'thread_id': None, 'session_id': 's', 'title': 'Session', 'status': 'completed',
        'received_at': 1, 'started_at': 2, 'completed_at': 3, 'updated_at': 3,
        'archived_at': None, 'promoted_task_id': None, 'transcript': 'PRIVATE MESSAGE',
    }])
    result = api.list_session_mirrors(include_archived=False, limit=100, board=None)
    assert len(result['mirrors']) == 1
    assert 'transcript' not in result['mirrors'][0]
    assert result['mirrors'][0]['session_id'] == 's'


def test_archive_and_delete_return_404_for_missing_mirror(fake_board, monkeypatch):
    monkeypatch.setattr(api.kbsm, 'archive_mirror', lambda *_: False)
    monkeypatch.setattr(api.kbsm, 'delete_mirror', lambda *_: False)
    with pytest.raises(HTTPException) as archived:
        api.archive_session_mirror(55, board=None)
    assert archived.value.status_code == 404
    with pytest.raises(HTTPException) as deleted:
        api.delete_session_mirror(55, board=None)
    assert deleted.value.status_code == 404


def test_promote_requires_nonempty_user_title(fake_board, monkeypatch):
    called = []
    monkeypatch.setattr(api, '_with_board_pinned', lambda _b, fn: fn())
    monkeypatch.setattr(api.kbsm, 'promote_mirror', lambda *_a, **kw: called.append(kw) or 'task-id')
    with pytest.raises(HTTPException) as exc:
        api.promote_session_mirror(4, api.PromoteMirrorBody(title=' '), board=None)
    assert exc.value.status_code == 400
    assert called == []


def test_promote_passes_board_and_user_authored_fields(fake_board, monkeypatch):
    received = {}
    monkeypatch.setattr(api.kanban_db, 'get_current_board', lambda: 'default')
    monkeypatch.setattr(api, '_with_board_pinned', lambda _b, fn: fn())

    def promote(conn, mirror_id, **kwargs):
        received.update(kwargs)
        return 'task-id'

    monkeypatch.setattr(api.kbsm, 'promote_mirror', promote)
    result = api.promote_session_mirror(4, api.PromoteMirrorBody(title='User title', body='User description'), board=None)
    assert result == {'task_id': 'task-id'}
    assert received == {'title': 'User title', 'body': 'User description', 'created_by': 'dashboard', 'board': 'default'}
