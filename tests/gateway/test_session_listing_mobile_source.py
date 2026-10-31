"""#146814: the stored-session picker must offer `source='mobile'` rows.

Desktop resume of a session whose runtime is gone goes through
``stored_picker_sessions``. A row whose source is neither a local label nor a
messaging platform is skipped *before any db read* ("neither a local nor a
messaging label (acp, unknown, ...): nothing this picker resumes"), so
phone-viewport dashboard sessions — the client self-reports ``source='mobile'``
— could neither be resumed nor deleted from the Desktop. These tests pin the
source gate: ``mobile`` is local and reaches the resume path, unknown labels are
still skipped.
"""

from __future__ import annotations

import contextlib

import pytest

from gateway import session_listing


class _Actor:
    profile_id = 'p1'
    transport_id = 't-native'
    subject = 'subj'
    capabilities = frozenset({'session:read', 'session:create'})


class _Conn:
    """Just enough sqlite surface for `local_receipt`'s single SELECT."""

    class _Cursor:
        def fetchone(self):
            return None

    def execute(self, *args, **kwargs):
        return self._Cursor()


class _DB:
    def __init__(self, rows):
        self.rows = rows
        self.tips = []
        self.sessions = []

    def list_sessions_rich(self, **kwargs):
        return list(self.rows)

    def get_compression_tip(self, sid):
        self.tips.append(sid)
        return sid

    def get_session(self, sid):
        self.sessions.append(sid)
        source = {r['id']: r['source'] for r in self.rows}[sid]
        return {'source': source, 'chat_id': f'chat-{sid}'}

    def _own_profile_name(self):
        return 'default'

    @contextlib.contextmanager
    def _read_ctx(self):
        # Both `local_receipt` (no stored receipt for this session) and
        # `authorize_history` (stubbed above) go through this context manager.
        yield _Conn()


class _Authority:
    def __init__(self, rows):
        self.db = _DB(rows)
        self.profile_id = 'p1'
        self.sessions = {}
        self._native_legacy_transports = {}

    def _handle(self, ref):  # pragma: no cover - only reached by a live runtime
        raise AssertionError('unexpected _handle call')


def _authority(actor, rows):
    auth = _Authority(rows)
    auth._native_legacy_transports[actor.transport_id] = actor
    return auth


@pytest.fixture(autouse=True)
def _stub_policy(monkeypatch):
    """Non-interactive stand-ins for the legacy-row and authorization gates."""
    monkeypatch.setattr(session_listing, 'validate_legacy_row', lambda *a, **k: None)
    monkeypatch.setattr(session_listing, 'authorize_history', lambda *a, **k: True)


def test_mobile_rows_reach_the_resume_path():
    actor = _Actor()
    auth = _authority(actor, [{'id': 's-mobile', 'source': 'mobile', 'title': 'phone chat'}])

    out = session_listing.stored_picker_sessions(auth, actor, limit=5)

    assert auth.db.sessions == ['s-mobile'], 'a mobile row must be looked up, not skipped'
    assert [row['id'] for row in out['sessions']] == ['s-mobile']
    assert out['scope'] == 'stored'


def test_unknown_sources_are_skipped_before_any_db_read():
    actor = _Actor()
    auth = _authority(actor, [{'id': 's-acp', 'source': 'acp'}])

    out = session_listing.stored_picker_sessions(auth, actor, limit=5)

    assert auth.db.tips == [] and auth.db.sessions == []
    assert out['sessions'] == []
