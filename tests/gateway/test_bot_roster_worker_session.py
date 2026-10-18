"""Regression for #140977: the canonical Bots roster answers ``worker_session``.

The unified gateway's ``profiles.list`` producer filled ``canonical_session`` / ``last_session``
but left ``worker_session`` at the ``None`` initializer it was born with
(``gateway/session_group_controls.py``, from 3e896bc390). The field is declared on the wire contract
(``ProfileWorkerSession``) and the Bot Mode roster reads it to light a profile up while a kanban/tool
worker runs (``workerActiveAt`` is ``bot?.worker_session?.last_active``), so with no producer that
indicator could never light on the unified gateway.

The producer reuses the legacy roster's semantics (``tui_gateway/methods_profiles``
``_latest_profile_session_rows``, #90268): the newest DENIED-source row (``kanban`` / ``tool`` /
``oneshot``, ``INTERNAL_LISTING_SOURCES``), never an archived or hidden row, with the worker's
``last_activity_at`` heartbeat (#72016) as ``last_active``. Both roster rows read through the two
functions exercised here -- ``_bot_sessions`` for this profile's own row, ``_sibling_sessions`` for a
sibling a host-scoped socket routes to -- so one fix reaches both, out of the cached read the other
two fields already share (no extra cost for the 5 s poll, #117257).
"""
import pytest

from gateway.session_contract import Principal
from gateway.session_group_controls import _bot_sessions, _read_bot_sessions, _sibling_sessions
from hermes_state import SessionDB

READER = Principal('uid:desk', 'alpha', frozenset({'session:operator'}), 'transport:test')


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / 'state.db')
    yield store
    store.close()


def _authority(db, **extra):
    class _Authority:  # weak-referenceable: ``_sibling_sessions`` tracks degraded authorities in a WeakSet
        def __init__(self):
            self.db, self.retiring, self.retired, self.profile_id = db, False, False, 'alpha'
            self.__dict__.update(extra)

    return _Authority()


def _session(db, sid, *, source='kanban', title='', last_active=0, hidden=0, archived=0):
    """A stored row with a fixed activity stamp, so "newest worker" is decided by this test."""
    db.create_session(sid, source=source)
    with db._lock:
        db._conn.execute(
            'UPDATE sessions SET title=?, last_activity_at=?, started_at=?, hidden=?, archived=? WHERE id=?',
            (title, last_active, last_active, hidden, archived, sid))


def test_the_newest_live_worker_is_reported_with_the_contract_fields(db):
    _session(db, 'work-old', title='Task 4', last_active=1000)
    _session(db, 'work-new', title='Task 9', last_active=3000)
    _session(db, 'tool-run', source='tool', title='Subagent', last_active=2000)
    _session(db, 'oneshot-run', source='oneshot', title='One-shot', last_active=2500)
    _session(db, 'chat', source='gui', title='Chat', last_active=5000)  # newest, but a conversation
    _session(db, 'work-archived', title='Retired', last_active=9000, archived=1)
    _session(db, 'work-hidden', title='Internal', last_active=9500, hidden=1)

    fields = _read_bot_sessions(db, 'uid:desk', False)

    assert fields['worker_session'] == {'id': 'work-new', 'source': 'kanban',
                                        'title': 'Task 9', 'last_active': 3000}, fields['worker_session']
    # The conversation fields keep their own contract: the worker never becomes the last session.
    assert (fields['last_session'] or {}).get('id') != 'work-new'
    assert [key for key in fields] == ['last_session', 'worker_session', 'canonical_session']


def test_a_profile_without_workers_reports_none(db):
    _session(db, 'chat', source='gui', title='Chat', last_active=1000)

    assert _read_bot_sessions(db, 'uid:desk', False)['worker_session'] is None


def test_the_own_row_and_a_host_scoped_sibling_row_both_carry_the_worker(db, tmp_path):
    _session(db, 'work-alpha', title='Task 1', last_active=2000)
    authority = _authority(db)

    # The own row (``_profiles`` does ``row.update(_bot_sessions(authority, actor))``).
    own = _bot_sessions(authority, READER)
    assert own['worker_session']['id'] == 'work-alpha'
    assert own['worker_session']['last_active'] == 2000
    # A sibling a host-scoped socket routes to (``_sibling_sessions(authority, reader)``).
    assert _sibling_sessions(authority, READER)['worker_session']['id'] == 'work-alpha'
    # A retiring sibling is not read at all, so its row stays metadata-only.
    authority.retiring = True
    assert _sibling_sessions(authority, READER) == {}
