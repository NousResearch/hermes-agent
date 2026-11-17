"""Regression: a lost *human* turn settles with resend advice instead of pausing its chat.

``recover_session_inputs`` fences a turn that was ``started`` on a dead epoch ``unknown``. For a
cron/webhook/kanban row ``settle_recovered_unattended`` already settles it (#141357). A person's
turn was left alone on purpose — only they know whether to resend — so the session paused until
they typed ``/discard``. On a headless gateway they are not at boot, so the chat sat paused and
every message queued behind it stalled. ``settle_recovered_human`` settles it ``interrupted`` with
``FAILED_TURN_NOTICE`` (resend advice) and never replays it.
"""
import pytest


def _row(session_id='s1', principal='uid:1000', status='unknown', payload=None):
    return {'admission_id': 'a1', 'target_session_id': session_id, 'status': status,
            'principal_id': principal, 'payload': payload or {'text': 'continue'}, 'generation': 1}


def test_a_persons_lost_turn_is_resumable():
    from gateway.session_unattended_unknown import _resumable_unknown
    assert _resumable_unknown([_row()], {'s1': 'discord'}) == [_row()]


def test_unattended_turns_are_not_double_handled():
    """Cron/webhook/-z rows belong to settle_recovered_unattended, never to the human pass."""
    from gateway.session_unattended_unknown import _resumable_unknown
    rows = [_row('s1', principal='cron-owner'), _row('s2', payload={'local_automation_v1': {}}),
            _row('s3', payload={'text': 'x', 'unattended': True})]
    assert _resumable_unknown(rows, {'s1': 'cron', 's2': 'discord', 's3': 'oneshot'}) == []


def test_a_non_unknown_row_is_never_touched():
    """Only the fence state is in scope; a live or settled turn must not be resolved."""
    from gateway.session_unattended_unknown import _resumable_unknown
    rows = [_row('s1', status='started'), _row('s2', status='terminal'), _row('s3', status='queued')]
    assert _resumable_unknown(rows, {'s1': 'discord', 's2': 'discord', 's3': 'discord'}) == []


def test_oneshot_sessions_are_skipped():
    """A ``-z`` session has no chat to resend into; pausing it is still the honest outcome."""
    from gateway.session_unattended_unknown import _resumable_unknown
    assert _resumable_unknown([_row('s1')], {'s1': 'oneshot'}) == []


@pytest.mark.parametrize('source', ['discord', 'telegram', 'cli', 'tui', 'api_server', ''])
def test_every_interactive_surface_resumes(source):
    from gateway.session_unattended_unknown import _resumable_unknown
    assert len(_resumable_unknown([_row()], {'s1': source})) == 1


def test_unknown_session_source_still_resumes():
    """A row whose session is gone (stale reference) is the caller's problem, not a silent skip."""
    from gateway.session_unattended_unknown import _resumable_unknown
    assert len(_resumable_unknown([_row()], {})) == 1


def test_human_settlement_end_to_end(tmp_path, monkeypatch):
    """The boot pass really resolves the row: terminal/interrupted and a resend notice committed."""
    import contextlib
    import sqlite3

    from agent.turn_failure_copy import FAILED_TURN_NOTICE
    from gateway import session_unattended_unknown as mod

    db_path = tmp_path / 'state.db'
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, runtime_revision INTEGER DEFAULT 0)")
    con.execute("CREATE TABLE session_admissions (admission_id TEXT PRIMARY KEY, seq INTEGER,"
                " target_session_id TEXT, principal_id TEXT, payload_json TEXT, status TEXT,"
                " generation INTEGER, owner_epoch INTEGER, outcome TEXT)")
    con.execute("INSERT INTO sessions (id, source) VALUES ('s_human','discord')")
    con.execute("INSERT INTO session_admissions VALUES ('a_human',1,'s_human','uid:1000','{\"text\": \"continue\"}',"
                "'unknown',1,0,NULL)")
    con.commit()
    con.close()

    resolved, closed = [], []

    class _Db:
        def __init__(self, path):
            self.db_path = str(path)

        @contextlib.contextmanager
        def _read_ctx(self):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _fake_resolve(db, *, epoch, admission_id, generation, _terminal_write=None, _captured=None):
        resolved.append(admission_id)
        _terminal_write(None, {'admission_id': admission_id})
        return {'status': 'terminal', 'outcome': 'interrupted'}

    monkeypatch.setattr('hermes_state_runtime.resolve_unknown_session_input', _fake_resolve)
    monkeypatch.setattr('hermes_state_runtime._row', lambda r: dict(r))
    monkeypatch.setattr('gateway.session_ingress_media.release_admission_media', lambda db, aid: None)
    monkeypatch.setattr('gateway.session_results.close_discarded_turn',
                        lambda db, conn, row, notice=None: closed.append((row['admission_id'], notice)))

    assert mod.settle_recovered_human(_Db(db_path), epoch=1) == 1
    assert resolved == ['a_human']
    assert closed == [('a_human', FAILED_TURN_NOTICE)]
