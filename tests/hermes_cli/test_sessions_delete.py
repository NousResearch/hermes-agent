import sys

import pytest

def test_sessions_delete_accepts_unique_id_prefix(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260315_092437_c9a6ff"

        def get_session(self, session_id):
            return {"id": session_id, "pinned": 0}

        def delete_session(self, session_id, **kwargs):
            captured["deleted"] = session_id
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "delete", "20260315_092437_c9a6", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert captured == {
        "resolved_from": "20260315_092437_c9a6",
        "deleted": "20260315_092437_c9a6ff",
        "closed": True,
    }
    assert "Deleted session '20260315_092437_c9a6ff'." in output

@pytest.mark.parametrize('status', ['queued', 'unknown'])
def test_sessions_delete_with_unfinished_turn_points_to_discard(tmp_path, monkeypatch, capsys, status):
    """P3: the ledger's session_busy / unknown_execution refusal is a user-facing answer (exit 1
    with what to do), not an uncaught RuntimeStoreError traceback."""
    import hermes_cli.main as main_mod
    import hermes_state
    import hermes_state_runtime as rt
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s1', source='cli')
    epoch = rt.begin_runtime_epoch(db, instance_id='a')
    rt.admit_session_input(db, epoch=epoch, principal_id='p', session_id='s1', request_id='r', payload={'text': 'x'})
    if status == 'unknown':
        rt.claim_session_input(db, epoch=epoch, session_id='s1')
        rt.recover_session_inputs(db, epoch=rt.begin_runtime_epoch(db, instance_id='b'))
    monkeypatch.setattr(hermes_state, 'SessionDB', lambda *args, **kwargs: db)
    monkeypatch.setattr(sys, 'argv', ['hermes', 'sessions', 'delete', 's1', '--yes'])
    with pytest.raises(SystemExit) as exited:
        main_mod.main()
    output = capsys.readouterr().out
    assert exited.value.code == 1
    assert ('hermes sessions discard s1' in output) == (status == 'unknown'), output
    assert 'queued or running turn' in output or status == 'unknown', output
    reopened = SessionDB(tmp_path / 'state.db')
    assert reopened.get_session('s1') is not None
    reopened.close()


def _run_prune(monkeypatch, capsys, argv_tail, candidates=None, skipped_open=0):
    """Run `hermes sessions prune <argv_tail>` against a FakeDB, capturing
    the filter kwargs passed to list_prune_candidates. Auto-confirms."""
    import hermes_cli.main as main_mod
    import hermes_state

    seen = {}
    rows = candidates if candidates is not None else [
        {
            "id": "20260101_000000_aaaaaa",
            "source": "cron",
            "title": "oldest run",
            "started_at": 1_600_000_000.0,
            "last_active": 1_600_000_050.0,
            "ended_at": 1_600_000_100.0,
            "message_count": 2,
            "archived": 0,
        },
        {
            "id": "20260601_000000_bbbbbb",
            "source": "cron",
            "title": "newest run",
            "started_at": 1_700_000_000.0,
            "last_active": 1_700_000_050.0,
            "ended_at": 1_700_000_100.0,
            "message_count": 4,
            "archived": 0,
        },
    ]

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            seen.update(kwargs)
            return rows

        def count_open_prune_matches(self, **kwargs):
            # Same filters as the preview. `whole_lineages` is the preview's selection mode, not a
            # filter: an open row is never a compression ancestor, so the count never takes it.
            assert kwargs == {k: v for k, v in seen.items() if k != "whole_lineages"}
            return skipped_open

        def count_prune_matches(self, pinned_only=False, **kwargs):
            return 0 if pinned_only else len(rows)

        def prune_sessions(self, **kwargs):
            return len(rows)

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys, "argv", ["hermes", "sessions", "prune", *argv_tail]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    main_mod.main()
    return seen, capsys.readouterr().out

def test_sessions_prune_bare_keeps_90_day_default(monkeypatch, capsys):
    """A truly bare `hermes sessions prune` keeps the implicit 90-day cutoff."""
    import time as _time

    filters, _out = _run_prune(monkeypatch, capsys, [])
    assert filters["last_active_before"] is not None
    assert filters["last_active_before"] == pytest.approx(
        _time.time() - 90 * 86400, abs=60
    )
