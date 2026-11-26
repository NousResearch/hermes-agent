import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from persistence.housekeeping.maintenance import PruneCandidate, PruneResult


class _FakeStore:
    """The storage interface handle the sessions dispatcher opens."""

    def __init__(self, **repositories):
        self.scope = SimpleNamespace(profile_dir=Path("."))
        self.admissions = SimpleNamespace(retired_candidates=lambda kind: [])
        for name, repository in repositories.items():
            setattr(self, name, repository)

    def ready(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _use_store(monkeypatch, store):
    import contextlib

    import persistence_assembly
    monkeypatch.setattr(persistence_assembly, "open_persistence", lambda scope, eager=False: store)
    monkeypatch.setattr(persistence_assembly, "open_persistence_reader", lambda target: contextlib.nullcontext(store))


def test_sessions_delete_accepts_unique_id_prefix(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    captured = {}

    class FakeSessions:
        def resolve_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260315_092437_c9a6ff"

        def get(self, session_id):
            return SimpleNamespace(session_id=session_id, pinned=False)

        def delete(self, session_id, **kwargs):
            captured["deleted"] = session_id
            return (session_id,)

    store = _FakeStore(sessions=FakeSessions())
    store.close = lambda: captured.__setitem__("closed", True)
    _use_store(monkeypatch, store)
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
    with what to do), not an uncaught AuthorityRefusal traceback."""
    import hermes_cli.main as main_mod
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import session_row
    from tests.fixtures.storage_seed import seed_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, 's1', 'cli')
        epoch = store.runtime.begin_epoch(instance_id='a')
        store.admissions.admit(epoch=epoch, principal_id='p', session_id='s1', request_id='r', payload={'text': 'x'})
        if status == 'unknown':
            store.admissions.claim(epoch=epoch, session_id='s1')
            store.admissions.recover(epoch=store.runtime.begin_epoch(instance_id='b'))
    monkeypatch.setattr(sys, 'argv', ['hermes', 'sessions', 'delete', 's1', '--yes'])
    with pytest.raises(SystemExit) as exited:
        main_mod.main()
    output = capsys.readouterr().out
    assert exited.value.code == 1
    assert ('hermes sessions discard s1' in output) == (status == 'unknown'), output
    assert 'queued or running turn' in output or status == 'unknown', output
    with open_persistence(profile_scope(home=tmp_path)) as reopened:
        assert session_row(reopened, 's1') is not None


def _run_prune(monkeypatch, capsys, argv_tail, candidates=None, skipped_open=0, sub="prune"):
    """Run `hermes sessions <sub> <argv_tail>` (prune/archive) against a fake store handle, capturing
    the bounds and criteria passed to list_prune_candidates. Auto-confirms."""
    import hermes_cli.main as main_mod
    from persistence.conversation.sessions import ArchiveChange

    seen = {}
    rows = candidates if candidates is not None else [
        PruneCandidate(
            session_id="20260101_000000_aaaaaa", source="cron", title="oldest run", model=None,
            started_at=1_600_000_000.0, last_active=1_600_000_050.0, ended_at=1_600_000_100.0,
            message_count=2, archived=0,
        ),
        PruneCandidate(
            session_id="20260601_000000_bbbbbb", source="cron", title="newest run", model=None,
            started_at=1_700_000_000.0, last_active=1_700_000_050.0, ended_at=1_700_000_100.0,
            message_count=4, archived=0,
        ),
    ]

    class FakeMaintenance:
        def list_prune_candidates(self, older_than_days=None, source=None, *, criteria, whole_lineages=False):
            seen.update(older_than_days=older_than_days, source=source, criteria=criteria,
                        whole_lineages=whole_lineages)
            return tuple(rows)

        def count_open_prune_matches(self, **kwargs):
            # Same filters as the preview. `whole_lineages` is the preview's selection mode, not a
            # filter: an open row is never a compression ancestor, so the count never takes it.
            assert kwargs == {k: v for k, v in seen.items() if k != "whole_lineages"}
            return skipped_open

        def count_prune_matches(self, older_than_days=None, source=None, *, criteria, pinned_only=False):
            return 0 if pinned_only else len(rows)

        def prune_sessions(self, older_than_days=90, source=None, **kwargs):
            return PruneResult(count=len(rows))

    class FakeSessions:
        def set_archived(self, session_id, archived, notify=True):
            return ArchiveChange(changed=True)

    _use_store(monkeypatch, _FakeStore(maintenance=FakeMaintenance(), sessions=FakeSessions()))
    monkeypatch.setattr(
        sys, "argv", ["hermes", "sessions", sub, *argv_tail]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    main_mod.main()
    return seen, capsys.readouterr().out

def test_sessions_prune_bare_keeps_90_day_default(monkeypatch, capsys):
    """A truly bare `hermes sessions prune` keeps the implicit 90-day cutoff."""
    import time as _time

    filters, _out = _run_prune(monkeypatch, capsys, [])
    assert filters["criteria"].last_active_before is not None
    assert filters["criteria"].last_active_before == pytest.approx(
        _time.time() - 90 * 86400, abs=60
    )


def test_archive_include_open_flag_reaches_selection(monkeypatch, capsys):
    """`sessions archive --include-open` lifts the ended-only selector: desktop sessions never set
    ended_at, so without the flag they can never match any filter (#139597)."""
    filters, _out = _run_prune(
        monkeypatch, capsys, ["--title", "dupes", "--include-open", "--yes"], sub="archive")
    assert filters["criteria"].include_open is True


def test_archive_without_flag_reports_open_skips(monkeypatch, capsys):
    """Without --include-open, matches that are still open are skipped silently no longer: the
    CLI says how many and names the opt-in flag."""
    filters, out = _run_prune(
        monkeypatch, capsys, ["--title", "dupes", "--yes"], skipped_open=2, sub="archive")
    assert filters["criteria"].include_open is False
    assert "2 matching sessions still open" in out
    assert "--include-open" in out
