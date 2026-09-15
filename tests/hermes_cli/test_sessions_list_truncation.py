"""B7: `hermes sessions list` silently dropped rows past --limit.

The cap lived inside the DB query (`list_sessions_rich(..., limit=args.limit)`),
so a user with 35 sessions saw 20 rows and assumed that was everything. `list`
now fetches limit+1 to detect truncation and prints a dim footer when rows
were cut (mirroring the `pets.py` precedent). --limit 0 now means unlimited
(SQLite LIMIT 0 previously returned zero rows).
"""

from argparse import Namespace

import hermes_cli.sessions_cmd as sc
from hermes_state import SessionDB


def _args(**kw):
    base = dict(
        sessions_action="list",
        session_id=None, title=None, yes=True, source=None, path=None,
        from_source=None, dry_run=False, older_than=None, newer_than=None,
        before=None, after=None, limit=20, workspace=None,
    )
    base.update(kw)
    return Namespace(**base)


def _seed(n):
    db = SessionDB()
    for i in range(n):
        db.create_session(session_id=f"sess-{i:03d}", source="cli")
    db.close()


def _rows(out):
    return [line for line in out.splitlines() if "sess-" in line]


def test_truncation_footer_shown_when_capped(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(25)
    rc = sc.cmd_sessions(_args(limit=20))
    assert rc in (0, None)
    out = capsys.readouterr().out
    rows = _rows(out)
    assert len(rows) == 20, f"expected exactly --limit rows, got {len(rows)}"
    assert "… and more" in out
    assert "--limit 0" in out


def test_no_footer_when_under_limit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(5)
    rc = sc.cmd_sessions(_args(limit=20))
    assert rc in (0, None)
    out = capsys.readouterr().out
    assert len(_rows(out)) == 5
    assert "… and more" not in out


def test_limit_zero_shows_everything(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(25)
    rc = sc.cmd_sessions(_args(limit=0))
    assert rc in (0, None)
    out = capsys.readouterr().out
    assert len(_rows(out)) == 25
    assert "… and more" not in out
