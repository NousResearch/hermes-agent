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

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
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
            assert kwargs == seen
            return skipped_open

        def count_prune_matches(self, **kwargs):
            return len(rows)

        def prune_sessions(self, **kwargs):
            return len(rows)

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
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


def test_sessions_prune_preview_shows_oldest_newest(monkeypatch, capsys):
    """Confirmation preview surfaces count + oldest/newest session times."""
    from hermes_cli.session_filters import format_epoch

    _filters, out = _run_prune(monkeypatch, capsys, ["--source", "cron"])
    assert "2 session(s) match" in out
    assert f"oldest activity {format_epoch(1_600_000_050.0)}" in out
    assert f"newest activity {format_epoch(1_700_000_050.0)}" in out


def test_sessions_prune_surfaces_matching_open_sessions(monkeypatch, capsys):
    _filters, out = _run_prune(
        monkeypatch,
        capsys,
        ["--source", "cron"],
        candidates=[],
        skipped_open=2,
    )

    assert "2 open sessions also match these filters" in out
    assert "prune only deletes ended sessions" in out
    assert "hermes sessions delete <id>" in out
    assert "No sessions match" in out
