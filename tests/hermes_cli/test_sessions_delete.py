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


def test_sessions_delete_reports_not_found_when_prefix_is_unknown(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            return None

        def delete_session(self, session_id, **kwargs):
            raise AssertionError("delete_session should not be called when resolution fails")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "delete", "missing-prefix", "--yes"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    assert "Session 'missing-prefix' not found." in output


def test_sessions_delete_handles_eoferror_on_confirm(monkeypatch, capsys):
    """sessions delete should not crash when stdin is closed (non-TTY)."""
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "20260315_092437_c9a6ff"

        def delete_session(self, session_id, **kwargs):
            raise AssertionError("delete_session should not be called when cancelled")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys, "argv",
        ["hermes", "sessions", "delete", "20260315_092437_c9a6"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))

    main_mod.main()

    output = capsys.readouterr().out
    assert "Cancelled" in output


def test_sessions_prune_handles_eoferror_on_confirm(monkeypatch, capsys):
    """sessions prune should not crash when stdin is closed (non-TTY)."""
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            return [
                {
                    "id": "20260315_092437_c9a6ff",
                    "source": "cli",
                    "title": "old session",
                    "started_at": 0.0,
                    "ended_at": 1.0,
                    "message_count": 3,
                    "archived": 0,
                }
            ]

        def prune_sessions(self, **kwargs):
            raise AssertionError("prune_sessions should not be called when cancelled")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys, "argv",
        ["hermes", "sessions", "prune"],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))

    main_mod.main()

    output = capsys.readouterr().out
    assert "Cancelled" in output


def _run_prune(monkeypatch, capsys, argv_tail, candidates=None):
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
            "ended_at": 1_600_000_100.0,
            "message_count": 2,
            "archived": 0,
        },
        {
            "id": "20260601_000000_bbbbbb",
            "source": "cron",
            "title": "newest run",
            "started_at": 1_700_000_000.0,
            "ended_at": 1_700_000_100.0,
            "message_count": 4,
            "archived": 0,
        },
    ]

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            seen.update(kwargs)
            return rows

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
    assert filters["started_before"] is not None
    assert filters["started_before"] == pytest.approx(
        _time.time() - 90 * 86400, abs=60
    )


def test_sessions_prune_source_matches_all_ages(monkeypatch, capsys):
    """--source alone suppresses the implicit 90-day cutoff (all ages)."""
    filters, _out = _run_prune(monkeypatch, capsys, ["--source", "cron"])
    assert filters["started_before"] is None
    assert filters["started_after"] is None
    assert filters["source"] == "cron"


def test_sessions_prune_source_with_explicit_time_respected(monkeypatch, capsys):
    """--source + explicit --older-than keeps the user's bound."""
    import time as _time

    filters, _out = _run_prune(
        monkeypatch, capsys, ["--source", "cron", "--older-than", "30"]
    )
    assert filters["started_before"] == pytest.approx(
        _time.time() - 30 * 86400, abs=60
    )
    assert filters["source"] == "cron"


def test_sessions_prune_preview_shows_oldest_newest(monkeypatch, capsys):
    """Confirmation preview surfaces count + oldest/newest session times."""
    from hermes_cli.session_filters import format_epoch

    _filters, out = _run_prune(monkeypatch, capsys, ["--source", "cron"])
    assert "2 session(s) match" in out
    assert f"oldest {format_epoch(1_600_000_000.0)}" in out
    assert f"newest {format_epoch(1_700_000_000.0)}" in out
