"""Tests for the repeatable --exclude-source flag on `hermes sessions list`
and `hermes sessions browse`.

Covers the real CLI parsers (via main()) and the forwarding of the combined
exclusion list — the implicit 'tool' exclusion plus any user-supplied
--exclude-source values — to SessionDB.list_sessions_rich. DB-level
exclusion semantics are covered in tests/test_hermes_state.py.
"""

import sys


def _run_sessions(monkeypatch, capsys, argv_tail):
    """Run `hermes sessions <argv_tail>` against a FakeDB, capturing the
    kwargs passed to list_sessions_rich. Returns (kwargs, stdout)."""
    import hermes_cli.main as main_mod
    import hermes_state

    seen = {}

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            seen.update(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    main_mod.main()
    return seen, capsys.readouterr().out


# ─── sessions list ───────────────────────────────────────────────────────────

class TestSessionsListExcludeSource:
    def test_default_keeps_implicit_tool_exclusion(self, monkeypatch, capsys):
        """Bare `sessions list` still hides third-party tool sessions."""
        seen, _out = _run_sessions(monkeypatch, capsys, ["list"])
        assert seen["source"] is None
        assert seen["exclude_sources"] == ["tool"]

    def test_repeated_flag_adds_to_implicit_tool_exclusion(self, monkeypatch, capsys):
        """Repeated --exclude-source values are forwarded alongside 'tool'."""
        seen, _out = _run_sessions(
            monkeypatch,
            capsys,
            ["list", "--exclude-source", "cron", "--exclude-source", "telegram"],
        )
        assert seen["source"] is None
        assert seen["exclude_sources"] == ["tool", "cron", "telegram"]

    def test_exclude_tool_is_not_duplicated(self, monkeypatch, capsys):
        """Explicitly excluding 'tool' doesn't produce a duplicate entry."""
        seen, _out = _run_sessions(
            monkeypatch, capsys, ["list", "--exclude-source", "tool"]
        )
        assert seen["exclude_sources"] == ["tool"]

    def test_source_alone_still_disables_implicit_exclusion(self, monkeypatch, capsys):
        """--source without --exclude-source keeps current main's semantics:
        no exclusion at all (tool sessions are shown for explicit --source)."""
        seen, _out = _run_sessions(monkeypatch, capsys, ["list", "--source", "cli"])
        assert seen["source"] == "cli"
        assert seen["exclude_sources"] is None

    def test_source_with_excludes_forwards_only_user_values(self, monkeypatch, capsys):
        """With --source, the implicit 'tool' exclusion stays disabled but
        the user's --exclude-source values are still forwarded."""
        seen, _out = _run_sessions(
            monkeypatch,
            capsys,
            ["list", "--source", "cli", "--exclude-source", "cron"],
        )
        assert seen["source"] == "cli"
        assert seen["exclude_sources"] == ["cron"]


# ─── sessions browse ─────────────────────────────────────────────────────────

class TestSessionsBrowseExcludeSource:
    def test_default_keeps_implicit_tool_exclusion(self, monkeypatch, capsys):
        seen, out = _run_sessions(monkeypatch, capsys, ["browse"])
        assert seen["source"] is None
        assert seen["exclude_sources"] == ["tool"]
        assert "No sessions found." in out

    def test_repeated_flag_adds_to_implicit_tool_exclusion(self, monkeypatch, capsys):
        seen, _out = _run_sessions(
            monkeypatch,
            capsys,
            ["browse", "--exclude-source", "cron", "--exclude-source", "telegram"],
        )
        assert seen["source"] is None
        assert seen["exclude_sources"] == ["tool", "cron", "telegram"]

    def test_exclude_tool_is_not_duplicated(self, monkeypatch, capsys):
        seen, _out = _run_sessions(
            monkeypatch, capsys, ["browse", "--exclude-source", "tool"]
        )
        assert seen["exclude_sources"] == ["tool"]

    def test_source_alone_still_disables_implicit_exclusion(self, monkeypatch, capsys):
        seen, _out = _run_sessions(monkeypatch, capsys, ["browse", "--source", "cli"])
        assert seen["source"] == "cli"
        assert seen["exclude_sources"] is None

    def test_source_with_excludes_forwards_only_user_values(self, monkeypatch, capsys):
        seen, _out = _run_sessions(
            monkeypatch,
            capsys,
            ["browse", "--source", "cli", "--exclude-source", "cron"],
        )
        assert seen["source"] == "cli"
        assert seen["exclude_sources"] == ["cron"]
