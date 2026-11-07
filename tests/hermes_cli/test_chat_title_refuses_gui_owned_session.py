"""`-c <title>` / `--resume <title>` / `/resume <title>` never land a CLI turn in a
desktop-owned session.

A programmatic ``hermes chat -c "<window title>" --create-if-missing -q ...``
that titles a conversation the Desktop app created used to resolve the title
to the desktop's live session and run the turn there: the headless copy took
the per-session turn lease and answered inside the desktop's window while the
desktop's own prompt parked on the lease. A title is not ownership: by-title
resolution refuses sessions whose ``source`` is a GUI surface and names the
explicit ``--resume <id>`` (launch) / ``/resume <id>`` (REPL) override; an
exact id still resolves, and ``--create-if-missing`` never falls through to a
duplicate titled session. CLI- and TUI-created sessions resolve as before.
"""

import pytest

DESKTOP_TITLE = "Quarterly report"
SID = "20260101_000000_000001"  # literal: fixed across runs, no hash seeding
CLI_SID = "20260101_000000_000002"
TUI_SID = "20260101_000000_000003"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed(title, source, sid=SID):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session(sid, source=source)
        db.set_session_title(sid, title)
        return sid
    finally:
        db.close()


def _args(**kw):
    base = {
        "continue_last": None,
        "resume": None,
        "create_if_missing": False,
        "in_dir": None,
    }
    base.update(kw)
    return type("Args", (), base)()


class TestTitleResolutionRefusesGuiOwnedSessions:
    def test_title_match_on_desktop_session_raises(self, isolated_home):
        from hermes_cli.main import SessionOwnedByGuiError, _resolve_session_by_name_or_id

        sid = _seed(DESKTOP_TITLE, "desktop")
        with pytest.raises(SessionOwnedByGuiError) as ei:
            _resolve_session_by_name_or_id(DESKTOP_TITLE)
        assert ei.value.session_id == sid
        assert ei.value.source == "desktop"
        # The override is spelled per surface: launch CLI vs REPL.
        assert f"--resume {sid}" in ei.value.describe()
        assert f"/resume {sid}" in ei.value.describe(override_command="/resume")

    def test_exact_id_of_desktop_session_still_resolves(self, isolated_home):
        """The explicit id is the deliberate override."""
        from hermes_cli.main import _resolve_session_by_name_or_id

        sid = _seed(DESKTOP_TITLE, "desktop")
        assert _resolve_session_by_name_or_id(sid) == sid

    def test_lookup_only_caller_can_opt_out_of_the_refusal(self, isolated_home):
        """A surface that lists/inspects by title (never starts a turn) keeps
        resolving desktop sessions via refuse_gui_owned=False."""
        from hermes_cli.main import _resolve_session_by_name_or_id

        sid = _seed(DESKTOP_TITLE, "desktop")
        assert _resolve_session_by_name_or_id(DESKTOP_TITLE, refuse_gui_owned=False) == sid

    def test_source_probe_failure_falls_back_to_resolving(self, isolated_home, monkeypatch):
        """A raising ownership probe must never turn a hit into a miss — a miss
        would send `-c <title> --create-if-missing` down its duplicate-creation
        path (hermes_cli.main._session_db swallows body errors, so the resolver
        carries the probe in its own guard)."""
        import hermes_state

        from hermes_cli.main import _resolve_session_by_name_or_id

        _seed(DESKTOP_TITLE, "desktop")
        calls = {"n": 0}
        real = hermes_state.SessionDB.get_session

        def flaky(self, sid, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:  # 1st call: id lookup (miss); 2nd: the source probe
                raise RuntimeError("database is locked")
            return real(self, sid, *a, **kw)

        monkeypatch.setattr(hermes_state.SessionDB, "get_session", flaky)
        assert _resolve_session_by_name_or_id(DESKTOP_TITLE) is not None

    def test_cli_and_tui_titles_still_resolve(self, isolated_home):
        """Terminal- and TUI-created sessions (never GUI-created) resolve as before."""
        from hermes_cli.main import _resolve_session_by_name_or_id

        bot = _seed("Bot Chat", "tui", sid=TUI_SID)
        cli = _seed("notes", "cli", sid=CLI_SID)
        assert _resolve_session_by_name_or_id("Bot Chat") == bot
        assert _resolve_session_by_name_or_id("notes") == cli

    def test_continue_by_title_exits_1_naming_owner_and_override(
        self, isolated_home, capsys
    ):
        """`-c <title>` (launch CLI) refuses without binding, exit 1 on stderr."""
        import hermes_cli.main as main_mod

        _seed(DESKTOP_TITLE, "desktop")
        args = _args(continue_last=DESKTOP_TITLE, create_if_missing=True)

        with pytest.raises(SystemExit) as ei:
            main_mod._resolve_continue_arg(args, use_tui=False)
        assert ei.value.code == 1
        assert args.resume is None
        err = capsys.readouterr().err
        assert SID in err and "desktop" in err and f"--resume {SID}" in err
        # The REPL spelling must not leak onto the launch-CLI surface.
        assert f"/resume {SID}" not in err

    def test_continue_create_if_missing_never_reaches_the_create_branch(
        self, isolated_home, monkeypatch
    ):
        """`-c <title> --create-if-missing` must not mint a duplicate titled session."""
        import hermes_cli.main as main_mod

        def _boom(_title):
            raise AssertionError("create-if-missing branch must not be reached")

        monkeypatch.setattr(main_mod, "_create_titled_session", _boom)
        _seed(DESKTOP_TITLE, "desktop")
        args = _args(continue_last=DESKTOP_TITLE, create_if_missing=True)

        with pytest.raises(SystemExit) as ei:
            main_mod._resolve_continue_arg(args, use_tui=False)
        assert ei.value.code == 1

        from hermes_state import SessionDB

        db = SessionDB()
        try:
            row = db.get_session_by_title(DESKTOP_TITLE)
            assert row["id"] == SID  # still the desktop's single row
            assert db.resolve_session_by_title(DESKTOP_TITLE) == SID
        finally:
            db.close()

    def test_resume_by_title_exits_1_before_agent_init(self, isolated_home, capsys):
        """`--resume <title>` (launch CLI) refuses identically, before agent init."""
        import hermes_cli.main as main_mod

        _seed(DESKTOP_TITLE, "desktop")
        args = _args(resume=DESKTOP_TITLE)

        with pytest.raises(SystemExit) as ei:
            main_mod._resolve_chat_session_args(args, use_tui=False)
        assert ei.value.code == 1
        assert args.resume == DESKTOP_TITLE  # untouched: never bound to the id
        err = capsys.readouterr().err
        assert SID in err and f"--resume {SID}" in err

    def test_repl_resume_title_prints_and_stays_put(self, isolated_home):
        """REPL `/resume <title>` prints the refusal (with the REPL spelling)
        and keeps the current session instead of switching."""
        from unittest.mock import MagicMock, patch

        from cli import HermesCLI

        _seed(DESKTOP_TITLE, "desktop")
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.session_id = "current_session"
        cli_obj._session_db = MagicMock()

        with patch("cli._cprint") as mock_cprint:
            result = cli_obj._resolve_resume_target(DESKTOP_TITLE)

        assert result is None  # refused: no (session_id, meta) returned
        assert cli_obj.session_id == "current_session"  # not switched
        printed = " ".join(call.args[0] for call in mock_cprint.call_args_list)
        assert SID in printed and f"/resume {SID}" in printed
        cli_obj._session_db.get_session.assert_not_called()

    def test_repl_resume_by_id_still_resolves(self, isolated_home):
        """The REPL keeps the exact-id override: /resume <id> resolves."""
        from unittest.mock import MagicMock

        from cli import HermesCLI

        _seed(DESKTOP_TITLE, "desktop")
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.session_id = "current_session"
        cli_obj._session_db = MagicMock()
        cli_obj._session_db.get_session.return_value = {"id": SID, "title": DESKTOP_TITLE}
        cli_obj._session_db.resolve_resume_session_id.return_value = SID

        result = cli_obj._resolve_resume_target(SID)

        assert result == (SID, {"id": SID, "title": DESKTOP_TITLE})
