"""Tests for ``chat --session-id`` — start a FRESH session with a pre-registered session id.

Mirrors Claude Code's ``--session-id``: an orchestrator (pixel-agents) pre-registers the id as its
routing key BEFORE spawning ``hermes chat --session-id <ID>``, so the hooks fired from
``agent/conversation_loop.py`` see ``agent.session_id == <ID>``.

Coverage:
  * parse/propagation at BOTH positions (``hermes chat --session-id X`` and
    ``hermes --session-id X chat``) — the SUPPRESS-default convention from
    test_argparse_flag_propagation.py means the top-level value must not be clobbered;
  * malformed ids rejected at PARSE time (exit 2, one-line argparse error, no traceback);
  * the resume/continue conflict: a runtime usage error in cmd_chat (exit 2) rather than silent
    precedence — an argparse mutually-exclusive group cannot span the top/sub parser pair;
  * the ``_init_runtime_state`` seam that actually sets ``self.session_id``.
"""

import io
import sys
import types
import uuid
from contextlib import redirect_stderr
from types import SimpleNamespace

import pytest


VALID_ID = "20260225_143052_a1b2c3"
OTHER_ID = "20260225_144000_ffeecc"


def _parse(argv):
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    return parser.parse_args(argv)


class TestParserAcceptsAndPropagates:
    def test_chat_subparser_sets_session_id(self):
        args = _parse(["chat", "--session-id", VALID_ID])
        assert args.session_id == VALID_ID

    def test_top_level_flag_propagates_through_chat_subparser(self):
        # Regression mirror: a plain None default on the chat subparser would clobber this.
        args = _parse(["--session-id", VALID_ID, "chat"])
        assert args.session_id == VALID_ID

    def test_absent_flag_is_none(self):
        # Top level declares default=None, so the attr always exists for cmd_chat.
        assert _parse(["chat"]).session_id is None

    @pytest.mark.parametrize("session_id", [
        VALID_ID,                                   # CLI/TUI mint shape
        str(uuid.uuid4()),                          # acp_adapter / /v1/responses shape
        "cron_job-1_20260225_143052",               # cron shape (legacy job ids are real)
        "bg_143052_a1b2c3",                         # /bg shape
        "api-0123456789abcdef",                     # chat-completions derived shape
    ])
    def test_every_recognized_mint_shape_is_accepted(self, session_id):
        assert _parse(["chat", "--session-id", session_id]).session_id == session_id

    @pytest.mark.parametrize("bad", ["", "hello", "2026-02-25T14-30-52", "latest", "20260225143052_a1b2c3"])
    def test_malformed_id_rejected_at_parse_time(self, bad):
        err = io.StringIO()
        with pytest.raises(SystemExit) as exc, redirect_stderr(err):
            _parse(["chat", "--session-id", bad])
        assert exc.value.code == 2
        text = err.getvalue()
        assert "Traceback" not in text
        assert "--session-id" in text and "shape" in text

    def test_malformed_id_rejected_when_flag_precedes_subcommand(self):
        err = io.StringIO()
        with pytest.raises(SystemExit) as exc, redirect_stderr(err):
            _parse(["--session-id", "nope", "chat"])
        assert exc.value.code == 2
        assert "--session-id" in err.getvalue()


class TestInitRuntimeStateSeam:
    """The exact seam where a fresh session's id is chosen (cli_init_mixin).

    Exercises CLIInitMixin._init_runtime_state directly — full HermesCLI construction drags the
    whole REPL graph in; only the session-store/UI hooks are stubbed. ``cli`` is faked in
    sys.modules: the method imports it ONLY for the ``_hermes_home`` path constant, and on this
    machine the live checkout sits INSIDE the real Hermes home, so a real ``import cli`` would run
    ``hermes_bootstrap``'s module-level manifest read and trip the test home-I/O guard
    (pre-existing machine hazard — test_cli_init.py and test_argparse_flag_propagation.py fail the
    same way at HEAD here).
    """

    def _stub(self, monkeypatch, tmp_path):
        from hermes_cli.cli_init_mixin import CLIInitMixin

        fake_cli = types.SimpleNamespace(_hermes_home=tmp_path)
        monkeypatch.setitem(sys.modules, "cli", fake_cli)

        class _Stub(CLIInitMixin):
            def _init_session_store(self):
                self._session_db = None
                self._session_db_unavailable = False

            def _init_ui_state(self):
                pass

        return _Stub()

    def test_valid_id_respected_on_fresh_init(self, monkeypatch, tmp_path):
        stub = self._stub(monkeypatch, tmp_path)
        stub._init_runtime_state(None, session_id=VALID_ID)
        assert stub.session_id == VALID_ID
        assert stub._resumed is False

    def test_fresh_init_without_id_still_mints(self, monkeypatch, tmp_path):
        from hermes_state_ids import is_known_session_id

        stub = self._stub(monkeypatch, tmp_path)
        stub._init_runtime_state(None)
        assert is_known_session_id(stub.session_id)

    def test_resume_path_unchanged(self, monkeypatch, tmp_path):
        stub = self._stub(monkeypatch, tmp_path)
        stub._init_runtime_state(VALID_ID)
        assert stub.session_id == VALID_ID
        assert stub._resumed is True

    def test_malformed_id_raises_value_error(self, monkeypatch, tmp_path):
        # cli.main's caller (cmd_chat) turns ValueError into "Error: …" + exit 1; no traceback.
        with pytest.raises(ValueError, match="invalid session id"):
            self._stub(monkeypatch, tmp_path)._init_runtime_state(None, session_id="not-a-session-id")

    def test_resume_plus_session_id_raises(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="cannot be combined"):
            self._stub(monkeypatch, tmp_path)._init_runtime_state(VALID_ID, session_id=OTHER_ID)


@pytest.mark.allow_real_home_io
class TestCmdChatWiring:
    """cmd_chat forwards the id to cli.main and refuses resume/--session-id combos.

    Marked ``allow_real_home_io``: importing ``hermes_cli.main`` runs
    ``hermes_bootstrap.activate_dependencies``' module-level manifest read; harmless in CI (a
    checkout outside the home finds no manifest) but the test home-I/O guard refuses the READ on
    machines whose checkout lives inside the real Hermes home. The test itself never touches home
    files — ``cli`` is a fake module in sys.modules.
    """

    def _harness(self, monkeypatch, argv):
        import hermes_cli.main as main_mod

        args = _parse(argv)
        captured = {}
        fake_cli = types.ModuleType("cli")
        fake_cli.main = lambda **kwargs: captured.update(kwargs)
        fake_banner = types.ModuleType("hermes_cli.banner")
        fake_banner.prefetch_update_check = lambda: None
        fake_skills = types.ModuleType("tools.skills_sync")
        fake_skills.sync_skills = lambda quiet=True: None

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        return main_mod, args, captured

    @pytest.mark.parametrize("argv", [
        ["chat", "--session-id", VALID_ID],
        ["--session-id", VALID_ID, "chat"],
    ])
    def test_session_id_forwarded_to_cli_main(self, monkeypatch, argv):
        main_mod, args, captured = self._harness(monkeypatch, argv)
        main_mod.cmd_chat(args)
        assert captured["session_id"] == VALID_ID

    def test_absent_flag_not_forwarded(self, monkeypatch):
        main_mod, args, captured = self._harness(monkeypatch, ["chat"])
        main_mod.cmd_chat(args)
        # cmd_chat's None filter drops it — cli.main keeps its own default.
        assert "session_id" not in captured

    @pytest.mark.parametrize("argv, names_conflict", [
        (["chat", "-r", VALID_ID, "--session-id", OTHER_ID], "--resume/-r"),
        (["--resume", VALID_ID, "chat", "--session-id", OTHER_ID], "--resume/-r"),
        (["-c", "--session-id", VALID_ID, "chat"], "-c/--continue"),
        (["chat", "--continue", VALID_ID, "--session-id", OTHER_ID], "-c/--continue"),
    ])
    def test_resume_conflict_is_usage_error(self, argv, names_conflict):
        import hermes_cli.main as main_mod

        args = _parse(argv)
        err = io.StringIO()
        with pytest.raises(SystemExit) as exc, redirect_stderr(err):
            main_mod.cmd_chat(args)
        assert exc.value.code == 2
        text = err.getvalue()
        assert "--session-id cannot be combined with" in text
        assert names_conflict in text
        assert "Traceback" not in text

    def test_conflict_check_runs_before_any_session_side_effects(self, monkeypatch):
        # The first line of cmd_chat is the guard: nothing else may have run (no cli import).
        import hermes_cli.main as main_mod

        called = []
        monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda a: called.append("safe_mode"))
        args = _parse(["chat", "-r", VALID_ID, "--session-id", OTHER_ID])
        with pytest.raises(SystemExit), redirect_stderr(io.StringIO()):
            main_mod.cmd_chat(args)
        assert called == []


@pytest.mark.allow_real_home_io
class TestSetChatArgDefaults:
    def test_session_id_defaulted_when_chat_not_parsed(self):
        import hermes_cli.main as main_mod

        ns = SimpleNamespace()
        main_mod._set_chat_arg_defaults(ns)
        assert ns.session_id is None

    def test_bare_namespace_conflict_helper_is_noop(self):
        import hermes_cli.main as main_mod

        ns = SimpleNamespace(session_id=None, resume=None, continue_last=None)
        main_mod._reject_session_id_conflict(ns)  # must not exit
