"""Tests for the defensive subparser routing workaround (bpo-9338).

``hermes_cli.main.main()`` sets ``subparsers.required=True`` when argv contains
a known subcommand name.  This forces deterministic routing on Python versions
where argparse fails to match subcommand tokens because the parent parser has
an ``nargs='?'`` optional argument (``--continue``).  The symptom without the
workaround is ``unrecognized arguments: model`` for a plain ``hermes model``.

If the subcommand token was actually consumed as a flag value (e.g.
``hermes -c model`` to resume a session *named* "model"), the ``required=True``
parse raises ``SystemExit`` and the code falls back to ``required=False``.

These tests drive the **real** ``main()`` parser rather than a hand-built
replica, so they stay honest if the production argv handling changes.  Command
handlers are patched to capture the resolved namespace instead of executing.
"""

import pytest

import hermes_cli.main as main_module


@pytest.fixture
def route(monkeypatch):
    """Drive the real ``main()`` and return which handler got which namespace.

    Returns a callable taking an argv list (without the ``hermes`` prog name)
    and returning ``(handler_name, argparse.Namespace)``.
    """
    captured: dict = {}

    def _record(name):
        def _handler(args):
            captured["handler"] = name
            captured["args"] = args

        return _handler

    # Skip plugin discovery / shell-hook registration: irrelevant to routing
    # and expensive (plus it can prompt for hook consent).
    monkeypatch.setattr(main_module, "_prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(main_module, "cmd_chat", _record("chat"))
    monkeypatch.setattr(main_module, "cmd_model", _record("model"))

    def _route(argv):
        captured.clear()
        monkeypatch.setattr("sys.argv", ["hermes"] + list(argv))
        main_module.main()
        assert "handler" in captured, f"no handler ran for argv={argv!r}"
        return captured["handler"], captured["args"]

    return _route


class TestSubparserRoutingFallback:
    """Verify the bpo-9338 defensive routing works for all key cases."""

    def test_direct_subcommand(self, route):
        handler, args = route(["model"])
        assert handler == "model"
        assert args.command == "model"

    def test_subcommand_with_flags(self, route):
        handler, args = route(["--yolo", "model"])
        assert handler == "model"
        assert args.command == "model"
        assert args.yolo is True

    def test_bare_hermes_defaults_to_chat(self, route):
        handler, args = route([])
        assert handler == "chat"
        assert args.command is None

    def test_flags_only_defaults_to_chat(self, route):
        handler, args = route(["--yolo"])
        assert handler == "chat"
        assert args.command is None
        assert args.yolo is True

    def test_continue_flag_alone(self, route):
        handler, args = route(["-c"])
        assert handler == "chat"
        assert args.continue_last is True

    def test_continue_with_session_name(self, route):
        handler, args = route(["-c", "myproject"])
        assert handler == "chat"
        assert args.continue_last == "myproject"

    def test_continue_with_subcommand_name_as_session(self, route):
        """Session named 'model' is a session name, not a subcommand.

        This is the fallback branch: the ``required=True`` parse raises
        SystemExit because 'model' was eaten by ``-c``, so routing retries
        with ``required=False``.
        """
        handler, args = route(["-c", "model"])
        assert handler == "chat"
        assert args.continue_last == "model"

    def test_continue_with_session_then_subcommand(self, route):
        handler, args = route(["-c", "myproject", "model"])
        assert handler == "model"
        assert args.command == "model"
        assert args.continue_last == "myproject"

    def test_chat_with_query(self, route):
        handler, args = route(["chat", "-q", "hello"])
        assert handler == "chat"
        assert args.command == "chat"
        assert args.query == "hello"

    def test_resume_flag(self, route):
        handler, args = route(["-r", "abc123"])
        assert handler == "chat"
        assert args.resume == "abc123"

    def test_resume_with_subcommand(self, route):
        handler, args = route(["-r", "abc123", "chat"])
        assert handler == "chat"
        assert args.command == "chat"
        assert args.resume == "abc123"

    def test_skills_flag_with_subcommand(self, route):
        handler, args = route(["-s", "myskill", "model"])
        assert handler == "model"
        assert args.command == "model"
        assert args.skills == ["myskill"]

    def test_all_flags_with_subcommand(self, route):
        handler, args = route(["--yolo", "-w", "-s", "myskill", "model"])
        assert handler == "model"
        assert args.command == "model"
        assert args.yolo is True
        assert args.worktree is True
        assert args.skills == ["myskill"]


class TestHelpNotDuplicated:
    """``--help`` must print usage exactly once (#10230).

    The ``required=True`` attempt can exit 0 for help/version.  Re-parsing in
    that case would print the same help text a second time, so the fallback
    re-raises instead of retrying.
    """

    @pytest.mark.parametrize("argv", [["--help"], ["chat", "--help"]])
    def test_help_prints_usage_once(self, argv, monkeypatch, capsys):
        monkeypatch.setattr(main_module, "_prepare_agent_startup", lambda args: None)
        monkeypatch.setattr("sys.argv", ["hermes"] + argv)

        with pytest.raises(SystemExit) as exc:
            main_module.main()

        assert exc.value.code == 0
        assert capsys.readouterr().out.count("usage: ") == 1
