"""Seeded interactive ``-q`` behavior (Aug 2026).

On a real TTY, ``hermes chat -q "…"`` seeds a normal interactive session with
the prompt submitted literally as the first turn. Legacy answer-and-exit is
preserved for ``--oneshot``, ``-Q/--quiet``, and every non-TTY invocation
(kanban workers, cron, pipes, A2A). The seeded prompt bypasses slash-command
routing, ``!`` shell dispatch, and file-drop detection.

Context: Omarchy prompted-agent launches (basecamp/omarchy#8705) needed a
"start interactive, seeded with this prompt" mode with literal prompt
handling, like other coding agents.
"""

import sys
import types

import pytest


@pytest.fixture()
def cli_mod():
    import cli

    return cli


class TestShouldSeedInteractive:
    def _tty(self, monkeypatch, cli_mod, stdin=True, stdout=True):
        monkeypatch.setattr(
            cli_mod.sys, "stdin", types.SimpleNamespace(isatty=lambda: stdin)
        )
        monkeypatch.setattr(
            cli_mod.sys, "stdout", types.SimpleNamespace(isatty=lambda: stdout)
        )

    def test_tty_query_seeds_interactive(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod)
        assert cli_mod._should_seed_interactive("hi", None, quiet=False, oneshot=False)

    def test_image_only_also_seeds(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod)
        assert cli_mod._should_seed_interactive(
            None, "/tmp/x.png", quiet=False, oneshot=False
        )

    def test_oneshot_flag_forces_legacy(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod)
        assert not cli_mod._should_seed_interactive(
            "hi", None, quiet=False, oneshot=True
        )

    def test_quiet_forces_legacy(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod)
        assert not cli_mod._should_seed_interactive(
            "hi", None, quiet=True, oneshot=False
        )

    def test_non_tty_stdin_forces_legacy(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod, stdin=False)
        assert not cli_mod._should_seed_interactive(
            "hi", None, quiet=False, oneshot=False
        )

    def test_non_tty_stdout_forces_legacy(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod, stdout=False)
        assert not cli_mod._should_seed_interactive(
            "hi", None, quiet=False, oneshot=False
        )

    def test_no_query_no_image_never_seeds(self, monkeypatch, cli_mod):
        self._tty(monkeypatch, cli_mod)
        assert not cli_mod._should_seed_interactive(
            None, None, quiet=False, oneshot=False
        )

    def test_isatty_failure_forces_legacy(self, monkeypatch, cli_mod):
        def _boom():
            raise OSError("no tty")

        monkeypatch.setattr(
            cli_mod.sys, "stdin", types.SimpleNamespace(isatty=_boom)
        )
        assert not cli_mod._should_seed_interactive(
            "hi", None, quiet=False, oneshot=False
        )


class TestSeededQueryMessage:
    def test_str_returns_text(self, cli_mod):
        msg = cli_mod._SeededQueryMessage("!echo pwned")
        assert str(msg) == "!echo pwned"
        assert msg.images == []

    def test_images_are_copied(self, cli_mod):
        imgs = ["/tmp/a.png"]
        msg = cli_mod._SeededQueryMessage("hi", imgs)
        assert msg.images == imgs
        assert msg.images is not imgs


class TestChatParserOneshotFlag:
    """The chat subcommand's --oneshot must not collide with top-level -z."""

    def _parse(self, argv):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        return parser.parse_args(argv)

    def test_chat_oneshot_sets_distinct_dest(self):
        args = self._parse(["chat", "-q", "hello", "--oneshot"])
        assert args.oneshot_exit is True
        # Top-level -z prompt dest untouched — dispatch sites check
        # `args.oneshot` truthiness and would treat True as a prompt.
        assert getattr(args, "oneshot", None) in (None, False)

    def test_chat_without_oneshot_defaults_false(self):
        args = self._parse(["chat", "-q", "hello"])
        assert args.oneshot_exit is False

    def test_top_level_oneshot_prompt_unaffected(self):
        args = self._parse(["-z", "what is up"])
        assert args.oneshot == "what is up"
        assert getattr(args, "oneshot_exit", False) is False


class TestRunCommand:
    """``--run-command`` dispatches a registered slash command (#109971).

    A plugin's ``register_command`` handler had no non-interactive entry point:
    ``-q "/mycommand"`` is literal by design (the class above is that design),
    and piping into ``hermes chat`` needs a real PTY. So a plugin author could
    not drive their own handler from a script or from CI.
    """

    def test_the_sentinel_defaults_to_literal(self, cli_mod):
        assert cli_mod._SeededQueryMessage("/mycommand").run_command is False

    def test_the_opt_in_is_keyword_only(self, cli_mod):
        # Positional would put it where `images` goes, and a truthy path list
        # would silently turn an ordinary -q into a command dispatch.
        with pytest.raises(TypeError):
            cli_mod._SeededQueryMessage("/mycommand", None, True)

    def test_the_flag_reaches_the_sentinel(self, cli_mod, monkeypatch):
        seen = {}

        class _Cli:
            def run(self):
                seen["message"] = self._seeded_first_message

        cli = _Cli()
        cli_mod._run_single_query_mode(
            cli, "/mycommand args", None, False, False, run_command=True)

        assert seen["message"].text == "/mycommand args"
        assert seen["message"].run_command is True

    def test_a_command_seeds_even_without_a_tty(self, cli_mod, monkeypatch):
        """The point of the flag is CI, where nothing is a terminal. The
        one-shot fallback would answer the text with the model instead of
        running the handler."""
        monkeypatch.setattr(
            cli_mod.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr(
            cli_mod.sys, "stdout", types.SimpleNamespace(isatty=lambda: False))
        assert not cli_mod._should_seed_interactive(
            "/mycommand", None, quiet=False, oneshot=False)

        ran = {}

        class _Cli:
            def run(self):
                ran["message"] = self._seeded_first_message

        cli_mod._run_single_query_mode(
            _Cli(), "/mycommand", None, False, False, run_command=True)

        assert ran["message"].run_command is True


class TestRunCommandUnwrap:
    """The unwrap helper stashes the opt-in for the dispatch gate."""

    def _unwrap(self, cli_mod, message):
        from hermes_cli.cli_process_notifications import (
            CLIProcessNotificationsMixin,
        )

        holder = types.SimpleNamespace()
        return (
            CLIProcessNotificationsMixin._tui_unwrap_input(holder, message),
            holder,
        )

    def test_a_plain_seeded_query_is_not_a_command(self, cli_mod):
        (text, _voice, seeded), holder = self._unwrap(
            cli_mod, cli_mod._SeededQueryMessage("/mycommand"))

        assert (text, seeded) == ("/mycommand", True)
        assert holder._seeded_runs_command is False

    def test_a_run_command_message_is(self, cli_mod):
        (text, _voice, seeded), holder = self._unwrap(
            cli_mod, cli_mod._SeededQueryMessage("/mycommand", run_command=True))

        assert (text, seeded) == ("/mycommand", True)
        assert holder._seeded_runs_command is True

    def test_ordinary_text_clears_the_flag(self, cli_mod):
        """The attribute lives on the CLI across turns, so a command turn must
        not leave the next ordinary turn dispatching slashes."""
        _first, holder = self._unwrap(
            cli_mod, cli_mod._SeededQueryMessage("/mycommand", run_command=True))
        from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin

        CLIProcessNotificationsMixin._tui_unwrap_input(holder, "just text")

        assert holder._seeded_runs_command is False


class TestRunCommandParser:
    def _parse(self, argv):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        return parser.parse_args(argv)

    def test_the_flag_carries_the_command(self):
        args = self._parse(["chat", "--run-command", "/mycommand args"])
        assert args.run_command == "/mycommand args"
        assert args.query is None

    def test_it_is_mutually_exclusive_with_query(self):
        # Two different dispatch rules for one text: the literal one and the
        # command one. Refusing is better than picking.
        with pytest.raises(SystemExit) as exc:
            self._parse(["chat", "-q", "hi", "--run-command", "/x"])
        assert exc.value.code == 2

    def test_main_passes_it_through(self):
        import inspect

        import cli

        assert "run_command" in inspect.signature(cli.main).parameters
        from pathlib import Path

        wiring = Path(cli.__file__).read_text(encoding="utf-8")
        # The one line that makes the flag live; nothing else here can see it.
        assert "run_command=True" in wiring


class TestRunCommandDispatchGate:
    """The one line the flag exists for: a seeded message reaching the slash
    dispatcher instead of the model.

    Driven through the real ``_tui_process_one_input``, because a gate that
    reads the right attribute and is never consulted looks identical from
    every other angle. ``_tui_run_slash_input`` returns None, which is the
    method's own "the command handled it, stop here" signal, so nothing after
    the gate runs.
    """

    class _Anything:
        """Callable, and attribute-tolerant all the way down."""

        def __call__(self, *_a, **_k):
            return None

        def __getattr__(self, _name):
            return TestRunCommandDispatchGate._Anything()

    def _cli(self, cli_mod):
        seen = {"slash": [], "bang": []}

        class _Stub:
            _pending_resume_sessions = ()
            _status_bar_suppressed_after_resize = True
            _tui_unwrap_input = cli_mod.HermesCLI._tui_unwrap_input

            def _typed_voice_stop(self, _text):
                return False

            def handle_bang_shell(self, text):
                seen["bang"].append(text)
                return False

            def _tui_run_slash_input(self, text):
                seen["slash"].append(text)
                return None

            def __getattr__(self, name):
                # Everything past the gate is rendering and turn machinery this
                # cell is not about; a no-op keeps the method running to the end
                # instead of turning an unrelated attribute into a failure.
                seen.setdefault("touched", []).append(name)
                return TestRunCommandDispatchGate._Anything()

        return _Stub(), seen

    def _run(self, cli_mod, message):
        stub, seen = self._cli(cli_mod)
        cli_mod.HermesCLI._tui_process_one_input(stub, message)
        return seen

    def test_a_run_command_message_reaches_the_dispatcher(self, cli_mod):
        seen = self._run(
            cli_mod, cli_mod._SeededQueryMessage("/mycommand args", run_command=True))

        assert seen["slash"] == ["/mycommand args"]

    def test_a_plain_seeded_query_does_not(self, cli_mod):
        """The literal contract this class's sibling pins, unchanged."""
        seen = self._run(cli_mod, cli_mod._SeededQueryMessage("/mycommand args"))

        assert seen["slash"] == []

    def test_a_shell_escape_stays_literal_even_with_the_opt_in(self, cli_mod):
        """`!` is not part of the opt-in. A seeded shell escape is the
        injection the sentinel exists to prevent, and --run-command is a
        request to run a REGISTERED COMMAND, not a request to run anything."""
        seen = self._run(
            cli_mod, cli_mod._SeededQueryMessage("!echo pwned", run_command=True))

        assert seen["bang"] == []
        assert seen["slash"] == []
