"""Tests for parent→subparser flag propagation.

When flags like --yolo, -w, -s exist on both the parent parser and the 'chat'
subparser, placing the flag BEFORE the subcommand (e.g. 'hermes --yolo chat')
must not silently drop the flag value.

Regression test for: argparse subparser default=False overwriting parent's
parsed True when the same argument is defined on both parsers.

Fix: chat subparser uses default=argparse.SUPPRESS for all duplicated flags,
so the subparser only sets the attribute when the user explicitly provides it.
"""

import argparse
import sys

import pytest


class TestChatVerboseArg:
    """Verify chat --verbose preserves config fallback when absent."""

    def test_chat_without_verbose_leaves_attribute_unset(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["chat"])

        assert not hasattr(args, "verbose")


    def test_cmd_chat_forwards_none_when_verbose_is_absent(self, monkeypatch):
        import types
        import sys

        import hermes_cli.main as main_mod
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=main_mod.cmd_chat)
        args = parser.parse_args(["chat"])
        captured = {}
        fake_cli = types.ModuleType("cli")

        def fake_main(**kwargs):
            captured.update(kwargs)

        setattr(fake_cli, "main", fake_main)
        fake_banner = types.ModuleType("hermes_cli.banner")
        setattr(fake_banner, "prefetch_update_check", lambda: None)
        fake_skills_sync = types.ModuleType("tools.skills_sync")
        setattr(fake_skills_sync, "sync_skills", lambda quiet=True: None)

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)

        main_mod.cmd_chat(args)

        assert captured["quiet"] is False
        assert "verbose" not in captured


class TestAcceptHooksOnAgentSubparsers:
    """Verify --accept-hooks is accepted at every agent-subcommand
    position (before the subcommand, between group/subcommand, and
    after the leaf subcommand) for gateway/cron/mcp/acp.  Regression
    against prior behaviour where the flag only worked on the root
    parser and `chat`, so `hermes gateway run --accept-hooks` failed
    with `unrecognized arguments`."""

    ARGVS = [
        ["--accept-hooks", "gateway", "run", "--help"],
        ["gateway", "--accept-hooks", "run", "--help"],
        ["gateway", "run", "--accept-hooks", "--help"],
        ["--accept-hooks", "cron", "tick", "--help"],
        ["cron", "--accept-hooks", "tick", "--help"],
        ["cron", "tick", "--accept-hooks", "--help"],
        ["cron", "run", "--accept-hooks", "dummy-id", "--help"],
        ["--accept-hooks", "mcp", "serve", "--help"],
        ["mcp", "--accept-hooks", "serve", "--help"],
        ["mcp", "serve", "--accept-hooks", "--help"],
        ["acp", "--accept-hooks", "--help"],
    ]

    # One driver subprocess parses ALL argvs: hermes_cli.main is a very heavy
    # import (previously 11 separate `python -m hermes_cli.main` spawns with a
    # 15s timeout each — a cold import on a loaded CI worker regularly blew
    # that deadline, making this test flaky). Importing once and parsing 11
    # times removes the repeated-import cost entirely; the generous timeout
    # only trips on a genuine hang. `--help` exits via SystemExit(0), which
    # the driver catches per argv.
    _DRIVER = r"""
import io, json, sys
from contextlib import redirect_stdout, redirect_stderr

import hermes_cli.main as main_mod

argvs = json.loads(sys.argv[1])
results = []
for argv in argvs:
    sys.argv = ["hermes", *argv]
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            main_mod.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the driver
        code = -1
        err.write(repr(exc))
    results.append({"argv": argv, "code": code, "stderr": err.getvalue()[:300]})
print(json.dumps(results))
"""

    def test_accepted_at_every_position(self):
        """Every `hermes <argv>` must exit 0 (help) rather than failing
        with `unrecognized arguments`."""
        import json
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", self._DRIVER, json.dumps(self.ARGVS)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, (
            f"driver failed rc={result.returncode}\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )
        for entry in json.loads(result.stdout.strip().splitlines()[-1]):
            assert entry["code"] == 0, (
                f"argv={entry['argv']!r} returned {entry['code']}\n"
                f"stderr: {entry['stderr']}"
            )
            assert "unrecognized arguments" not in entry["stderr"]


class TestNoStreamingFlag:
    """Verify --no-streaming propagation, forwarding, and TUI interaction."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["chat", "--no-streaming"],
            ["--no-streaming", "chat"],
            ["--no-streaming"],
        ],
        ids=["after-chat", "before-chat", "implicit-chat"],
    )
    def test_parser_sets_no_streaming_for_every_chat_surface(self, argv):
        """The real parser preserves the override at every chat flag position."""
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(argv)
        assert args.no_streaming is True

    def test_cmd_chat_forwards_no_streaming_to_cli_main(self, monkeypatch):
        """cmd_chat passes no_streaming into cli.main() kwargs."""
        import sys
        import types

        import hermes_cli.main as main_mod
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=main_mod.cmd_chat)
        args = parser.parse_args(["chat", "--no-streaming"])
        captured = {}
        fake_cli = types.ModuleType("cli")

        def fake_main(**kwargs):
            captured.update(kwargs)

        setattr(fake_cli, "main", fake_main)
        fake_banner = types.ModuleType("hermes_cli.banner")
        setattr(fake_banner, "prefetch_update_check", lambda: None)
        fake_skills_sync = types.ModuleType("tools.skills_sync")
        setattr(fake_skills_sync, "sync_skills", lambda quiet=True: None)

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: False)

        main_mod.cmd_chat(args)

        assert captured.get("no_streaming") is True

    def test_chat_without_no_streaming_omits_attribute(self, monkeypatch):
        """hermes chat without --no-streaming passes no_streaming=False to cli.main()."""
        import sys
        import types

        import hermes_cli.main as main_mod
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=main_mod.cmd_chat)
        args = parser.parse_args(["chat"])
        captured = {}

        def fake_main(**kwargs):
            captured.update(kwargs)

        fake_cli = types.ModuleType("cli")
        setattr(fake_cli, "main", fake_main)
        fake_banner = types.ModuleType("hermes_cli.banner")
        setattr(fake_banner, "prefetch_update_check", lambda: None)
        fake_skills_sync = types.ModuleType("tools.skills_sync")
        setattr(fake_skills_sync, "sync_skills", lambda quiet=True: None)

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: False)

        main_mod.cmd_chat(args)

        # When not explicitly set, no_streaming defaults to False
        assert captured.get("no_streaming", None) is False

    def test_no_streaming_warns_in_tui_mode(self, monkeypatch, capsys):
        """When TUI is selected and --no-streaming is passed, a warning is printed."""
        import sys
        import types

        import hermes_cli.main as main_mod
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=main_mod.cmd_chat)
        args = parser.parse_args(["chat", "--no-streaming"])

        fake_cli = types.ModuleType("cli")
        fake_banner = types.ModuleType("hermes_cli.banner")
        setattr(fake_banner, "prefetch_update_check", lambda: None)
        fake_skills_sync = types.ModuleType("tools.skills_sync")
        setattr(fake_skills_sync, "sync_skills", lambda quiet=True: None)

        monkeypatch.setitem(sys.modules, "cli", fake_cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
        monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: True)
        # _launch_tui is called with positional + keyword args
        monkeypatch.setattr(main_mod, "_launch_tui", lambda *a, **kw: None)

        main_mod.cmd_chat(args)

        captured = capsys.readouterr()
        assert "not supported in TUI mode" in captured.err


    @pytest.mark.parametrize(
        ("configured", "no_streaming", "expected"),
        [
            (True, False, True),
            (True, True, False),
            (False, False, False),
            (False, True, False),
            (None, False, False),
        ],
    )
    def test_display_streaming_precedence(self, monkeypatch, configured, no_streaming, expected):
        """The CLI override wins, while an absent override preserves config/default behavior."""
        import cli as cli_mod
        from hermes_cli.cli_init_mixin import CLIInitMixin

        display = dict(cli_mod.CLI_CONFIG["display"])
        if configured is None:
            display.pop("streaming", None)
        else:
            display["streaming"] = configured
        config = {"display": display}
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", config)
        monkeypatch.setattr(cli_mod, "_configure_output_history", lambda **_kwargs: None)

        instance = CLIInitMixin()
        instance._init_display_options(verbose=None, compact=None, no_streaming=no_streaming)

        assert instance.streaming_enabled is expected
        assert config["display"].get("streaming") is configured

    def test_no_streaming_override_is_invocation_scoped(self, monkeypatch):
        """Disabling one instance neither mutates config nor disables the next instance."""
        import cli as cli_mod
        from hermes_cli.cli_init_mixin import CLIInitMixin

        display = dict(cli_mod.CLI_CONFIG["display"])
        display["streaming"] = True
        config = {"display": display}
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", config)
        monkeypatch.setattr(cli_mod, "_configure_output_history", lambda **_kwargs: None)

        disabled = CLIInitMixin()
        disabled._init_display_options(verbose=None, compact=None, no_streaming=True)
        unchanged = CLIInitMixin()
        unchanged._init_display_options(verbose=None, compact=None, no_streaming=False)

        assert disabled.streaming_enabled is False
        assert unchanged.streaming_enabled is True
        assert config["display"]["streaming"] is True


class TestChatSubparserInheritedValueFlags:
    """Verify -t/--toolsets, -m/--model and --provider survive parent→chat
    subparser dispatch.

    Regression test for #28780: `hermes -t web chat` silently dropped the
    toolset because the chat subparser re-declared `-t/--toolsets` with
    `default=None`, which clobbered the top-level parser's value during
    subparser dispatch.

    Uses the real `hermes_cli._parser.build_top_level_parser()` rather than
    the hand-rolled replica above so this also fails if the production
    parser drifts back to `default=None` on these flags.
    """

    @pytest.fixture
    def real_parser(self):
        from hermes_cli._parser import build_top_level_parser
        parser, _subparsers, _chat = build_top_level_parser()
        return parser


    def test_all_three_flags_before_chat(self, real_parser):
        """Issue #28780 reporter's case generalized: passing every inherited
        value flag before `chat` must preserve all of them simultaneously."""
        args, _ = real_parser.parse_known_args([
            "-t", "web",
            "-m", "anthropic/claude-sonnet-4",
            "--provider", "openrouter",
            "chat",
        ])
        assert args.toolsets == "web"
        assert args.model == "anthropic/claude-sonnet-4"
        assert args.provider == "openrouter"


    def test_chat_subparser_inherited_value_flags_use_suppress(self):
        """Contract test for the underlying invariant.

        Any chat-subparser flag whose `dest` also exists on the top-level
        parser MUST declare `default=argparse.SUPPRESS`, otherwise the
        subparser silently overwrites the top-level value with its own
        default during dispatch. This is the structural class behind #28780.
        """
        from hermes_cli._parser import build_top_level_parser
        parser, _subparsers, chat_parser = build_top_level_parser()

        top_level_dests = {
            a.dest for a in parser._actions
            if a.option_strings and a.dest != "help"
        }

        offenders = []
        for action in chat_parser._actions:
            if not action.option_strings or action.dest == "help":
                continue
            if action.dest not in top_level_dests:
                continue
            if action.default is not argparse.SUPPRESS:
                offenders.append((action.option_strings, action.dest, action.default))

        assert not offenders, (
            "Chat subparser redeclares these top-level flags without "
            "default=argparse.SUPPRESS; they will silently clobber the "
            "top-level value when used as `hermes <flag> <value> chat`:\n  "
            + "\n  ".join(f"{opts} dest={dest} default={d!r}"
                          for opts, dest, d in offenders)
        )
