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
