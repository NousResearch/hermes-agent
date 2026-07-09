"""Tests for ``--max-turns`` on the oneshot (``-z``) path.

The flag already existed on ``hermes chat``; these cover accepting it at the top
level (so it pairs with ``-z/--oneshot``) and threading it through to the agent.
"""

from unittest.mock import patch

from hermes_cli._parser import build_top_level_parser


class TestOneshotMaxTurnsParsing:
    def test_max_turns_pairs_with_oneshot(self):
        args = build_top_level_parser().parse_args(["-z", "hi", "--max-turns", "5"])
        assert args.max_turns == 5

    def test_absent_max_turns_reads_as_none(self):
        args = build_top_level_parser().parse_args(["-z", "hi"])
        assert getattr(args, "max_turns", None) is None

    def test_top_level_flag_survives_chat_subcommand(self):
        # `hermes --max-turns 7 chat`: the chat subparser's --max-turns default is
        # argparse.SUPPRESS, so the top-level value must NOT be clobbered back.
        args = build_top_level_parser().parse_args(["--max-turns", "7", "chat"])
        assert args.max_turns == 7


class TestOneshotMaxTurnsThreading:
    def test_run_oneshot_forwards_max_turns_to_agent(self):
        from hermes_cli import oneshot

        with patch.object(oneshot, "_write_usage_file"), patch.object(
            oneshot, "_run_agent", return_value=("ok", {})
        ) as run_agent:
            oneshot.run_oneshot("hello", max_turns=5)

        assert run_agent.call_args.kwargs.get("max_turns") == 5

    def test_run_oneshot_forwards_none_when_unset(self):
        from hermes_cli import oneshot

        with patch.object(oneshot, "_write_usage_file"), patch.object(
            oneshot, "_run_agent", return_value=("ok", {})
        ) as run_agent:
            oneshot.run_oneshot("hello")

        assert run_agent.call_args.kwargs.get("max_turns") is None
