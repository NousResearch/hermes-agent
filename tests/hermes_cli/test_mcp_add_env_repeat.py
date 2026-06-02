"""Regression test: ``hermes mcp add --env KEY=VALUE`` must accumulate
across repeated flags rather than keeping only the final one.

Before the fix the ``--env`` argument used ``nargs="*"`` which made a
second ``--env`` flag overwrite the previous value, so

    hermes mcp add foo --command bar \
        --env A=1 \
        --env B=2

silently dropped ``A=1`` and persisted only ``B=2`` to ``config.yaml``.

Fix: switch to ``action="append"`` so repeated flags accumulate.  We
replicate the relevant parser shape here rather than importing the real
builder, mirroring ``test_mcp_add_command_dest.py``.
"""

import argparse


def _build_parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")

    mcp_p = subparsers.add_parser("mcp")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_action")

    mcp_add = mcp_sub.add_parser("add")
    mcp_add.add_argument("name")
    mcp_add.add_argument("--command", dest="mcp_command")
    mcp_add.add_argument("--env", action="append", default=[])

    return parser


class TestMcpAddEnvRepeat:
    def test_repeated_env_flags_accumulate(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mcp",
                "add",
                "alpaca",
                "--command",
                "uvx",
                "--env",
                "ALPACA_API_KEY=xxx",
                "--env",
                "ALPACA_SECRET_KEY=yyy",
            ]
        )
        assert args.env == ["ALPACA_API_KEY=xxx", "ALPACA_SECRET_KEY=yyy"]

    def test_no_env_flag_is_empty_list(self):
        parser = _build_parser()
        args = parser.parse_args(["mcp", "add", "foo", "--command", "npx"])
        assert args.env == []

    def test_single_env_flag(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["mcp", "add", "foo", "--command", "npx", "--env", "X=1"]
        )
        assert args.env == ["X=1"]
