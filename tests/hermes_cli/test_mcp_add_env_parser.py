"""Production-parser regressions for ``hermes mcp add --env``."""

import argparse

from hermes_cli.subcommands.mcp import build_mcp_parser


def _parse_mcp_add(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_mcp_parser(subparsers, cmd_mcp=lambda _args: None)
    return parser.parse_args(["mcp", "add", "demo", "--command", "demo-mcp", *argv])


def test_repeated_env_flags_accumulate_groups() -> None:
    args = _parse_mcp_add(
        "--env",
        "FIRST=1",
        "SECOND=two",
        "--env",
        "THIRD=3",
        "FOURTH=four",
    )

    assert args.env == [
        ["FIRST=1", "SECOND=two"],
        ["THIRD=3", "FOURTH=four"],
    ]


def test_single_env_flag_keeps_grouped_assignments() -> None:
    args = _parse_mcp_add("--env", "FIRST=1", "SECOND=two")

    assert args.env == [["FIRST=1", "SECOND=two"]]


def test_env_flags_before_args_preserve_command_remainder() -> None:
    args = _parse_mcp_add(
        "--env",
        "FIRST=1",
        "--args",
        "mcp",
        "gateway",
        "run",
        "--profile",
        "research",
    )

    assert args.env == [["FIRST=1"]]
    assert args.args == ["mcp", "gateway", "run", "--profile", "research"]
