"""Regression test: ``hermes mcp add --args`` must accept every token after
the flag, including values that look like flags.

Before this fix ``mcp add`` declared ``--args`` with ``nargs="*"`` on a
subparser.  In that configuration argparse silently consumed only the
first token (``/c``) and then bounced the rest back to the top-level
parser as unrecognized arguments:

    $ hermes mcp add NAME --command cmd.exe --args /c npx -y pkg --autoConnect
    hermes: error: unrecognized arguments: -y pkg --autoConnect

This made the documented WSL→Windows Chrome bridge command (see
``docs/guides/use-mcp-with-hermes.md``) unusable from the CLI — users
either had to hand-edit ``config.yaml`` or wrap the args in a shell
script.

The fix switches to ``nargs=argparse.REMAINDER`` so the stdio server
receives every token after ``--args`` verbatim.  Trade-off: ``--args``
must be the last flag in the ``hermes mcp add`` invocation (documented
in the new inline comment in ``hermes_cli/subcommands/mcp.py``).

We replicate the relevant parser shape here rather than importing the
real builder, mirroring ``test_mcp_add_command_dest.py``.
"""

import argparse


def _build_parser():
    """Minimal replica of the slice of the hermes parser that exhibits
    the bug: top-level subparsers (dest="command"), ``mcp add`` with
    ``--command`` (dest="mcp_command"), and ``--args``.
    """
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat")

    mcp_p = subparsers.add_parser("mcp")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_action")

    mcp_add = mcp_sub.add_parser("add")
    mcp_add.add_argument("name")
    mcp_add.add_argument("--url")
    mcp_add.add_argument("--command", dest="mcp_command")
    mcp_add.add_argument("--args", nargs=argparse.REMAINDER, default=[])

    return parser


class TestMcpAddArgsRemainder:
    def test_documented_chrome_devtools_command_parses(self):
        """The WSL→Windows Chrome bridge command from the docs must parse.

        This is the exact incantation from
        ``docs/guides/use-mcp-with-hermes.md``:

            hermes mcp add chrome-devtools-win \\
                --command cmd.exe \\
                --args /c npx -y chrome-devtools-mcp@latest \\
                       --autoConnect --no-usage-statistics
        """
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mcp", "add", "chrome-devtools-win",
                "--command", "cmd.exe",
                "--args", "/c", "npx", "-y",
                "chrome-devtools-mcp@latest",
                "--autoConnect", "--no-usage-statistics",
            ]
        )

        assert args.command == "mcp"
        assert args.mcp_action == "add"
        assert args.name == "chrome-devtools-win"
        assert args.mcp_command == "cmd.exe"
        assert args.args == [
            "/c", "npx", "-y",
            "chrome-devtools-mcp@latest",
            "--autoConnect", "--no-usage-statistics",
        ]

    def test_args_with_path_token(self):
        """``--args`` with a filesystem path must not split the path."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mcp", "add", "fs",
                "--command", "npx",
                "--args", "-y",
                "@modelcontextprotocol/server-filesystem",
                "/home/user/my-project",
            ]
        )
        assert args.args == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/user/my-project",
        ]

    def test_args_with_single_token(self):
        """``--args`` with a single token still works (no regression)."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "mcp", "add", "github",
                "--command", "npx",
                "--args", "-y", "@modelcontextprotocol/server-github",
            ]
        )
        assert args.args == ["-y", "@modelcontextprotocol/server-github"]

    def test_args_omitted_defaults_to_empty_list(self):
        """Omitting ``--args`` must default to ``[]`` so cmd_mcp_add can
        iterate without a None check."""
        parser = _build_parser()
        args = parser.parse_args(
            ["mcp", "add", "ink", "--url", "https://mcp.example.com"]
        )
        assert args.args == []
