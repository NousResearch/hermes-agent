"""Regression test: ``hermes mcp add --command`` must not clobber the
top-level ``args.command`` subparser dest.

The top-level argparse parser uses ``dest="command"`` for its subparsers
(``hermes_cli/_parser.py``).  The dispatcher in ``hermes_cli/main.py``
reads ``args.command`` to decide which command to run; if it is ``None``
it falls through to interactive chat.

The ``mcp add`` subparser exposes a ``--command`` flag (the stdio command
for an MCP server, e.g. ``npx``).  Without an explicit ``dest=``, argparse
derives the dest from the flag name and writes ``args.command = None``
when the flag is omitted, overwriting the top-level ``"mcp"`` value.  As a
result, ``hermes mcp add foo --url ...`` silently launches chat instead
of registering an MCP server.

The fix: declare the flag with ``dest="mcp_command"``.  The CLI flag name
is unchanged; only the in-memory attribute moves.

We replicate the relevant parser shape here rather than importing the
real builder, mirroring ``test_argparse_flag_propagation.py`` and
``test_subparser_routing_fallback.py``.
"""

import argparse


def _build_parser():
    """Minimal replica of the slice of the hermes parser that exhibits
    the bug: top-level subparsers (dest="command") and ``mcp add`` with
    its ``--command`` flag.
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

    return parser


class TestMcpAddCommandDest:
    def test_url_invocation_preserves_top_level_command(self):
        """`hermes mcp add foo --url ...` must keep args.command == "mcp".

        Before the dest fix this was clobbered to None, sending the
        dispatcher into the chat fallback.
        """
        parser = _build_parser()
        args = parser.parse_args(
            ["mcp", "add", "foo", "--url", "https://example.com/mcp"]
        )

        assert args.command == "mcp"
        assert args.mcp_action == "add"
        assert args.name == "foo"
        assert args.url == "https://example.com/mcp"
        assert args.mcp_command is None

    def test_command_flag_writes_to_mcp_command_dest(self):
        """`--command npx` must populate args.mcp_command, not args.command."""
        parser = _build_parser()
        args = parser.parse_args(
            ["mcp", "add", "github", "--command", "npx"]
        )

        assert args.command == "mcp"
        assert args.mcp_command == "npx"

    def test_bare_mcp_add_does_not_clobber_command(self):
        """Even without --url or --command, args.command stays "mcp".

        Catches the regression at the parser layer regardless of which
        transport flag the user passes.
        """
        parser = _build_parser()
        args = parser.parse_args(["mcp", "add", "foo"])

        assert args.command == "mcp"
        assert args.mcp_command is None
        assert args.url is None
