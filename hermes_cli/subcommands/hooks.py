"""``hermes hooks`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_hooks_parser(subparsers, *, cmd_hooks: Callable) -> None:
    """Attach the ``hooks`` subcommand to ``subparsers``."""
    # =========================================================================
    hooks_parser = subparsers.add_parser(
        "hooks",
        help="Inspect and manage shell-script hooks",
        description=(
            "Inspect shell-script hooks declared in ~/.hermes/config.yaml, "
            "test them against synthetic payloads, and manage the first-use "
            "consent allowlist at ~/.hermes/shell-hooks-allowlist.json."
        ),
    )
    hooks_subparsers = hooks_parser.add_subparsers(dest="hooks_action")

    hooks_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List configured hooks with matcher, timeout, and consent status",
    )

    _hk_test = hooks_subparsers.add_parser(
        "test",
        help="Fire every hook matching <event> against a synthetic payload",
    )
    _hk_test.add_argument(
        "event",
        help="Hook event name (e.g. pre_tool_call, pre_llm_call, subagent_stop)",
    )
    _hk_test.add_argument(
        "--for-tool",
        dest="for_tool",
        default=None,
        help=(
            "Only fire hooks whose matcher matches this tool name "
            "(used for pre_tool_call / post_tool_call)"
        ),
    )
    _hk_test.add_argument(
        "--payload-file",
        dest="payload_file",
        default=None,
        help=(
            "Path to a JSON file whose contents are merged into the "
            "synthetic payload before execution"
        ),
    )

    _hk_revoke = hooks_subparsers.add_parser(
        "revoke",
        aliases=["remove", "rm"],
        help="Remove a command's allowlist entries (takes effect on next restart)",
    )
    _hk_revoke.add_argument(
        "command",
        help="The exact command string to revoke (as declared in config.yaml)",
    )

    hooks_subparsers.add_parser(
        "doctor",
        help=(
            "Check each configured hook: exec bit, allowlist, mtime drift, "
            "JSON validity, and synthetic run timing"
        ),
    )

    hooks_parser.set_defaults(func=cmd_hooks)
