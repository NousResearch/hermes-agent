"""
Top-level argparse construction for the hermes CLI.

Lives in its own module so other modules (e.g. ``relaunch.py``) can
introspect the parser to discover which flags exist without running the
``main`` fn.

Only the top-level parser and the ``chat`` subparser live here. Every other
subparser (model, gateway, sessions, …) is built inline in ``main.py``
because its dispatch is tightly coupled to module-level ``cmd_*`` functions.
"""

import argparse


# `--profile` / `-p` is consumed by ``main._apply_profile_override`` before
# argparse runs (it sets ``HERMES_HOME`` and strips itself from ``sys.argv``),
# so it isn't on the parser. Listed here so all "carry over on relaunch"
# metadata lives in one file.
PRE_ARGPARSE_INHERITED_FLAGS: list[tuple[str, bool]] = [
    ("--profile", True),
    ("-p", True),
]


def _inherited_flag(parser, *args, **kwargs):
    """Register a flag that ``hermes_cli.relaunch`` should carry over when
    the CLI re-execs itself (e.g. after ``sessions browse`` picks a session,
    or after the setup wizard launches chat).

    Equivalent to ``parser.add_argument(...)`` plus tagging the resulting
    Action with ``inherit_on_relaunch = True`` so the relaunch table builder
    can find it via introspection.
    """
    action = parser.add_argument(*args, **kwargs)
    action.inherit_on_relaunch = True
    return action


_EPILOGUE = """
Examples:
    hermes                        Start interactive chat
    hermes chat -q "Hello"        Single query mode
    hermes --tui                  Launch the modern TUI (or set display.interface: tui)
    hermes --cli                  Force the classic REPL (overrides display.interface: tui)
    hermes -c                     Resume the most recent session
    hermes -c "my project"        Resume a session by name (latest in lineage)
    hermes --resume <session_id>  Resume a specific session by ID
    hermes --resume latest        Resume the most recent session (same as -c)
    hermes --tui --resume latest --in ./dir   Resume ./dir's latest session in the TUI
    hermes setup                  Run setup wizard
    hermes logout                 Clear stored authentication
    hermes auth add <provider>    Add a pooled credential
    hermes auth list              List pooled credentials
    hermes auth remove <p> <t>    Remove pooled credential by index, id, or label
    hermes auth reset <provider>  Clear exhaustion status for a provider
    hermes model                  Select default model
    hermes fallback [list]        Show fallback provider chain
    hermes fallback add           Add a fallback provider (same picker as `hermes model`)
    hermes fallback remove        Remove a fallback provider from the chain
    hermes config                 View configuration
    hermes config edit            Edit config in $EDITOR
    hermes config set model gpt-4 Set a config value
    hermes gateway                Run messaging gateway
    hermes -s hermes-agent-dev,github-auth
    hermes -w                     Start in isolated git worktree
    hermes gateway install        Install gateway background service
    hermes sessions list          List past sessions
    hermes sessions browse        Interactive session picker
    hermes sessions rename ID T   Rename/title a session
    hermes logs                   View agent.log (last 50 lines)
    hermes logs -f                Follow agent.log in real time
    hermes logs errors            View errors.log
    hermes logs --since 1h        Lines from the last hour
    hermes debug share             Upload debug report for support
    hermes console                Open the safe Hermes command console
    hermes update                 Update to latest version
    hermes dashboard              Start web UI dashboard (port 9119)
    hermes dashboard --stop       Stop running dashboard processes
    hermes dashboard --status     List running dashboard processes

For more help on a command:
    hermes <command> --help
"""


def build_top_level_parser():
    """Build the top-level parser, the subparsers action, and the ``chat`` subparser.

    Returns ``(parser, subparsers, chat_parser)``. The caller wires
    ``chat_parser.set_defaults(func=cmd_chat)`` and continues registering
    other subparsers via ``subparsers.add_parser(...)``.
    """
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Hermes Agent - AI assistant with tool-calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOGUE,
    )

    parser.add_argument(
        "--version", "-V", action="store_true", help="Show version and exit"
    )
    parser.add_argument(
        "-z",
        "--oneshot",
        metavar="PROMPT",
        default=None,
        help=(
            "One-shot mode: send a single prompt and print ONLY the final "
            "response text to stdout. No banner, no spinner, no tool "
            "previews, no session_id line. Tools, memory, rules, and "
            "AGENTS.md in the CWD are loaded as normal; approvals are "
            "auto-bypassed. Intended for scripts / pipes."
        ),
    )
    parser.add_argument(
        "--usage-file",
        metavar="PATH",
        default=None,
        help=(
            "One-shot mode only: after the run, write a JSON usage report "
            "(estimated cost, token counts, model, api_calls) to PATH. "
            "The report is written even when the run fails, so pipelines "
            "can always account for spend. No effect outside -z/--oneshot."
        ),
    )
    # --model / --provider are accepted at the top level so they can pair
    # with -z without needing the `chat` subcommand.  If neither -z nor a
    # subcommand consumes them, they fall through harmlessly as None.
    # Mirrors `hermes chat --model ... --provider ...` semantics.
    _inherited_flag(
        parser,
        "-m",
        "--model",
        default=None,
        help=(
            "Model override for this invocation (e.g. anthropic/claude-sonnet-4.6). "
            "Applies to -z/--oneshot and --tui. Also settable via HERMES_INFERENCE_MODEL env var."
        ),
    )
    _inherited_flag(
        parser,
        "--provider",
        default=None,
        help=(
            "Provider override for this invocation (e.g. openrouter, anthropic). "
            "Applies to -z/--oneshot and --tui. The persistent provider lives in config.yaml "
            "under model.provider — use `hermes setup` or edit the file to change it."
        ),
    )
    _inherited_flag(
        parser,
        "--reasoning",
        default=None,
        metavar="LEVEL",
        help=(
            "Reasoning effort for this invocation: none, minimal, low, medium, "
            "high, xhigh, max, or ultra. Overrides agent.reasoning_effort in "
            "config.yaml for this run only; the persistent level lives there "
            "(or per-model under agent.reasoning_overrides)."
        ),
    )
    parser.add_argument(
        "-t",
        "--toolsets",
        default=None,
        help="Comma-separated toolsets to enable for this invocation. Applies to -z/--oneshot and --tui.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        metavar="SESSION",
        default=None,
        help=(
            "Resume a previous session by ID or title, or pass 'latest' for "
            "the most recent session (workspace-scoped, like -c with no name)"
        ),
    )
    parser.add_argument(
        "--no-restore-cwd",
        action="store_true",
        default=False,
        help="Don't cd into a resumed session's recorded working directory.",
    )
    parser.add_argument(
        "--in",
        dest="in_dir",
        metavar="DIR",
        default=None,
        help=(
            "Change into DIR before starting or resuming. Combined with "
            "'--resume latest' or -c, the most recent session for DIR's "
            "workspace is picked, and the session stays in DIR (skips the "
            "recorded-cwd restore)."
        ),
    )
    parser.add_argument(
        "--continue",
        "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION_NAME",
        help="Resume a session by name, or the most recent if no name given",
    )
    parser.add_argument(
        "--worktree",
        "-w",
        action="store_true",
        default=False,
        help="Run in an isolated git worktree (for parallel agents)",
    )
    _inherited_flag(
        parser,
        "--accept-hooks",
        action="store_true",
        default=False,
        help=(
            "Auto-approve any unseen shell hooks declared in config.yaml "
            "without a TTY prompt.  Equivalent to HERMES_ACCEPT_HOOKS=1 or "
            "hooks_auto_accept: true in config.yaml.  Use on CI / headless "
            "runs that can't prompt."
        ),
    )
    _inherited_flag(
        parser,
        "--skills",
        "-s",
        action="append",
        default=None,
        help="Preload one or more skills for the session (repeat flag or comma-separate)",
    )
    _inherited_flag(
        parser,
        "--yolo",
        action="store_true",
        default=False,
        help="Bypass all dangerous command approval prompts (use at your own risk)",
    )
    _inherited_flag(
        parser,
        "--pass-session-id",
        action="store_true",
        default=False,
        help="Include the session ID in the agent's system prompt",
    )
    _inherited_flag(
        parser,
        "--ignore-user-config",
        action="store_true",
        default=False,
        help="Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded)",
    )
    _inherited_flag(
        parser,
        "--ignore-rules",
        action="store_true",
        default=False,
        help="Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills",
    )
    _inherited_flag(
        parser,
        "--safe-mode",
        action="store_true",
        default=False,
        help="Troubleshooting mode: disable ALL customizations — user config, AGENTS.md/memory injection, plugins, and MCP servers (implies --ignore-user-config and --ignore-rules)",
    )
    _inherited_flag(
        parser,
        "--tui",
        action="store_true",
        default=False,
        help="Launch the modern TUI instead of the classic REPL",
    )
    _inherited_flag(
        parser,
        "--cli",
        action="store_true",
        default=False,
        help="Force the classic prompt_toolkit REPL (overrides display.interface=tui)",
    )
    _inherited_flag(
        parser,
        "--dev",
        dest="tui_dev",
        action="store_true",
        default=False,
        help="With --tui: run TypeScript sources via tsx (skip dist build)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # =========================================================================
    # chat command
    # =========================================================================
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat with the agent",
        description="Start an interactive chat session with Hermes Agent",
    )
    _query_group = chat_parser.add_mutually_exclusive_group()
    _query_group.add_argument(
        "-q", "--query", help="Single query (non-interactive mode)"
    )
    _query_group.add_argument(
        "--query-file",
        metavar="PATH",
        help=(
            "Read the single query from a file instead of the command line "
            "('-' reads stdin). Safe for arbitrary text: nothing is shell-"
            "interpreted, so quotes, $(...), and backticks are preserved "
            "verbatim. Mutually exclusive with -q."
        ),
    )
    chat_parser.add_argument(
        "--image", help="Optional local image path to attach to a single query"
    )
    # `default=argparse.SUPPRESS` on flags that are ALSO declared on the
    # top-level parser: when the user writes `hermes -m foo chat`, argparse
    # first sets `args.model = "foo"` from the top-level parser, then
    # dispatches to the chat subparser. Without SUPPRESS the chat subparser's
    # own default (`None`) would silently clobber the top-level value because
    # the subparser shares the same namespace and `dest`. SUPPRESS keeps the
    # subparser action a no-op unless the user actually passes the flag after
    # the subcommand. Matches the pattern already used for `-s/--skills` and
    # the relaunch-inherited flags `-r/--resume`, `-c/--continue`,
    # `-w/--worktree`, `--yolo`, etc. (see tests/hermes_cli/
    # test_argparse_flag_propagation.py).
    _inherited_flag(
        chat_parser,
        "-m", "--model",
        default=argparse.SUPPRESS,
        help="Model to use (e.g., anthropic/claude-sonnet-4)",
    )
    chat_parser.add_argument(
        "-t", "--toolsets",
        default=argparse.SUPPRESS,
        help="Comma-separated toolsets to enable",
    )
    _inherited_flag(
        chat_parser,
        "--reasoning",
        default=argparse.SUPPRESS,
        metavar="LEVEL",
        help=(
            "Reasoning effort for this session: none, minimal, low, medium, "
            "high, xhigh, max, or ultra. Overrides agent.reasoning_effort for "
            "this run only (same levels as the /reasoning slash command)."
        ),
    )
    _inherited_flag(
        chat_parser,
        "-s",
        "--skills",
        action="append",
        default=argparse.SUPPRESS,
        help="Preload one or more skills for the session (repeat flag or comma-separate)",
    )
    _inherited_flag(
        chat_parser,
        "--provider",
        # No `choices=` here: user-defined providers from config.yaml `providers:`
        # are also valid values, and runtime resolution (resolve_runtime_provider)
        # handles validation/error reporting consistently with the top-level
        # `--provider` flag.
        default=argparse.SUPPRESS,
        help="Inference provider (default: auto). Built-in or a user-defined name from `providers:` in config.yaml.",
    )
    chat_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Verbose output",
    )
    chat_parser.add_argument(
        "-Q",
        "--quiet",
        action="store_true",
        help="Quiet mode for programmatic use: suppress banner, spinner, and tool previews. Only output the final response and session info.",
    )
    chat_parser.add_argument(
        "--resume",
        "-r",
        metavar="SESSION_ID",
        default=argparse.SUPPRESS,
        help=(
            "Resume a previous session by ID (shown on exit), or 'latest' "
            "for the most recent session"
        ),
    )
    chat_parser.add_argument(
        "--no-restore-cwd",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Don't cd into a resumed session's recorded working directory.",
    )
    chat_parser.add_argument(
        "--in",
        dest="in_dir",
        metavar="DIR",
        default=argparse.SUPPRESS,
        help=(
            "Change into DIR before starting or resuming (scopes "
            "'--resume latest' / -c lookups to DIR's workspace)."
        ),
    )
    chat_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_last",
        nargs="?",
        const=True,
        default=argparse.SUPPRESS,
        metavar="SESSION_NAME",
        help="Resume a session by name, or the most recent if no name given",
    )
    chat_parser.add_argument(
        "--create-if-missing",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "With -c/--continue <name>: if no session matches the name, "
            "create a new session with that title and proceed (instead of "
            "failing with a not-found error). Programmatic callers that "
            "want 'send to this named thread, making it if needed'."
        ),
    )
    chat_parser.add_argument(
        "--worktree",
        "-w",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Run in an isolated git worktree (for parallel agents on the same repo)",
    )
    _inherited_flag(
        chat_parser,
        "--accept-hooks",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Auto-approve any unseen shell hooks declared in config.yaml "
            "without a TTY prompt (see also HERMES_ACCEPT_HOOKS env var and "
            "hooks_auto_accept: in config.yaml)."
        ),
    )
    chat_parser.add_argument(
        "--checkpoints",
        action="store_true",
        default=False,
        help="Enable filesystem checkpoints before destructive file operations (use /rollback to restore)",
    )
    chat_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help="Maximum tool-calling iterations per conversation turn (default: 500, or agent.max_turns in config)",
    )
    chat_parser.add_argument(
        "--run-budget",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="run_budget",
        help=(
            "Optional wall-clock budget in seconds for each conversation run. "
            "At 80%% elapsed the agent gets a one-time wrap-up notice, and "
            "implicit provider stale timeouts are capped to the remaining "
            "budget so one hung call can't consume the run. Unset = off. "
            "Also configurable as agent.run_budget_seconds in config.yaml. "
            "Intended for one-shot/eval invocations with a hard ceiling."
        ),
    )
    _inherited_flag(
        chat_parser,
        "--yolo",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Bypass all dangerous command approval prompts (use at your own risk)",
    )
    _inherited_flag(
        chat_parser,
        "--pass-session-id",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Include the session ID in the agent's system prompt",
    )
    _inherited_flag(
        chat_parser,
        "--ignore-user-config",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded). Useful for isolated CI runs, reproduction, and third-party integrations.",
    )
    _inherited_flag(
        chat_parser,
        "--ignore-rules",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills. Combine with --ignore-user-config for a fully isolated run.",
    )
    _inherited_flag(
        chat_parser,
        "--safe-mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Troubleshooting mode: disable ALL customizations — user config, AGENTS.md/memory injection, plugins, and MCP servers (implies --ignore-user-config and --ignore-rules). Use to isolate whether a problem comes from your setup or from Hermes itself.",
    )
    chat_parser.add_argument(
        "--source",
        default=None,
        help="Session source tag for filtering (default: cli). Use 'tool' for third-party integrations that should not appear in user session lists.",
    )
    _inherited_flag(
        chat_parser,
        "--tui",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Launch the modern TUI instead of the classic REPL",
    )
    _inherited_flag(
        chat_parser,
        "--cli",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Force the classic prompt_toolkit REPL (overrides display.interface=tui)",
    )
    _inherited_flag(
        chat_parser,
        "--dev",
        dest="tui_dev",
        action="store_true",
        default=argparse.SUPPRESS,
        help="With --tui: run TypeScript sources via tsx (skip dist build)",
    )

    return parser, subparsers, chat_parser
