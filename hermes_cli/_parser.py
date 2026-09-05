"""Top-level argparse construction for the hermes CLI.

Only the top-level parser and the ``chat`` subparser live here. Every other subparser (model,
gateway, sessions, …) is built by ``hermes_cli/subcommands/<group>.py`` and wired in
``main._build_cli_parser`` with its ``cmd_*`` handler injected.
"""

import argparse
from functools import lru_cache

# `--profile` / `-p` is consumed by ``main._apply_profile_override`` before argparse runs
# (it sets ``HERMES_HOME`` and strips itself from ``sys.argv``), so it isn't on the parser.
# Listed here so all "carry over on relaunch" metadata lives in one file.
PRE_ARGPARSE_INHERITED_FLAGS: list[tuple[str, bool]] = [("--profile", True), ("-p", True)]

# Static snapshot fallback for ``top_level_value_flag_sets`` — used only if introspecting the
# live parser fails. The derived path is authoritative; tests/hermes_cli/
# test_top_level_value_flags_parity.py fails CI if the parser grows a value-taking flag this
# snapshot lacks AND derivation regresses.
_VALUE_FLAGS_FALLBACK: frozenset[str] = frozenset({
    "-z", "--oneshot", "-m", "--model", "--provider", "--reasoning", "-t", "--toolsets",
    "-r", "--resume", "-s", "--skills", "--usage-file", "--in",
})
_OPTIONAL_VALUE_FLAGS_FALLBACK: frozenset[str] = frozenset({"-c", "--continue"})


@lru_cache(maxsize=1)
def top_level_value_flag_sets() -> tuple[frozenset[str], frozenset[str]]:
    """(required-value, optional-value) top-level flags, derived from the REAL parser.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0) so the argv scanners in
    ``main.py`` (``_first_positional_argv``, ``_apply_profile_override``) can never drift from the
    argparse surface — the drift that made ``hermes --reasoning high chat …`` misread ``high`` as
    the subcommand and forced eager plugin discovery.

    Mirrors the ``update_cmd._holder_value_flags`` precedent, including the handwritten-snapshot fallback
    for a broken parser import. Cached per process. See #93530.
    """
    try:
        parser = build_top_level_parser()[0]
        required: set[str] = set()
        optional: set[str] = set()
        for action in parser._actions:
            if not action.option_strings or action.nargs == 0:
                continue
            target = optional if action.nargs == "?" else required
            target.update(action.option_strings)
        return frozenset(required), frozenset(optional)
    except Exception:
        return _VALUE_FLAGS_FALLBACK, _OPTIONAL_VALUE_FLAGS_FALLBACK


def _inherited_flag(parser, *args, **kwargs):
    """``parser.add_argument`` + tag the Action ``inherit_on_relaunch`` for ``hermes_cli.relaunch``."""
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


def _add_top_level_flags(parser: argparse.ArgumentParser) -> None:
    """Top-level (pre-subcommand) flags; ``-m/--provider`` pair with ``-z`` without ``chat``."""
    add, inherited = parser.add_argument, _inherited_flag
    add("--version", "-V", action="store_true", help="Show version and exit")
    add("-z", "--oneshot", metavar="PROMPT", default=None, help=(
        "One-shot mode: send a single prompt and print ONLY the final "
        "response text to stdout. No banner, no spinner, no tool "
        "previews, no session_id line. Tools, memory, rules, and "
        "AGENTS.md in the CWD are loaded as normal; approvals are "
        "auto-bypassed. Intended for scripts / pipes."))
    add("--usage-file", metavar="PATH", default=None, help=(
        "One-shot mode only: after the run, write a JSON usage report "
        "(estimated cost, token counts, model, api_calls) to PATH. "
        "The report is written even when the run fails, so pipelines "
        "can always account for spend. No effect outside -z/--oneshot."))
    # --model / --provider are accepted at the top level so they can pair with -z without the
    # `chat` subcommand; if neither -z nor a subcommand consumes them, they fall through as None.
    inherited(parser, "-m", "--model", default=None, help=(
        "Model override for this invocation (e.g. anthropic/claude-sonnet-4.6). "
        "Applies to -z/--oneshot and --tui. Also settable via HERMES_INFERENCE_MODEL env var."))
    inherited(parser, "--provider", default=None, help=(
        "Provider override for this invocation (e.g. openrouter, anthropic). "
        "Applies to -z/--oneshot and --tui. The persistent provider lives in config.yaml "
        "under model.provider — use `hermes setup` or edit the file to change it."))
    inherited(parser, "--reasoning", default=None, metavar="LEVEL", help=(
        "Reasoning effort for this invocation: none, minimal, low, medium, "
        "high, xhigh, max, or ultra. Overrides agent.reasoning_effort in "
        "config.yaml for this run only; the persistent level lives there "
        "(or per-model under agent.reasoning_overrides)."))
    add("-t", "--toolsets", default=None,
        help="Comma-separated toolsets to enable for this invocation. Applies to -z/--oneshot and --tui.")
    add("--resume", "-r", metavar="SESSION", default=None, help=(
        "Resume a previous session by ID or title, or pass 'latest' for "
        "the most recent session (workspace-scoped, like -c with no name)"))
    add("--no-restore-cwd", action="store_true", default=False,
        help="Don't cd into a resumed session's recorded working directory.")
    add("--in", dest="in_dir", metavar="DIR", default=None, help=(
        "Change into DIR before starting or resuming. Combined with "
        "'--resume latest' or -c, the most recent session for DIR's "
        "workspace is picked, and the session stays in DIR (skips the "
        "recorded-cwd restore)."))
    add("--continue", "-c", dest="continue_last", nargs="?", const=True, default=None,
        metavar="SESSION_NAME", help="Resume a session by name, or the most recent if no name given")
    add("--worktree", "-w", action="store_true", default=False,
        help="Run in an isolated git worktree (for parallel agents)")
    inherited(parser, "--accept-hooks", action="store_true", default=False, help=(
        "Auto-approve any unseen shell hooks declared in config.yaml "
        "without a TTY prompt.  Equivalent to HERMES_ACCEPT_HOOKS=1 or "
        "hooks_auto_accept: true in config.yaml.  Use on CI / headless "
        "runs that can't prompt."))
    inherited(parser, "--skills", "-s", action="append", default=None,
              help="Preload one or more skills for the session (repeat flag or comma-separate)")
    inherited(parser, "--yolo", action="store_true", default=False,
              help="Bypass all dangerous command approval prompts (use at your own risk)")
    inherited(parser, "--pass-session-id", action="store_true", default=False,
              help="Include the session ID in the agent's system prompt")
    inherited(parser, "--ignore-user-config", action="store_true", default=False,
              help="Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded)")
    inherited(parser, "--ignore-rules", action="store_true", default=False,
              help="Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills")
    inherited(parser, "--safe-mode", action="store_true", default=False,
              help="Troubleshooting mode: disable ALL customizations — user config, AGENTS.md/memory injection, plugins, and MCP servers (implies --ignore-user-config and --ignore-rules)")
    inherited(parser, "--tui", action="store_true", default=False,
              help="Launch the modern TUI instead of the classic REPL")
    inherited(parser, "--cli", action="store_true", default=False,
              help="Force the classic prompt_toolkit REPL (overrides display.interface=tui)")
    inherited(parser, "--dev", dest="tui_dev", action="store_true", default=False,
              help="With --tui: run TypeScript sources via tsx (skip dist build)")


def _build_chat_parser(subparsers) -> argparse.ArgumentParser:
    """The ``chat`` subparser (also the implicit default command).

    Flags ALSO declared on the top-level parser use ``default=argparse.SUPPRESS``: for
    ``hermes -m foo chat`` argparse first sets ``args.model`` from the top-level parser, then
    dispatches to the chat subparser, which shares the namespace and ``dest`` — a plain ``None``
    default would silently clobber the top-level value. SUPPRESS keeps the subparser action a no-op
    unless the flag is actually passed after the subcommand (tests/hermes_cli/
    test_argparse_flag_propagation.py).
    """
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat with the agent",
        description="Start an interactive chat session with Hermes Agent")
    add, inherited, SUPPRESS = chat_parser.add_argument, _inherited_flag, argparse.SUPPRESS
    _query_group = chat_parser.add_mutually_exclusive_group()
    _query_group.add_argument("-q", "--query", help=(
        "Query to run. On a real TTY the prompt seeds an interactive "
        "session (submitted literally as the first turn); combined with "
        "--oneshot or -Q, or on a non-TTY, it answers and exits."))
    _query_group.add_argument("--query-file", metavar="PATH", help=(
        "Read the single query from a file instead of the command line "
        "('-' reads stdin). Safe for arbitrary text: nothing is shell-"
        "interpreted, so quotes, $(...), and backticks are preserved "
        "verbatim. Mutually exclusive with -q."))
    # Distinct dest: the top-level `-z/--oneshot PROMPT` is value-taking and its dispatch sites do
    # `if args.oneshot: _run_and_exit_oneshot(args.oneshot)` — a shared boolean dest would be
    # passed as the prompt.
    add("--oneshot", dest="oneshot_exit", action="store_true", default=False, help=(
        "With -q/--query-file: answer the query and exit (legacy "
        "single-query behavior) instead of seeding an interactive "
        "session. Implied on non-TTY stdio and by -Q/--quiet."))
    add("--image", help="Optional local image path to attach to a single query")
    inherited(chat_parser, "-m", "--model", default=SUPPRESS,
              help="Model to use (e.g., anthropic/claude-sonnet-4)")
    add("-t", "--toolsets", default=SUPPRESS, help="Comma-separated toolsets to enable")
    inherited(chat_parser, "--reasoning", default=SUPPRESS, metavar="LEVEL", help=(
        "Reasoning effort for this session: none, minimal, low, medium, "
        "high, xhigh, max, or ultra. Overrides agent.reasoning_effort for "
        "this run only (same levels as the /reasoning slash command)."))
    inherited(chat_parser, "-s", "--skills", action="append", default=SUPPRESS,
              help="Preload one or more skills for the session (repeat flag or comma-separate)")
    # No `choices=` on --provider: user-defined providers from config.yaml `providers:` are valid
    # too; runtime resolution (resolve_runtime_provider) validates, same as the top-level flag.
    inherited(chat_parser, "--provider", default=SUPPRESS,
              help="Inference provider (default: auto). Built-in or a user-defined name from `providers:` in config.yaml.")
    add("-v", "--verbose", action="store_true", default=SUPPRESS, help="Verbose output")
    add("-Q", "--quiet", action="store_true",
        help="Quiet mode for programmatic use: suppress banner, spinner, and tool previews. Only output the final response and session info.")
    add("--resume", "-r", metavar="SESSION_ID", default=SUPPRESS, help=(
        "Resume a previous session by ID (shown on exit), or 'latest' "
        "for the most recent session"))
    add("--no-restore-cwd", action="store_true", default=SUPPRESS,
        help="Don't cd into a resumed session's recorded working directory.")
    add("--in", dest="in_dir", metavar="DIR", default=SUPPRESS, help=(
        "Change into DIR before starting or resuming (scopes "
        "'--resume latest' / -c lookups to DIR's workspace)."))
    add("--continue", "-c", dest="continue_last", nargs="?", const=True, default=SUPPRESS,
        metavar="SESSION_NAME", help="Resume a session by name, or the most recent if no name given")
    add("--create-if-missing", action="store_true", default=SUPPRESS, help=(
        "With -c/--continue <name>: if no session matches the name, "
        "create a new session with that title and proceed (instead of "
        "failing with a not-found error). Programmatic callers that "
        "want 'send to this named thread, making it if needed'."))
    add("--allow-self-bot-chat", action="store_true", default=SUPPRESS, help=(
        "Override the self-DM guard for `chat -c \"Bot Chat\"`: permitted "
        "only when the launching context is an agent session of the SAME "
        "profile. The guard exists because that shape forks a second, "
        "unregistered agent context of the profile (its board actions are "
        "attributed to the profile and can clobber a live claimant's run). "
        "Cross-profile relays never trip the guard."))
    add("--worktree", "-w", action="store_true", default=SUPPRESS,
        help="Run in an isolated git worktree (for parallel agents on the same repo)")
    inherited(chat_parser, "--accept-hooks", action="store_true", default=SUPPRESS, help=(
        "Auto-approve any unseen shell hooks declared in config.yaml "
        "without a TTY prompt (see also HERMES_ACCEPT_HOOKS env var and "
        "hooks_auto_accept: in config.yaml)."))
    add("--checkpoints", action="store_true", default=False,
        help="Enable filesystem checkpoints before destructive file operations (use /rollback to restore)")
    add("--max-turns", type=int, default=None, metavar="N",
        help="Maximum tool-calling iterations per conversation turn (default: 500, or agent.max_turns in config)")
    add("--run-budget", type=float, default=None, metavar="SECONDS", dest="run_budget", help=(
        "Optional wall-clock budget in seconds for each conversation run. "
        "At 80%% elapsed the agent gets a one-time wrap-up notice, and "
        "implicit provider stale timeouts are capped to the remaining "
        "budget so one hung call can't consume the run. Unset = off. "
        "Also configurable as agent.run_budget_seconds in config.yaml. "
        "Intended for one-shot/eval invocations with a hard ceiling."))
    inherited(chat_parser, "--yolo", action="store_true", default=SUPPRESS,
              help="Bypass all dangerous command approval prompts (use at your own risk)")
    inherited(chat_parser, "--pass-session-id", action="store_true", default=SUPPRESS,
              help="Include the session ID in the agent's system prompt")
    inherited(chat_parser, "--ignore-user-config", action="store_true", default=SUPPRESS,
              help="Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded). Useful for isolated CI runs, reproduction, and third-party integrations.")
    inherited(chat_parser, "--ignore-rules", action="store_true", default=SUPPRESS,
              help="Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills. Combine with --ignore-user-config for a fully isolated run.")
    inherited(chat_parser, "--safe-mode", action="store_true", default=SUPPRESS,
              help="Troubleshooting mode: disable ALL customizations — user config, AGENTS.md/memory injection, plugins, and MCP servers (implies --ignore-user-config and --ignore-rules). Use to isolate whether a problem comes from your setup or from Hermes itself.")
    add("--source", default=None,
        help="Session source tag for filtering (default: cli). Use 'tool' for third-party integrations that should not appear in user session lists.")
    inherited(chat_parser, "--tui", action="store_true", default=SUPPRESS,
              help="Launch the modern TUI instead of the classic REPL")
    inherited(chat_parser, "--cli", action="store_true", default=SUPPRESS,
              help="Force the classic prompt_toolkit REPL (overrides display.interface=tui)")
    inherited(chat_parser, "--dev", dest="tui_dev", action="store_true", default=SUPPRESS,
              help="With --tui: run TypeScript sources via tsx (skip dist build)")
    return chat_parser


def build_top_level_parser():
    """Build the top-level parser, the subparsers action, and the ``chat`` subparser.

    Returns ``(parser, subparsers, chat_parser)``; the caller wires
    ``chat_parser.set_defaults(func= cmd_chat)`` and registers further subparsers via
    ``subparsers.add_parser(...)``.
    """
    parser = argparse.ArgumentParser(
        prog="hermes", description="Hermes Agent - AI assistant with tool-calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOGUE)
    _add_top_level_flags(parser)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    return parser, subparsers, _build_chat_parser(subparsers)
