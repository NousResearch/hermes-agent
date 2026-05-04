"""CLI subcommand: ``hermes send`` — pipe text from shell scripts to any
configured messaging platform (Telegram, Discord, Slack, Signal, SMS, etc.).

This is a thin wrapper around ``tools.send_message_tool.send_message_tool``
that exposes its functionality as a standalone CLI entry point so ops
scripts, cron jobs, CI hooks, and monitoring daemons can reuse the gateway's
already-configured credentials without having to reimplement each platform's
REST API client.

Design notes:

* No LLM, no agent loop — the subcommand just resolves arguments, reads the
  message body, calls the shared tool function, and prints/returns the
  result. It is intentionally fast, cheap, and side-effect-only.
* For platforms that send via bot token (Telegram, Discord, Slack, Signal,
  SMS, WhatsApp-CloudAPI, …) no running gateway is required. The tool
  talks directly to each platform's REST endpoint. For platforms that rely
  on a persistent adapter connection (plugin platforms, Matrix in some
  modes, …) a live gateway is needed; the underlying tool surfaces that
  error to the caller.
* Exit codes follow the classic Unix convention:
    0 — delivery (or list) succeeded
    1 — delivery failed at the platform level
    2 — usage / argument / config error (argparse already uses 2)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


_USAGE_EXIT = 2
_FAILURE_EXIT = 1
_SUCCESS_EXIT = 0


def _read_message_body(
    positional: Optional[str],
    file_path: Optional[str],
) -> Optional[str]:
    """Resolve the message body from (in order):

    1. An explicit positional message argument.
    2. ``--file PATH`` or ``--file -`` (where ``-`` means stdin).
    3. Piped stdin when it is not attached to a TTY.

    Returns ``None`` when nothing is available — callers must treat that as
    a usage error.
    """
    if positional:
        return positional

    if file_path:
        if file_path == "-":
            return sys.stdin.read()
        try:
            return Path(file_path).read_text()
        except OSError as exc:
            print(f"hermes send: cannot read {file_path}: {exc}", file=sys.stderr)
            sys.exit(_USAGE_EXIT)

    # Piped input: only consume stdin when it is not a TTY. Reading from a
    # TTY would block the user in a half-broken "type your message" state,
    # which is a poor default for an ops CLI.
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data

    return None


def _resolve_target(arg_to: Optional[str]) -> Optional[str]:
    """Return a cleaned ``--to`` value, or ``None`` when nothing is set."""
    if arg_to and arg_to.strip():
        return arg_to.strip()
    return None


def _emit_result(
    result_json: str,
    *,
    json_mode: bool,
    quiet: bool,
) -> int:
    """Print the tool result in the requested format and return the exit code.

    The underlying ``send_message_tool`` always returns a JSON string. We
    parse it, decide success/failure, and format accordingly.
    """
    try:
        payload = json.loads(result_json) if result_json else {}
    except json.JSONDecodeError:
        # Shouldn't happen with the shared tool, but be defensive — pass the
        # raw string through so the user can still see what went wrong.
        payload = {"error": "invalid JSON from send_message_tool", "raw": result_json}

    if json_mode:
        print(json.dumps(payload, indent=2))
    elif quiet:
        pass
    else:
        if payload.get("error"):
            print(f"hermes send: {payload['error']}", file=sys.stderr)
        elif payload.get("success"):
            note = payload.get("note")
            if note:
                print(note)
            else:
                print("sent")
        else:
            # Unknown shape — dump it so nothing is silently dropped.
            print(json.dumps(payload, indent=2))

    if payload.get("error"):
        return _FAILURE_EXIT
    if payload.get("skipped"):
        return _SUCCESS_EXIT
    if payload.get("success"):
        return _SUCCESS_EXIT
    # Unknown / unexpected — treat as failure so scripts notice.
    return _FAILURE_EXIT


def _list_targets(platform_filter: Optional[str], *, json_mode: bool) -> int:
    """Print the channel directory (all configured targets across platforms).

    Uses ``load_directory()`` for structured JSON output and
    ``format_directory_for_display()`` for the human-readable rendering that
    the send_message tool itself shows to the model — keeps the two surfaces
    identical.
    """
    try:
        from gateway.channel_directory import (
            format_directory_for_display,
            load_directory,
        )
    except Exception as exc:
        print(f"hermes send: failed to load channel directory: {exc}", file=sys.stderr)
        return _FAILURE_EXIT

    try:
        raw = load_directory()
    except Exception as exc:
        print(f"hermes send: failed to read channel directory: {exc}", file=sys.stderr)
        return _FAILURE_EXIT

    platforms = dict(raw.get("platforms") or {})

    if platform_filter:
        key = platform_filter.strip().lower()
        filtered = {k: v for k, v in platforms.items() if k.lower() == key}
        if not filtered:
            print(
                f"hermes send: no targets found for platform '{platform_filter}'. "
                f"Configured: {', '.join(sorted(platforms)) or '(none)'}",
                file=sys.stderr,
            )
            return _FAILURE_EXIT
        platforms = filtered

    if json_mode:
        print(json.dumps({"platforms": platforms}, indent=2, default=str))
        return _SUCCESS_EXIT

    if not any(platforms.values()):
        print("No messaging platforms configured or no channels discovered yet.")
        print("Set one up with `hermes gateway setup`, or run the gateway once so")
        print("channel discovery can populate ~/.hermes/channel_directory.json.")
        return _SUCCESS_EXIT

    # Human display — when unfiltered, reuse the shared formatter the agent
    # already sees. When filtered, build a minimal view ourselves.
    if platform_filter is None:
        print(format_directory_for_display())
        return _SUCCESS_EXIT

    for plat_name in sorted(platforms):
        channels = platforms[plat_name]
        print(f"{plat_name}:")
        if not channels:
            print("  (no channels discovered yet)")
            continue
        for ch in channels:
            name = ch.get("name", "?")
            chat_id = ch.get("id") or ch.get("chat_id") or ""
            suffix = f"  [{chat_id}]" if chat_id and chat_id != name else ""
            print(f"  {plat_name}:{name}{suffix}")
        print()

    return _SUCCESS_EXIT


def _load_hermes_env() -> None:
    """Populate ``os.environ`` from ``~/.hermes/.env`` AND bridge top-level
    ``config.yaml`` keys into the environment so the underlying gateway
    config loader sees platform credentials and home channel IDs.

    ``send_message_tool`` reads tokens and home-channel IDs via
    ``os.getenv(...)`` on each call. The gateway process does two things at
    startup that ``hermes send`` must replicate when invoked standalone:

    1. ``load_dotenv(~/.hermes/.env)`` — brings bot tokens into the env.
    2. Bridge top-level simple values from ``~/.hermes/config.yaml`` into
       ``os.environ`` (without overriding existing env vars). This is where
       ``TELEGRAM_HOME_CHANNEL`` and friends live when the user saved them
       via ``hermes config set``.

    See ``gateway/run.py`` for the canonical version of this bridge — we
    intentionally reimplement the minimum needed here so ``hermes send``
    doesn't pull in the full gateway module just to resolve a home channel.
    """
    # Step 1: dotenv
    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None  # type: ignore[assignment]

    try:
        from hermes_cli.config import get_hermes_home
        home = get_hermes_home()
    except Exception:
        return

    env_path = home / ".env"
    if load_dotenv and env_path.exists():
        try:
            load_dotenv(str(env_path), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                load_dotenv(str(env_path), override=True, encoding="latin-1")
            except Exception:
                pass
        except Exception:
            pass

    # Step 2: bridge top-level config.yaml values into the environment so
    # gateway.config.load_gateway_config() sees them. Scalars only; don't
    # override values already in the env.
    import os
    config_path = home / "config.yaml"
    if not config_path.exists():
        return

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        return

    try:
        from hermes_cli.config import _expand_env_vars
        raw = _expand_env_vars(raw)
    except Exception:
        pass

    if not isinstance(raw, dict):
        return

    for key, val in raw.items():
        if not isinstance(val, (str, int, float, bool)):
            continue
        if key in os.environ:
            continue
        os.environ[key] = str(val)


def cmd_send(args: argparse.Namespace) -> None:
    """Entry point wired into the top-level argparse dispatcher."""

    # Bridge ~/.hermes/.env and ~/.hermes/config.yaml into os.environ so the
    # gateway config loader (invoked downstream by send_message_tool and by
    # the channel directory) can see platform credentials and home channels.
    _load_hermes_env()

    # --list short-circuits everything else.
    if getattr(args, "list_targets", False):
        # When `--list telegram` is used, argparse stores "telegram" in the
        # `message` positional (since list_targets takes no argument).
        platform_filter = getattr(args, "message", None)
        exit_code = _list_targets(platform_filter, json_mode=getattr(args, "json", False))
        sys.exit(exit_code)

    target = _resolve_target(getattr(args, "to", None))
    if not target:
        print(
            "hermes send: --to PLATFORM[:channel[:thread]] is required\n"
            "Examples:\n"
            "  hermes send --to telegram \"hello\"\n"
            "  hermes send --to discord:#ops --file report.md\n"
            "  hermes send --list      # list available targets",
            file=sys.stderr,
        )
        sys.exit(_USAGE_EXIT)

    message = _read_message_body(
        getattr(args, "message", None),
        getattr(args, "file", None),
    )
    if message is None or not message.strip():
        print(
            "hermes send: no message provided. Pass text as a positional "
            "argument, use --file PATH, or pipe data via stdin.",
            file=sys.stderr,
        )
        sys.exit(_USAGE_EXIT)

    # Optional: prepend a subject line. Useful for alerting scripts that
    # want a consistent header without inlining it into every call.
    subject = getattr(args, "subject", None)
    if subject:
        message = f"{subject}\n\n{message.lstrip()}"

    # Import lazily so `hermes send --help` stays fast and does not pull in
    # the full tool registry / gateway config stack.
    from tools.send_message_tool import send_message_tool

    # send_message_tool auto-loads gateway config + env and routes to the
    # appropriate platform adapter (bot-token path for Telegram/Discord/Slack/
    # Signal/SMS/WhatsApp; live-adapter path for plugin platforms).
    #
    # It expects the standard tool-call dict and returns a JSON string.
    tool_args = {
        "action": "send",
        "target": target,
        "message": message,
    }

    result = send_message_tool(tool_args)
    exit_code = _emit_result(
        result,
        json_mode=getattr(args, "json", False),
        quiet=getattr(args, "quiet", False),
    )
    sys.exit(exit_code)


def register_send_subparser(subparsers) -> argparse.ArgumentParser:
    """Create the ``send`` subparser and return it.

    Kept as a standalone function so the top-level parser builder can wire
    it in next to the other messaging subcommands without cluttering
    ``_parser.py`` or ``main.py``.
    """
    parser = subparsers.add_parser(
        "send",
        help="Send a message to a configured platform (scripts, cron jobs, CI).",
        description=(
            "Pipe text from any shell script to any messaging platform Hermes "
            "is already configured for. Reuses the gateway's platform "
            "credentials (~/.hermes/.env + ~/.hermes/config.yaml) — no LLM, "
            "no agent loop, no running gateway required for bot-token "
            "platforms like Telegram/Discord/Slack/Signal."
        ),
        epilog=(
            "Examples:\n"
            "  hermes send --to telegram \"deploy finished\"\n"
            "  echo \"RAM 92%\" | hermes send --to telegram:-1001234567890\n"
            "  hermes send --to discord:#ops --file /tmp/report.md\n"
            "  hermes send --to slack:#eng --subject \"[CI]\" --file build.log\n"
            "  hermes send --list                  # all platforms\n"
            "  hermes send --list telegram         # filter by platform\n"
            "\n"
            "Exit codes: 0 ok, 1 delivery/backend error, 2 usage error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--to",
        metavar="TARGET",
        default=None,
        help=(
            "Delivery target. Format: 'platform' (home channel), "
            "'platform:chat_id', 'platform:chat_id:thread_id', or "
            "'platform:#channel-name'. Examples: telegram, "
            "telegram:-1001234567890:17585, discord:#ops, slack:C0123ABCD, "
            "signal:+15551234567."
        ),
    )

    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Message text. If omitted, read from --file or stdin.",
    )

    # Legacy / convenience positional removed — use --to for clarity.

    parser.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        default=None,
        help="Read message body from PATH. Use '-' to force stdin.",
    )

    parser.add_argument(
        "-s",
        "--subject",
        metavar="LINE",
        default=None,
        help="Prepend a subject/header line before the message body.",
    )

    parser.add_argument(
        "-l",
        "--list",
        dest="list_targets",
        action="store_true",
        default=False,
        help="List available targets. Optional positional filter: `hermes send --list telegram`.",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress stdout on success (exit code only).",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit raw JSON result instead of human-readable output.",
    )

    parser.set_defaults(func=cmd_send)
    return parser


__all__ = ["cmd_send", "register_send_subparser"]
