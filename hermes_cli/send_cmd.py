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
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional


_USAGE_EXIT = 2
_FAILURE_EXIT = 1
_SUCCESS_EXIT = 0
_TELEGRAM_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")


def _hash_identifier(value: Any) -> dict[str, Any]:
    """Return non-reversible metadata for a private identifier."""
    text = "" if value is None else str(value)
    if not text:
        return {"present": False}
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return {
        "present": True,
        "sha256_12": digest[:12],
        "length": len(text),
    }


def _redact_error_text(value: Any) -> str:
    """Redact exception text before it reaches dry-run output."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(str(value), force=True)
    except Exception:
        return str(value)


def _message_preflight_summary(
    message: str,
    *,
    positional: Optional[str],
    file_path: Optional[str],
) -> dict[str, Any]:
    """Summarize message input without echoing the message body."""
    if file_path:
        source = "stdin" if file_path == "-" else "file"
    elif positional is not None:
        source = "argument"
    else:
        source = "stdin"

    encoded = message.encode("utf-8", errors="replace")
    summary: dict[str, Any] = {
        "source": source,
        "chars": len(message),
        "bytes": len(encoded),
        "lines": 0 if not message else message.count("\n") + 1,
        "body_sha256_12": hashlib.sha256(encoded).hexdigest()[:12],
        "content_printed": False,
    }

    if file_path and file_path != "-":
        path = Path(file_path).expanduser()
        summary["file"] = {
            "name": path.name,
            "path_sha256_12": hashlib.sha256(
                str(path).encode("utf-8", errors="replace")
            ).hexdigest()[:12],
            "path_printed": False,
        }
        try:
            summary["file"]["bytes_on_disk"] = path.stat().st_size
        except OSError:
            pass

    return summary


def _append_check(
    checks: list[dict[str, str]],
    name: str,
    passed: bool,
    detail: str,
) -> bool:
    checks.append(
        {"name": name, "status": "pass" if passed else "fail", "detail": detail}
    )
    return passed


def _parse_target_for_preflight(
    target: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str], bool]:
    """Parse enough target metadata for a redacted dry-run."""
    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    target_ref = parts[1].strip() if len(parts) > 1 else None
    chat_id = None
    thread_id = None
    explicit = False

    if platform_name == "telegram" and target_ref:
        match = _TELEGRAM_TARGET_RE.fullmatch(target_ref)
        if match:
            chat_id = match.group(1)
            thread_id = match.group(2)
            explicit = True

    return platform_name, target_ref, chat_id, thread_id, explicit


def _build_send_preflight(
    *,
    target: str,
    message: str,
    positional: Optional[str],
    file_path: Optional[str],
) -> dict[str, Any]:
    """Build a redacted, no-network send dry-run result."""
    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    platform_name, target_ref, chat_id, thread_id, explicit = (
        _parse_target_for_preflight(target)
    )
    ok = True

    ok &= _append_check(
        checks, "target_present", bool(target), "delivery target was supplied"
    )
    ok &= _append_check(
        checks, "message_present", bool(message.strip()), "message body was supplied"
    )
    ok &= _append_check(
        checks, "no_network", True, "dry-run does not call platform APIs"
    )

    target_summary: dict[str, Any] = {
        "platform": platform_name,
        "target_mode": (
            "explicit" if explicit else ("named_or_alias" if target_ref else "home_channel")
        ),
        "target_ref_present": bool(target_ref),
        "target_ref_printed": False,
        "chat_id": _hash_identifier(chat_id),
        "thread_id": _hash_identifier(thread_id),
    }

    if platform_name != "telegram":
        ok &= _append_check(
            checks,
            "telegram_scope",
            False,
            "post-campaign dry-run preflight is currently limited to Telegram",
        )
    else:
        _append_check(checks, "telegram_scope", True, "Telegram target selected")

    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError) as exc:
            ok &= _append_check(
                checks, "platform_known", False, _redact_error_text(exc)
            )
            platform = None

        pconfig = config.platforms.get(platform) if platform is not None else None
        configured = bool(pconfig and getattr(pconfig, "enabled", False))
        ok &= _append_check(
            checks,
            "platform_configured",
            configured,
            (
                f"{platform_name} platform is configured"
                if configured
                else f"{platform_name} platform is not configured"
            ),
        )

        credential_present = bool(getattr(pconfig, "token", None)) or bool(
            os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        )
        ok &= _append_check(
            checks,
            "credential_present",
            credential_present,
            "Telegram bot credential is present" if credential_present else "Telegram bot credential is missing",
        )

        home = None
        if platform is not None and hasattr(config, "get_home_channel"):
            try:
                home = config.get_home_channel(platform)
            except Exception as exc:
                warnings.append(
                    f"home channel lookup failed: {_redact_error_text(exc)}"
                )

        if explicit:
            target_summary["chat_id"] = _hash_identifier(chat_id)
            target_summary["thread_id"] = _hash_identifier(thread_id)
            _append_check(
                checks, "target_resolved", True, "explicit Telegram target parsed"
            )
        elif target_ref:
            warnings.append(
                "named or alias targets are not resolved in dry-run to avoid "
                "reading channel directory contents"
            )
            _append_check(
                checks,
                "target_resolved",
                True,
                "named or alias target left unresolved by design",
            )
        else:
            home_chat_id = getattr(home, "chat_id", None)
            target_summary["chat_id"] = _hash_identifier(home_chat_id)
            home_present = bool(home_chat_id)
            ok &= _append_check(
                checks,
                "home_channel_present",
                home_present,
                "Telegram home channel is configured" if home_present else "Telegram home channel is missing",
            )

    except Exception as exc:
        ok &= _append_check(checks, "config_load", False, _redact_error_text(exc))

    client_present = importlib.util.find_spec("telegram") is not None
    ok &= _append_check(
        checks,
        "telegram_client_library_present",
        client_present,
        "python-telegram-bot import is available" if client_present else "python-telegram-bot import is missing",
    )

    return {
        "ok": bool(ok),
        "dry_run": True,
        "would_send": False,
        "network_performed": False,
        "target": target_summary,
        "message": _message_preflight_summary(
            message,
            positional=positional,
            file_path=file_path,
        ),
        "checks": checks,
        "warnings": warnings,
    }


def _format_preflight_result(payload: dict[str, Any], *, json_mode: bool) -> str:
    """Format a dry-run preflight result without private target/body data."""
    if json_mode:
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    status = "PASS" if payload.get("ok") else "FAIL"
    target = payload.get("target") or {}
    message = payload.get("message") or {}
    lines = [
        f"hermes send dry-run: {status}",
        f"platform: {target.get('platform', '?')}",
        f"target: {target.get('target_mode', '?')} (redacted)",
        (
            "message: "
            f"{message.get('chars', 0)} chars, "
            f"{message.get('bytes', 0)} bytes, "
            f"{message.get('lines', 0)} lines"
        ),
        "network: none; no message sent",
    ]
    for check in payload.get("checks", []):
        lines.append(f"- {check['status']}: {check['name']} - {check['detail']}")
    for warning in payload.get("warnings", []):
        lines.append(f"- warning: {warning}")
    return "\n".join(lines) + "\n"


def _emit_preflight_result(
    payload: dict[str, Any],
    *,
    json_mode: bool,
    quiet: bool,
    output_path: Optional[str] = None,
) -> int:
    """Print or privately write a dry-run preflight result."""
    text = _format_preflight_result(payload, json_mode=json_mode)
    if output_path:
        from hermes_cli.private_artifacts import write_private_text

        target = write_private_text(output_path, text)
        if not quiet:
            status = "PASS" if payload.get("ok") else "FAIL"
            print(f"hermes send dry-run: {status}; receipt written to {target}")
    elif not quiet:
        print(text, end="")

    return _SUCCESS_EXIT if payload.get("ok") else _FAILURE_EXIT


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
            return Path(file_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
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

    if getattr(args, "dry_run", False):
        payload = _build_send_preflight(
            target=target,
            message=message,
            positional=getattr(args, "message", None),
            file_path=getattr(args, "file", None),
        )
        sys.exit(
            _emit_preflight_result(
                payload,
                json_mode=getattr(args, "json", False),
                quiet=getattr(args, "quiet", False),
                output_path=getattr(args, "output", None),
            )
        )

    if getattr(args, "output", None):
        print(
            "hermes send: --output is only valid with --dry-run/--preflight",
            file=sys.stderr,
        )
        sys.exit(_USAGE_EXIT)

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
            "  hermes send --to telegram --file docs/HERMES_FINAL_REPORT.md "
            "--dry-run --json --output /tmp/hermes-telegram-preflight.json\n"
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
        "--dry-run",
        "--preflight",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "Validate Telegram send readiness and print redacted target/message "
            "metadata without calling the platform API or sending a message."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "With --dry-run/--preflight, write the redacted receipt to PATH "
            "using owner-only file permissions."
        ),
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
