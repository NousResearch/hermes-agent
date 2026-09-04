"""CLI subcommand: ``hermes send`` — pipe text from shell scripts to any configured messaging platform
(Telegram, Discord, Slack, Signal, SMS, etc.).
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


def _fail(msg: str, exit_code: int | None = None) -> int:
    """Print ``msg`` to stderr; exit with ``exit_code`` when given, else return ``_FAILURE_EXIT``."""
    print(msg, file=sys.stderr)
    if exit_code is not None:
        sys.exit(exit_code)
    return _FAILURE_EXIT


def _read_message_body(positional: Optional[str], file_path: Optional[str]) -> Optional[str]:
    """Resolve the message body: positional arg, then ``--file PATH`` / ``--file -`` (stdin), then
    piped stdin when not attached to a TTY. ``None`` when nothing is available (a usage error)."""
    if positional:
        return positional
    if file_path:
        if file_path == "-":
            return sys.stdin.read()
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _fail(
                f"hermes send: {file_path} is not a text file. --file reads the "
                "message *body* (logs, reports, markdown).\n"
                "To send an image/document/audio file as a native attachment, "
                "reference it with MEDIA: in the message text instead:\n"
                f'  hermes send --to telegram "MEDIA:{file_path}"\n'
                f'  hermes send --to telegram "optional caption MEDIA:{file_path}"\n'
                "Add [[as_document]] to deliver an image as an uncompressed file:\n"
                f'  hermes send --to telegram "[[as_document]] MEDIA:{file_path}"',
                _USAGE_EXIT)
        except OSError as exc:
            _fail(f"hermes send: cannot read {file_path}: {exc}", _USAGE_EXIT)

    # Reading from a TTY would block the user in a half-broken "type your message" state.
    return (sys.stdin.read() or None) if not sys.stdin.isatty() else None


def _emit_result(result_json: str, *, json_mode: bool, quiet: bool) -> int:
    """Print the ``send_message_tool`` JSON result in the requested format; return the exit code.
    Unknown / unexpected shapes are failures so scripts notice."""
    try:
        payload = json.loads(result_json) if result_json else {}
    except json.JSONDecodeError:
        # Pass the raw string through so the user can still see what went wrong.
        payload = {"error": "invalid JSON from send_message_tool", "raw": result_json}
    if json_mode:
        print(json.dumps(payload, indent=2))
    elif not quiet:
        if payload.get("error"):
            print(f"hermes send: {payload['error']}", file=sys.stderr)
        elif payload.get("success"):
            print(payload.get("note") or "sent")
        else:
            print(json.dumps(payload, indent=2))  # unknown shape — dump it, drop nothing
    if not payload.get("error") and (payload.get("skipped") or payload.get("success")):
        return _SUCCESS_EXIT
    return _FAILURE_EXIT


def _list_targets(platform_filter: Optional[str], *, json_mode: bool) -> int:
    """Print the channel directory (all configured targets across platforms), reusing the
    ``format_directory_for_display`` rendering the send_message tool shows the model."""
    try:
        from gateway.channel_directory import format_directory_for_display, load_directory
    except Exception as exc:
        return _fail(f"hermes send: failed to load channel directory: {exc}")
    try:
        raw = load_directory()
    except Exception as exc:
        return _fail(f"hermes send: failed to read channel directory: {exc}")
    platforms = dict(raw.get("platforms") or {})

    # Merge in configured-but-undiscovered platforms (e.g. a fresh SimpleX setup used only for
    # outbound sends) so `--list` never hides a working send target.
    try:
        from gateway.config import load_gateway_config
        for plat in load_gateway_config().get_connected_platforms():
            plat_name = getattr(plat, "value", str(plat))
            if plat_name not in ("local", "api_server", "webhook"):
                platforms.setdefault(plat_name, [])
    except Exception:
        pass  # directory contents alone are still useful; don't fail --list on a config problem
    if platform_filter:
        key = platform_filter.strip().lower()
        filtered = {k: v for k, v in platforms.items() if k.lower() == key}
        if not filtered:
            return _fail(
                f"hermes send: no targets found for platform '{platform_filter}'. "
                f"Configured: {', '.join(sorted(platforms)) or '(none)'}")
        platforms = filtered
    if json_mode:
        print(json.dumps({"platforms": platforms}, indent=2, default=str))
        return _SUCCESS_EXIT
    if not platforms:
        print("No messaging platforms configured or no channels discovered yet.")
        print("Set one up with `hermes gateway setup`, or run the gateway once so")
        print("channel discovery can populate ~/.hermes/channel_directory.json.")
        return _SUCCESS_EXIT

    # Unfiltered: the shared formatter over the merged view. Filtered: a minimal view of our own.
    if platform_filter is None:
        print(format_directory_for_display(platforms))
        return _SUCCESS_EXIT
    for plat_name in sorted(platforms):
        print(f"{plat_name}:")
        if not platforms[plat_name]:
            print("  (no channels discovered yet)")
            continue
        for ch in platforms[plat_name]:
            name = ch.get("name", "?")
            chat_id = ch.get("id") or ch.get("chat_id") or ""
            print(f"  {plat_name}:{name}" + (f"  [{chat_id}]" if chat_id and chat_id != name else ""))
        print()
    return _SUCCESS_EXIT


def _load_hermes_env() -> None:
    """Populate ``os.environ`` from ``~/.hermes/.env`` AND bridge top-level ``config.yaml`` keys into
    the environment so the gateway config loader sees platform credentials and home channels."""
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
            # utf-8-sig strips a leading BOM (PowerShell 5.1 / Notepad); plain "utf-8" would keep
            # U+FEFF on the first key name and silently drop it from os.environ.
            load_dotenv(str(env_path), override=True, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:  # utf-8-sig can't strip a BOM once we fall back to latin-1.
                import codecs
                import io
                raw = env_path.read_bytes().removeprefix(codecs.BOM_UTF8)
                load_dotenv(stream=io.StringIO(raw.decode("latin-1")), override=True)
            except Exception:
                pass
        except Exception:
            pass

    # Bridge top-level config.yaml scalars into the environment (never overriding existing values).
    import os
    config_path = home / "config.yaml"
    if not config_path.exists():
        return
    try:
        # Raw read is deliberate — only keys the user actually wrote get bridged.
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw(config_path)
    except Exception:
        return
    try:
        from hermes_cli.config import _expand_env_vars
        raw = _expand_env_vars(raw)
    except Exception:
        pass

    # Managed scope: administrator-pinned values win here too (fail-open via the helper).
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    if not isinstance(raw, dict):
        return
    for key, val in raw.items():
        if isinstance(val, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(val)


def cmd_send(args: argparse.Namespace) -> None:
    """Entry point wired into the top-level argparse dispatcher."""
    _load_hermes_env()  # the downstream gateway config loader reads credentials from os.environ
    if getattr(args, "list_targets", False):  # --list short-circuits everything else
        # `hermes send --list telegram` lands "telegram" in the `message` positional.
        exit_code = _list_targets(getattr(args, "message", None), json_mode=getattr(args, "json", False))
        sys.exit(exit_code)
    target = (getattr(args, "to", None) or "").strip()
    if not target:
        _fail(
            "hermes send: --to PLATFORM[:channel[:thread]] is required\n"
            "Examples:\n"
            "  hermes send --to telegram \"hello\"\n"
            "  hermes send --to discord:#ops --file report.md\n"
            "  hermes send --list      # list available targets",
            _USAGE_EXIT)
    message = _read_message_body(getattr(args, "message", None), getattr(args, "file", None))
    if message is None or not message.strip():
        _fail(
            "hermes send: no message provided. Pass text as a positional "
            "argument, use --file PATH, or pipe data via stdin.",
            _USAGE_EXIT)

    # Optional subject line: a consistent header for alerting scripts.
    subject = getattr(args, "subject", None)
    if subject:
        message = f"{subject}\n\n{message.lstrip()}"

    # Lazy import keeps `hermes send --help` fast (no tool registry / gateway config stack).
    from tools.send_message_tool import send_message_tool

    # Routes to the platform adapter (bot-token path for built-ins, live-adapter path for plugin
    # platforms); takes the standard tool-call dict and returns a JSON string.
    result = send_message_tool({"action": "send", "target": target, "message": message})
    sys.exit(_emit_result(result, json_mode=getattr(args, "json", False), quiet=getattr(args, "quiet", False)))


# (flags, add_argument kwargs) in --help order.
_SEND_ARGUMENTS = (
    (("-t", "--to"), dict(metavar="TARGET", default=None, help=(
        "Delivery target. Format: 'platform' (home channel), "
        "'platform:chat_id', 'platform:chat_id:thread_id', or "
        "'platform:#channel-name'. Examples: telegram, "
        "telegram:-1001234567890:17585, discord:#ops, slack:C0123ABCD, signal:+15551234567."))),
    (("message",), dict(nargs="?", default=None, help="Message text. If omitted, read from --file or stdin.")),
    (("-f", "--file"), dict(metavar="PATH", default=None, help=(
        "Read message body from PATH (text only). Use '-' to force stdin. "
        "To send an image/document as an attachment, use MEDIA:<path> in the message text instead."))),
    (("-s", "--subject"), dict(metavar="LINE", default=None, help="Prepend a subject/header line before the message body.")),
    (("-l", "--list"), dict(dest="list_targets", action="store_true", default=False,
                            help="List available targets. Optional positional filter: `hermes send --list telegram`.")),
    (("-q", "--quiet"), dict(action="store_true", default=False, help="Suppress stdout on success (exit code only).")),
    (("--json",), dict(action="store_true", default=False, help="Emit raw JSON result instead of human-readable output.")),
)


def register_send_subparser(subparsers) -> argparse.ArgumentParser:
    """Create the ``send`` subparser and return it."""
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
            "  hermes send --to telegram \"MEDIA:/tmp/chart.png\"   # send a media attachment\n"
            "  hermes send --list                  # all platforms\n"
            "  hermes send --list telegram         # filter by platform\n"
            "\n"
            "Exit codes: 0 ok, 1 delivery/backend error, 2 usage error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for flags, kwargs in _SEND_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
    parser.set_defaults(func=cmd_send)
    return parser


__all__ = ["cmd_send", "register_send_subparser"]
