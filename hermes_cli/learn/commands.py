"""Shared Learn command handler for CLI and slash-command surfaces."""

from __future__ import annotations

import argparse
import shlex
from typing import Any

from . import analyzer, runtime, state


def _format_status(status: dict[str, Any]) -> str:
    mode = status.get("mode", "off")
    runtime_state = status.get("state", "stopped")
    count = int(status.get("collected_event_count") or 0)
    if mode == "off":
        headline = "Learn is off."
    else:
        headline = f"Learn is {runtime_state} in {mode} mode."
    return (
        f"{headline}\n"
        f"  Events: {count}\n"
        f"  Storage: {status.get('storage_path')}\n"
        f"  Retention: {status.get('retention_days')} day(s)"
    )


def _suggestions_hint(surface: str) -> str:
    return "Run /suggestions to review." if surface != "cli" else "Run /suggestions to review, or `hermes suggestions` from a shell."


def handle_learn_command(args: str = "", *, surface: str = "cli") -> str:
    """Dispatch a Learn command and return user-facing text."""
    try:
        parts = shlex.split(args or "")
    except ValueError:
        parts = (args or "").split()

    sub = parts[0].lower() if parts else "status"
    rest = parts[1:]

    if sub in {"status", "state"}:
        return _format_status(state.get_status())

    if sub == "start":
        mode = rest[0] if rest else "learn"
        try:
            status = state.start(mode=mode)
        except ValueError as exc:
            return f"Learn start failed: {exc}"
        runtime.ensure_running()
        return f"Learn started in {status['mode']} mode.\n{_format_status(status)}"

    if sub == "pause":
        status = state.pause()
        runtime.stop_runtime()
        return f"Learn paused.\n{_format_status(status)}"

    if sub == "resume":
        status = state.resume()
        runtime.ensure_running()
        return f"Learn resumed.\n{_format_status(status)}"

    if sub == "stop":
        status = state.stop()
        runtime.stop_runtime()
        return f"Learn stopped.\n{_format_status(status)}"

    if sub in {"review", "suggest", "suggestions"}:
        created = analyzer.create_usage_suggestions()
        if not created:
            return "No new Learn suggestions right now."
        noun = "suggestion" if len(created) == 1 else "suggestions"
        titles = ", ".join(str(item.get("title") or item.get("id") or "untitled") for item in created)
        return f"Created {len(created)} Learn {noun}: {titles}.\n{_suggestions_hint(surface)}"

    if sub in {"delete-data", "delete", "clear"}:
        status = state.delete_data()
        runtime.stop_runtime()
        return f"Learn data deleted.\n{_format_status(status)}"

    if sub in {"config", "configure"}:
        allowlist: list[str] | None = None
        denylist: list[str] | None = None
        retention_days: int | None = None
        i = 0
        while i < len(rest):
            token = rest[i]
            if token == "--allowlist" and i + 1 < len(rest):
                allowlist = [item.strip() for item in rest[i + 1].split(",") if item.strip()]
                i += 2
            elif token == "--denylist" and i + 1 < len(rest):
                denylist = [item.strip() for item in rest[i + 1].split(",") if item.strip()]
                i += 2
            elif token == "--retention-days" and i + 1 < len(rest):
                try:
                    retention_days = int(rest[i + 1])
                except ValueError:
                    return "Learn config failed: --retention-days must be a number."
                i += 2
            else:
                return "Usage: learn config [--allowlist a,b] [--denylist x,y] [--retention-days N]"
        status = state.update_config(allowlist=allowlist, denylist=denylist, retention_days=retention_days)
        return f"Learn config updated.\n{_format_status(status)}"

    return (
        "Usage:\n"
        "  learn status\n"
        "  learn start [learn]\n"
        "  learn pause|resume|stop\n"
        "  learn review\n"
        "  learn delete-data\n"
        "  learn config [--allowlist a,b] [--denylist x,y] [--retention-days N]"
    )


def _print_command(args: argparse.Namespace) -> None:
    pieces = [getattr(args, "learn_command", None) or "status"]
    mode = getattr(args, "mode", None)
    if mode:
        pieces.append(mode)
    if getattr(args, "allowlist", None):
        pieces.extend(["--allowlist", ",".join(args.allowlist)])
    if getattr(args, "denylist", None):
        pieces.extend(["--denylist", ",".join(args.denylist)])
    if getattr(args, "retention_days", None) is not None:
        pieces.extend(["--retention-days", str(args.retention_days)])
    print(handle_learn_command(" ".join(shlex.quote(str(piece)) for piece in pieces)))


def register_cli(parent: argparse.ArgumentParser) -> None:
    sub = parent.add_subparsers(dest="learn_command")
    parent.set_defaults(func=_print_command)

    sub.add_parser("status", help="Show Learn status").set_defaults(func=_print_command)
    p_start = sub.add_parser("start", help="Start Learn")
    p_start.add_argument("mode", nargs="?", default="learn", choices=["learn"])
    p_start.set_defaults(func=_print_command)
    sub.add_parser("pause", help="Pause Learn").set_defaults(func=_print_command)
    sub.add_parser("resume", help="Resume Learn").set_defaults(func=_print_command)
    sub.add_parser("stop", help="Stop Learn").set_defaults(func=_print_command)
    sub.add_parser("review", help="Create pending usage suggestions").set_defaults(func=_print_command)
    sub.add_parser("delete-data", help="Delete collected Learn data").set_defaults(func=_print_command)
    p_config = sub.add_parser("config", help="Update Learn collection controls")
    p_config.add_argument("--allowlist", nargs="*", default=None)
    p_config.add_argument("--denylist", nargs="*", default=None)
    p_config.add_argument("--retention-days", type=int, default=None)
    p_config.set_defaults(func=_print_command)
