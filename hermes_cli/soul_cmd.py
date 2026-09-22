"""Handlers for the non-interactive ``hermes soul`` command."""

from __future__ import annotations

from pathlib import Path
import sys


def _soul_set(args) -> None:
    from hermes_constants import display_hermes_home, get_hermes_home
    from utils import atomic_write_text

    source_path = getattr(args, "file", None)
    try:
        content = (
            Path(source_path).expanduser().read_text(encoding="utf-8")
            if source_path is not None
            else args.text
        )
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Error: could not read SOUL.md source: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    target = get_hermes_home() / "SOUL.md"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, content, preserve_mode=True, create_mode=0o644)
    except OSError as exc:
        print(f"Error: could not write {target}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"✓ Updated SOUL.md in {display_hermes_home()}")


SOUL_ACTIONS = {"set": _soul_set}


def cmd_soul(args):
    """Manage the current profile's SOUL.md file."""
    handler = SOUL_ACTIONS.get(getattr(args, "soul_action", None))
    if handler is not None:
        return handler(args)
