"""Hermes Agent CLI integration for the ``hermes-voice-chat`` skill.

This module is the *integration shim*. The real implementation lives in the
skill at ``~/.hermes/skills/voice/hermes-voice-chat/`` and exposes the public
``voice_chat(turn=True)`` entrypoint there. This file imports it lazily so the
CLI works whether or not the skill is installed: missing-skill degrades to a
clean help-message + non-zero exit instead of a stack trace.

Importing the skill also requires its directory on ``sys.path``; we add it
idempotently on first import.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

# Path is the canonical location. The default profile's skills dir is also
# possible; users can override with HERMES_VOICE_SKILL_DIR.
DEFAULT_SKILL_DIR = Path.home() / ".hermes" / "skills" / "voice" / "hermes-voice-chat"


def _load_skill_module() -> Optional[ModuleType]:
    """Return the ``voice_cli`` module from the skill, or None if missing."""
    skill_dir = Path(os.environ.get("HERMES_VOICE_SKILL_DIR") or DEFAULT_SKILL_DIR)
    if not (skill_dir / "voice_cli.py").is_file():
        return None
    spath = str(skill_dir)
    if spath not in sys.path:
        sys.path.insert(0, spath)
    try:
        return importlib.import_module("voice_cli")
    except Exception:
        return None


def build_voice_parser(subparsers):
    """Register the ``voice`` subcommand on ``hermes``.

    Falls back to a parser that prints an installation hint when the skill
    is missing. Always returns the registered parser object so
    ``_build_cli_parser`` can chain other registrations.
    """
    mod = _load_skill_module()
    if mod is None:
        parser = subparsers.add_parser(
            "voice",
            help=(
                "Start a realtime voice chat session with Hermes Agent "
                "(install the `hermes-voice-chat` skill first)."
            ),
        )
        parser.set_defaults(
            func=lambda args: _missing_skill_error(),
        )
        return parser
    return mod.build_voice_parser(subparsers)


def _missing_skill_error() -> int:
    print(
        "The `hermes-voice-chat` skill is not installed.\n"
        f"Expected at: {DEFAULT_SKILL_DIR}\n"
        "Install it (or set $HERMES_VOICE_SKILL_DIR) and re-run.",
        file=sys.stderr,
    )
    return 2
