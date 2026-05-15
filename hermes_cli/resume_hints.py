"""Helpers for shell-visible session resume hints.

The command printed at the end of a session must recreate the same profile
context that produced the session.  Named profiles store their sessions under a
profile-specific HERMES_HOME, so a plain ``hermes --resume <id>`` may look in the
default profile and report ``Session not found``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _shell_double_quote(value: str) -> str:
    """Return a double-quoted shell token, preserving existing hint style.

    Inside double quotes, POSIX shells still expand ``$...`` and backticks, so
    escape those too.  Newlines are rendered as spaces to keep the hint a
    single copy/pasteable command line.
    """
    escaped = (
        value.replace("\r", " ")
        .replace("\n", " ")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return '"' + escaped + '"'


def _active_profile_name_for_cli_hint() -> str | None:
    """Return the active named profile for ``hermes -p ...`` hints, if any.

    We intentionally derive this from the active HERMES_HOME path rather than
    only from ``get_active_profile_name()``: that function returns ``"custom"``
    for unrecognized homes, which is ambiguous with a real named profile called
    ``custom``.  A profile flag is only safe when HERMES_HOME is exactly one
    child under the profiles root.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.profiles import _PROFILE_ID_RE, _get_profiles_root

        hermes_home = get_hermes_home().resolve()
        profiles_root = _get_profiles_root().resolve()
        rel = hermes_home.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
            return parts[0]
    except Exception:
        return None
    return None


def _base_command_parts(*, tui: bool = False) -> list[str]:
    parts = ["hermes"]
    profile_name = _active_profile_name_for_cli_hint()
    if profile_name:
        parts.extend(["-p", profile_name])
    if tui:
        parts.append("--tui")
    return parts


def _join_parts(parts: Iterable[str]) -> str:
    return " ".join(parts)


def build_resume_command(session_id: str, *, tui: bool = False) -> str:
    """Build a profile-aware ``hermes --resume`` command."""
    return _join_parts([*_base_command_parts(tui=tui), "--resume", str(session_id)])


def build_continue_command(title: str, *, tui: bool = False) -> str:
    """Build a profile-aware resume-by-title command."""
    return _join_parts([*_base_command_parts(tui=tui), "-c", _shell_double_quote(str(title))])
