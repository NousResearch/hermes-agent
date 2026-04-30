"""Emit OSC escape sequences so the terminal tab shows: "<persona>: <session title>".

Drop this file in as ``hermes_cli/terminal_title.py`` in the Hermes Agent repo and
wire it into the chat loop (see WIRING.md in this directory).

Format emitted (OSC 0 — sets both window title and icon title):

    ESC ] 0 ; <persona>: <session title> BEL

Plus OSC 7 for working directory, which Warp / iTerm2 / Kitty / WezTerm use to
display the cwd subtitle:

    ESC ] 7 ; file://<host><abs-cwd> BEL

No-ops gracefully when stderr/stdout aren't TTYs (pipes, captured output, ACP
JSON-RPC mode), when ``TERM=dumb``, or when ``HERMES_DISABLE_TAB_TITLE=1``.
"""
from __future__ import annotations

import atexit
import os
import re
import socket
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

# OSC framing. BEL terminator works everywhere; ST (\x1b\\) is the formal one.
_OSC = "\x1b]"
_BEL = "\x07"

_MAX_TITLE_LEN = 120
_DEFAULT_PERSONA = "Hermes"

# Strip everything that could break the OSC frame or look like garbage in a tab.
_UNSAFE = re.compile(r"[\x00-\x1f\x7f]")

_prior_title_set = False  # whether we've stored anything to restore on exit


# ---------------------------------------------------------------------------
# Output channel
# ---------------------------------------------------------------------------

def _tty_stream():
    """Return a writable TTY stream, or None if we shouldn't emit.

    Prefers ``/dev/tty`` so the escape lands on the real terminal even when
    stdout/stderr are redirected. Falls back to stderr if it's a tty.
    """
    if os.environ.get("HERMES_DISABLE_TAB_TITLE"):
        return None
    term = os.environ.get("TERM", "")
    if term in ("", "dumb"):
        return None
    # /dev/tty is the controlling terminal regardless of stdio redirection.
    try:
        return open("/dev/tty", "w", buffering=1, encoding="utf-8")
    except OSError:
        pass
    if sys.stderr.isatty():
        return sys.stderr
    return None


def _write(seq: str) -> None:
    stream = _tty_stream()
    if stream is None:
        return
    try:
        stream.write(seq)
        stream.flush()
    except Exception:
        pass
    finally:
        # If we opened /dev/tty, close it; don't close stderr.
        if stream is not sys.stderr:
            try:
                stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------

def _clean(text: str, max_len: int = _MAX_TITLE_LEN) -> str:
    if not text:
        return ""
    text = _UNSAFE.sub("", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "\u2026"  # …
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_tab_title(persona: Optional[str], session_title: Optional[str]) -> None:
    """Set the terminal tab/window title to ``"<persona>: <session_title>"``.

    Empty / missing values are tolerated. Called frequently; cheap and safe.
    """
    persona_clean = _clean(persona or _DEFAULT_PERSONA, max_len=40)
    title_clean = _clean(session_title or "", max_len=_MAX_TITLE_LEN - len(persona_clean) - 2)

    if title_clean:
        full = f"{persona_clean}: {title_clean}"
    else:
        full = persona_clean

    global _prior_title_set
    _prior_title_set = True
    # OSC 0 sets both window and icon title at once.
    _write(f"{_OSC}0;{full}{_BEL}")


def set_cwd(path: Optional[str | os.PathLike] = None) -> None:
    """Emit OSC 7 so the terminal knows the current working directory."""
    p = Path(path or os.getcwd()).resolve()
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    # Path components must be percent-encoded (per RFC 3986); leave the slashes.
    encoded = quote(str(p), safe="/")
    _write(f"{_OSC}7;file://{host}{encoded}{_BEL}")


def reset_tab_title() -> None:
    """Clear our title; most terminals will fall back to the shell's default."""
    if _prior_title_set:
        _write(f"{_OSC}0;{_BEL}")


# Restore on normal interpreter exit.
atexit.register(reset_tab_title)


# ---------------------------------------------------------------------------
# Persona / title resolution helpers (callers can use or ignore)
# ---------------------------------------------------------------------------

_PERSONA_NAME_RX = re.compile(r"^\s*-?\s*Name\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_PERSONA_HEADING_RX = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def resolve_persona_name(soul_path: Optional[Path] = None,
                         config_value: Optional[str] = None) -> str:
    """Best-effort persona name.

    Resolution order:
      1. Explicit ``config_value`` (caller-provided, e.g. ``agent.persona_name``).
      2. ``Name: <X>`` line in SOUL.md.
      3. First markdown ``# heading`` in SOUL.md.
      4. ``"Hermes"`` fallback.
    """
    if config_value:
        cleaned = _clean(config_value, max_len=40)
        if cleaned:
            return cleaned

    if soul_path is None:
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        soul_path = home / "SOUL.md"

    try:
        text = soul_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return _DEFAULT_PERSONA

    m = _PERSONA_NAME_RX.search(text)
    if m:
        return _clean(m.group(1), max_len=40) or _DEFAULT_PERSONA

    m = _PERSONA_HEADING_RX.search(text)
    if m:
        return _clean(m.group(1), max_len=40) or _DEFAULT_PERSONA

    return _DEFAULT_PERSONA


def resolve_session_title(session_id: str) -> Optional[str]:
    """Look up the session title from the Hermes SQLite store, or return None."""
    try:
        from hermes_state import SessionDB  # type: ignore[import-not-found]
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        return db.get_session_title(session_id)
    except Exception:
        return None


# Convenience one-liner for callers that already know both.
def update_for_session(session_id: str,
                       persona_override: Optional[str] = None,
                       title_override: Optional[str] = None) -> None:
    """Resolve persona + title (with optional overrides) and emit."""
    persona = persona_override or resolve_persona_name()
    title = title_override or resolve_session_title(session_id) or ""
    set_tab_title(persona, title)
