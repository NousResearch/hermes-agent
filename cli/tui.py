"""prompt_toolkit TUI helpers for the Hermes classic CLI."""

from __future__ import annotations

import os
import sys


def _preserve_ctrl_enter_newline() -> bool:
    """Detect environments where Ctrl+Enter must produce a newline, not submit.

    Native Windows, WSL, SSH sessions, and Windows Terminal all send Ctrl+Enter
    as bare LF (c-j). On those terminals c-j must NOT be bound to submit;
    binding it to submit makes Ctrl+Enter (intended as 'newline like Alt+Enter')
    submit instead. Local POSIX TTYs that deliver Enter as LF (docker exec,
    some thin PTYs without SSH) still need c-j bound to submit, so we keep
    that binding for those.

    See issue #22379.
    """
    if sys.platform == "win32":
        return True
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return True
    if os.environ.get("WT_SESSION"):
        return True
    if "microsoft" in os.environ.get("WSL_DISTRO_NAME", "").lower():
        return True
    # WSL detection — env vars can be scrubbed under sudo, also peek /proc.
    for p in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            continue
    return False


def _bind_prompt_submit_keys(kb, handler) -> None:
    """Bind terminal Enter forms to the submit handler.

    Enter is always submit. On POSIX we also bind c-j (LF) to submit because
    some thin PTYs (docker exec, certain SSH flavors) deliver Enter as LF
    instead of CR — without this, Enter appears dead on those terminals.

    Exception: on Windows, WSL, SSH sessions, and Windows Terminal,
    c-j is the wire encoding of Ctrl+Enter (a distinct keystroke from
    plain Enter / c-m). We leave c-j unbound there so the c-j newline
    handler registered separately can fire — giving the user an
    Enter-involving newline keystroke without terminal settings changes.
    See _preserve_ctrl_enter_newline() and issue #22379.
    """
    kb.add("enter")(handler)
    if sys.platform != "win32" and not _preserve_ctrl_enter_newline():
        kb.add("c-j")(handler)


def _disable_prompt_toolkit_cpr_warning(app) -> None:
    """Let prompt_toolkit fall back from CPR without printing into the prompt."""
    try:
        app.renderer.cpr_not_supported_callback = None
    except Exception:
        pass


