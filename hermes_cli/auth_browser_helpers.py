"""SSH / remote-session detection and graphical-browser helpers.

Extracted verbatim from ``hermes_cli.auth`` (SSH / remote session
detection section) — the loopback-OAuth helpers that decide whether a
browser can actually be opened from the host running Hermes:

- ``_is_remote_session`` — SSH / cloud-shell detection: loopback OAuth
  can't reach a local browser from a remote VM (GCP Cloud Shell,
  Codespaces, EC2 Instance Connect, Gitpod, Replit, ...).
- ``_CONSOLE_BROWSER_NAMES`` / ``_can_open_graphical_browser`` — refuse
  to auto-open text-mode browsers (w3m/lynx/links/browsh) that would
  hijack the user's TTY instead of opening a real windowed browser.
- ``_ssh_user_at_host`` / ``_print_loopback_ssh_hint`` — print the
  ``ssh -N -L`` tunnel hint for remote loopback OAuth flows.

``hermes_cli.auth`` re-exports every name (eager import at the vacated
site plus a PEP 562 module ``__getattr__`` fallback), so bare-name call
sites inside the monolith, downstream ``from hermes_cli.auth import ...``
importers, and test monkeypatches all keep resolving unchanged.

The only non-verbatim body adjustment is the function-local
``from hermes_cli.auth import _is_remote_session`` seam inside
``_print_loopback_ssh_hint``: it imports through the monolith's re-export
at call time so ``monkeypatch.setattr(auth, "_is_remote_session", ...)``
keeps steering the moved code (round-trip seam — same pattern as the
tui-launch helpers extracted from ``hermes_cli.main``).

``OAUTH_OVER_SSH_DOCS_URL`` is imported at the BOTTOM of this module to
break the import cycle: whichever module is imported first, the
cross-import is each module's last statement, so the other module is
always fully initialized by the time the name is fetched.
"""

from __future__ import annotations

import os
import sys
from typing import FrozenSet
from urllib.parse import urlparse

def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    Historically only SSH was checked, but #26923 surfaced that
    **browser-only remote consoles** (GCP Cloud Shell, GitHub
    Codespaces, AWS EC2 Instance Connect, Gitpod, Replit, etc.) hit
    the exact same problem — the user has a browser on their laptop
    but the loopback listener is bound on the remote VM that the
    laptop's browser can't reach.  These environments typically don't
    set ``SSH_CLIENT`` / ``SSH_TTY``, so the SSH-only check left
    them with no guidance and no fallback.
    """
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        return True
    # Browser-only remote IDEs / cloud shells.  Keep this list narrow
    # (well-known, documented env vars set by the host platform) so
    # we don't falsely trip on a developer's local shell.
    for var in (
        "CLOUD_SHELL",         # GCP Cloud Shell
        "CODESPACES",          # GitHub Codespaces
        "CODESPACE_NAME",      # GitHub Codespaces (alt)
        "GITPOD_WORKSPACE_ID", # Gitpod
        "REPL_ID",             # Replit
        "STACKBLITZ",          # StackBlitz
    ):
        if os.getenv(var):
            return True
    return False


# Console/text-mode browsers that ``webbrowser`` will happily launch INSIDE
# the terminal.  Opening one of these is worse than not opening anything —
# it hijacks the user's TTY with an unusable text browser (the xAI OAuth
# "Account Management" page rendered in w3m, reported May 2026) instead of
# letting them copy the URL to a real browser.  When the resolved browser is
# one of these we refuse to auto-open and fall back to the print-the-URL
# path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset(
    {
        "w3m",
        "lynx",
        "links",
        "links2",
        "elinks",
        "www-browser",
        "browsh",  # TUI browser — still hijacks the terminal
    }
)


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    ``webbrowser.open()`` resolves to whatever the platform offers, and on a
    headless / CLI-only Linux box with no GUI browser installed that is often
    a text-mode browser (w3m/lynx/links) which launches inside the terminal
    and takes over the user's session.  This guard distinguishes "a real
    windowed browser will pop up" from "a console browser will hijack the
    TTY", so callers can fall back to printing the URL instead.

    Heuristics:
      * Respect ``$BROWSER`` — if it names a known console browser, refuse.
      * On Linux, require a display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``)
        unless ``$BROWSER`` points at something graphical; no display server
        almost always means no GUI browser.
      * Ask ``webbrowser.get()`` what it resolved to and refuse when the
        underlying command is a known console browser.
      * macOS and Windows always have a usable default GUI browser.
    """
    import webbrowser as _webbrowser

    def _names_console_browser(value: str) -> bool:
        token = value.strip().split()[0] if value.strip() else ""
        base = os.path.basename(token).lower()
        return base in _CONSOLE_BROWSER_NAMES

    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False

    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        # An explicit graphical $BROWSER can work without $DISPLAY in odd
        # setups, but a console $BROWSER already returned False above, so the
        # only way to reach here with a $BROWSER set is a graphical one.
        if not has_display and not browser_env:
            return False

    try:
        controller = _webbrowser.get()
    except Exception:
        # No browser resolvable at all → definitely don't auto-open.
        return False

    candidate = (
        getattr(controller, "name", "")
        or getattr(controller, "basename", "")
        or ""
    )
    if candidate and _names_console_browser(candidate):
        return False

    return True


def _ssh_user_at_host() -> str:
    """Return best-effort 'user@hostname' for the SSH tunnel hint command.

    Falls back to placeholder tokens when the values cannot be determined so
    the hint is always syntactically valid even if not copy-pasteable.
    """
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when running a loopback-redirect OAuth flow on a
    remote host. The auth server (Spotify, MCP servers, ...) will redirect the
    user's browser to ``127.0.0.1:<port>/callback``. If the browser is on a
    different machine than the loopback listener (the usual SSH case), the
    redirect can't reach the listener without a local port forward.

    The hint is best-effort: silent if we don't think we're remote, or if we
    can't parse a host/port out of the redirect URI.

    Pass ``docs_url`` for a provider-specific guide; the generic OAuth-over-SSH
    guide is always shown after it.
    """
    from hermes_cli.auth import _is_remote_session  # noqa: PLC0415
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host = parsed.hostname or ""
    port = parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print()
    print(divider)
    print("Remote session detected — SSH tunnel required")
    print(divider)
    print(f"Hermes is waiting for the OAuth callback on {redirect_uri}")
    print("but your browser is on a different machine. Run this command")
    print("in a NEW terminal on your local machine BEFORE opening the URL:")
    print()
    print(f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}")
    print()
    print("Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}")
    print(divider)
    print()


# Bottom-of-module cross-import (cycle-break): ``hermes_cli.auth``
# re-exports the moved names mid-file; importing *this* module first
# triggers ``hermes_cli.auth``, whose re-export finds this module fully
# initialized (its own bottom import is its last statement). Importing
# ``hermes_cli.auth`` first runs its mid-file re-export after this module
# has completed, and this import — the last statement — finds auth fully
# loaded with ``OAUTH_OVER_SSH_DOCS_URL`` defined.
from hermes_cli.auth import OAUTH_OVER_SSH_DOCS_URL  # noqa: E402

