"""``/handoff desktop`` — continue the current CLI session in the Hermes Desktop app.

Inspired by Poke's "one agent, every surface" model as it resurfaced in the Devin CLI
(``/open desktop``, Sep 2026): the terminal is one view of a conversation, the app is
another, and moving between them should not cost the thread. Hermes sessions are owned by
one process at a time (the active-session lease), so — exactly like ``/handoff <platform>``
— the CLI leg ends here and Desktop resumes the stored session through the
``hermes://session/<id>?profile=<name>`` deep link the app registers on install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from urllib.parse import quote


def _cp(*lines: str) -> None:
    """``_cprint`` each line (lazy import: cli.py imports the CLI mixins)."""
    from cli import _cprint
    for line in lines:
        _cprint(line)


# Profile names the launcher cannot route: ``get_active_profile_name`` labels a HERMES_HOME
# outside ``~/.hermes`` and ``~/.hermes/profiles/<name>`` as "custom".
_UNROUTABLE_PROFILES = frozenset({"custom"})


def desktop_session_url(session_id: str, profile: str | None) -> str:
    """The ``hermes://session/<id>[?profile=<name>]`` link Desktop resolves to a saved chat.

    ``profile`` is a routing hint (which backend to make live before resuming); Desktop
    still opens the chat when the hint is missing.
    """
    url = f"hermes://session/{quote(str(session_id), safe='')}"
    if profile and profile not in _UNROUTABLE_PROFILES:
        url += f"?profile={quote(profile, safe='')}"
    return url


def launch_os_url(url: str) -> str | None:
    """Hand ``url`` to the OS default opener. ``None`` on success, else a one-line reason.

    ``webbrowser.open`` is deliberately not used: on a headless Linux box it resolves to a
    text-mode browser that cannot dispatch a custom scheme and reports success anyway.
    """
    try:
        if sys.platform == "win32":
            os.startfile(url)  # noqa: S606 — protocol dispatch via the shell association
            return None
        opener = "open" if sys.platform == "darwin" else shutil.which("xdg-open")
        if not opener:
            return "xdg-open is not installed, so no hermes:// handler can be reached from here."
        proc = subprocess.run(
            [opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, timeout=20, check=False,
        )
    except Exception as exc:  # opener missing/crashed/timed out — all mean "not opened"
        return f"could not run the URL opener: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode(errors="replace").strip()
        return f"the URL opener exited {proc.returncode}" + (f": {detail}" if detail else "") + \
            " (is Hermes Desktop installed?)"
    return None


def handoff_to_desktop(cli) -> bool:
    """Run ``/handoff desktop`` for ``cli``. Returns False when the CLI should exit (the app has
    the session now); True keeps the CLI session intact after printing why."""
    session_title = cli._handoff_prepare_session()  # refuses mid-turn; ensures the DB row
    if session_title is None:
        return True
    profile = None
    try:
        from hermes_cli.profiles import current_profile_name
        profile = current_profile_name()
    except Exception:
        pass
    url = desktop_session_url(cli.session_id, profile)
    reason = launch_os_url(url)
    if reason:
        _cp(f"  Could not open Hermes Desktop: {reason}",
            f"  Open this link once the app is installed: {url}",
            "  Your CLI session is intact.")
        return True
    # From here the app owns the session: hand back the active-session slot before exiting so
    # Desktop's first turn can claim it, and — as with a platform handoff — do not finalize the
    # row on the way out (that stamps end_reason on a session the app is about to reopen, #88234).
    from cli import _handed_off_session_ids
    _handed_off_session_ids.add(cli.session_id)
    with_release = getattr(cli, "_release_active_session", None)
    if callable(with_release):
        with_release()
    _cp("", f"  ↻ Opening '{session_title}' in Hermes Desktop.",
        f"  Resume it on this CLI later with: /resume {session_title}", "")
    cli._should_exit = True  # same exit semantics as /quit and a completed platform handoff
    return False
