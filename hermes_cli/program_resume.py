"""OSC 7577 Program Resume reporting for the interactive CLI surfaces.

``terminal_notify`` encodes the sequences. This module decides which command brings Hermes back and
keeps one pane's record in step with it: report when the command changes, clear when the user ends
Hermes, keep the record when the terminal goes away.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import lru_cache

from hermes_cli.terminal_notify import (
    _is_valid_osc7577_argv0,
    osc7577_token,
    program_resume_clear_sequence,
    program_resume_sequence,
)

logger = logging.getLogger(__name__)

_ENCODE_ERRORS = (TypeError, ValueError, UnicodeError)


def _entrypoint() -> list[str]:
    """The command that runs this install: the launcher this process came from, else this install's
    own command for ``python -m`` and source runs (``resolve_hermes_bin`` would otherwise fall back to
    whichever ``hermes`` is on PATH), which needs no PYTHONPATH from the old shell."""
    if not sys.argv[0].lower().endswith((".py", ".pyc")):
        from hermes_cli.relaunch import resolve_hermes_bin

        bin_path = resolve_hermes_bin()
        if bin_path and _is_valid_osc7577_argv0(bin_path):
            return [bin_path]
    from hermes_cli.chat_launch import hermes_command

    return hermes_command()


def _profile_args() -> list[str] | None:
    """``-p NAME`` for this process's home, default included: the resumed command starts in a fresh
    environment, where a sticky profile or HERMES_HOME may differ. None for a home no ``-p`` value
    names there: a HERMES_HOME outside the profiles root, or a custom root (Docker's /opt/data, a
    HERMES_DATA_DIR_SUFFIX build), whose ``-p default`` means ``~/.hermes`` in a fresh environment."""
    import os

    from hermes_cli.profiles import get_active_profile_name, profile_matches_home
    from hermes_constants import get_default_hermes_root, hermes_root_for_environment

    root_vars = {"HERMES_HOME", "HERMES_DATA_DIR_SUFFIX"}  # the resumed command carries neither
    fresh_root = hermes_root_for_environment({k: v for k, v in os.environ.items() if k.upper() not in root_vars})
    if fresh_root is None or fresh_root.resolve() != get_default_hermes_root().resolve():
        return None
    name = get_active_profile_name()
    return ["-p", name] if profile_matches_home(name) else None


_BYPASS_FLAGS = ("--safe-mode", "--ignore-user-config")


def bypass_flags(args=None) -> tuple[str, ...]:
    """The config-bypass flag this launch runs under: from *args* or the env opt-ins. The TUI and
    classic launches set the env at startup; a gateway chat bypass launch does not (it passes *args*).
    Safe mode implies the other."""
    from hermes_cli.gateway_chat import env_launch_flags

    launch = {**env_launch_flags(), **{name: True for name in ("safe_mode", "ignore_user_config")
                                        if getattr(args, name, False)}}
    if launch.get("safe_mode"):
        return ("--safe-mode",)
    return ("--ignore-user-config",) if launch.get("ignore_user_config") else ()


@lru_cache(maxsize=8)
def _cached_prefix(surface: str, flags: tuple[str, ...], home: str) -> tuple[str, ...] | None:
    try:
        profile = _profile_args()
        env_bypass = () if set(_BYPASS_FLAGS) & set(flags) else bypass_flags()
        return None if profile is None else (*_entrypoint(), *profile, f"--{surface}", *flags, *env_bypass)
    except Exception:
        logger.debug("program resume: launch command unresolved", exc_info=True)
        return None


def resume_argv_prefix(surface: str = "cli", *flags: str) -> tuple[str, ...] | None:
    """``hermes -p NAME --cli|--tui [flags] [bypass flag]``, resolved once per home; None when no command
    can bring this session back (report nothing)."""
    import os

    from hermes_constants import get_hermes_home

    if os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip():
        return None  # attached to a named gateway: a fresh launch would open this machine's store instead
    return _cached_prefix(surface, flags, str(get_hermes_home()))


def resume_argv(session_id: str | None, surface: str = "cli", *flags: str) -> tuple[str, ...] | None:
    """The resume command for *session_id* (``--resume ID`` appended), the bare launch without one, or None.
    A bypass launch has no bare form: started again without its explicit model it fails, while
    ``--resume`` reuses the session's frozen route."""
    prefix = resume_argv_prefix(surface, *flags)
    if prefix is None:
        return None
    if session_id:
        return (*prefix, "--resume", session_id)
    return None if set(_BYPASS_FLAGS) & set(prefix) else prefix


def session_saved(store, session_id: str) -> bool:
    """True when *store* holds a row for *session_id*: only then can ``--resume`` find it after a restart.
    No store, no row and an unreadable store are all False."""
    if store is None:
        return False
    try:
        return store.get_session(session_id) is not None
    except Exception:
        logger.debug("program resume: session row lookup failed", exc_info=True)
        return False


class ProgramResumeReporter:
    """One pane's OSC 7577 record, written through *write*.

    ``sync`` reports the command when it changed; no command (None) or one that cannot be encoded
    sends a clear for a reported record so it is not resumed in its place (best effort: a failed write
    is dropped). ``clear`` is for the user ending Hermes; ``keep`` closes
    the reporter without clearing (signal, terminal loss, relaunch handoff). A lock orders the writes: a SIGCONT
    re-report runs on its own thread and must not land after the user's clear.
    """

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self._last: tuple[str, ...] | None = None
        self._lock = threading.Lock()
        self.closed = False

    @property
    def active(self) -> bool:
        return not self.closed and osc7577_token() is not None

    @property
    def reported(self) -> bool:
        return self._last is not None

    def sync(self, argv: Sequence[str] | Callable[[], Sequence[str] | None] | None, *, force: bool = False) -> None:
        """*argv* may be a callable, resolved under the lock: a SIGCONT thread's choice of session must not
        land after a session switch reported the new one."""
        with self._lock:
            self._sync(argv() if callable(argv) else argv, force)

    def _sync(self, argv: Sequence[str] | None, force: bool) -> None:
        token = osc7577_token()
        if self.closed or token is None:
            return
        if argv is None:
            self._send_clear(token)
            return
        target = tuple(argv)
        if target == self._last and not force:
            return
        try:
            seq = program_resume_sequence(target, token=token)
        except _ENCODE_ERRORS:
            self._send_clear(token)
            return
        self._write(seq)
        self._last = target

    def clear(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if (token := osc7577_token()) is not None:
                self._send_clear(token)

    def keep(self) -> None:
        with self._lock:
            self.closed = True

    def _send_clear(self, token: str) -> None:
        if self._last is None:
            return
        self._last = None
        with suppress(*_ENCODE_ERRORS):
            self._write(program_resume_clear_sequence(token=token))
