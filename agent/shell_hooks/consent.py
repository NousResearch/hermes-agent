"""First-use consent: the per-profile allowlist of ``(event, command)`` pairs, the TTY prompt, the
opt-in channels (``--accept-hooks`` / ``HERMES_ACCEPT_HOOKS`` / ``hooks_auto_accept``) and the
script facts ``hermes hooks`` reports."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import ExitStack, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_cli._subprocess_compat import split_command_line

try:
    import fcntl  # POSIX only; Windows falls back to best-effort without flock.
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home
from utils import atomic_json_write

from agent.shell_hooks.spec import ShellHookSpec
from agent.shell_hooks.wire import _utc_now_iso

logger = logging.getLogger(__name__)

ALLOWLIST_FILENAME = "shell-hooks-allowlist.json"

_TRUTHY = {"1", "true", "yes", "on"}

# Non-POSIX fallback for allowlist read-modify-write; separate from _registered_lock so an allowlist
# write never waits on (or nests inside) a registry update.
_allowlist_write_lock = threading.Lock()

_process_accept = False


def _entry_matches(e: Any, event: Optional[str], command: str) -> bool:
    return isinstance(e, dict) and (event is None or e.get("event") == event) and e.get("command") == command


def _approved_pairs(approvals: list[Any]) -> set[tuple[str, str]]:
    return {(e["event"], e["command"]) for e in approvals
            if isinstance(e, dict) and isinstance(e.get("event"), str) and isinstance(e.get("command"), str)}


def allowlisted_pairs() -> set[tuple[str, str]]:
    return _approved_pairs(load_allowlist()["approvals"])


def _consented(spec: ShellHookSpec, accept: bool, allowed: set[tuple[str, str]], *, prompt: bool) -> bool:
    """Allowlisted, or approved now (opt-in, or a TTY prompt when ``prompt``; recorded); ``allowed``
    caches one read."""
    pair = (spec.event, spec.command)
    if pair in allowed:
        return True
    if not (accept or prompt):
        return False
    if not _prompt_and_record(spec.event, spec.command, accept_hooks=accept, wakes=spec.async_rewake):
        return False
    allowed.add(pair)
    return True


def allowlist_path() -> Path:
    """Path to the per-user shell-hook allowlist file."""
    return get_hermes_home() / ALLOWLIST_FILENAME


def _read_allowlist() -> tuple[dict[str, Any], bool]:
    """``(allowlist, readable)``: the parsed allowlist (an empty skeleton when absent or broken) and
    whether it could be trusted; a missing file is readable (nothing approved yet), an unreadable or
    malformed one is not."""
    try:
        raw = json.loads(allowlist_path().read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"approvals": []}, True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {"approvals": []}, False
    if not isinstance(raw, dict):
        return {"approvals": []}, False
    readable = isinstance(raw.get("approvals"), list)
    if not readable:
        raw["approvals"] = []
    return raw, readable


def load_allowlist() -> dict[str, Any]:
    """Return the parsed allowlist, or an empty skeleton if absent."""
    return _read_allowlist()[0]


def save_allowlist(data: dict[str, Any]) -> None:
    """Atomic write; on OSError log and carry on (the approval is not saved)."""
    p = allowlist_path()
    try:
        atomic_json_write(p, data, sort_keys=True, mode=0o600)
    except OSError as exc:
        logger.warning("Failed to persist shell hook allowlist to %s: %s. The approval holds for this process "
                       "only (a gateway keeps a launch's --accept-hooks until it restarts); a gateway session "
                       "opened from a CLI prompt skips the hook, and the next run re-prompts.", p, exc)


@contextmanager
def _locked_update_approvals() -> Iterator[dict[str, Any]]:
    """Serialise allowlist read-modify-write across processes via a sibling flock file."""
    p = allowlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if fcntl is None:  # pragma: no cover — non-POSIX fallback
            stack.enter_context(_allowlist_write_lock)
        else:
            lock_fh = stack.enter_context(open(p.with_suffix(p.suffix + ".lock"), "a+", encoding="utf-8"))  # windows-footgun: ok (write/append mode, not a read)
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            stack.callback(_flock_unlock, lock_fh)
        data = load_allowlist()
        yield data
        save_allowlist(data)


def _flock_unlock(lock_fh: Any) -> None:
    with suppress(OSError):
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _prompt_and_record(event: str, command: str, *, accept_hooks: bool, wakes: bool = False) -> bool:
    """Approve an unseen ``(event, command)`` pair; True iff granted and recorded."""
    if accept_hooks:
        record_approval(event, command)
        logger.info("shell hook auto-approved via --accept-hooks / env / config: %s -> %s", event, command)
        return True
    if not sys.stdin.isatty():
        return False
    print(
        f"\n⚠ Hermes is about to register a shell hook that will run a\n  command on your behalf.\n\n"
        f"    Event:   {event}\n    Command: {command}\n"
        + ("    Mode:    async_rewake (runs in the background and can start\n"
           "             a new turn in this session when it exits 2)\n" if wakes else "")
        + "\n  Commands run with your full user credentials.  Only approve\n  commands you trust."
    )
    try:
        answer = input("Allow this hook to run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # keep the terminal tidy after ^C
        return False
    if answer in {"y", "yes"}:
        record_approval(event, command)
    return answer in {"y", "yes"}


def record_approval(event: str, command: str) -> None:
    entry = {"event": event, "command": command, "approved_at": _utc_now_iso(), "script_mtime_at_approval": script_mtime_iso(command)}
    with _locked_update_approvals() as data:
        data["approvals"] = [e for e in data.get("approvals", []) if not _entry_matches(e, event, command)] + [entry]


def revoke(command: str) -> int:
    """Remove every allowlist entry for ``command``; returns the count. A gateway unregisters it at its
    next allowlist check (``sync_from_config``, about every 30 s) unless an opt-in approves every hook; other live
    processes keep it until restart."""
    with _locked_update_approvals() as data:
        before = len(data.get("approvals", []))
        data["approvals"] = [e for e in data.get("approvals", []) if not _entry_matches(e, None, command)]
        return before - len(data["approvals"])

_SCRIPT_EXTENSIONS: tuple[str, ...] = (".sh", ".bash", ".zsh", ".fish", ".py", ".pyw", ".rb", ".pl", ".lua", ".js", ".mjs", ".cjs", ".ts")


def _command_script_path(command: str) -> str:
    """First token with a script extension, else the first path-like token, else the first token."""
    try:
        parts = split_command_line(command) or [command]
    except ValueError:
        return command
    return (next((p for p in parts if p.lower().endswith(_SCRIPT_EXTENSIONS)), None)
            or next((p for p in parts if "/" in p or p.startswith("~")), None) or parts[0])


def _resolve_effective_accept(cfg: dict[str, Any], accept_hooks_arg: bool) -> bool:
    """Any truthy opt-in channel wins: explicit arg, this process's --accept-hooks, HERMES_ACCEPT_HOOKS,
    hooks_auto_accept."""
    from utils import env_var_enabled
    if accept_hooks_arg or _process_accept or env_var_enabled("HERMES_ACCEPT_HOOKS"):
        return True
    cfg_val = cfg.get("hooks_auto_accept", False)
    return cfg_val if isinstance(cfg_val, bool) else isinstance(cfg_val, str) and cfg_val.strip().lower() in _TRUTHY


def accept_for_process() -> None:
    """This process's own ``--accept-hooks`` (``hermes gateway run --accept-hooks``): an opt-in in
    force for its lifetime, as ``HERMES_ACCEPT_HOOKS`` would be, without exporting that variable to
    every tool subprocess."""
    global _process_accept
    _process_accept = True


def allowlist_entry_for(event: str, command: str) -> Optional[dict[str, Any]]:
    """Return the allowlist record for this pair, if any."""
    return next((e for e in load_allowlist().get("approvals", []) if _entry_matches(e, event, command)), None)


def script_mtime_iso(command: str) -> Optional[str]:
    """ISO-8601 mtime of the resolved script path, or ``None`` if missing."""
    path = _command_script_path(command)
    try:
        mtime = os.path.getmtime(os.path.expanduser(path)) if path else None
    except OSError:
        return None
    return None if mtime is None else datetime.fromtimestamp(mtime, tz=UTC).isoformat().replace("+00:00", "Z")


def script_is_executable(command: str) -> bool:
    """Runnable as configured: a bare script needs X_OK, an interpreter-prefixed one only R_OK (as ``spawn_hook`` does)."""
    path = _command_script_path(command)
    expanded = os.path.expanduser(path)
    try:
        argv = split_command_line(command) if path and os.path.isfile(expanded) else None
    except ValueError:
        return False
    return argv is not None and os.access(expanded, os.X_OK if argv and argv[0] == path else os.R_OK)
