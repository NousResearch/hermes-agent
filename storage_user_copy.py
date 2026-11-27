"""Plain-language copy for "session storage is unavailable / could not be written" notices.

One table keyed by ``classify_persistence_error``'s cause bucket feeds every surface (CLI banner,
gateway home-channel warning, TUI/Desktop RPC errors) so they agree on what happened, what to do,
and a machine-readable ``code`` a GUI can attach a "Run doctor" button to.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from sqlite_support import _WAL_INCOMPAT_MARKERS
from storage_errors import classify_persistence_error, is_disk_full_error

STORAGE_RECOVERY_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/session-storage-recovery"


@dataclass(frozen=True)
class StorageFailure:
    cause: str      # classify_persistence_error bucket
    code: str       # machine-readable, stable: storage_locked | storage_readonly | storage_corrupt | disk_full | ...
    gloss: str      # what happened, one clause, lowercase start
    action: str     # what to do, one sentence naming the exact command


_DOCTOR = "Run `hermes {profile_arg}doctor --fix` to diagnose and repair."

# cause -> (code, gloss, action). "disk" is split by is_disk_full_error at lookup time.
_STORAGE_FAILURES: dict[str, tuple[str, str, str]] = {
    "locked": (
        "storage_locked",
        "the session database is locked by another Hermes process",
        "Wait a moment and try again; if it persists, stop the other Hermes process "
        "(`hermes {profile_arg}gateway stop`).",
    ),
    "disk_full": (
        "disk_full",
        "the disk is full",
        "Free some disk space, then try again.",
    ),
    "disk": (
        "storage_readonly",
        "the session database file is read-only or not writable",
        _DOCTOR,
    ),
    "corrupt": (
        "storage_corrupt",
        "the session database file is damaged",
        _DOCTOR + " Recovery: `hermes {profile_arg}sessions recover --source <state.db> --inspect-only`.",
    ),
    "session_row_missing": (
        "storage_session_missing",
        "this session's database row was deleted while the chat was still open and could not be recreated",
        "Try again; if it persists, run `hermes {profile_arg}doctor`.",
    ),
    "fts_index": (
        "storage_index_corrupt",
        "the session search index is damaged (the messages themselves are intact)",
        "Run `hermes {profile_arg}doctor --fix` (or `hermes {profile_arg}sessions repair`) to rebuild it.",
    ),
    "replaced": (
        "storage_replaced",
        "the session database file was replaced while Hermes was running",
        "Stop Hermes (`hermes {profile_arg}gateway stop`), run `hermes {profile_arg}doctor`, then start it again.",
    ),
    # Code stays `storage_replaced` (GUI clients key on it); the copy names the real remedy: every writer
    # on the profile must stop, doctor names the ones still holding the retired log.
    "deleted_wal": (
        "storage_replaced",
        "another Hermes process still holds an old copy of the session database's write-ahead log, "
        "so Hermes stopped writing to keep the file safe",
        "Nothing is lost. Quit every Hermes process on this profile (Desktop app, "
        "`hermes {profile_arg}gateway stop`, dashboard, cron), run `hermes {profile_arg}doctor` — it names "
        "any process still holding the log — then start Hermes again. Do not run `doctor --fix` or delete "
        "any state.db files while they run. Guide: " + STORAGE_RECOVERY_DOCS_URL,
    ),
    "compression": (
        "storage_busy",
        "another process is compressing this session",
        "Send your message again once compression finishes.",
    ),
    "compression_closed": (
        "storage_session_rotated",
        "this session was rotated by context compression",
        "Refresh the client (or start a new turn) and send your message again.",
    ),
    "turn_lease": (
        "storage_busy",
        "another Hermes process took over this session",
        "Wait for it to finish, then send your message again.",
    ),
    "unknown": (
        "storage_unavailable",
        "the session database could not be opened",
        _DOCTOR,
    ),
}


# Error text that names an exception's class names every refusal of the storage contract by this name.
_STORE_REFUSAL_NAME = "RuntimeStoreError"


def error_type_name(exc: BaseException) -> str:
    """The class name error text shows for *exc*: ``RuntimeStoreError`` for the storage contract's own refusal
    classes, the class of the error any other storage contract error chains (the SQLite or store error the backend
    raised), the class's own name for a gateway refusal subclass and any other error."""
    from persistence.core.errors import AuthorityRefusal, PersistenceError
    if isinstance(exc, AuthorityRefusal):
        return _STORE_REFUSAL_NAME if type(exc).__module__ == AuthorityRefusal.__module__ else type(exc).__name__
    if isinstance(exc, PersistenceError) and exc.__cause__ is not None:
        return type(exc.__cause__).__name__
    return type(exc).__name__


def describe_storage_failure(exc_or_str) -> StorageFailure:
    """Plain-language description of a persistence failure (never raises)."""
    cause = classify_persistence_error(exc_or_str)
    key = "disk_full" if cause == "disk" and is_disk_full_error(exc_or_str) else cause
    code, gloss, action = _STORAGE_FAILURES.get(key, _STORAGE_FAILURES["unknown"])
    # Pin the copy-pasteable command to the failing profile — see profile_cli_selector.
    from hermes_constants import profile_cli_selector

    return StorageFailure(
        cause=cause, code=code, gloss=gloss, action=action.replace("{profile_arg}", profile_cli_selector())
    )


def storage_failure_details(exc_or_str, limit: int = 200) -> str:
    """Raw cause for a trailing, secondary "Details:" line (never the lead sentence)."""
    text = " ".join(str(exc_or_str or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


_SESSION_DB_CONSEQUENCE = "Sessions will not be saved until this is fixed."
_NETWORK_DRIVE_HINT = " If the database lives on a network drive, move it to a local disk."
_NETWORK_DRIVE_GLOSS = "the session database could not be opened; it may be on a network or unsupported drive"
_NETWORK_DRIVE_ACTION = (
    "Move it to a local disk (`hermes {profile_arg}doctor` shows where it is), then start Hermes again."
)


# The most recent store open failure, per process; the unavailable notice below names it. Only a store open writes it.
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()


def set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear with None) the most recent open failure. An open never clears it on success: a concurrent
    open would erase the cause another thread's notice is about to format."""
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Most recent store open failure (None if none or never attempted)."""
    return _last_init_error


def format_session_db_unavailable(
    prefix: str = "Hermes can't open its session history right now",
    *,
    details: bool = False,
) -> str:
    """User-facing one-liner: ``<prefix>: <gloss>. <consequence> <action>[ network hint]``.

    The cause is the store's most recent open failure; the cause table above makes CLI, gateway
    and TUI agree. Chat surfaces (gateway, TUI) get the one-liner; ``details=True`` (the CLI
    banner) appends a ``Details: <raw cause>`` line for the raw SQLite text. Network filesystems
    (NFS/SMB/FUSE/ZFS) cannot host SQLite's write-ahead log: when the raw cause carries one of
    those markers the message names the network-drive suspicion, because ``hermes doctor --fix``
    cannot repair a mount — only moving the file can."""
    from hermes_constants import profile_cli_selector

    profile_arg = profile_cli_selector()
    cause = get_last_init_error()
    if not cause:
        return (
            f"{prefix}. {_SESSION_DB_CONSEQUENCE} Run `hermes {profile_arg}doctor` to check the "
            "storage location."
        )
    failure = describe_storage_failure(cause)
    gloss, action, hint = failure.gloss, failure.action, ""
    if any(m in cause.lower() for m in _WAL_INCOMPAT_MARKERS):
        if failure.cause == "unknown":
            gloss, action = _NETWORK_DRIVE_GLOSS, _NETWORK_DRIVE_ACTION.replace("{profile_arg}", profile_arg)
        else:
            hint = _NETWORK_DRIVE_HINT
    text = f"{prefix}: {gloss}. {_SESSION_DB_CONSEQUENCE} {action}{hint}"
    if details:
        text += f"\nDetails: {storage_failure_details(cause)}"
    return text
