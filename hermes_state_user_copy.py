"""Plain-language copy for "session storage is unavailable / could not be written" notices.

One table keyed by ``classify_persistence_error``'s cause bucket feeds every surface (CLI banner,
gateway home-channel warning, TUI/Desktop RPC errors) so they agree on what happened, what to do,
and a machine-readable ``code`` a GUI can attach a "Run doctor" button to.
"""

from __future__ import annotations

from dataclasses import dataclass

from hermes_state_errors import STORAGE_RECOVERY_DOCS_URL, classify_persistence_error, is_disk_full_error


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
    # on the profile must stop, doctor names the ones still holding the retired log (#110054).
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
