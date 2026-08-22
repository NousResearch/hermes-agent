from __future__ import annotations

from pathlib import Path
from typing import Any

import hermes_state

from ._authority import AUTHORITY
from ._context import (
    EXPECTED_GENERATION,
    ORIGINAL_SESSION_DB,
    ORIGINAL_TRACKED_CONNECT,
    PENDING_FAILURE,
)
from ._model import (
    StateDBGenerationConflictError,
    canonical_state_db_path,
    same_identity,
    stat_identity,
)


def _database_matches_expected(database: Any, expected_path: Path) -> bool:
    if isinstance(database, Path):
        candidate = database
    elif isinstance(database, str):
        if database == ":memory:" or database.startswith("file:"):
            return False
        candidate = Path(database)
    else:
        return False
    return canonical_state_db_path(candidate) == expected_path


def guarded_tracked_connect(database: Any, *args: Any, **kwargs: Any):
    expected = EXPECTED_GENERATION.get()
    if expected is None or not _database_matches_expected(database, expected[0]):
        return ORIGINAL_TRACKED_CONNECT(database, *args, **kwargs)

    path, identity = expected
    try:
        before = stat_identity(path)
    except OSError as exc:
        raise StateDBGenerationConflictError(
            f"state.db disappeared before writer connect: {exc}",
            path=path,
        ) from exc
    if not same_identity(before, identity):
        raise StateDBGenerationConflictError(
            f"state.db at {path} no longer matches the verified generation",
            path=path,
        )

    conn = ORIGINAL_TRACKED_CONNECT(database, *args, **kwargs)
    try:
        if not same_identity(stat_identity(path), identity):
            raise StateDBGenerationConflictError(
                f"state.db at {path} changed generation during writer connect",
                path=path,
            )
        return conn
    except BaseException:
        try:
            conn.close()
        except Exception:
            pass
        raise


def _install_connect_guard() -> None:
    current = hermes_state._connect_tracked_db
    if getattr(current, "_gateway_state_db_generation_guard", False):
        return
    guarded_tracked_connect._gateway_state_db_generation_guard = True
    guarded_tracked_connect.__wrapped__ = current
    hermes_state._connect_tracked_db = guarded_tracked_connect


def _install_session_store_failure_bridge() -> None:
    from gateway.session import SessionStore

    current = SessionStore._open_session_db_for_active_scope
    if getattr(current, "_gateway_state_db_failure_bridge", False):
        return

    def open_with_admission_failure_bridge(self):
        token = PENDING_FAILURE.set(None)
        try:
            result = current(self)
            failure = PENDING_FAILURE.get()
            if failure is None:
                return result
            path = canonical_state_db_path()
            lock = getattr(self, "_db_handles_lock", None)
            handles = getattr(self, "_db_handles", None)
            if lock is not None and isinstance(handles, dict):
                with lock:
                    if handles.get(path) is None:
                        handles.pop(path, None)
            raise failure
        finally:
            PENDING_FAILURE.reset(token)

    open_with_admission_failure_bridge._gateway_state_db_failure_bridge = True
    open_with_admission_failure_bridge.__wrapped__ = current
    SessionStore._open_session_db_for_active_scope = (
        open_with_admission_failure_bridge
    )


def _build_admitted_session_db_class():
    original = ORIGINAL_SESSION_DB

    class GatewayAdmittedSessionDB(original):
        _gateway_state_db_authority_wrapped = True
        _gateway_state_db_original_class = original

        def __init__(self, db_path: Path = None, read_only: bool = False):
            if read_only:
                original.__init__(self, db_path=db_path, read_only=True)
                return
            AUTHORITY.initialize_writable(
                self,
                db_path=db_path,
                original_init=original.__init__,
            )

        def close(self) -> None:
            try:
                original.close(self)
            finally:
                AUTHORITY.release(self)

    GatewayAdmittedSessionDB.__name__ = "SessionDB"
    GatewayAdmittedSessionDB.__qualname__ = "SessionDB"
    GatewayAdmittedSessionDB.__module__ = "hermes_state"
    return GatewayAdmittedSessionDB


def install_gateway_state_db_authority() -> type:
    """Install passive, reload-safe constructor and connection authority."""
    _install_connect_guard()
    current = hermes_state.SessionDB
    if not getattr(current, "_gateway_state_db_authority_wrapped", False):
        hermes_state.SessionDB = _build_admitted_session_db_class()
    _install_session_store_failure_bridge()
    return hermes_state.SessionDB
