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
    StateDBAdmissionError,
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


def _build_admitted_session_db_class():
    original = ORIGINAL_SESSION_DB

    class GatewayAdmittedSessionDB(original):
        _gateway_state_db_authority_wrapped = True
        _gateway_state_db_original_class = original

        def __init__(self, db_path: Path = None, read_only: bool = False):
            if read_only:
                original.__init__(self, db_path=db_path, read_only=True)
                return
            token = PENDING_FAILURE.set(None)
            try:
                AUTHORITY.initialize_writable(
                    self,
                    db_path=db_path,
                    original_init=original.__init__,
                )
            finally:
                PENDING_FAILURE.reset(token)

        def close(self) -> None:
            try:
                original.close(self)
            finally:
                AUTHORITY.release(self)

    GatewayAdmittedSessionDB.__name__ = "GatewayAdmittedSessionDB"
    GatewayAdmittedSessionDB.__qualname__ = "GatewayAdmittedSessionDB"
    GatewayAdmittedSessionDB.__module__ = __name__
    return GatewayAdmittedSessionDB


def _install_session_store_authority() -> type:
    """Inject authority at the gateway-owned writer construction seam only.

    ``hermes_state.SessionDB`` is shared by CLI doctor, sessions repair,
    console recovery, tests, and non-gateway runtimes. Replacing that class
    process-wide makes importing ``gateway`` seize authority over unrelated
    callers and changes their native exception/repair contracts. SessionStore
    is the gateway writer owner, so replace only its per-profile opener while
    preserving its cache and JSONL-fallback behavior byte-for-byte.
    """
    from gateway.session import SessionStore

    current = SessionStore._open_session_db_for_active_scope
    if getattr(current, "_gateway_state_db_authority", False):
        return getattr(current, "_gateway_state_db_class")

    native_open = current
    admitted_session_db = _build_admitted_session_db_class()

    def open_with_gateway_state_db_authority(self):
        # Preserve the established SessionStore monkeypatch seam exactly. Tests
        # and embedders replace hermes_state.SessionDB to inject construction
        # failures; the gateway authority must not turn those into integrity
        # verdicts or silently bypass their fallback/guard semantics. A normal
        # production process keeps the native class, so every real writer still
        # flows through the admitted constructor below.
        if hermes_state.SessionDB is not ORIGINAL_SESSION_DB:
            return native_open(self)

        from hermes_state import _default_db_path

        path = Path(_default_db_path())
        with self._db_handles_lock:
            if path in self._db_handles:
                return self._db_handles[path]
            db = None
            try:
                # Bind the authority to the exact path SessionStore resolved
                # for this profile scope. Passing None here lets a later
                # context/default resolver choose a different generation than
                # the cache key, defeating both profile routing and admission.
                db = admitted_session_db(db_path=path)
            except StateDBAdmissionError:
                # Integrity/generation refusal is an authoritative terminal
                # outcome, not permission to cache None and silently fall back.
                raise
            except RuntimeError as exc:
                if "live-system guard" in str(exc):
                    raise
                print(
                    "[gateway] Warning: SQLite session store unavailable, "
                    f"falling back to JSONL: {exc}"
                )
            except Exception as exc:
                print(
                    "[gateway] Warning: SQLite session store unavailable, "
                    f"falling back to JSONL: {exc}"
                )
            self._db_handles[path] = db
            return db

    open_with_gateway_state_db_authority._gateway_state_db_authority = True
    open_with_gateway_state_db_authority._gateway_state_db_class = (
        admitted_session_db
    )
    open_with_gateway_state_db_authority.__wrapped__ = current
    SessionStore._open_session_db_for_active_scope = (
        open_with_gateway_state_db_authority
    )
    return admitted_session_db


def install_gateway_state_db_authority() -> type:
    """Install passive, reload-safe gateway-only writer authority."""
    _install_connect_guard()
    return _install_session_store_authority()
