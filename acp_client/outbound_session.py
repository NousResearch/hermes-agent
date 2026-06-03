"""Outbound ACP session manager (mirror of ``acp_adapter/session.py``).

Where the server-side ``SessionManager`` maps an editor's ACP session onto a
Hermes ``AIAgent``, this manager tracks sessions where Hermes is the **client**
driving an external agent.  There is no local ``AIAgent`` — the "agent" is the
external CLI subprocess — so a session state holds the *external* session id,
the bound cwd, the chosen backend transport, a cancel event, and a mirror of
the conversation history.

Sessions persist to the shared SessionDB under ``source="acp_client"`` so they
survive a worker restart and can be reconnected via :meth:`get` / :meth:`load`
(design §2.6 reconnect; §2.2 module layout).  The external agent's session id
is used as the SessionDB row id so the two sides agree (design R5).
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SOURCE = "acp_client"


@dataclass
class OutboundSessionState:
    """Per-session state for an external ACP agent driven by Hermes."""

    session_id: str
    cwd: str = "."
    backend: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = field(default_factory=threading.Event)
    is_running: bool = False
    last_stop_reason: Optional[str] = None
    runtime_lock: Any = field(default_factory=Lock)


class OutboundSessionManager:
    """Thread-safe manager for outbound (Hermes-as-client) ACP sessions.

    Args:
        db: Optional :class:`SessionDB`.  When omitted the default
            (``~/.hermes/state.db``) is lazily created — tests inject a temp DB.
    """

    def __init__(self, db: Any = None):
        self._sessions: Dict[str, OutboundSessionState] = {}
        self._lock = Lock()
        self._db_instance = db  # None → lazy-init on first use

    # ---- public API --------------------------------------------------------

    def register(
        self, session_id: str, *, cwd: str = ".", backend: str = ""
    ) -> OutboundSessionState:
        """Register a session id minted by the external agent (or a local id).

        Mirrors ``SessionManager.create_session`` but does **not** create an
        AIAgent — the external CLI is the agent.  Persists immediately so a
        crashed worker can reconnect.
        """
        state = OutboundSessionState(session_id=session_id, cwd=cwd, backend=backend)
        with self._lock:
            self._sessions[session_id] = state
        self._persist(state)
        logger.info(
            "Registered outbound ACP session %s (backend=%s, cwd=%s)",
            session_id,
            backend,
            cwd,
        )
        return state

    def get(self, session_id: str) -> Optional[OutboundSessionState]:
        """Return the session, restoring from the DB if not in memory."""
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return state
        return self._restore(session_id)

    def mark_running(self, session_id: str, running: bool = True) -> None:
        state = self.get(session_id)
        if state is not None:
            state.is_running = running

    def record_history(self, session_id: str, role: str, content: str) -> None:
        """Append a mirror-history row and re-persist."""
        state = self.get(session_id)
        if state is None:
            return
        state.history.append({"role": role, "content": content})
        self._persist(state)

    def set_stop_reason(self, session_id: str, stop_reason: Optional[str]) -> None:
        state = self.get(session_id)
        if state is not None:
            state.last_stop_reason = stop_reason
            state.is_running = False
            self._persist(state)

    def cancel(self, session_id: str) -> bool:
        """Signal cancellation for a session.

        Sets the local ``cancel_event`` and clears ``is_running``.  The actual
        ``session/cancel`` RPC is sent by :class:`OutboundConnection`; this only
        manages local state.  Returns ``True`` if the session was known.
        """
        state = self.get(session_id)
        if state is None:
            return False
        state.cancel_event.set()
        state.is_running = False
        logger.info("Cancelled outbound ACP session %s", session_id)
        return True

    def fork(self, session_id: str, *, cwd: Optional[str] = None) -> Optional[OutboundSessionState]:
        """Deep-copy a session's history into a new locally-minted session."""
        original = self.get(session_id)
        if original is None:
            return None
        new_id = str(uuid.uuid4())
        state = OutboundSessionState(
            session_id=new_id,
            cwd=cwd or original.cwd,
            backend=original.backend,
            history=copy.deepcopy(original.history),
        )
        with self._lock:
            self._sessions[new_id] = state
        self._persist(state)
        logger.info("Forked outbound ACP session %s -> %s", session_id, new_id)
        return state

    def remove(self, session_id: str) -> bool:
        """Drop a session from memory and the DB.  Returns True if it existed."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        db_existed = self._delete_persisted(session_id)
        return existed or db_existed

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return lightweight info dicts (memory + DB) for ``acp_client`` rows."""
        rows: Dict[str, Dict[str, Any]] = {}
        db = self._get_db()
        if db is not None:
            try:
                for row in db.list_sessions_rich(source=SOURCE, limit=1000):
                    rows[str(row["id"])] = {
                        "session_id": str(row["id"]),
                        "cwd": self._cwd_from_row(row),
                        "backend": self._backend_from_row(row),
                        "in_memory": False,
                    }
            except Exception:
                logger.debug("Failed to list acp_client sessions from DB", exc_info=True)
        with self._lock:
            for sid, state in self._sessions.items():
                rows[sid] = {
                    "session_id": sid,
                    "cwd": state.cwd,
                    "backend": state.backend,
                    "is_running": state.is_running,
                    "in_memory": True,
                }
        return list(rows.values())

    # ---- persistence -------------------------------------------------------

    def _get_db(self):
        if self._db_instance is not None:
            return self._db_instance
        try:
            from hermes_state import SessionDB

            self._db_instance = SessionDB()
            return self._db_instance
        except Exception:
            logger.debug("SessionDB unavailable for acp_client persistence", exc_info=True)
            return None

    def _meta_json(self, state: OutboundSessionState) -> str:
        return json.dumps({"cwd": state.cwd, "backend": state.backend})

    def _persist(self, state: OutboundSessionState) -> None:
        db = self._get_db()
        if db is None:
            return
        try:
            existing = db.get_session(state.session_id)
            if existing is None:
                db.create_session(
                    session_id=state.session_id,
                    source=SOURCE,
                    model=state.backend or None,
                    model_config={"cwd": state.cwd, "backend": state.backend},
                )
            else:
                try:
                    with db._lock:
                        db._conn.execute(
                            "UPDATE sessions SET model_config = ? WHERE id = ?",
                            (self._meta_json(state), state.session_id),
                        )
                        db._conn.commit()
                except Exception:
                    logger.debug("Failed to update acp_client session meta", exc_info=True)
            db.replace_messages(state.session_id, state.history)
        except Exception:
            logger.warning(
                "Failed to persist acp_client session %s", state.session_id, exc_info=True
            )

    def _restore(self, session_id: str) -> Optional[OutboundSessionState]:
        db = self._get_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for acp_client session %s", session_id, exc_info=True)
            return None
        if row is None or row.get("source") != SOURCE:
            return None

        cwd = self._cwd_from_row(row)
        backend = self._backend_from_row(row)
        try:
            history = db.get_messages_as_conversation(session_id)
        except Exception:
            logger.warning(
                "Failed to load messages for acp_client session %s", session_id, exc_info=True
            )
            history = []

        state = OutboundSessionState(
            session_id=session_id,
            cwd=cwd,
            backend=backend,
            history=history,
        )
        with self._lock:
            self._sessions[session_id] = state
        logger.info(
            "Restored acp_client session %s from DB (%d messages)", session_id, len(history)
        )
        return state

    def _delete_persisted(self, session_id: str) -> bool:
        db = self._get_db()
        if db is None:
            return False
        try:
            return db.delete_session(session_id)
        except Exception:
            logger.debug("Failed to delete acp_client session %s", session_id, exc_info=True)
            return False

    @staticmethod
    def _cwd_from_row(row: Dict[str, Any]) -> str:
        mc = row.get("model_config")
        if mc:
            try:
                meta = json.loads(mc)
                if isinstance(meta, dict):
                    return str(meta.get("cwd", "."))
            except (json.JSONDecodeError, TypeError):
                pass
        return "."

    @staticmethod
    def _backend_from_row(row: Dict[str, Any]) -> str:
        mc = row.get("model_config")
        if mc:
            try:
                meta = json.loads(mc)
                if isinstance(meta, dict) and meta.get("backend"):
                    return str(meta["backend"])
            except (json.JSONDecodeError, TypeError):
                pass
        return str(row.get("model") or "")
