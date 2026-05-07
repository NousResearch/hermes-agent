"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``.  When the editor
reconnects after idle/restart, the ``load_session`` / ``resume_session`` calls
find the persisted session in the database and restore the full conversation
history.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home

import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _win_path_to_wsl(path: str) -> str | None:
    """Convert a Windows drive path to its WSL /mnt/<drive>/... equivalent."""
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return None
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def _translate_acp_cwd(cwd: str) -> str:
    """Translate Windows ACP cwd values when Hermes itself is running in WSL.

    Windows ACP clients can launch ``hermes acp`` inside WSL while still sending
    editor workspaces as Windows drive paths such as ``E:\\Projects``. Store
    and execute against the WSL mount path so agents, tools, and persisted ACP
    sessions all agree on the usable workspace. Native Linux/macOS keeps the
    original cwd unchanged.
    """
    from hermes_constants import is_wsl

    if not is_wsl():
        return cwd
    translated = _win_path_to_wsl(str(cwd))
    return translated if translated is not None else cwd


def _normalize_cwd_for_compare(cwd: str | None) -> str:
    raw = str(cwd or ".").strip()
    if not raw:
        raw = "."
    expanded = os.path.expanduser(raw)

    # Normalize Windows drive paths into the equivalent WSL mount form so
    # ACP history filters match the same workspace across Windows and WSL.
    translated = _win_path_to_wsl(expanded)
    if translated is not None:
        expanded = translated
    elif re.match(r"^/mnt/[A-Za-z]/", expanded):
        expanded = f"/mnt/{expanded[5].lower()}/{expanded[7:]}"

    return os.path.normpath(expanded)


def _build_session_title(title: Any, preview: Any, cwd: str | None) -> str:
    explicit = str(title or "").strip()
    if explicit:
        return explicit
    preview_text = str(preview or "").strip()
    if preview_text:
        return preview_text
    leaf = os.path.basename(str(cwd or "").rstrip("/\\"))
    return leaf or "New thread"


def _format_updated_at(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _updated_at_sort_key(value: Any) -> float:
    if value is None:
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return float("-inf")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(raw)
        except Exception:
            return float("-inf")


def _acp_stderr_print(*args, **kwargs) -> None:
    """Best-effort human-readable output sink for ACP stdio sessions.

    ACP reserves stdout for JSON-RPC frames, so any incidental CLI/status output
    from AIAgent must be redirected away from stdout. Route it to stderr instead.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _register_task_cwd(task_id: str, cwd: str) -> None:
    """Bind a task/session id to the editor's working directory for tools.

    Zed can launch Hermes from a Windows workspace while the ACP process runs
    inside WSL. In that case ACP sends cwd as e.g. ``E:\\Projects\\POTI``;
    local tools need the WSL mount equivalent or subprocess creation fails
    before the command can run.
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides
        register_task_env_overrides(task_id, {"cwd": _translate_acp_cwd(cwd)})
    except Exception:
        logger.debug("Failed to register ACP task cwd override", exc_info=True)


def _expand_acp_enabled_toolsets(
    toolsets: List[str] | None = None,
    mcp_server_names: List[str] | None = None,
) -> List[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    expanded: List[str] = []
    for name in list(toolsets or ["hermes-acp"]):
        if name and name not in expanded:
            expanded.append(name)

    for server_name in list(mcp_server_names or []):
        toolset_name = f"mcp-{server_name}"
        if server_name and toolset_name not in expanded:
            expanded.append(toolset_name)

    return expanded


def _clear_task_cwd(task_id: str) -> None:
    """Remove task-specific cwd overrides for an ACP session."""
    if not task_id:
        return
    try:
        from tools.terminal_tool import clear_task_env_overrides
        clear_task_env_overrides(task_id)
    except Exception:
        logger.debug("Failed to clear ACP task cwd override", exc_info=True)


@dataclass
class SessionState:
    """Tracks per-session state for an ACP-managed Hermes agent."""

    session_id: str
    agent: Any  # AIAgent instance
    cwd: str = "."
    model: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = None  # threading.Event
    is_running: bool = False
    queued_prompts: List[str] = field(default_factory=list)
    runtime_lock: Any = field(default_factory=Lock)
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the
    shared SessionDB so they survive process restarts and are searchable
    via ``session_search``.
    """

    def __init__(self, agent_factory=None, db=None):
        """
        Args:
            agent_factory: Optional callable that creates an AIAgent-like object.
                           Used by tests. When omitted, a real AIAgent is created
                           using the current Hermes runtime provider configuration.
            db:            Optional SessionDB instance. When omitted, the default
                           SessionDB (``~/.hermes/state.db``) is lazily created.
        """
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use

    # ---- public API ---------------------------------------------------------

    def create_session(self, cwd: str = ".") -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent."""
        import threading

        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=session_id, cwd=cwd)
        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", "") or "",
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the session for *session_id*, or ``None``.

        If the session is not in memory but exists in the database (e.g. after
        a process restart), it is transparently restored.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return state
        # Attempt to restore from database.
        return self._restore(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from memory and database. Returns True if it existed."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        db_existed = self._delete_persisted(session_id)
        if existed or db_existed:
            _clear_task_cwd(session_id)
        return existed or db_existed

    def fork_session(self, session_id: str, cwd: str = ".") -> Optional[SessionState]:
        """Deep-copy a session's history into a new session."""
        import threading

        cwd = _translate_acp_cwd(cwd)
        original = self.get_session(session_id)  # checks DB too
        if original is None:
            return None

        new_id = str(uuid.uuid4())
        agent = self._make_agent(
            session_id=new_id,
            cwd=cwd,
            model=original.model or None,
        )
        state = SessionState(
            session_id=new_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", original.model) or original.model,
            history=copy.deepcopy(original.history),
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[new_id] = state
        _register_task_cwd(new_id, cwd)
        self._persist(state)
        logger.info("Forked ACP session %s -> %s", session_id, new_id)
        return state

    def list_sessions(self, cwd: str | None = None) -> List[Dict[str, Any]]:
        """Return lightweight info dicts for all sessions (memory + database)."""
        normalized_cwd = _normalize_cwd_for_compare(cwd) if cwd else None
        db = self._get_db()
        persisted_rows: dict[str, dict[str, Any]] = {}

        if db is not None:
            try:
                for row in db.list_sessions_rich(source="acp", limit=1000):
                    persisted_rows[str(row["id"])] = dict(row)
            except Exception:
                logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        # Collect in-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                history_len = len(s.history)
                if history_len <= 0:
                    continue
                if normalized_cwd and _normalize_cwd_for_compare(s.cwd) != normalized_cwd:
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                preview = next(
                    (
                        str(msg.get("content") or "").strip()
                        for msg in s.history
                        if msg.get("role") == "user" and str(msg.get("content") or "").strip()
                    ),
                    persisted.get("preview") or "",
                )
                results.append(
                    {
                        "session_id": s.session_id,
                        "cwd": s.cwd,
                        "model": s.model,
                        "history_len": history_len,
                        "title": _build_session_title(persisted.get("title"), preview, s.cwd),
                        "updated_at": _format_updated_at(
                            persisted.get("last_active") or persisted.get("started_at") or time.time()
                        ),
                    }
                )

        # Merge any persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            if sid in seen_ids:
                continue
            message_count = int(row.get("message_count") or 0)
            if message_count <= 0:
                continue
            # Extract cwd from model_config JSON.
            session_cwd = "."
            mc = row.get("model_config")
            if mc:
                try:
                    session_cwd = json.loads(mc).get("cwd", ".")
                except (json.JSONDecodeError, TypeError):
                    pass
            if normalized_cwd and _normalize_cwd_for_compare(session_cwd) != normalized_cwd:
                continue
            results.append({
                "session_id": sid,
                "cwd": session_cwd,
                "model": row.get("model") or "",
                "history_len": message_count,
                "title": _build_session_title(row.get("title"), row.get("preview"), session_cwd),
                "updated_at": _format_updated_at(row.get("last_active") or row.get("started_at")),
            })

        results.sort(key=lambda item: _updated_at_sort_key(item.get("updated_at")), reverse=True)
        return results

    def update_cwd(self, session_id: str, cwd: str) -> Optional[SessionState]:
        """Update the working directory for a session and its tool overrides."""
        cwd = _translate_acp_cwd(cwd)
        state = self.get_session(session_id)  # checks DB too
        if state is None:
            return None
        state.cwd = cwd
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        return state

    def cleanup(self) -> None:
        """Remove all sessions (memory and database) and clear task-specific cwd overrides."""
        with self._lock:
            session_ids = list(self._sessions.keys())
            self._sessions.clear()
        for session_id in session_ids:
            _clear_task_cwd(session_id)
            self._delete_persisted(session_id)
        # Also remove any DB-only ACP sessions not currently in memory.
        db = self._get_db()
        if db is not None:
            try:
                rows = db.search_sessions(source="acp", limit=10000)
                for row in rows:
                    sid = row["id"]
                    _clear_task_cwd(sid)
                    db.delete_session(sid)
            except Exception:
                logger.debug("Failed to cleanup ACP sessions from DB", exc_info=True)

    def save_session(self, session_id: str) -> None:
        """Persist the current state of a session to the database.

        Called by the server after prompt completion, slash commands that
        mutate history, and model switches.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            self._persist(state)

    # ---- persistence via SessionDB ------------------------------------------

    def _get_db(self):
        """Lazily initialise and return the SessionDB instance.

        Returns ``None`` if the DB is unavailable (e.g. import error in a
        minimal test environment).

        Note: we resolve ``HERMES_HOME`` dynamically rather than relying on
        the module-level ``DEFAULT_DB_PATH`` constant, because that constant
        is evaluated at import time and won't reflect env-var changes made
        later (e.g. by the test fixture ``_isolate_hermes_home``).
        """
        if self._db_instance is not None:
            return self._db_instance
        try:
            from hermes_state import SessionDB
            hermes_home = get_hermes_home()
            self._db_instance = SessionDB(db_path=hermes_home / "state.db")
            return self._db_instance
        except Exception:
            logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
            return None

    def _persist(self, state: SessionState) -> None:
        """Write session state to the database.

        Creates the session record if it doesn't exist, then replaces all
        stored messages with the current in-memory history.
        """
        db = self._get_db()
        if db is None:
            return

        # Ensure model is a plain string (not a MagicMock or other proxy).
        model_str = str(state.model) if state.model else None
        session_meta = {"cwd": state.cwd}
        provider = getattr(state.agent, "provider", None)
        base_url = getattr(state.agent, "base_url", None)
        api_mode = getattr(state.agent, "api_mode", None)
        if isinstance(provider, str) and provider.strip():
            session_meta["provider"] = provider.strip()
        if isinstance(base_url, str) and base_url.strip():
            session_meta["base_url"] = base_url.strip()
        if isinstance(api_mode, str) and api_mode.strip():
            session_meta["api_mode"] = api_mode.strip()
        cwd_json = json.dumps(session_meta)

        try:
            # Ensure the session record exists.
            existing = db.get_session(state.session_id)
            if existing is None:
                db.create_session(
                    session_id=state.session_id,
                    source="acp",
                    model=model_str,
                    model_config={"cwd": state.cwd},
                )
            else:
                # Update model_config (contains cwd) if changed.
                try:
                    with db._lock:
                        db._conn.execute(
                            "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
                            (cwd_json, model_str, state.session_id),
                        )
                        db._conn.commit()
                except Exception:
                    logger.debug("Failed to update ACP session metadata", exc_info=True)

            # Replace stored messages with current history atomically so a
            # mid-rewrite failure rolls back and the previously persisted
            # conversation is preserved (salvaged from #13675).
            db.replace_messages(state.session_id, state.history)
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load a session from the database into memory, recreating the AIAgent."""
        import threading

        db = self._get_db()
        if db is None:
            return None

        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for ACP session %s", session_id, exc_info=True)
            return None

        if row is None:
            return None

        # Only restore ACP sessions.
        if row.get("source") != "acp":
            return None

        # Extract cwd from model_config.
        cwd = "."
        requested_provider = row.get("billing_provider")
        restored_base_url = row.get("billing_base_url")
        restored_api_mode = None
        mc = row.get("model_config")
        if mc:
            try:
                meta = json.loads(mc)
                if isinstance(meta, dict):
                    cwd = meta.get("cwd", ".")
                    requested_provider = meta.get("provider") or requested_provider
                    restored_base_url = meta.get("base_url") or restored_base_url
                    restored_api_mode = meta.get("api_mode") or restored_api_mode
            except (json.JSONDecodeError, TypeError):
                pass

        model = row.get("model") or None

        # Load conversation history.
        try:
            history = db.get_messages_as_conversation(session_id)
        except Exception:
            logger.warning("Failed to load messages for ACP session %s", session_id, exc_info=True)
            history = []

        try:
            agent = self._make_agent(
                session_id=session_id,
                cwd=cwd,
                model=model,
                requested_provider=requested_provider,
                base_url=restored_base_url,
                api_mode=restored_api_mode,
            )
        except Exception:
            logger.warning("Failed to recreate agent for ACP session %s", session_id, exc_info=True)
            return None

        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=model or getattr(agent, "model", "") or "",
            history=history,
            cancel_event=threading.Event(),
        )
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        logger.info("Restored ACP session %s from DB (%d messages)", session_id, len(history))
        return state

    def _delete_persisted(self, session_id: str) -> bool:
        """Delete a session from the database. Returns True if it existed."""
        db = self._get_db()
        if db is None:
            return False
        try:
            return db.delete_session(session_id)
        except Exception:
            logger.debug("Failed to delete ACP session %s from DB", session_id, exc_info=True)
            return False

    # ---- internal -----------------------------------------------------------

    def _make_agent(
        self,
        *,
        session_id: str,
        cwd: str,
        model: str | None = None,
        requested_provider: str | None = None,
        base_url: str | None = None,
        api_mode: str | None = None,
    ):
        if self._agent_factory is not None:
            return self._agent_factory()

        from run_agent import AIAgent
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or default_model)
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        configured_mcp_servers = [
            name
            for name, cfg in (config.get("mcp_servers") or {}).items()
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is not False
        ]

        kwargs = {
            "platform": "acp",
            "enabled_toolsets": _expand_acp_enabled_toolsets(
                ["hermes-acp"],
                mcp_server_names=configured_mcp_servers,
            ),
            "quiet_mode": True,
            "session_id": session_id,
            "model": model or default_model,
        }

        try:
            runtime = resolve_runtime_provider(requested=requested_provider or config_provider)
            kwargs.update(
                {
                    "provider": runtime.get("provider"),
                    "api_mode": api_mode or runtime.get("api_mode"),
                    "base_url": base_url or runtime.get("base_url"),
                    "api_key": runtime.get("api_key"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                }
            )
        except Exception:
            logger.debug("ACP session falling back to default provider resolution", exc_info=True)

        _register_task_cwd(session_id, cwd)
        agent = AIAgent(**kwargs)
        # ACP stdio transport requires stdout to remain protocol-only JSON-RPC.
        # Route any incidental human-readable agent output to stderr instead.
        agent._print_fn = _acp_stderr_print
        return agent
