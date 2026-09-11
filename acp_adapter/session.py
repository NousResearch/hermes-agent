"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``; ``load_session`` /
``resume_session`` after an editor reconnect restore the full history from there.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home, translate_cwd_for_wsl_backend, windows_path_to_wsl

import copy
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _plain_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _named_provider_identity(provider: Any, requested_provider: Any) -> str:
    """Prefer ``custom:<name>`` over the flattened runtime provider ``custom``.

    ``auto`` is a resolution sentinel, not a durable identity — keep the already
    resolved runtime provider so ACP choice ids stay ``openrouter:...`` (#101946).
    """
    requested = _plain_str(requested_provider) or ""
    flattened = _plain_str(provider) or ""
    requested_l = requested.lower()
    flattened_l = flattened.lower()
    if requested_l.startswith("custom:") and requested_l != "custom:":
        return requested
    if flattened_l.startswith("custom:") and flattened_l != "custom:":
        return flattened
    try:
        from hermes_cli.models import named_custom_provider_id

        durable = named_custom_provider_id(requested) or named_custom_provider_id(flattened)
        if durable:
            return durable
    except Exception:
        pass
    if requested_l in {"", "auto"}:
        return flattened
    return requested or flattened


def _split_prefixed_model(model: Optional[str], provider: Optional[str]) -> tuple[str, str]:
    """Split ``aihubmix:qwen3.8-flash`` into ``(custom:aihubmix, qwen3.8-flash)``.

    Unprefixed model ids are returned unchanged with *provider* preserved.
    """
    raw_model = _plain_str(model) or ""
    raw_provider = _plain_str(provider) or ""
    if not raw_model:
        return raw_provider, raw_model
    try:
        from hermes_cli.models import named_custom_provider_id, parse_model_input

        parsed_provider, parsed_model = parse_model_input(
            raw_model, raw_provider or "custom"
        )
        durable = named_custom_provider_id(parsed_provider) or parsed_provider
        if parsed_model != raw_model:
            return durable or parsed_provider, parsed_model
        if durable:
            return durable, parsed_model
        return raw_provider or parsed_provider, parsed_model
    except Exception:
        return raw_provider, raw_model


def persist_identity(state: "SessionState") -> Dict[str, Optional[str]]:
    """Return the ACP session identity that should be written to SessionDB.

    Fallback is turn-scoped: a transient aiberm → aihubmix failover must not
    become the restored primary after the next prompt or ACP process restart.
    Named custom providers (``custom:aiberm``) are preserved even though the
    live runtime provider is flattened to ``custom``.
    """
    agent = getattr(state, "agent", None)
    rt = getattr(agent, "_primary_runtime", None) if agent is not None else None
    if not isinstance(rt, dict):
        rt = {}
    fallback_on = getattr(agent, "_fallback_activated", False) is True
    use_primary = fallback_on and bool(
        _plain_str(rt.get("model")) or _plain_str(rt.get("provider")) or _plain_str(rt.get("base_url"))
    )

    if use_primary:
        model = _plain_str(rt.get("model")) or _plain_str(getattr(state, "model", None))
        provider = _named_provider_identity(rt.get("provider"), rt.get("requested_provider"))
        base_url = _plain_str(rt.get("base_url"))
        api_mode = _plain_str(rt.get("api_mode"))
        requested_provider = _plain_str(rt.get("requested_provider")) or provider or None
    else:
        live_model = getattr(agent, "model", None) if agent is not None else None
        model = _plain_str(getattr(state, "model", None)) or _plain_str(live_model)
        provider = _named_provider_identity(
            getattr(agent, "provider", None) if agent is not None else None,
            getattr(agent, "requested_provider", None) if agent is not None else None,
        )
        base_url = _plain_str(getattr(agent, "base_url", None)) if agent is not None else None
        api_mode = _plain_str(getattr(agent, "api_mode", None)) if agent is not None else None
        requested_provider = (
            _plain_str(getattr(agent, "requested_provider", None)) or provider or None
            if agent is not None
            else None
        )

    live_requested = requested_provider
    provider, model = _split_prefixed_model(model, provider or requested_provider)
    durable = _named_provider_identity(provider, requested_provider) or provider
    if (live_requested or "").strip().lower() == "auto":
        requested_provider = live_requested
    else:
        requested_provider = durable or live_requested or provider
    provider = durable or provider

    return {
        "model": model,
        "provider": provider or None,
        "requested_provider": requested_provider or provider or None,
        "base_url": base_url,
        "api_mode": api_mode,
    }


def _translate_acp_cwd(cwd: str) -> str:
    """Translate Windows ACP cwd values (``E:\\Projects``, ``\\\\wsl.localhost\\``) to POSIX form
    when Hermes runs in WSL so agents, tools, and persisted sessions agree; no-op elsewhere."""
    return translate_cwd_for_wsl_backend(str(cwd))


def _normalize_cwd_for_compare(cwd: str | None) -> str:
    expanded = os.path.expanduser(str(cwd or ".").strip() or ".")

    # Windows drive paths -> WSL mount form so history filters match across hosts.
    translated = windows_path_to_wsl(expanded)
    if translated is not None:
        expanded = translated
    elif re.match(r"^/mnt/[A-Za-z]/", expanded):
        expanded = f"/mnt/{expanded[5].lower()}/{expanded[7:]}"

    # realpath resolves symlink aliases (macOS ``/var`` vs ``/private/var``, ``/tmp`` vs
    # ``/private/tmp``) that otherwise drop a workspace's own sessions; it is lexical
    # for missing paths (e.g. WSL-translated drives).
    try:
        # Resolve symlink aliases so equivalent spellings of the same directory compare equal — macOS
        # reports editor workspaces as ``/var/...`` while sessions get stored under ``/private/var/...``
        # (and ``/tmp`` vs ``/private/tmp``), which made ACP history filters silently drop a workspace's own
        # sessions. WSL-translated Windows drives — keep the previous normpath behavior. Ported from
        # PrimeIntellect-ai/prime-agent#628.
        return os.path.realpath(expanded)
    except OSError:
        return os.path.normpath(expanded)


def _build_session_title(title: Any, preview: Any, cwd: str | None) -> str:
    leaf = os.path.basename(str(cwd or "").rstrip("/\\"))
    return str(title or "").strip() or str(preview or "").strip() or leaf or "New thread"


def _format_updated_at(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and value.strip()):
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _updated_at_sort_key(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return float("-inf")
    for parse in (lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp(), float):
        try:
            return parse(raw)
        except Exception:
            continue
    return float("-inf")


def _acp_stderr_print(*args, **kwargs) -> None:
    """Route incidental AIAgent output to stderr; ACP reserves stdout for JSON-RPC."""
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _register_task_cwd(task_id: str, cwd: str) -> None:
    """Bind a task/session id to the editor cwd for tools. Zed may send a Windows cwd while
    the ACP process runs in WSL; tools need the WSL mount or subprocess creation fails."""
    if not task_id:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides
        register_task_env_overrides(task_id, {"cwd": _translate_acp_cwd(cwd)})
    except Exception:
        logger.debug("Failed to register ACP task cwd override", exc_info=True)


def _expand_acp_enabled_toolsets(toolsets: List[str] | None = None,
                                 mcp_server_names: List[str] | None = None) -> List[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    names = [n for n in (toolsets or ["hermes-acp"]) if n]
    names += [f"mcp-{s}" for s in (mcp_server_names or []) if s]
    return list(dict.fromkeys(names))


def _parse_model_config(mc: Any) -> dict:
    """Decode a persisted model_config JSON blob; ``{}`` when absent/invalid/non-dict."""
    try:
        meta = json.loads(mc) if mc else None
    except (json.JSONDecodeError, TypeError):
        meta = None
    return meta if isinstance(meta, dict) else {}


def _session_info(sid: str, cwd: str, model: Any, history_len: int, title: Any, preview: Any,
                  updated_at: Any) -> Dict[str, Any]:
    return {"session_id": sid, "cwd": cwd, "model": model, "history_len": history_len,
            "title": _build_session_title(title, preview, cwd), "updated_at": _format_updated_at(updated_at)}


def _first_user_preview(history: List[Dict[str, Any]], default: str) -> str:
    return next((str(m.get("content") or "").strip() for m in history
                 if m.get("role") == "user" and str(m.get("content") or "").strip()), default)


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
    runtime_lock: Any = field(default_factory=threading.Lock)
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the shared
    SessionDB so they survive restarts and are searchable via ``session_search``."""

    def __init__(self, agent_factory=None, db=None):
        """``agent_factory``: AIAgent-like factory (tests); default builds a real AIAgent from
        the runtime provider config. ``db``: SessionDB; default lazily opens ``~/.hermes/state.db``."""
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use

    # ---- public API ---------------------------------------------------------

    def create_session(self, cwd: str = ".") -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent."""
        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=session_id, cwd=cwd)
        state = self._install_state(session_id, agent, cwd, getattr(agent, "model", "") or "", [])
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the session, transparently restoring it from the DB (e.g. after
        a process restart) when it is not in memory; ``None`` if unknown."""
        with self._lock:
            state = self._sessions.get(session_id)
        return state if state is not None else self._restore(session_id)

    def fork_session(self, session_id: str, cwd: str = ".") -> Optional[SessionState]:
        """Deep-copy a session's history into a new session."""
        cwd = _translate_acp_cwd(cwd)
        original = self.get_session(session_id)  # checks DB too
        if original is None:
            return None
        new_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=new_id, cwd=cwd, model=original.model or None)
        model = getattr(agent, "model", original.model) or original.model
        state = self._install_state(new_id, agent, cwd, model, copy.deepcopy(original.history))
        logger.info("Forked ACP session %s -> %s", session_id, new_id)
        return state

    def list_sessions(self, cwd: str | None = None) -> List[Dict[str, Any]]:
        """Return lightweight info dicts for all sessions (memory + database)."""
        normalized_cwd = _normalize_cwd_for_compare(cwd) if cwd else None
        db = self._get_db()
        persisted_rows: dict[str, dict[str, Any]] = {}
        try:
            for row in (db.list_sessions_rich(source="acp", limit=1000) if db is not None else ()):
                persisted_rows[str(row["id"])] = dict(row)
        except Exception:
            logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        def _matches(session_cwd: str) -> bool:
            return not normalized_cwd or _normalize_cwd_for_compare(session_cwd) == normalized_cwd

        # In-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                if not s.history or not _matches(s.cwd):
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                results.append(_session_info(
                    s.session_id, s.cwd, s.model, len(s.history), persisted.get("title"),
                    _first_user_preview(s.history, persisted.get("preview") or ""),
                    persisted.get("last_active") or persisted.get("started_at") or time.time(),
                ))

        # Then persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            message_count = int(row.get("message_count") or 0)
            session_cwd = _parse_model_config(row.get("model_config")).get("cwd", ".")
            if sid in seen_ids or message_count <= 0 or not _matches(session_cwd):
                continue
            results.append(_session_info(
                sid, session_cwd, row.get("model") or "", message_count, row.get("title"),
                row.get("preview"), row.get("last_active") or row.get("started_at"),
            ))

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

    def save_session(self, session_id: str) -> None:
        """Persist a session; called by the server after prompt completion,
        history-mutating slash commands, and model switches."""
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            self._persist(state)

    # ---- persistence via SessionDB ------------------------------------------

    def _install_state(self, session_id: str, agent: Any, cwd: str, model: str,
                       history: List[Dict[str, Any]], *, persist: bool = True) -> SessionState:
        """Build a SessionState, register it in memory, bind its cwd for tools, optionally persist."""
        state = SessionState(session_id=session_id, agent=agent, cwd=cwd, model=model,
                             history=history, cancel_event=threading.Event())
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        if persist:
            self._persist(state)
        return state

    def _get_db(self):
        """Lazily initialise the SessionDB; ``None`` if unavailable (e.g. import error in a
        minimal test env). ``HERMES_HOME`` is resolved here, not via the import-time
        ``DEFAULT_DB_PATH``, so test fixtures that change the env var later are honoured."""
        if self._db_instance is None:
            try:
                from hermes_state import SessionDB
                self._db_instance = SessionDB(db_path=get_hermes_home() / "state.db")
            except Exception:
                logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
        return self._db_instance

    def _persist(self, state: SessionState) -> None:
        """Create/update the session record, then sync the live message set."""
        db = self._get_db()
        if db is None:
            return

        identity = persist_identity(state)
        model_str = identity.get("model")
        if model_str:
            state.model = model_str
        session_meta = {"cwd": state.cwd}
        for key in ("provider", "requested_provider", "base_url", "api_mode"):
            value = identity.get(key)
            if isinstance(value, str) and value.strip():
                session_meta[key] = value.strip()

        try:
            if db.get_session(state.session_id) is None:
                if not state.history:
                    # Empty editor probes stay ephemeral; copied fork history persists.
                    return
                db.create_session(session_id=state.session_id, source="acp", model=model_str,
                                  model_config=session_meta)
            else:
                try:
                    db.update_session_meta(state.session_id, json.dumps(session_meta), model_str)
                except Exception:
                    logger.debug("Failed to update ACP session metadata", exc_info=True)

            # An agent that owns persistence to this same DB already flushed the transcript
            # incrementally (append_message) and keeps pre-compaction turns as archived
            # active=0 rows; replace_messages() would DELETE those (and, after a compression
            # id rotation, clobber the ended parent transcript). Skip it in that case.
            # Calling replace_messages() here would then be a redundant double-write that DELETEs exactly
            # those archived rows (and, after a compression-driven id rotation where agent.session_id no
            # longer equals state.session_id, clobbers the ended parent transcript) — silent data loss for
            # any ACP conversation long enough to compress. Only fall back to the destructive atomic replace
            # when the agent is NOT persisting itself to this DB (e.g. a test agent factory, or a fresh
            # create/fork whose copied history the agent has not flushed yet). That path still rolls back on
            # a mid-rewrite failure so the previously persisted conversation survives (salvaged from
            # #13675).
            agent = state.agent
            if getattr(agent, "_session_db", None) is db and getattr(agent, "_session_db_created", False):
                return
            # A non-owning agent (model switch, /restore: fresh agent, _session_db_created=False)
            # may still sit on archived rows, so replace ONLY the active=1 set: on a fresh
            # create/fork every row is active (== full replace), and archived rows survive.
            # Unconditional because an existence probe would fail OPEN on DB error and can
            # race a concurrent archive_and_compact. Still rolls back on mid-rewrite failure.
            db.replace_messages(state.session_id, state.history, active_only=True)
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load an ACP session from the database into memory, recreating the AIAgent."""
        db = self._get_db()
        if db is None:
            return None
        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for ACP session %s", session_id, exc_info=True)
            return None
        if row is None or row.get("source") != "acp":
            return None

        meta = _parse_model_config(row.get("model_config"))
        cwd, model = meta.get("cwd", "."), row.get("model") or None

        # repair_alternation: this list becomes the resumed agent's LIVE conversation; a durable
        # ``user;user`` violation in state.db would otherwise re-fire the pre-request repair every request.
        try:
            history = db.get_messages_as_conversation(session_id, repair_alternation=True)
        except Exception:
            logger.warning("Failed to load messages for ACP session %s", session_id, exc_info=True)
            history = []

        try:
            agent = self._make_agent(
                session_id=session_id, cwd=cwd, model=model, api_mode=meta.get("api_mode") or None,
                requested_provider=meta.get("requested_provider") or meta.get("provider") or row.get("billing_provider"),
                base_url=meta.get("base_url") or row.get("billing_base_url"))
        except Exception:
            logger.warning("Failed to recreate agent for ACP session %s", session_id, exc_info=True)
            return None
        state = self._install_state(session_id, agent, cwd, model or getattr(agent, "model", "") or "",
                                    history, persist=False)
        logger.info("Restored ACP session %s from DB (%d messages)", session_id, len(history))
        return state

    # ---- internal -----------------------------------------------------------

    def _make_agent(self, *, session_id: str, cwd: str, model: str | None = None,
                    requested_provider: str | None = None, base_url: str | None = None, api_mode: str | None = None):
        if self._agent_factory is not None:
            try:
                return self._agent_factory(
                    session_id=session_id,
                    cwd=cwd,
                    model=model,
                    requested_provider=requested_provider,
                    base_url=base_url,
                    api_mode=api_mode,
                )
            except TypeError:
                return self._agent_factory()

        from run_agent import AIAgent
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider

        config = load_config()
        model_cfg = config.get("model")
        default_model, config_provider = "", None
        if isinstance(model_cfg, dict):
            default_model, config_provider = str(model_cfg.get("default") or ""), model_cfg.get("provider")
        elif isinstance(model_cfg, str):
            default_model = model_cfg.strip()

        configured_mcp_servers = [
            name for name, cfg in (config.get("mcp_servers") or {}).items()
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is not False
        ]

        incoming_model = model or default_model
        incoming_provider = requested_provider or config_provider
        if incoming_model:
            split_provider, split_model = _split_prefixed_model(
                incoming_model, incoming_provider
            )
            incoming_model = split_model or incoming_model
            if split_provider and (
                incoming_provider or (model and ":" in str(model))
            ):
                incoming_provider = split_provider
        requested_provider = incoming_provider

        kwargs = {
            "platform": "acp", "quiet_mode": True, "session_id": session_id, "session_db": self._get_db(),
            "enabled_toolsets": _expand_acp_enabled_toolsets(["hermes-acp"], mcp_server_names=configured_mcp_servers),
            "model": incoming_model or default_model,
        }
        try:
            from hermes_cli.fallback_config import get_fallback_chain

            fallback_chain = get_fallback_chain(config)
            if fallback_chain:
                kwargs["fallback_model"] = fallback_chain
        except Exception:
            logger.debug("ACP session could not load fallback_model", exc_info=True)

        try:
            runtime = resolve_runtime_provider(
                requested=requested_provider or config_provider,
                target_model=kwargs.get("model") or None,
            )
            kwargs.update({
                "provider": runtime.get("provider"),
                "requested_provider": requested_provider or runtime.get("requested_provider") or runtime.get("provider"),
                "api_mode": api_mode or runtime.get("api_mode"),
                "base_url": base_url or runtime.get("base_url"), "api_key": runtime.get("api_key"),
                "command": runtime.get("command"), "args": list(runtime.get("args") or []),
            })
        except Exception:
            logger.debug("ACP session falling back to default provider resolution", exc_info=True)
            if requested_provider:
                kwargs["requested_provider"] = requested_provider

        _register_task_cwd(session_id, cwd)

        # Bounded wait for the background MCP discovery started by entry.py: the agent
        # snapshots tools once at build and never re-reads the registry, so without this
        # join a slow-but-reachable server would be invisible all session. ensure_* also
        # (re)starts discovery if the entry spawn never ran or connected zero servers.
        # Bounded by ``mcp_discovery_timeout`` (config.yaml, ~1.5s); late servers are
        # picked up by HermesACPAgent._schedule_mcp_late_refresh.
        try:
            from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build

            ensure_mcp_discovery_before_agent_build(logger=logger, thread_name="acp-mcp-discovery")
        except Exception:
            logger.debug("ACP: bounded MCP discovery wait failed", exc_info=True)

        agent = AIAgent(**kwargs)
        # Codex app-server sessions spawn lazily on the first turn; stamp the ACP
        # workspace so the Codex runtime starts from the editor cwd, not ours.
        agent.session_cwd = cwd
        # ACP stdio: stdout is protocol-only JSON-RPC; agent chatter goes to stderr.
        agent._print_fn = _acp_stderr_print
        return agent


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from threading import Lock  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
