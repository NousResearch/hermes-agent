"""``HermesCLI._session_db`` backed by the shared gateway session (CALLMAP §3.5).

Main's classic CLI holds a ``SessionDB`` writer. Rewired, the gateway is the single writer:

- reads (resume lookup, recap history, titles, the /resume picker, safety checks) go to a
  read-only ``state.db`` reader in this process, which the maintainer allows;
- every write is a gateway verb (``session.create`` / ``session.resume`` / ``session.mutate`` /
  ``session.detach``) or a documented no-op where the gateway already owns the work
  (transcript flushes, token accounting, maintenance);
- anything not yet on the gateway raises ``GatewayClientError`` (a ``ValueError``), so main's
  existing ``except`` paths print it instead of writing locally.

The session id is the gateway's: :meth:`open_session` creates or resumes the session before the
banner and adopts the id it returns.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import uuid

from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection

logger = logging.getLogger(__name__)

# Read-only SessionDB methods main's CLI calls (served by the WAL reader).
_READS = frozenset({
    "get_session", "get_session_title", "get_session_by_title", "resolve_resume_session_id",
    "get_messages_as_conversation", "get_resume_conversations", "assert_resume_safe",
    "is_kanban_owned_session", "list_sessions_rich", "search_sessions", "get_next_title_in_lineage",
    "get_handoff_state", "get_meta", "get_compression_tip", "resolve_session_by_title", "get_messages",
    "session_count", "sanitize_title", "session_yolo_enabled",
})
# Writes the gateway already performs itself (persistence, accounting, maintenance).
_GATEWAY_OWNED = frozenset({
    "create_session", "reopen_session", "append_messages_batch", "flush_token_counts",
    "update_session_model", "patch_session_model_config", "maybe_auto_archive",
    "maybe_auto_prune_and_vacuum", "set_meta", "update_token_counts", "close",
})
# Mutations the owner refuses without the execution generation they were read at (hermes_state_runtime).
_GENERATION_FENCED = frozenset({"delete", "rewind", "reset", "branch", "model", "compress"})
_COUNTED_REPAIRS = frozenset({"prune_empty_ghost_sessions", "finalize_orphaned_compression_sessions"})
# Launch options main's ``cli.main`` takes that are creation policy on the gateway session.
_LAUNCH_POLICY = ("model", "provider", "toolsets", "reasoning", "api_key", "base_url", "max_turns",
                  "ignore_rules", "checkpoints", "pass_session_id", "skills")


def _noop(*_a, **_k):
    return None


def _zero(*_a, **_k):
    return 0


class GatewaySessionStore:
    """Truthy while the gateway connection is usable (main's ``if self._session_db:`` guards)."""

    def __init__(self, connection=None):
        self.connection = connection or GatewayConnection.shared()
        self.snapshots: dict[str, dict] = {}
        self.current: dict | None = None
        self._reader = None

    def __bool__(self):
        return not self.connection.closed

    # -- reads -----------------------------------------------------------------------------
    def _db(self):
        """The read-only reader, opened on first use. No state.db yet (the gateway creates it on its
        first start, after the CLI's startup maintenance probe) is retried on the next read."""
        if self._reader is None:
            from hermes_constants import get_hermes_home
            path = get_hermes_home() / "state.db"
            if not path.exists():
                return None
            try:
                from hermes_state import SessionDB
                self._reader = SurfaceLabelReader(SessionDB(db_path=path, read_only=True))
            except Exception as exc:  # health: allow BLE001 -- reads find nothing; retried on the next read
                logger.debug("read-only state.db unavailable: %s", exc)
        return self._reader

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in _READS:
            if name in {"sanitize_title", "session_yolo_enabled"}:
                from hermes_state import SessionDB
                return getattr(SessionDB, name)
            db = self._db()
            if db is None:
                return _noop
            return getattr(db, name)
        if name in _GATEWAY_OWNED:
            return _noop
        if name in _COUNTED_REPAIRS:
            return _zero

        def refused(*_a, **_k):
            raise GatewayClientError(f"{name} is not on the gateway session yet; nothing was written")
        return refused

    # -- session lifecycle -----------------------------------------------------------------
    def call(self, method, **params):
        return self.connection.call(method, **params)

    def _adopt(self, snapshot):
        sid = snapshot["stored_session_id"]
        self.snapshots[sid] = snapshot
        self.current = snapshot
        self.connection.state(sid).adopt_snapshot(snapshot)
        return sid

    def attach(self, session_id):
        """``session.resume`` once per session id this viewer uses (idempotent)."""
        if session_id in self.snapshots:
            return self.snapshots[session_id]
        snapshot = self.call("session.resume", session_id=session_id)
        self._adopt(snapshot)
        return snapshot

    def open_session(self, cli, launch=None):
        """Create (or resume) the gateway session for this CLI and adopt its id. A resume target the
        gateway does not know is left as is: main's own "Session not found" path reports it."""
        if getattr(cli, "_resumed", False):
            try:
                return self._adopt(self.attach(cli.session_id))
            except GatewayClientError as exc:
                if str(exc) in {"not_found", "permission_denied"}:
                    return None
                raise
        return self.create(launch)

    def create(self, launch=None, title=None, *, record_cwd=True):
        """``session.create`` with the launch policy (and ``title``: the owner's resolve -> create ->
        title step); adopts and returns the new session id. ``record_cwd=False`` (``/new``,
        ``--create-if-missing``) leaves the row's cwd empty, as main's explicit creates did."""
        from agent.session_source import CLI_FAMILY_SOURCES, session_source_for
        params = _creation_policy(launch or {})
        params["cwd"] = os.path.realpath(os.getenv("TERMINAL_CWD") or os.getcwd())
        # Main's row label: ``cli``, ``oneshot`` for a finite -q/-Q run, or an explicit --source.
        params["source"] = session_source_for("cli")
        # Main's row stamp (run_agent._launch_cwd_for_session): this process's cwd (``-w`` never
        # chdirs, so the repo root), only for a CLI-family label on a local terminal backend.
        local = (os.getenv("TERMINAL_ENV") or "local").strip().lower() in ("", "local")
        params["row_cwd"] = os.getcwd() if record_cwd and local and params["source"] in CLI_FAMILY_SOURCES else ""
        if title:
            params["title"] = title
        snapshot = self.call("session.create", request_id=uuid.uuid4().hex, **params)
        return self._adopt(snapshot)

    def branch(self, session_id, title=None):
        """``session.mutate branch``: the gateway forks the whole active transcript into a child it
        attaches this viewer to; returns the child id (main's /branch then switches to it)."""
        snapshot = self.call("session.resume", session_id=session_id)
        payload = {"title": title} if title else {}
        receipt = self.call("session.mutate", session_id=session_id, request_id=uuid.uuid4().hex,
                            expected_revision=snapshot["revision"],
                            expected_generation=snapshot["execution_generation"],
                            operation="branch", payload=payload) or {}
        child = receipt.get("branched_session_id")
        if not child:
            raise GatewayClientError("branch_failed")
        self._adopt(self.attach(child))
        return child

    def _mutate(self, session_id, operation, payload):
        snapshot = self.call("session.resume", session_id=session_id)
        fence = {}
        if operation in _GENERATION_FENCED:
            fence["expected_generation"] = snapshot["execution_generation"]
        return self.call("session.mutate", session_id=session_id, request_id=uuid.uuid4().hex,
                         expected_revision=snapshot["revision"], operation=operation, payload=payload, **fence)

    def set_session_title(self, session_id, title):
        try:
            self._mutate(session_id, "rename", {"title": title})
        except GatewayClientError as exc:
            # The owner's SessionDB ValueError text never crosses the wire (bounded reason codes):
            # print main's "already in use" copy from the reader when another row holds the title.
            holder = self._title_holder(session_id, title)
            if holder:
                raise ValueError(f"Title '{title}' is already in use by session {holder}") from exc
            raise
        if session_id in self.snapshots:
            self.connection.state(session_id).title = title
        return True

    def rewind_user_turn(self, session_id, user_ordinal, *, warm_history=None, require_retryable=False, **_kw):
        """Main's ``SessionDB.rewind_user_turn`` (``/undo N``, ``/retry``) as ``session.mutate rewind``.
        The user turn is located on the gateway's snapshot transcript (its rows carry ``row_id``),
        checked against the CLI's warm view exactly as main checks it, then rewound by that row id."""
        from agent.context_compressor import history_before_user_originated_turn, retryable_user_text
        from agent.message_content import flatten_message_text
        from agent.session_persistence import _is_ephemeral_scaffolding
        from hermes_state_rewind import (
            _HISTORY_CHANGED, RewindOutcome, RewindTargetUnavailableError, _comparison_content, _user_indices)
        snapshot = self.call("session.resume", session_id=session_id)
        durable = list(snapshot.get("messages") or [])
        durable_user = _user_indices(durable)
        if user_ordinal < 0:
            user_ordinal = max(len(durable_user) + user_ordinal, 0)
        if user_ordinal >= len(durable_user):
            raise RewindTargetUnavailableError("target user message is no longer in session history")
        target = durable[durable_user[user_ordinal]]
        durable_prefix, live_view = history_before_user_originated_turn(durable, durable_user[user_ordinal])
        prefix = [{k: v for k, v in m.items() if k != "row_id"} for m in durable_prefix]
        if warm_history is not None:
            warm = [m for m in warm_history if not _is_ephemeral_scaffolding(m)]
            warm_user = _user_indices(warm)
            if len(warm_user) != len(durable_user):
                raise RuntimeError(_HISTORY_CHANGED)
            prefix, warm_live = history_before_user_originated_turn(warm, warm_user[user_ordinal])
            if _comparison_content(live_view) != _comparison_content(warm_live):
                raise RuntimeError(_HISTORY_CHANGED)
        live_text = retryable_user_text(live_view.get("content")) if require_retryable else None
        if not isinstance(target.get("row_id"), int):
            raise RuntimeError("rewind target has no durable row identity")
        receipt = self.call("session.mutate", session_id=session_id, request_id=uuid.uuid4().hex,
                            expected_revision=snapshot["revision"],
                            expected_generation=snapshot["execution_generation"],
                            operation="rewind", payload={"target_message_id": target["row_id"]}) or {}
        return RewindOutcome(
            prefix=prefix, live_view=live_view,
            live_text=live_text if live_text is not None else flatten_message_text(live_view.get("content")),
            rewound_count=int(receipt.get("rewound_count") or 0), turns_undone=len(durable_user) - user_ordinal)

    def set_session_yolo(self, session_id, enabled):
        """Main's ``/yolo`` persist hook: the gateway session's own bypass (``config.set yolo``, persisted
        on the route), since approvals run there. The CLI's in-process flag stays main's display state."""
        self.call("config.set", session_id=session_id, key="yolo", value="1" if enabled else "0")

    def _title_holder(self, session_id, title):
        db = self._db()
        if db is None or not title:
            return None
        with db._read_ctx() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE title = ? AND id != ?", (title, session_id)).fetchone()
        return row[0] if row else None

    def delete_session(self, session_id, sessions_dir=None):
        self.snapshots.pop(session_id, None)
        self._mutate(session_id, "delete", {})
        return True

    def end_session(self, session_id, reason=None):
        """Main's ``end_session(sid, reason)``: detach this viewer with ``end_reason``; the owner ends
        the idle local row with that reason once no viewer is left (``ended_at`` / ``end_reason``)."""
        snapshot = self.snapshots.pop(session_id, None)
        subscription = (snapshot or {}).get("subscription_id")
        if subscription and not self.connection.closed:
            params = {"end_reason": reason} if reason else {}
            try:
                self.call("session.detach", session_id=session_id, subscription_id=subscription, **params)
            except GatewayClientError as exc:
                logger.debug("session.detach failed: %s", exc)
        if self.current is snapshot:
            self.current = None

    def delete_session_if_empty(self, session_id, sessions_dir=None):
        """Main's quit-immediately prune, as a gateway delete: only a row with no messages, no title
        and no children (checked on the read-only reader; the gateway deletes the lineage)."""
        db = self._db()
        if db is None:
            return False
        try:
            with db._read_ctx() as conn:
                empty = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? AND title IS NULL"
                    " AND NOT EXISTS (SELECT 1 FROM messages WHERE messages.session_id = sessions.id)"
                    " AND NOT EXISTS (SELECT 1 FROM sessions child WHERE child.parent_session_id = sessions.id)",
                    (session_id,)).fetchone()
        except Exception as exc:  # health: allow BLE001 -- a failed probe keeps the row (never deletes on doubt)
            logger.debug("empty-session probe failed: %s", exc)
            return False
        if not empty:
            return False
        try:
            return self.delete_session(session_id)
        except GatewayClientError as exc:
            logger.debug("empty session prune refused: %s", exc)
            return False

    # -- banner --------------------------------------------------------------------------
    def banner_inventory(self):
        """The current session snapshot's banner fields (``gateway/session_local_inventory``), as
        ``build_welcome_banner`` kwargs; None until a session is open."""
        info = (self.current or {}).get("info") or {}
        if "tools" not in info:
            return None
        grouped = info.get("tools") or {}
        toolset_of = {name: toolset for toolset, names in grouped.items() for name in names or []}
        availability = info.get("tool_availability") or {
            "available_toolsets": [], "unavailable_toolsets": [], "lazy_tools": [], "disabled_tools": []}
        for item in availability.get("unavailable_toolsets", []):
            for name in item.get("tools", []):
                toolset_of.setdefault(name, item.get("id", item.get("name", "")))
        return {
            "tools": [{"type": "function", "function": {"name": name}} for names in grouped.values()
                      for name in names or []],
            "get_toolset_for_tool": toolset_of.get,
            "availability": availability,
            "skills_by_category": info.get("skills") or {},
            "mcp_status": list(info.get("mcp_servers") or []),
            "enabled_toolsets": info.get("enabled_toolsets"),
            "context_length": info.get("context_length"),
            "cwd": info.get("cwd"),
            # Main's startup lines under the banner, computed in the gateway (``source: cli`` only).
            "tool_availability_warnings": info.get("tool_availability_warnings"),
            "browser_backend_notice": info.get("browser_backend_notice"),
        }


# Interactive surfaces whose gateway rows are labelled ``local`` (``created_source`` keeps the surface;
# gateway/session_a2a.storage_source). Producer labels (oneshot, tool, kanban, ...) are stored as is.
_LOCAL_LABELLED = frozenset({"cli", "tui", "gui"})


class SurfaceLabelReader:
    """A read-only ``SessionDB`` whose source-filtered listings read main's labels: ``source='cli'``
    also matches a gateway row labelled ``local`` that the CLI created (``created_source='cli'``), so
    main's /resume picker, /sessions, /history and bare ``-c`` find the sessions the CLI made."""

    def __init__(self, db):
        self._wrapped = db

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def _labelled(self, method, requested, limit, offset, call):
        wanted = {requested} if isinstance(requested, str) else set(requested or ())
        surfaces = wanted & _LOCAL_LABELLED
        if not surfaces:
            return call(requested, limit, offset)
        # Over-fetch, then keep only the local rows those surfaces created (ordering is the reader's own).
        rows = call(sorted(wanted | {"local"}), (limit + offset) * 3 + 50, 0)
        kept = [row for row in rows if row.get("source") != "local" or row.get("created_source") in surfaces]
        return kept[offset:offset + limit]

    def search_sessions(self, source=None, limit=20, offset=0, **kw):
        return self._labelled("search_sessions", source, limit, offset,
                              lambda src, lim, off: self._wrapped.search_sessions(source=src, limit=lim, offset=off, **kw))

    def list_sessions_rich(self, source=None, sources=None, limit=20, offset=0, **kw):
        requested = sources if sources else ([source] if source else None)
        if not requested:
            return self._wrapped.list_sessions_rich(limit=limit, offset=offset, **kw)
        return self._labelled("list_sessions_rich", requested, limit, offset,
                              lambda src, lim, off: self._wrapped.list_sessions_rich(sources=src, limit=lim, offset=off, **kw))


_CANONICAL_LOCAL_ID = re.compile(r"^local-([0-9a-f]{8})[0-9a-f]{56}$")


def session_label(session_id):
    """Main's 8-character banner label for a canonical durable id (``local-`` + 64 hex); every other
    id is shown as is. Display only: RPCs, /resume hints and the exit summary keep the full id."""
    match = _CANONICAL_LOCAL_ID.match(session_id or "")
    return match.group(1) if match else session_id


def _creation_policy(launch):
    """The explicit launch flags as ``session.create`` policy (the thin client's own mapping);
    unset flags leave the gateway to resolve the profile default, as main's CLI did."""
    from hermes_cli.gateway_chat import _requested_policy, env_launch_flags
    fields = {key: launch.get(key) for key in _LAUNCH_POLICY}
    if isinstance(fields.get("toolsets"), (list, tuple)):
        fields["toolsets"] = ",".join(str(item) for item in fields["toolsets"])
    policy = _requested_policy(argparse.Namespace(**fields, source=None))
    policy.pop("source", None)
    if launch.get("run_budget") is not None:  # main's `--run-budget` (the thin client refuses it)
        policy["run_budget"] = float(launch["run_budget"])
    if launch.get("verbose"):  # main's `-v` (the thin client refuses it)
        policy["verbose"] = True
    policy.update(env_launch_flags())
    return policy


def session_store():
    return GatewaySessionStore()


def create_titled_session(title):
    """``chat -c <title> --create-if-missing`` for the classic CLI: the gateway's ``session.create``
    with ``title`` (it resolves-or-creates under one owner lock), so this process writes no row.
    Main's ``_create_titled_session`` minted the id and wrote the row itself, with no cwd stamp
    (``row_cwd`` ""). Returns the id, or None when the gateway refused (the caller prints main's
    "could not be created" copy)."""
    params = _creation_policy({})
    params["cwd"] = os.path.realpath(os.getenv("TERMINAL_CWD") or os.getcwd())
    params["source"] = "cli"  # main's titled row is `cli` (hermes_cli.main._create_titled_session)
    params["row_cwd"] = ""
    try:
        snapshot = session_store().call("session.create", request_id=uuid.uuid4().hex, title=title, **params)
    except GatewayClientError as exc:
        logger.warning("Failed to create titled session %r on the gateway: %s", title, exc)
        return None
    return snapshot["stored_session_id"]
