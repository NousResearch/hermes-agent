"""#47 Phase 2 — C1/C2 Direct-dispatch ownership gate (pre/post tool-call hooks).

The Kanban admission gate (``gateway/session_kanban.py``) already refuses a card
whose canonical work key is owned by another live session. The DIRECT path — a
chat session dispatching a delegate/subagent batch straight from ``goal`` /
``context`` — has no equivalent gate. This module is that gate.

What it does
------------
* :func:`scan_dispatch_scope` — the RFC v2 §2 UNION scan, copied verbatim from
  the tested reference in ``tests/gateway/test_cross_mode_integration.py``
  (``_scan_batch`` + ``_scopes_from_text``): top-level ``goal`` + ``context``
  plus **every** ``tasks[].goal`` / ``tasks[].context``. A conflicting issue
  hidden inside ``tasks[]`` must not be bypassable.
* :func:`release_for_dispatch` — thin wrapper over the ledger's
  ``release_ownership`` (owner-only, reason-tagged).
* :func:`pre_tool_call` — the ``pre_tool_call`` callback. When
  ``HERMES_OWNERSHIP_DIRECT_ENFORCE`` is ON it opens the ledger RW, scans the
  batch, and claims **all** scopes or **none** (all-or-nothing). A live foreign
  owner on ANY scope blocks the whole batch. ANY exception (missing ledger,
  broken schema, lock contention) blocks the tool call — fail-closed, because
  the hook mechanism itself also fails closed on a raise/timeout.
* :func:`post_tool_call` — the ``post_tool_call`` callback. When the call
  reported an error (``status='error'`` or an ``error_type``), it releases the
  scopes the pre-hook claimed for that ``tool_call_id``. Fail-open: a release
  failure must never block or corrupt the tool result.

Enforcement flag
----------------
``HERMES_OWNERSHIP_DIRECT_ENFORCE`` — **default OFF**. Truthy values are
``1/true/yes/on`` (case-insensitive); anything else (including unset) is OFF.
While OFF the pre-hook is a no-op and never touches the ledger, so the plugin is
safe to ship bundled (``kind: backend`` auto-load) before the flag is enabled.

State
-----
Module-level ``_CLAIMED`` (``tool_call_id`` -> claim records) guarded by a
``threading.Lock``: parallel tool calls each release exactly their own scopes.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from typing import Any, Optional

from cross_session_ownership import (
    VERDICT_OWNED,
    OwnershipError,
    claim_for_dispatch,
    claim_ownership,
    github_scope,
    init_schema,
    inspect_ownership,
    release_ownership,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_REPO_NAME",
    "DEFAULT_REPO_OWNER",
    "claimed_scopes",
    "enforcement_enabled",
    "post_tool_call",
    "pre_tool_call",
    "release_for_dispatch",
    "scan_dispatch_scope",
]


# --------------------------------------------------------------------------
# Scope scan — the tested RFC v2 §2 algorithm, copied from
# tests/gateway/test_cross_mode_integration.py (``_scan_batch`` /
# ``_scopes_from_text``). Do NOT diverge: the integration suite pins this exact
# union semantics (top-level goal+context AND every tasks[].goal/context).
#
# ONE deliberate divergence: a bare ``GH-n`` / ``#n`` ref is no longer
# hard-mapped onto ``DEFAULT_REPO_*``. Its repo comes from the SERVER-SIDE
# dispatch context (``default_repo`` — the card / session repo), never from the
# model-controlled batch / ``tasks[]`` payload. ``DEFAULT_REPO_*`` applies ONLY
# when the server supplies no repo at all. Ref matching itself is unchanged, so
# the set of refs found in a text is exactly the one the integration suite pins.
# --------------------------------------------------------------------------
DEFAULT_REPO_OWNER = "Andrew288369"
DEFAULT_REPO_NAME = "ANDREW-GRP"

_ISSUE_REF_RE = re.compile(r"(?i)\b(?:gh[-_]?(\d+)|#(\d+))\b")


def _normalize_repo(repo: Any) -> Optional[tuple]:
    """Normalize a dispatch-context repo value to ``(owner, name)`` or ``None``.

    Accepts ``"owner/name"``, ``(owner, name)`` / ``[owner, name]``, or a
    mapping with ``owner`` + ``name`` (or ``repo_owner`` + ``repo_name``).
    """
    if repo is None:
        return None
    if isinstance(repo, str):
        owner, sep, name = repo.strip().partition("/")
        return (owner, name) if sep and owner and name else None
    if isinstance(repo, (tuple, list)) and len(repo) == 2:
        owner, name = str(repo[0]).strip(), str(repo[1]).strip()
        return (owner, name) if owner and name else None
    if isinstance(repo, dict):
        owner = repo.get("owner", repo.get("repo_owner"))
        name = repo.get("name", repo.get("repo_name"))
        if owner and name:
            return str(owner).strip(), str(name).strip()
    return None


def _scopes_from_text(text: Any, default_repo: Any = None) -> set:
    """Issue refs in a goal/context string -> canonical ``gh:`` scopes.

    Every ref's repo comes from ``default_repo`` — the SERVER-SIDE dispatch
    context (the card / session repo), as ``"owner/name"`` or ``(owner, name)``.
    ``DEFAULT_REPO_OWNER`` / ``DEFAULT_REPO_NAME`` apply ONLY when the server
    supplies no repo at all. Model-controlled payload fields (batch ``repo`` /
    ``tasks[].source_repo`` / ...) are deliberately NOT consulted: a crafted
    dispatch must not be able to choose its own ledger key.

    Previously the repo was hard-mapped to ``DEFAULT_REPO_*``, so a dispatch
    about another repo was claimed on the default repo's issue — the wrong
    ledger key, producing a false conflict or a false free. Ref matching is
    unchanged: the same refs are found, only their repo is now derived.
    """
    owner, name = _normalize_repo(default_repo) or (DEFAULT_REPO_OWNER, DEFAULT_REPO_NAME)
    out = set()
    for m in _ISSUE_REF_RE.finditer(text or ""):
        number = m.group(1) or m.group(2)
        if number:
            out.add(github_scope(owner, name, int(number)))
    return out


def scan_dispatch_scope(
    batch: Optional[dict],
    *,
    include_tasks: bool = True,
    default_repo: Any = None,
) -> list:
    """The UNION of top-level ``goal`` + ``context`` + every ``tasks[].goal/context``.

    ``tasks`` may be a JSON-array string (small models emit the batch that way):
    it is parsed with the SAME normalizer ``delegate_task`` uses before the
    scan. Without this the string is iterated char-by-char, no element is a
    dict, and every ``tasks[]`` scope is silently skipped (0 scopes -> the
    pre-hook admits an unowned batch).

    ``include_tasks=False`` reproduces the v1 defect (F2): a conflicting issue
    hidden inside ``tasks[]`` is never inspected. Production callers must leave
    it at the default ``True``.

    The repo for a bare ref comes ONLY from the explicit ``default_repo`` (the
    server-side card / session context) or, when that is absent, from
    ``DEFAULT_REPO_*``. The batch's own repo field and each ``tasks[]`` entry's
    repo field are MODEL-CONTROLLED input and are deliberately ignored: a
    crafted dispatch must not be able to choose the ledger key. Ref matching is
    unchanged from the integration suite's pinned reference.
    """
    batch = batch or {}
    batch_repo = _normalize_repo(default_repo)
    texts = [
        (batch.get("goal") or "", batch_repo),
        (batch.get("context") or "", batch_repo),
    ]
    tasks = batch.get("tasks")
    if isinstance(tasks, str):
        try:
            from tools.delegate_tool_tasks import _recover_tasks_from_json_string
            parsed, _error = _recover_tasks_from_json_string(tasks)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            tasks = parsed
    if include_tasks:
        for task in tasks or []:
            if isinstance(task, dict):
                texts.append((task.get("goal") or "", batch_repo))
                texts.append((task.get("context") or "", batch_repo))
    scopes: set = set()
    for text, repo in texts:
        scopes |= _scopes_from_text(text, repo)
    return sorted(scopes)


# --------------------------------------------------------------------------
# Ledger access
# --------------------------------------------------------------------------
def enforcement_enabled() -> bool:
    """``HERMES_OWNERSHIP_DIRECT_ENFORCE`` — default OFF (safe to ship bundled)."""
    return os.environ.get("HERMES_OWNERSHIP_DIRECT_ENFORCE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ledger_path():
    """The Kanban DB path (``HERMES_KANBAN_DB`` wins) — same resolver the gate uses."""
    from hermes_cli import kanban_db as kb

    return kb.kanban_db_path()


def _open_ledger_rw() -> sqlite3.Connection:
    """Open the ownership ledger read-write and ensure its schema exists.

    Used by both the pre-hook claim and the post-hook release. Any failure
    (missing path, unreadable file, broken schema) propagates: the caller turns
    it into a block (pre) or a logged no-op (post).
    """
    conn = sqlite3.connect(str(_ledger_path()), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        init_schema(conn)
    except Exception as exc:  # a broken pre-existing table must not look like an empty ledger
        conn.close()
        raise OwnershipError(f"ownership ledger schema unavailable: {exc}") from exc
    return conn


def release_for_dispatch(
    conn: Optional[sqlite3.Connection],
    scope: str,
    session_id: str,
    *,
    reason: Optional[str] = None,
) -> dict:
    """Release one scope this dispatch claimed (owner-only).

    ``conn=None`` opens the ledger RW itself. Returns the ledger result dict and
    never raises for an ordinary non-ownership (``no_owner`` / ``not_owner``).
    """
    owns_conn = conn is None
    if owns_conn:
        conn = _open_ledger_rw()
    try:
        return release_ownership(conn, scope, session_id, reason=reason)
    finally:
        if owns_conn:
            conn.close()


# --------------------------------------------------------------------------
# pre_tool_call — claim all scopes or block (fail-closed)
# --------------------------------------------------------------------------
# Tool calls that can carry a dispatch batch. Kept explicit so an unrelated
# tool with a ``goal``-shaped arg is never gated.
_DISPATCH_TOOLS = frozenset({
    "delegate_task", "delegate", "dispatch_task", "kanban_dispatch",
    "kanban_dispatch_batch", "run_task", "subagent_dispatch", "task_dispatch",
})

# tool_call_id -> {"session_id": str, "scopes": [scope, ...]}
_CLAIMED: dict[str, dict] = {}
_LOCK = threading.Lock()


def _extract_batch(args: Optional[dict]) -> Optional[dict]:
    """The dispatch batch inside a tool-call argument dict, or ``None``.

    Accepts either the batch itself (``goal``/``context``/``tasks`` at top level)
    or the common wrapped forms (``batch=`` / ``dispatch=`` / ``request=``).
    """
    if not isinstance(args, dict):
        return None
    if any(k in args for k in ("goal", "context", "tasks")):
        return args
    for key in ("batch", "dispatch", "request"):
        inner = args.get(key)
        if isinstance(inner, dict) and any(k in inner for k in ("goal", "context", "tasks")):
            return inner
    return None


def _verify_ambient_session(session_id: str) -> bool:
    """Verify the claimed ``session_id`` against the ambient session identity.

    Returns True when the ambient ``HERMES_SESSION_ID`` (ContextVar-aware, so
    an in-process binding wins over ``os.environ``) is bound and equals
    ``session_id``.

    PERMISSIVE FALLBACK (mandatory): when no ambient session is bound (env
    unset/empty — tests and non-session callers), returns True. Fail-closed
    here would break every hook call that does not carry ``HERMES_SESSION_ID``
    (the hook is invoked by callers that have no session context at all).
    """
    from gateway.session_context import get_session_env
    ambient = (get_session_env('HERMES_SESSION_ID') or '').strip()
    if not ambient:
        return True
    return ambient == session_id


def _claim_all_or_block(session_id: str, scopes: list) -> Optional[dict]:
    """Claim every scope in one transaction, or return a block directive.

    All-or-nothing: a conflict on ANY scope blocks before a single write, and a
    mid-claim failure rolls the whole transaction back (no partial claim).
    Raises on any ledger error — the caller turns that into a block.
    """
    conn = _open_ledger_rw()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conflicts = []
            for scope in scopes:
                ins = inspect_ownership(conn, scope)
                if ins["verdict"] == VERDICT_OWNED and ins["owner"] != session_id:
                    conflicts.append({"scope": scope, "owner": ins["owner"]})
            if conflicts:
                conn.execute("ROLLBACK")
                return {
                    "action": "block",
                    "message": (
                        "BLOCKED: dispatch scope conflict — "
                        + ", ".join(f"{c['scope']} owned by {c['owner']}" for c in conflicts)
                    ),
                }
            for scope in scopes:
                res = claim_ownership(conn, scope, session_id, join_txn=True,
                                      verify_session=_verify_ambient_session)
                if not res["ok"]:
                    conn.execute("ROLLBACK")
                    return {
                        "action": "block",
                        "message": (
                            f"BLOCKED: dispatch claim refused for {scope}: "
                            f"{res.get('reason')} (owner={res.get('owner')})"
                        ),
                    }
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        return None
    finally:
        conn.close()


def _record_claim(tool_call_id: str, session_id: str, scopes: list) -> None:
    if not tool_call_id:
        return
    with _LOCK:
        _CLAIMED[tool_call_id] = {"session_id": session_id, "scopes": list(scopes)}


def claimed_scopes(tool_call_id: str) -> list:
    """Scopes recorded for a ``tool_call_id`` (0 if none / already released)."""
    with _LOCK:
        rec = _CLAIMED.get(tool_call_id)
        return list(rec["scopes"]) if rec else []


def pre_tool_call(
    tool_name: str = "",
    args: Optional[dict] = None,
    session_id: str = "",
    task_id: str = "",
    tool_call_id: str = "",
    default_repo: Any = None,
    **_: Any,
) -> Optional[dict]:
    """``pre_tool_call`` hook — claim the dispatch scope(s) or block.

    Returns ``{"action": "block", "message": ...}`` to veto, or ``None`` to
    proceed. NEVER raises: any ledger/claim failure is converted into a block
    directive (fail-closed), because the hook mechanism also fails closed on a
    raise or timeout.

    ``default_repo`` is the SERVER-SIDE dispatch-context repo (task / card /
    session) as ``"owner/name"`` or ``(owner, name)``. It is used for bare
    ``GH-n`` / ``#n`` refs, so a dispatch about another repo is claimed on THAT
    repo's issue instead of being assumed onto ``DEFAULT_REPO_*``. It is the
    ONLY repo source: a repo carried on the tool-call args / batch /
    ``tasks[]`` entry is model-controlled input and is deliberately ignored, so
    a crafted dispatch cannot choose the ledger key.
    """
    if not enforcement_enabled():
        return None
    if not session_id or not tool_call_id:
        return None
    if tool_name and tool_name not in _DISPATCH_TOOLS:
        return None
    batch = _extract_batch(args)
    if batch is None:
        return None
    ctx_repo = _normalize_repo(default_repo)
    scopes = scan_dispatch_scope(batch, default_repo=ctx_repo)
    if not scopes:
        return None
    try:
        block = _claim_all_or_block(session_id, scopes)
    except Exception as exc:
        logger.warning("ownership_gate pre_tool_call claim failed (fail-closed): %s", exc)
        return {
            "action": "block",
            "message": (
                f"BLOCKED: ownership ledger unavailable for dispatch "
                f"({type(exc).__name__}: {str(exc)[:160]})"
            ),
        }
    if block is not None:
        return block
    _record_claim(tool_call_id, session_id, scopes)
    return None


# --------------------------------------------------------------------------
# post_tool_call — release on failure (fail-open)
# --------------------------------------------------------------------------
def post_tool_call(
    tool_name: str = "",
    args: Optional[dict] = None,
    session_id: str = "",
    task_id: str = "",
    tool_call_id: str = "",
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    **_: Any,
) -> None:
    """``post_tool_call`` hook — release this call's scopes when it failed.

    Fail-open: a release error is logged and swallowed, never raised (post hooks
    fail open by design, and a release must not corrupt the tool result).
    """
    if not enforcement_enabled() or not tool_call_id:
        return
    with _LOCK:
        rec = _CLAIMED.get(tool_call_id)
    if not rec:
        return
    failed = (status == "error") or bool(error_type)
    if not failed:
        # The dispatch succeeded: the claim stays outstanding (it is the live
        # owner record for that scope), so keep the tracking entry.
        return
    with _LOCK:
        _CLAIMED.pop(tool_call_id, None)
    reason = f"tool_failed:{error_type or status or 'error'}"
    for scope in rec["scopes"]:
        try:
            release_for_dispatch(None, scope, rec["session_id"], reason=reason)
        except Exception as exc:  # fail-open: never let a release break the call
            logger.debug("ownership_gate post_tool_call release failed: %s", exc)


def reset_state() -> None:
    """Test helper — clear the module-level claim ledger."""
    with _LOCK:
        _CLAIMED.clear()
