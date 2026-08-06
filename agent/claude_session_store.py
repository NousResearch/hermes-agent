"""``SessionStore`` adapter: the Claude Agent SDK's transcript mirror → Hermes' DB.

The SDK's ``SessionStore`` protocol is how an embedder gets a durable copy of
the Claude-native transcript the CLI subprocess writes.  Without it a Claude
session lives and dies with ``~/.claude/projects/<key>/<uuid>.jsonl`` on one
machine; with it, Hermes owns a copy in ``state.db`` and can resume, rewind,
and branch a Claude conversation the same way it does for every other
provider.

Two boundaries this module is careful about (see
``docs/design/claude-subscription-via-agent-sdk.md``):

* **Entries are opaque.**  Nothing here reads inside a transcript entry except
  to pull ``uuid`` for idempotency.  The shape is the CLI's internal JSONL
  union; interpreting it would be Hermes reimplementing Claude's context, and
  the SDK's own contract is only that entries round-trip *deep-equal*, not
  byte-equal.
* **Summaries are folded by the SDK, not by us.**  ``fold_session_summary`` is
  a pure function the SDK exports precisely so adapters do not have to know
  what a summary contains.  We call it inside the DB's write transaction so
  concurrent appends cannot interleave read-fold-write on the sidecar, and we
  stamp ``mtime`` afterwards because the fold deliberately never does.

``SessionDB`` is synchronous SQLite and the SDK calls the store from an async
context, so every method hops through :func:`asyncio.to_thread`.  No
connection is opened here: the adapter reuses the ``SessionDB`` instance the
agent already holds, whose writer connection is lock-guarded and whose readers
are thread-local and tracked.  That is deliberate — this repo has a history of
SQLite fd leaks from per-operation connections (see
``relatorio-issue-69678-sqlite-fd-leaks.md``), and the cheapest way to not
leak a connection is to not open one.

``claude-agent-sdk`` is an optional extra: this module imports cleanly without
it, and only :func:`build_claude_session_store` needs the SDK present (for the
exported ``fold_session_summary``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Discriminator written into every ``provider_*`` row this adapter owns.
#: Matches ``agent.claude_runtime.RUNTIME_LABEL``; duplicated as a literal so
#: importing the store does not drag in the runtime.
RUNTIME = "claude_agent_sdk"

#: Entry types that represent a real conversation message in Claude's JSONL.
#: Used only to find the boundary of a *visible* user turn — never to
#: interpret an entry's payload.


def is_visible_user_entry(entry: Any) -> bool:
    """True when *entry* is the user message that opened a visible turn.

    Claude's JSONL records tool results as ``type="user"`` entries too, and
    marks synthetic lines with ``isMeta`` / ``isCompactSummary``.  Only a
    plain user message with real content corresponds to something the Hermes
    transcript shows, which is the boundary a rewind has to land on.
    """
    if not isinstance(entry, dict) or entry.get("type") != "user":
        return False
    if entry.get("isMeta") is True or entry.get("isCompactSummary") is True:
        return False
    if entry.get("isSidechain") is True:
        return False
    message = entry.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return False
    return True


class HermesClaudeSessionStore:
    """``SessionStore`` backed by Hermes' ``SessionDB``.

    Implements all six protocol methods.  The four "optional" ones are not
    optional in practice for Hermes:

    * ``list_sessions`` / ``list_session_summaries`` back the session picker
      and ``--continue``-style resolution.
    * ``list_subkeys`` is what restores subagent transcripts on resume —
      without it a resumed session silently loses every subagent's history.
    * ``delete`` must cascade to subkeys and drop the summary, or a deleted
      session leaves orphaned subagent transcripts behind forever (the SDK
      never deletes from a store on its own; retention is ours).

    Duck-typed rather than subclassing ``SessionStore``: the SDK explicitly
    never uses ``isinstance`` for this, and not subclassing keeps the module
    importable without the optional extra.
    """

    def __init__(self, session_db: Any, *, summary_fold: Any = None) -> None:
        self._db = session_db
        self._summary_fold = summary_fold

    # -- protocol ----------------------------------------------------------

    async def append(self, key: Dict[str, Any], entries: List[Any]) -> None:
        """Mirror a batch of transcript entries.

        Deduplication by ``entry["uuid"]`` happens in SQLite (a partial unique
        index), because a failed batch is retried up to twice more and a retry
        can re-deliver entries that already landed.  Entries without a uuid
        are appended unconditionally — the CLI emits uuid-less lines (titles,
        tags, mode markers) that legitimately repeat.
        """
        project_key, session_id, subpath = _split(key)
        fold = None
        if self._summary_fold is not None and not subpath:
            def fold(previous, batch, _key=key):
                return self._summary_fold(previous, _key, batch)

        await asyncio.to_thread(
            self._db.append_provider_transcript_entries,
            RUNTIME,
            project_key,
            session_id,
            subpath,
            list(entries),
            summary_fold=fold,
        )

    async def load(self, key: Dict[str, Any]) -> Optional[List[Any]]:
        project_key, session_id, subpath = _split(key)
        return await asyncio.to_thread(
            self._db.load_provider_transcript_entries,
            RUNTIME,
            project_key,
            session_id,
            subpath,
        )

    async def list_sessions(self, project_key: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._db.list_provider_transcript_sessions, RUNTIME, project_key
        )

    async def list_session_summaries(self, project_key: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._db.list_provider_transcript_summaries, RUNTIME, project_key
        )

    async def delete(self, key: Dict[str, Any]) -> None:
        project_key, session_id, subpath = _split(key)
        await asyncio.to_thread(
            self._db.delete_provider_transcript,
            RUNTIME,
            project_key,
            session_id,
            # None (not "") means "the main key" — which cascades. An explicit
            # subpath deletes only that one transcript.
            subpath or None,
        )

    async def list_subkeys(self, key: Dict[str, Any]) -> List[str]:
        project_key, session_id, _ = _split(key)
        return await asyncio.to_thread(
            self._db.list_provider_transcript_subkeys,
            RUNTIME,
            project_key,
            session_id,
        )


def _split(key: Dict[str, Any]) -> tuple:
    """``SessionKey`` → ``(project_key, session_id, subpath)``.

    ``subpath`` is absent for a main transcript and opaque otherwise — it is
    used purely as a storage key suffix, never parsed.
    """
    return (
        str(key.get("project_key") or ""),
        str(key.get("session_id") or ""),
        str(key.get("subpath") or ""),
    )


def build_claude_session_store(session_db: Any) -> Optional[HermesClaudeSessionStore]:
    """Build the adapter for *session_db*, or None when it cannot be used.

    Returns None when there is no session DB (persistence disabled, a
    background-review fork, a test harness) so the caller can simply omit
    ``session_store`` — a Claude turn without a mirror still works, it just
    is not durable.
    """
    if session_db is None:
        return None
    try:
        from claude_agent_sdk import fold_session_summary
    except Exception:  # pragma: no cover - optional extra absent
        logger.debug("claude_agent_sdk absent; session store disabled", exc_info=True)
        return None
    return HermesClaudeSessionStore(session_db, summary_fold=fold_session_summary)


__all__ = [
    "RUNTIME",
    "HermesClaudeSessionStore",
    "build_claude_session_store",
    "is_visible_user_entry",
]
