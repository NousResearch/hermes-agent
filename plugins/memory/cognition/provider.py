"""Cognition memory provider — session-end durable memory with fetch-first merge.

Single-active provider with internal multi-repo routing: exactly one provider
named ``cognition`` is active at a time (``memory.provider=cognition``), and
facts are routed internally to the configured ``repos`` by ``origin_repo``
instead of spreading across competing providers.

Lifecycle: turns accumulate in memory; :meth:`on_session_end` extracts durable
facts and enqueues them on the durable push queue (SQLite FIFO). Pushing is
fixed to session end — ``sync_turn`` intentionally does no persistence. A
background-free opportunistic drain moves due queue entries into the commit
log via fetch-first merge (keep-both on conflict). ``Pending`` surfaced in the
status tool and prompt always means unpushed (in queue, not yet committed).

Dream is read-only: ``cognition_dream`` synthesizes committed state and
surfaces conflicts, and no write-back path exists by design (non-goal).
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .budget import DEFAULT_BUDGET_TOKENS, apply_budget
from .merge import CommitLog, committed_snapshot, fetch_first_push
from .provenance import archive_entry, make_provenance, validate_provenance
from .queue import DEFAULT_MAX_ENTRIES, DurablePushQueue

logger = logging.getLogger(__name__)

PROVIDER_NAME = "cognition"

# Human-readable memory mirrors: when present under the profile home, their
# content is pinned into recall ahead of everything else.
MEMORY_MIRRORS = ("MEMORY.md", "USER.md")

# Heuristic lasting-statement signals for session-end extraction. Deliberately
# narrow: session-end push is fixed and automatic, so extraction must prefer
# missing a fact over storing chatter.
_EXTRACT_PATTERNS = (
    re.compile(r"\bI\s+(prefer|like|love|use|want|need|always|never|usually)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(favorite|preferred|default)\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(decided|agreed|chose|will)\b", re.IGNORECASE),
    re.compile(r"\bremember\s+that\b", re.IGNORECASE),
    re.compile(r"\bthe\s+project\s+(uses|needs|requires|depends\s+on)\b", re.IGNORECASE),
)

_TOOL_SCHEMAS = [
    {
        "name": "cognition_search",
        "description": (
            "Search committed cognition memories (facts pushed at past session ends). "
            "Use before answering anything that may depend on prior context — "
            "preferences, decisions, project facts. Searches committed state only; "
            "Pending (unpushed) queue entries are not searched."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {"type": "integer", "description": "Max results (default 10)."},
                "repo": {"type": "string", "description": "Restrict to one routed repo."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cognition_status",
        "description": (
            "Show the cognition push-queue backlog. Pending means unpushed: facts "
            "extracted at session end that have not committed yet — plus retry and "
            "tombstone counts when pushes fail."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "cognition_dream",
        "description": (
            "Read-only synthesis of committed memories: a digest of what is known, "
            "including unresolved keep-both conflicts to surface. Dream never writes — "
            "it reads committed state only and cannot push, edit, or delete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Restrict the digest to one routed repo."},
            },
        },
    },
]

_DEFAULTS = {
    "repos": ["default"],
    "budget_tokens": DEFAULT_BUDGET_TOKENS,
    "queue_max_entries": DEFAULT_MAX_ENTRIES,
    "queue_path": "cognition/queue.db",
    "commit_log_path": "cognition/committed.json",
    "archive_path": "cognition/archive.jsonl",
    "author": "",
}


def _load_config(hermes_home: str = "") -> Dict[str, Any]:
    """Layered config: defaults < ``memory.cognition`` block < sidecar file.

    Key names are DRAFT (``memory.cognition={repos, budget_tokens, queue,
    archive}``); confirm against ``hermes plugins doctor`` schema output.
    """
    config: Dict[str, Any] = dict(_DEFAULTS)
    try:  # canonical config block; absent outside a configured home
        from hermes_cli.config import load_config_readonly  # lazy: heavy chain
        from hermes_cli.config import cfg_get

        block = cfg_get(load_config_readonly(), "memory", "cognition", default={}) or {}
        if isinstance(block, dict):
            nested = dict(block)
            queue = nested.pop("queue", None)
            if isinstance(queue, dict):
                if "path" in queue:
                    nested["queue_path"] = queue["path"]
                if "max_entries" in queue:
                    nested["queue_max_entries"] = queue["max_entries"]
            archive = nested.pop("archive", None)
            if isinstance(archive, dict) and "path" in archive:
                nested["archive_path"] = archive["path"]
            config.update({k: v for k, v in nested.items() if v not in (None, "")})
    except Exception:
        pass
    if hermes_home:
        try:
            with open(os.path.join(hermes_home, "cognition.json"), "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            if isinstance(sidecar, dict):
                config.update({k: v for k, v in sidecar.items() if v not in (None, "")})
        except (OSError, ValueError):
            pass
    repos = config.get("repos", ["default"])
    if isinstance(repos, str):
        repos = [r.strip() for r in repos.split(",") if r.strip()]
    config["repos"] = repos or ["default"]
    return config


def _host_git_author() -> str:
    """Host ``git config user.name`` fallback (reads ~/.gitconfig directly)."""
    try:
        parser = configparser.ConfigParser()
        parser.read(os.path.expanduser("~/.gitconfig"))
        return (parser.get("user", "name", fallback="") or "").strip()
    except Exception:
        return ""


def _resolve_repos(value: Any) -> List[str]:
    if isinstance(value, str):
        return [r.strip() for r in value.split(",") if r.strip()] or ["default"]
    if isinstance(value, (list, tuple)):
        return [str(r) for r in value if str(r).strip()] or ["default"]
    return ["default"]


class CognitionMemoryProvider(MemoryProvider):
    """Session-end durable memory with fetch-first merge and budgeted recall."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._overrides = dict(config) if config else {}
        self._config = dict(_DEFAULTS)
        self._config.update(self._overrides)
        self._session_id = ""
        self._hermes_home = ""
        self._queue: Optional[DurablePushQueue] = None
        self._log: Optional[CommitLog] = None
        self._author = ""
        self._repos = ["default"]
        self._budget_tokens = DEFAULT_BUDGET_TOKENS

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        # Local-first SQLite/JSON state: always available, no network or creds.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._config = _load_config(self._hermes_home)
        self._config.update(self._overrides)  # explicit construction wins over disk
        self._repos = _resolve_repos(self._config.get("repos"))
        try:
            self._budget_tokens = int(self._config.get("budget_tokens", DEFAULT_BUDGET_TOKENS))
        except (TypeError, ValueError):
            self._budget_tokens = DEFAULT_BUDGET_TOKENS
        try:
            queue_max = int(self._config.get("queue_max_entries", DEFAULT_MAX_ENTRIES))
        except (TypeError, ValueError):
            queue_max = DEFAULT_MAX_ENTRIES
        base = self._hermes_home or "."
        self._queue = DurablePushQueue(
            os.path.join(base, str(self._config.get("queue_path", "cognition/queue.db"))),
            max_entries=queue_max,
        )
        self._log = CommitLog(
            os.path.join(base, str(self._config.get("commit_log_path", "cognition/committed.json")))
        )
        # Author identity: configured author is authoritative; host git config
        # is the fallback (setup panel documents this precedence).
        configured = str(self._config.get("author") or "").strip()
        self._author = configured or _host_git_author() or "unknown"

    def unavailable_reason(self) -> str:
        return ""

    def identity_signature(self) -> Dict[str, Any]:
        return {"author": self._author, "repos": list(self._repos)}

    # -- recall ----------------------------------------------------------

    def _mirrors(self) -> List[str]:
        """Human-readable memory mirrors pinned ahead of extracted facts."""
        blocks = []
        for filename in MEMORY_MIRRORS:
            path = os.path.join(self._hermes_home, filename) if self._hermes_home else filename
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read().strip()
            except OSError:
                continue
            if content:
                blocks.append("## %s (pinned)\n%s" % (filename, content))
        return blocks

    def _drain_due(self) -> None:
        """Push every due queue entry through fetch-first merge (best-effort)."""
        if self._queue is None or self._log is None:
            return
        for item in self._queue.due():
            try:
                payload = dict(item["payload"])
                entry = dict(payload)
                entry["ref"] = item["ref"]
                fetch_first_push(self._log, entry, payload.get("base"))
                self._queue.ack(item["ref"])
            except Exception as exc:  # retry later; never break recall on push
                logger.warning("Cognition push failed for %s: %s", item["ref"], exc)
                try:
                    self._queue.fail(item["ref"])
                except Exception:
                    pass

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._log is None:
            return ""
        self._drain_due()  # commit what is due so recall sees fresh state
        snapshot = committed_snapshot(self._log)
        text, kept, dropped = apply_budget(snapshot, self._budget_tokens)
        blocks = list(self._mirrors())
        if kept:
            blocks.append("## Cognition Memory\n" + text)
        pending = self._queue.pending_count() if self._queue else 0
        if pending:
            blocks.append("[cognition: %d Pending (unpushed) entr%s in the session-end queue]"
                          % (pending, "y" if pending == 1 else "ies"))
        _ = (query, session_id, dropped)  # query shapes future ranking; unused today
        return "\n".join(blocks)

    def system_prompt_block(self) -> str:
        lines = [
            "# Cognition Memory",
            "Active. Single provider; facts route internally to repos: %s." % ", ".join(self._repos),
            "Push is automatic at session end — there is no per-turn save.",
            "Pending means unpushed: extracted at session end, not yet committed.",
            "Recall blends %s with committed facts under a %d-token prefetch budget."
            % (" + ".join(MEMORY_MIRRORS), self._budget_tokens),
            "Dream (cognition_dream) is read-only: it synthesizes committed state and "
            "surfaces conflicts; it never writes back.",
        ]
        return "\n".join(lines)

    # -- session-end push (fixed; sync_turn persists nothing) -------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", **kwargs) -> None:
        """No-op by design: cognition pushes once at session end, never per turn."""
        return None

    def extract_facts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Heuristic lasting-statement extraction from a session transcript."""
        facts = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            text = content.strip()
            if len(text) < 10 or len(text) > 2000:
                continue
            if any(pattern.search(text) for pattern in _EXTRACT_PATTERNS):
                facts.append(text[:400])
        return facts

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract durable facts and enqueue them for push (durable first)."""
        if self._queue is None:
            return
        session = self._session_id or "session"
        repo = self._repos[0]
        for fact in self.extract_facts(messages or []):
            payload = {
                "content": fact,
                "base": None,  # fresh ref: nothing committed under it yet
                "salience": 0.5,
                "provenance": make_provenance(
                    source_session=session,
                    author=self._author,
                    origin_repo=repo,
                    trace_id=uuid.uuid4().hex,
                ),
            }
            try:
                validate_provenance(payload)
                self._queue.enqueue(payload, session_id=session)
            except Exception as exc:
                logger.warning("Cognition dropped an unprovable fact: %s", exc)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [dict(schema) for schema in _TOOL_SCHEMAS]

    def _tool_search(self, args: Dict[str, Any]) -> str:
        if self._log is None:
            return json.dumps({"error": "cognition is not initialized"})
        query = str(args.get("query", "")).strip().lower()
        try:
            top_k = max(1, min(int(args.get("top_k", 10)), 50))
        except (TypeError, ValueError):
            top_k = 10
        repo = args.get("repo")
        snapshot = committed_snapshot(self._log)
        if repo:
            snapshot = [e for e in snapshot
                        if isinstance(e.get("provenance"), dict)
                        and e["provenance"].get("origin_repo") == repo]
        scored = []
        for entry in snapshot:
            content = str(entry.get("content", "")).lower()
            if query and query not in content:
                continue
            scored.append(entry)
        items = [
            {"ref": e.get("ref"), "content": e.get("content"),
             "repo": (e.get("provenance") or {}).get("origin_repo")}
            for e in scored[:top_k]
        ]
        if not items:
            return json.dumps({"result": "No relevant memories found."})
        return json.dumps({"results": items, "count": len(items)})

    def _tool_status(self, args: Dict[str, Any]) -> str:
        _ = args
        if self._queue is None or self._log is None:
            return json.dumps({"error": "cognition is not initialized"})
        summary = self._queue.backlog()
        return json.dumps({
            "pending_unpushed": summary["pending"],
            "oldest_pending_age_secs": round(summary["oldest_age_secs"], 1),
            "next_retry_in_secs": round(summary["next_due_in_secs"], 1),
            "tombstones": summary["tombstones"],
            "committed": len(self._log.refs()),
            "repos": list(self._repos),
        })

    def _tool_dream(self, args: Dict[str, Any]) -> str:
        """Read-only synthesis over committed state (no write path exists)."""
        if self._log is None:
            return json.dumps({"error": "cognition is not initialized"})
        repo = args.get("repo")
        snapshot = committed_snapshot(self._log)
        if repo:
            snapshot = [e for e in snapshot
                        if isinstance(e.get("provenance"), dict)
                        and e["provenance"].get("origin_repo") == repo]
        conflicts = [e for e in snapshot if "-conflict-" in str(e.get("ref", ""))]
        digest_lines = ["# Dream digest (read-only — committed state only)"]
        for entry in sorted(snapshot, key=lambda e: float(e.get("salience", 0.0)), reverse=True)[:20]:
            digest_lines.append("- [%s] %s" % (entry.get("ref"), entry.get("content")))
        if conflicts:
            digest_lines.append("## Unresolved conflicts (keep-both — needs a human choice)")
            for entry in conflicts:
                digest_lines.append("- %s keeps both %s and %s"
                                    % (entry.get("ref"), entry.get("conflict_with"),
                                       entry.get("ref")))
        if len(snapshot) > 20:
            digest_lines.append("[dream: %d further committed entries omitted]" % (len(snapshot) - 20))
        return json.dumps({"digest": "\n".join(digest_lines),
                           "committed": len(snapshot),
                           "conflicts": len(conflicts)})

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handlers = {
            "cognition_search": self._tool_search,
            "cognition_status": self._tool_status,
            "cognition_dream": self._tool_dream,
        }
        if tool_name not in handlers:
            return json.dumps({"error": "Unknown tool: %s" % tool_name})
        try:
            return handlers[tool_name](args or {})
        except Exception as exc:
            return json.dumps({"error": "cognition_%s failed: %s"
                               % (tool_name, exc)})

    # -- setup / lifecycle ---------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "repos", "description": "Repo names routed inside this one provider (comma-separated)",
             "required": False, "default": "default"},
            {"key": "budget_tokens", "description": "Prefetch token budget (default 8000)",
             "required": False, "default": "8000", "type": "integer", "minimum": 1000, "maximum": 64000},
            {"key": "queue_max_entries", "description": "Pending-push bound before oldest is dropped with a tombstone",
             "required": False, "default": "1000", "type": "integer", "minimum": 10, "maximum": 100000},
            {"key": "archive_path", "description": "Eviction-preserving archive file (JSONL)",
             "required": False, "default": "cognition/archive.jsonl"},
            {"key": "author", "description": "Author identity stamped on provenance (authoritative; falls back to host git config user.name)",
             "required": False, "default": ""},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Merge-write non-secret setup values to $HERMES_HOME/cognition.json."""
        path = os.path.join(hermes_home, "cognition.json")
        current: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, ValueError):
            pass
        current.update(values)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def archive_and_evict(self, ref: str, reason: str) -> bool:
        """Archive a committed entry, then evict it (archive always first)."""
        if self._log is None:
            return False
        base = self._hermes_home or "."
        archive_path = os.path.join(
            base, str(self._config.get("archive_path", "cognition/archive.jsonl")))
        entry = self._log.get(ref)
        if entry is None:
            return False
        archive_entry(archive_path, entry, reason=reason)
        self._log.remove(ref)
        return True

    def backup_paths(self) -> List[str]:
        base = self._hermes_home or "."
        return [os.path.join(base, "cognition")]

    def shutdown(self) -> None:
        try:
            self._drain_due()
        finally:
            if self._queue is not None:
                self._queue.close()
                self._queue = None


def register(ctx) -> None:
    """Register cognition as a memory provider plugin (single-active)."""
    ctx.register_memory_provider(CognitionMemoryProvider())
