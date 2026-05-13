"""YantrikDB memory plugin — self-maintaining memory for Hermes.

YantrikDB is a memory provider that actively maintains what it stores
instead of behaving like an append-only vector index:

- ``think()`` runs a bounded maintenance pass — canonicalizes duplicates,
  flags superseded or contradictory facts, and mines co-occurrence
  patterns.
- Contradiction tracking surfaces conflicting claims for the agent to
  resolve explicitly (``resolve_conflict``) rather than overwriting
  silently.
- Recency-aware ranking prefers fresher facts over stale ones without
  deleting the older records — every result is annotated with a
  ``why_retrieved`` reason list so recall is explainable.
- A knowledge graph promotes related memories in recall through entity
  edges created via ``relate``.

The plugin is a thin HTTP client against an externally managed
``yantrikdb-server`` (see README for deployment). The plugin never
starts, stops, or upgrades the backend — same pattern as ``honcho``.

Config via env + $HERMES_HOME/yantrikdb.json:
  YANTRIKDB_URL              — default http://localhost:7438
  YANTRIKDB_TOKEN            — required, Bearer token from `yantrikdb token create`
  YANTRIKDB_NAMESPACE        — default "hermes"; combined with agent_workspace:agent_identity
  YANTRIKDB_TOP_K            — default 10
  YANTRIKDB_READ_TIMEOUT     — default 15.0 seconds
  YANTRIKDB_CONNECT_TIMEOUT  — default 5.0 seconds
  YANTRIKDB_RETRY_TOTAL      — default 3 retries on transient 5xx
  YANTRIKDB_MAX_TEXT_LEN     — default 25000 chars; text is truncated client-side above this
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
    _HERMES_AVAILABLE = True
except ImportError:
    # Plugin is being imported outside a Hermes runtime — typically because
    # `yantrikdb-hermes-plugin` was pip-installed and the user is invoking
    # the CLI installer (`yantrikdb-hermes install <hermes_root>`). The
    # provider class never runs in this path; we just need module load to
    # succeed so the CLI's `from yantrikdb_hermes_plugin.cli import main`
    # works. Inside Hermes, the real imports take over.
    _HERMES_AVAILABLE = False

    class MemoryProvider:  # type: ignore[no-redef]
        """Stub used only when the plugin is imported outside Hermes.

        Real `agent.memory_provider.MemoryProvider` is an ABC; this stub is
        a plain class so the YantrikDBMemoryProvider subclass below still
        defines successfully when Hermes isn't on the import path. Hermes
        instantiates the provider via filesystem-loaded source (a fresh
        import from `plugins/memory/yantrikdb/`), so this stub is never
        the one Hermes sees in practice.
        """

    def tool_error(message: str) -> str:  # type: ignore[no-redef]
        import json
        return json.dumps({"error": message})

from .client import (
    DEFAULT_NAMESPACE,
    YantrikDBAuthError,
    YantrikDBClient,
    YantrikDBClientError,
    YantrikDBConfig,
    YantrikDBError,
    YantrikDBServerError,
    YantrikDBTransientError,
)
from .embedded import make_backend

logger = logging.getLogger(__name__)

# Circuit breaker — after N consecutive transient/server/auth failures, the
# plugin short-circuits for _BREAKER_COOLDOWN seconds so a flapping server
# does not hammer Hermes' event loop. 4xx errors are deterministic caller
# mistakes and do NOT count toward the breaker. Matches mem0's 5/120s.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN = 120.0

_PREFETCH_JOIN_SECS = 3.0
_SYNC_JOIN_SECS = 5.0
_SESSION_END_JOIN_SECS = 10.0


# ---------------------------------------------------------------------------
# Tool schemas
#
# Framing: these are memory maintenance operations, not opaque cognition.
# Descriptions tell the agent when to reach for each tool, what the tool
# will mutate, and how to read its structured result.
# ---------------------------------------------------------------------------

REMEMBER_SCHEMA = {
    "name": "yantrikdb_remember",
    "description": (
        "Store a durable memory. Use for decisions, preferences, facts about "
        "people, and project context. Skip ephemeral task state and anything "
        "derivable from code or git. Text over 25k characters is truncated "
        "client-side with a visible marker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The fact or decision to remember, as a complete sentence.",
            },
            "importance": {
                "type": "number",
                "description": (
                    "0.0-1.0 resistance to temporal ranking decay. "
                    "0.8+ for critical decisions, 0.5-0.7 for useful context, "
                    "0.3-0.5 for background detail. Default 0.6."
                ),
            },
            "domain": {
                "type": "string",
                "description": (
                    "Optional tag for filtered recall: 'work', 'preference', "
                    "'people', 'architecture', 'infrastructure', etc."
                ),
            },
        },
        "required": ["text"],
    },
}

RECALL_SCHEMA = {
    "name": "yantrikdb_recall",
    "description": (
        "Explainable recall — semantic search ranked by relevance × recency × "
        "importance with knowledge-graph boosting. Each result includes a "
        "`why_retrieved` list (semantic_match / graph-connected / "
        "keyword_match / important / emotionally weighted, etc.) so the "
        "agent can see why each memory ranked. Call before making claims "
        "about the user or past decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query. Be specific.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default 10, capped at 50).",
            },
            "domain": {
                "type": "string",
                "description": "Optional domain filter (e.g. 'work').",
            },
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "yantrikdb_forget",
    "description": (
        "Tombstone (soft-delete) a memory by its rid. Use when the user "
        "retracts a fact or you are resolving a conflict by dropping the "
        "obsolete record. Tombstoned memories stop surfacing in recall but "
        "remain for audit."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rid": {
                "type": "string",
                "description": "Memory id returned by yantrikdb_recall or yantrikdb_remember.",
            },
        },
        "required": ["rid"],
    },
}

THINK_SCHEMA = {
    "name": "yantrikdb_think",
    "description": (
        "Run a bounded memory maintenance pass: canonicalize near-duplicates, "
        "flag contradictory or superseded facts, optionally mine co-occurrence "
        "patterns. Returns structured counts plus a `triggers` list suggesting "
        "follow-up actions. Call at natural break points (end of a project "
        "phase, a long user pause) — it is the most expensive operation and "
        "should not run per turn. This is YantrikDB's differentiating feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "consolidation_limit": {
                "type": "integer",
                "description": "Max memories to consolidate this run (default: server-configured).",
            },
            "run_pattern_mining": {
                "type": "boolean",
                "description": "Also mine temporal and co-occurrence patterns. Default false.",
            },
        },
        "required": [],
    },
}

CONFLICTS_SCHEMA = {
    "name": "yantrikdb_conflicts",
    "description": (
        "List open contradictions detected by yantrikdb_think. Each conflict "
        "includes the two (or more) memory ids, a detection reason, and a "
        "priority. YantrikDB never silently overwrites — contradictions "
        "surface here until the agent resolves them explicitly."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RESOLVE_CONFLICT_SCHEMA = {
    "name": "yantrikdb_resolve_conflict",
    "description": (
        "Resolve a contradiction from yantrikdb_conflicts. Strategies: "
        "'keep_winner' (pick the winner_rid, tombstone the other), "
        "'merge' (emit a merged new_text, tombstone both), "
        "'keep_both' (record both as context-dependent), "
        "'dismiss' (close without action, e.g. false positive)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conflict_id": {"type": "string", "description": "Id from yantrikdb_conflicts."},
            "strategy": {
                "type": "string",
                "description": "One of: keep_winner, merge, keep_both, dismiss.",
            },
            "winner_rid": {
                "type": "string",
                "description": "Required for 'keep_winner'. The rid to preserve.",
            },
            "new_text": {
                "type": "string",
                "description": "Required for 'merge'. The consolidated text that replaces both.",
            },
            "resolution_note": {
                "type": "string",
                "description": "Optional audit note explaining the choice.",
            },
        },
        "required": ["conflict_id", "strategy"],
    },
}

RELATE_SCHEMA = {
    "name": "yantrikdb_relate",
    "description": (
        "Record a relationship edge between two entities in the knowledge "
        "graph (e.g. 'Alice works_at Acme', 'ProjectX uses React'). "
        "Edges boost recall — memories near related entities rank higher "
        "and show `graph-connected via X` in their why_retrieved reasons."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Source entity."},
            "target": {"type": "string", "description": "Target entity."},
            "relationship": {
                "type": "string",
                "description": (
                    "Relationship verb, snake_case by convention "
                    "(e.g. 'works_at', 'uses', 'reports_to')."
                ),
            },
        },
        "required": ["entity", "target", "relationship"],
    },
}

STATS_SCHEMA = {
    "name": "yantrikdb_stats",
    "description": (
        "Operational snapshot: active memory count, tombstoned count, "
        "graph edges, entities, open conflicts, pending triggers. Use to "
        "decide whether to call yantrikdb_think, spot runaway growth, or "
        "see whether conflicts have accumulated."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

# -- Skills (v0.3.0+) -------------------------------------------------
#
# Skills are procedural memory: reusable patterns the agent distills
# from observed success and pulls back next session. They live in the
# shared ``skill_substrate`` namespace alongside skills authored by
# other YantrikDB consumers — Hermes-authored entries are tagged
# ``metadata.source=hermes`` so any consumer can filter Hermes-authored
# skills in or out cleanly. Outcomes are append-only; success/failure
# rollup is the agent's pedagogy decision, not the substrate's.
#
# Note: this is a peer surface to the filesystem skills Hermes already
# has at $HERMES_HOME/skills/. Filesystem = human-authored, durable,
# version-controlled. YantrikDB skills = agent-authored, runtime-
# evolving, semantic-search-queryable. Different kinds of canonical;
# the model picks the right substrate by lifecycle, not by competition.

SKILL_SEARCH_SCHEMA = {
    "name": "yantrikdb_skill_search",
    "description": (
        "Semantic search over agent-authored skills in the shared "
        "skill_substrate. Use to find a procedural pattern relevant to "
        "the current task before acting — agent-authored skills capture "
        "what worked in previous similar situations. Distinct from "
        "yantrikdb_recall (episodic/semantic memory): skills are "
        "procedures, memories are facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of the task or pattern.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default 10).",
            },
            "applies_to": {
                "type": "string",
                "description": (
                    "Optional filter: only return skills tagged for this "
                    "context (e.g. 'git', 'deploy', 'pgsql')."
                ),
            },
        },
        "required": ["query"],
    },
}

SKILL_DEFINE_SCHEMA = {
    "name": "yantrikdb_skill_define",
    "description": (
        "Distill a procedural pattern into a reusable skill. Use when "
        "you've observed a sequence that worked and want it findable next "
        "session. The body should be a self-contained instruction — "
        "future you (or another agent) reads it and follows it. Skill "
        "naming convention: dot-separated lowercase, e.g. "
        "'git.commit_clean', 'deploy.rolling_restart'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": (
                    "Dot-separated lowercase identifier matching "
                    "^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)+$, length 4-200. "
                    "First segment is the broad category, later segments "
                    "narrow it (e.g. 'workflow.git.commit_clean')."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "The procedural instructions, 50-5000 chars. Self-"
                    "contained — assume the reader has no prior context."
                ),
            },
            "skill_type": {
                "type": "string",
                "enum": ["procedure", "reference", "lesson", "pattern", "rule"],
                "description": (
                    "procedure: do these steps. reference: information to "
                    "consult. lesson: things to remember from a past mistake. "
                    "pattern: recognizable shape. rule: invariant to maintain."
                ),
            },
            "applies_to": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tags identifying when this skill is relevant. Lowercase "
                    "+ digits + underscores ONLY (no hyphens, no dots, no "
                    "spaces): each entry must match ^[a-z][a-z0-9_]*$. "
                    "Max 10 entries. Examples: ['git', 'commit'], "
                    "['postgres', 'migration', 'rollback']."
                ),
            },
            "triggers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional natural-language phrases that should activate this skill.",
            },
            "version": {
                "type": "string",
                "description": "Optional version string (e.g. '1.0.0').",
            },
            "supersedes_skill_id": {
                "type": "string",
                "description": "Optional skill_id this skill replaces.",
            },
        },
        "required": ["skill_id", "body", "skill_type", "applies_to"],
    },
}

SKILL_OUTCOME_SCHEMA = {
    "name": "yantrikdb_skill_outcome",
    "description": (
        "Record whether a skill succeeded when used. Append-only: each "
        "call adds an event to the outcome log; success/failure rollup "
        "is the agent's call, not the substrate's. Use after attempting "
        "a skill to feed back signal for future search ranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "The skill_id that was applied.",
            },
            "succeeded": {
                "type": "boolean",
                "description": "True if the skill produced the intended outcome.",
            },
            "note": {
                "type": "string",
                "description": "Optional context — what worked, what didn't, what to adjust.",
            },
        },
        "required": ["skill_id", "succeeded"],
    },
}

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    REMEMBER_SCHEMA,
    RECALL_SCHEMA,
    FORGET_SCHEMA,
    THINK_SCHEMA,
    CONFLICTS_SCHEMA,
    RESOLVE_CONFLICT_SCHEMA,
    RELATE_SCHEMA,
    STATS_SCHEMA,
    SKILL_SEARCH_SCHEMA,
    SKILL_DEFINE_SCHEMA,
    SKILL_OUTCOME_SCHEMA,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_namespace(base: str, kwargs: dict[str, Any]) -> str:
    """Scope the namespace to ``{base}:{agent_workspace}:{agent_identity}``.

    Per HANDOFF §3 — one tenant namespace per agent identity so cross-session
    consolidation works, while preventing a second agent from polluting
    the first agent's memories.
    """
    workspace = (kwargs.get("agent_workspace") or "").strip()
    identity = (kwargs.get("agent_identity") or "").strip()
    parts = [base]
    if workspace:
        parts.append(workspace)
    if identity:
        parts.append(identity)
    return ":".join(parts)


def _format_recall_block(results: list[dict[str, Any]], limit: int = 8) -> str:
    """Markdown bullet list of recalled memories for prompt injection."""
    if not results:
        return ""
    lines: list[str] = []
    for r in results[:limit]:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        score = r.get("score")
        tag = f" _(score {score:.2f})_" if isinstance(score, (int, float)) else ""
        lines.append(f"- {text}{tag}")
    return "\n".join(lines)


def _estimate_importance(text: str) -> float:
    """Cheap heuristic for ambient writes via sync_turn.

    We can't tell a greeting from a decision without an LLM call; err
    conservative. ``think()`` consolidates noise on session end.
    """
    n = len(text.strip())
    if n < 40:
        return 0.3
    if n < 200:
        return 0.5
    return 0.6


def _coerce_float(raw: Any, *, default: float) -> float:
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _coerce_int(raw: Any, default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class YantrikDBMemoryProvider(MemoryProvider):
    """MemoryProvider implementation backed by yantrikdb-server over HTTP."""

    def __init__(self) -> None:
        self._config: YantrikDBConfig | None = None
        self._client: YantrikDBClient | None = None
        self._client_lock = threading.Lock()

        self._namespace: str = DEFAULT_NAMESPACE
        self._session_id: str = ""
        self._cron_skipped: bool = False

        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None

        self._failure_count: int = 0
        self._breaker_open_until: float = 0.0
        self._breaker_lock = threading.Lock()

    # -- Identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "yantrikdb"

    # -- Setup / config ---------------------------------------------------

    def is_available(self) -> bool:
        """Ready when the configured backend can be reached. No network call.

        - embedded mode: available iff `yantrikdb` Python package is importable.
        - http mode: available iff a token is configured.
        """
        cfg = YantrikDBConfig.load()
        if cfg.mode == "embedded":
            try:
                import yantrikdb._yantrikdb_rust  # noqa: F401
                return True
            except ImportError:
                return False
        return bool(cfg.token)

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "token",
                "description": "YantrikDB bearer token (from `yantrikdb token create`).",
                "secret": True,
                "required": True,
                "env_var": "YANTRIKDB_TOKEN",
                "url": "https://yantrikdb.com/server/quickstart/",
            },
            {
                "key": "url",
                "description": "YantrikDB HTTP endpoint.",
                "default": "http://localhost:7438",
                "env_var": "YANTRIKDB_URL",
            },
            {
                "key": "namespace",
                "description": "Tenant namespace prefix (combined with agent_workspace:agent_identity).",
                "default": "hermes",
                "env_var": "YANTRIKDB_NAMESPACE",
            },
            {
                "key": "top_k",
                "description": "Default max results for recall.",
                "default": "10",
                "env_var": "YANTRIKDB_TOP_K",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret config to $HERMES_HOME/yantrikdb.json."""
        path = Path(hermes_home) / "yantrikdb.json"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- Lifecycle --------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")
        if agent_context in ("cron", "flush") or platform == "cron":
            logger.debug(
                "YantrikDB skipped: cron/flush context (agent_context=%s, platform=%s)",
                agent_context, platform,
            )
            self._cron_skipped = True
            return

        self._session_id = session_id or ""
        hermes_home_raw = kwargs.get("hermes_home")
        hermes_home = Path(hermes_home_raw) if hermes_home_raw else None
        self._config = YantrikDBConfig.load(hermes_home)

        # Embedded mode is self-contained (`pip install` and go); HTTP mode
        # requires a token. is_available() short-circuits at the provider
        # level, but be defensive here too.
        if self._config.mode == "http" and not self._config.token:
            logger.debug("YantrikDB http mode but no token — plugin inactive")
            return

        self._namespace = _derive_namespace(self._config.namespace, kwargs)
        try:
            backend = make_backend(self._config)
        except YantrikDBError as e:
            logger.warning("YantrikDB %s mode init failed: %s", self._config.mode, e)
            return
        with self._client_lock:
            self._client = backend

        try:
            self._client.health()
            target = self._config.url if self._config.mode == "http" else (
                self._config.db_path or "default"
            )
            logger.info(
                "YantrikDB connected: mode=%s target=%s namespace=%s",
                self._config.mode, target, self._namespace,
            )
        except YantrikDBError as e:
            logger.warning(
                "YantrikDB health check failed (%s) — will retry on demand.", e,
            )

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=_SYNC_JOIN_SECS)
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _require_client(self) -> YantrikDBClient:
        """Return the client or raise — keeps dispatch paths type-clean."""
        if self._client is None:
            raise RuntimeError("YantrikDB client not initialized")
        return self._client

    # -- Prompt / prefetch -----------------------------------------------

    def system_prompt_block(self) -> str:
        if self._cron_skipped or self._client is None:
            return ""
        return (
            "# YantrikDB Memory\n"
            f"Active. Namespace: `{self._namespace}`.\n"
            "Self-maintaining memory: canonicalizes duplicates, surfaces "
            "contradictions, ranks with recency awareness, and explains recall. "
            "Use `yantrikdb_recall` before claiming facts about the user or "
            "past decisions — each result includes a why_retrieved reason list. "
            "When a new decision or relationship surfaces, call "
            "`yantrikdb_remember` or `yantrikdb_relate`. Run `yantrikdb_think` "
            "at natural break points to consolidate duplicates and surface "
            "contradictions — then `yantrikdb_conflicts` lists what needs "
            "resolving and `yantrikdb_resolve_conflict` closes each out."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._cron_skipped or self._client is None:
            return ""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=_PREFETCH_JOIN_SECS)
        with self._prefetch_lock:
            result, self._prefetch_result = self._prefetch_result, ""
        if not result:
            return ""
        return f"## YantrikDB Recall\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._cron_skipped or self._client is None or not query:
            return
        if self._breaker_open():
            return

        client = self._client
        namespace = self._namespace

        def _run() -> None:
            try:
                resp = client.recall(query, namespace=namespace, top_k=5)
                block = _format_recall_block(resp.get("results", []), limit=5)
                if block:
                    with self._prefetch_lock:
                        self._prefetch_result = block
                self._record_success()
            except YantrikDBClientError as e:
                logger.debug("YantrikDB prefetch rejected: %s", e)
            except YantrikDBError as e:
                self._record_failure()
                logger.debug("YantrikDB prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="yantrikdb-prefetch",
        )
        self._prefetch_thread.start()

    # -- Turn sync --------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Persist the user message after a completed turn.

        Assistant-message extraction is intentionally out of v1 scope
        (HANDOFF §10.1) — storing LLM output as fact amplifies
        hallucination. think() cleans up ambient noise at session end.
        """
        if self._cron_skipped or self._client is None or self._config is None:
            return
        if not self._config.sync_user_messages:
            return
        if self._breaker_open():
            return
        text = (user_content or "").strip()
        if not text:
            return

        client = self._client
        snapshot_sid = self._session_id or session_id
        namespace = self._namespace

        def _run() -> None:
            try:
                client.remember(
                    text,
                    namespace=namespace,
                    importance=_estimate_importance(text),
                    metadata={"session_id": snapshot_sid, "role": "user"},
                )
                self._record_success()
            except YantrikDBClientError as e:
                logger.debug("YantrikDB sync_turn rejected: %s", e)
            except YantrikDBError as e:
                self._record_failure()
                logger.debug("YantrikDB sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=_SYNC_JOIN_SECS)
        self._sync_thread = threading.Thread(
            target=_run, daemon=True, name="yantrikdb-sync",
        )
        self._sync_thread.start()

    # -- Tool dispatch ----------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        # Called by Hermes at register-time (before initialize()) to index
        # tool names → provider for routing. Must return the static schema
        # list regardless of client state; runtime readiness is enforced
        # in handle_tool_call().
        #
        # Skills are opt-in (YANTRIKDB_SKILLS_ENABLED=true). When the flag
        # is off, the three skill schemas are filtered out so the model
        # never sees them — keeping the tool surface minimal for users who
        # don't need the agentic skill loop. When `_config` is None (we
        # haven't initialized yet), expose all schemas so register-time
        # routing has every tool indexed; runtime check below short-
        # circuits if skills are disabled at call time.
        if self._cron_skipped:
            return []
        skills_enabled = bool(
            self._config.skills_enabled if self._config else
            YantrikDBConfig.load().skills_enabled
        )
        if skills_enabled:
            return list(ALL_TOOL_SCHEMAS)
        return [s for s in ALL_TOOL_SCHEMAS if not s["name"].startswith("yantrikdb_skill_")]

    def handle_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        if self._cron_skipped or self._client is None:
            return tool_error("YantrikDB is not active for this session.")
        if self._breaker_open():
            return tool_error(
                "YantrikDB temporarily unavailable (circuit breaker open). "
                "Will retry automatically."
            )

        try:
            if tool_name == "yantrikdb_remember":
                return self._do_remember(args)
            if tool_name == "yantrikdb_recall":
                return self._do_recall(args)
            if tool_name == "yantrikdb_forget":
                return self._do_forget(args)
            if tool_name == "yantrikdb_think":
                return self._do_think(args)
            if tool_name == "yantrikdb_conflicts":
                return self._do_conflicts()
            if tool_name == "yantrikdb_resolve_conflict":
                return self._do_resolve_conflict(args)
            if tool_name == "yantrikdb_relate":
                return self._do_relate(args)
            if tool_name == "yantrikdb_stats":
                return self._do_stats()
            if tool_name.startswith("yantrikdb_skill_"):
                if not (self._config and self._config.skills_enabled):
                    return tool_error(
                        "Skills are disabled. Set YANTRIKDB_SKILLS_ENABLED=true "
                        "to enable yantrikdb_skill_search / _define / _outcome."
                    )
                if tool_name == "yantrikdb_skill_search":
                    return self._do_skill_search(args)
                if tool_name == "yantrikdb_skill_define":
                    return self._do_skill_define(args)
                if tool_name == "yantrikdb_skill_outcome":
                    return self._do_skill_outcome(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except YantrikDBAuthError as e:
            self._record_failure()
            return tool_error(
                f"YantrikDB auth rejected: {e}. Check YANTRIKDB_TOKEN."
            )
        except YantrikDBClientError as e:
            return tool_error(f"YantrikDB rejected the request: {e}")
        except (YantrikDBTransientError, YantrikDBServerError) as e:
            self._record_failure()
            return tool_error(f"YantrikDB unavailable: {e}")
        except YantrikDBError as e:
            self._record_failure()
            return tool_error(f"YantrikDB error: {e}")

    def _do_remember(self, args: dict[str, Any]) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            return tool_error("Missing required parameter: text")
        importance = _coerce_float(args.get("importance"), default=0.6)
        resp = self._require_client().remember(
            text,
            namespace=self._namespace,
            importance=importance,
            domain=args.get("domain"),
            metadata={"session_id": self._session_id},
        )
        self._record_success()
        return json.dumps({"rid": resp.get("rid"), "stored": True})

    def _do_recall(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        default_top_k = self._config.top_k if self._config else 10
        top_k = min(_coerce_int(args.get("top_k"), default_top_k), 50)
        resp = self._require_client().recall(
            query,
            namespace=self._namespace,
            top_k=top_k,
            domain=args.get("domain"),
        )
        self._record_success()
        results = resp.get("results", []) or []
        compact = [
            {
                "rid": r.get("rid"),
                "text": r.get("text"),
                "score": r.get("score"),
                "importance": r.get("importance"),
                "domain": r.get("domain"),
                "created_at": r.get("created_at"),
                # Explainable recall — server returns a list of reasons per result.
                "why_retrieved": r.get("why_retrieved") or [],
            }
            for r in results
        ]
        return json.dumps({"count": len(compact), "results": compact})

    def _do_forget(self, args: dict[str, Any]) -> str:
        rid = (args.get("rid") or "").strip()
        if not rid:
            return tool_error("Missing required parameter: rid")
        resp = self._require_client().forget(rid)
        self._record_success()
        return json.dumps({"rid": rid, "found": bool(resp.get("found", False))})

    def _do_think(self, args: dict[str, Any]) -> str:
        resp = self._require_client().think(
            run_pattern_mining=bool(args.get("run_pattern_mining", False)),
            consolidation_limit=args.get("consolidation_limit"),
        )
        self._record_success()
        return json.dumps({
            "consolidated": resp.get("consolidation_count", 0),
            "conflicts_found": resp.get("conflicts_found", 0),
            "patterns_new": resp.get("patterns_new", 0),
            "patterns_updated": resp.get("patterns_updated", 0),
            "personality_updated": resp.get("personality_updated", False),
            "duration_ms": resp.get("duration_ms"),
            "triggers": resp.get("triggers", []),
        })

    def _do_conflicts(self) -> str:
        resp = self._require_client().conflicts()
        self._record_success()
        conflicts = resp.get("conflicts", []) or []
        return json.dumps({"count": len(conflicts), "conflicts": conflicts})

    def _do_resolve_conflict(self, args: dict[str, Any]) -> str:
        conflict_id = (args.get("conflict_id") or "").strip()
        strategy = (args.get("strategy") or "").strip()
        if not conflict_id or not strategy:
            return tool_error(
                "Missing required parameters: conflict_id, strategy"
            )
        if strategy == "keep_winner" and not args.get("winner_rid"):
            return tool_error("strategy='keep_winner' requires winner_rid")
        if strategy == "merge" and not args.get("new_text"):
            return tool_error("strategy='merge' requires new_text")

        resp = self._require_client().resolve_conflict(
            conflict_id,
            strategy=strategy,
            winner_rid=args.get("winner_rid"),
            new_text=args.get("new_text"),
            resolution_note=args.get("resolution_note"),
        )
        self._record_success()
        return json.dumps({
            "conflict_id": resp.get("conflict_id", conflict_id),
            "strategy": resp.get("strategy", strategy),
            "resolved": True,
        })

    def _do_relate(self, args: dict[str, Any]) -> str:
        entity = (args.get("entity") or "").strip()
        target = (args.get("target") or "").strip()
        relationship = (args.get("relationship") or "").strip()
        if not (entity and target and relationship):
            return tool_error(
                "Missing required parameters: entity, target, relationship"
            )
        resp = self._require_client().relate(
            entity, target, relationship,
        )
        self._record_success()
        return json.dumps({"edge_id": resp.get("edge_id"), "stored": True})

    def _do_stats(self) -> str:
        resp = self._require_client().stats()
        self._record_success()
        return json.dumps({
            "active_memories": resp.get("active_memories", 0),
            "consolidated_memories": resp.get("consolidated_memories", 0),
            "tombstoned_memories": resp.get("tombstoned_memories", 0),
            "edges": resp.get("edges", 0),
            "entities": resp.get("entities", 0),
            "operations": resp.get("operations", 0),
            "open_conflicts": resp.get("open_conflicts", 0),
            "pending_triggers": resp.get("pending_triggers", 0),
        })

    def _do_skill_search(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required parameter: query")
        default_top_k = self._config.top_k if self._config else 10
        top_k = min(_coerce_int(args.get("top_k"), default_top_k), 50)
        resp = self._require_client().skill_search(
            query, top_k=top_k, applies_to=args.get("applies_to"),
        )
        self._record_success()
        skills = resp.get("skills") or resp.get("results") or []
        compact = [
            {
                "rid": s.get("rid"),
                "skill_id": (s.get("metadata") or {}).get("skill_id"),
                "skill_type": (s.get("metadata") or {}).get("skill_type"),
                "applies_to": (s.get("metadata") or {}).get("applies_to", []),
                "body": s.get("text"),
                "score": s.get("score"),
                "source": (s.get("metadata") or {}).get("source"),
                "why_retrieved": s.get("why_retrieved") or [],
            }
            for s in skills
        ]
        return json.dumps({"count": len(compact), "skills": compact})

    def _do_skill_define(self, args: dict[str, Any]) -> str:
        skill_id = (args.get("skill_id") or "").strip()
        body = args.get("body") or ""
        skill_type = (args.get("skill_type") or "").strip()
        applies_to = args.get("applies_to") or []
        if not skill_id or not body or not skill_type or not applies_to:
            return tool_error(
                "Missing required parameters: skill_id, body, skill_type, applies_to"
            )
        resp = self._require_client().skill_define(
            skill_id=skill_id,
            body=body,
            skill_type=skill_type,
            applies_to=applies_to,
            triggers=args.get("triggers"),
            on_conflict=args.get("on_conflict", "reject"),
            version=args.get("version"),
            supersedes_skill_id=args.get("supersedes_skill_id"),
        )
        self._record_success()
        return json.dumps({
            "rid": resp.get("rid"),
            "skill_id": resp.get("skill_id", skill_id),
            "stored": bool(resp.get("stored", True)),
        })

    def _do_skill_outcome(self, args: dict[str, Any]) -> str:
        skill_id = (args.get("skill_id") or "").strip()
        if not skill_id:
            return tool_error("Missing required parameter: skill_id")
        if "succeeded" not in args:
            return tool_error("Missing required parameter: succeeded")
        resp = self._require_client().skill_outcome(
            skill_id,
            bool(args.get("succeeded")),
            note=args.get("note"),
        )
        self._record_success()
        return json.dumps({
            "rid": resp.get("rid"),
            "skill_id": resp.get("skill_id", skill_id),
            "recorded": bool(resp.get("recorded", True)),
        })

    # -- Optional hooks ---------------------------------------------------

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Run consolidation at session end, flushing pending writes first."""
        if self._cron_skipped or self._client is None or self._config is None:
            return
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=_SESSION_END_JOIN_SECS)
        if not self._config.auto_think_on_session_end:
            return
        if self._breaker_open():
            return
        try:
            stats = self._client.think(
                run_pattern_mining=False, run_personality=False,
            )
            logger.info(
                "YantrikDB session-end think: consolidated=%s conflicts=%s duration_ms=%s",
                stats.get("consolidation_count"),
                stats.get("conflicts_found"),
                stats.get("duration_ms"),
            )
        except YantrikDBError as e:
            logger.debug("YantrikDB session-end think failed: %s", e)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Preserve high-salience memories across context compression.

        Seeds recall with the tail of the about-to-be-compressed messages
        and returns a markdown block. Hermes' compressor includes this
        in the summary prompt so insights don't get dropped.
        """
        if self._cron_skipped or self._client is None:
            return ""
        if self._breaker_open():
            return ""
        tail = " ".join(
            (m.get("content") or "") for m in messages[-6:] if isinstance(m, dict)
        ).strip()
        if not tail:
            return ""
        try:
            resp = self._client.recall(
                tail[:2000], namespace=self._namespace, top_k=8,
            )
            block = _format_recall_block(resp.get("results", []), limit=8)
            if not block:
                return ""
            return f"## YantrikDB memories to preserve\n{block}"
        except YantrikDBError as e:
            logger.debug("YantrikDB on_pre_compress failed: %s", e)
            return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in MEMORY.md / USER.md additions into YantrikDB."""
        if self._cron_skipped or self._client is None:
            return
        if action != "add" or target not in ("memory", "user") or not content:
            return
        if self._breaker_open():
            return

        client = self._client
        text = content
        session_id = self._session_id
        namespace = self._namespace
        domain = "user" if target == "user" else "work"

        def _run() -> None:
            try:
                client.remember(
                    text,
                    namespace=namespace,
                    importance=0.7,
                    domain=domain,
                    metadata={
                        "source": "hermes_memory_md",
                        "target": target,
                        "session_id": session_id,
                    },
                )
                self._record_success()
            except YantrikDBError as e:
                self._record_failure()
                logger.debug("YantrikDB on_memory_write failed: %s", e)

        threading.Thread(
            target=_run, daemon=True, name="yantrikdb-memwrite",
        ).start()

    # -- Circuit breaker --------------------------------------------------

    def _breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._failure_count < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._failure_count = 0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._failure_count = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._failure_count += 1
            if self._failure_count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN
                logger.warning(
                    "YantrikDB circuit breaker tripped after %d failures — "
                    "pausing for %ds.",
                    self._failure_count, _BREAKER_COOLDOWN,
                )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register YantrikDB as a memory provider plugin."""
    ctx.register_memory_provider(YantrikDBMemoryProvider())


__all__ = [
    "ALL_TOOL_SCHEMAS",
    "CONFLICTS_SCHEMA",
    "FORGET_SCHEMA",
    "RECALL_SCHEMA",
    "RELATE_SCHEMA",
    "REMEMBER_SCHEMA",
    "RESOLVE_CONFLICT_SCHEMA",
    "STATS_SCHEMA",
    "THINK_SCHEMA",
    "YantrikDBMemoryProvider",
    "register",
]
