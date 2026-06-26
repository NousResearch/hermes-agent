"""Atlas memory plugin — MemoryProvider interface.

Backs Hermes long-term memory with Atlas, Blake's RDF-grounded personal
knowledge substrate. Recall is sourced from Atlas's /v1/memory/hermes/read
endpoint (active facts with confidence + life-context); writes go to
/v1/memory/hermes/write (classified, provenance-tracked RDF triples).

This is the Hermes-side adapter for army-of-one Plan 011-C.2. The transport
contract was defined by Plan 012 (memory_routes.py in army-of-one). Unlike
mem0/supermemory (server-side turn extraction), Atlas writes are EXPLICIT
facts: the agent decides what's worth remembering and stores it verbatim via
the atlas_remember tool, or the built-in memory tool mirrors writes through
the on_memory_write hook.

Design (mirrors mem0 provider patterns):
  - Non-blocking prefetch via background thread + cache
  - Circuit breaker: pause API calls after consecutive failures
  - Graceful degradation: every Atlas failure is swallowed; MemoryManager
    guarantees a failing provider never blocks the agent or the built-in
    provider.

Config via environment variables:
  ATLAS_BASE_URL       — Atlas API base URL (required; e.g.
                         http://atlas.agentic-stack.internal:8000 in cloud,
                         http://localhost:8000 locally)
  ATLAS_BEARER_TOKEN   — Bearer for LAN/VPC auth (optional for localhost)
  ATLAS_AGENT_NAME     — Agent identifier for fact attribution (default: hermes)
  ATLAS_MAX_AGE_DAYS   — Exclude facts older than N days (default: 90)

Or via $HERMES_HOME/atlas.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down Atlas.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Conservative timeouts — Atlas read is fast (in-process SPARQL) but the
# network hop (Cloud Map within the VPC) adds latency. Match Plan 012 spec.
_READ_TIMEOUT_SECS = 2.0
_WRITE_TIMEOUT_SECS = 3.0
# /v1/ask is the heavyweight retrieval+rerank+synth pipeline; allow longer.
_ASK_TIMEOUT_SECS = 10.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/atlas.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "base_url": os.environ.get("ATLAS_BASE_URL", ""),
        "token": os.environ.get("ATLAS_BEARER_TOKEN", ""),
        "agent_name": os.environ.get("ATLAS_AGENT_NAME", "hermes"),
        "max_age_days": int(os.environ.get("ATLAS_MAX_AGE_DAYS", "90")),
    }

    config_path = get_hermes_home() / "atlas.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "atlas_recall",
    "description": (
        "Retrieve stored facts about Blake from Atlas — preferences, people, "
        "projects, decisions, and context drawn from his unified knowledge "
        "substrate. Returns active facts ranked by recency + confidence. "
        "Use at conversation start or when you need durable context."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ASK_SCHEMA = {
    "name": "atlas_ask",
    "description": (
        "Ask Atlas a question about Blake's history, contacts, commitments, "
        "past decisions, or anything from his ingested corpus (Gmail, Calendar, "
        "Pipedrive, GitHub, Claude transcripts). Routes through Atlas's full "
        "/v1/ask retrieval pipeline (BM25 + vector + rerank + synthesis) and "
        "returns a cited answer. Use for recall questions like 'what's my last "
        "Pipedrive activity for Apex Capital?' or 'what did I commit to Greg "
        "about the 3pm?'. Do NOT use for arbitrary 'what is X' world-knowledge "
        "questions. Preserve any [cite:<chunk_id>] markers verbatim in your "
        "response to Blake so he can audit the answer's grounding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask Atlas. Required, non-empty.",
            },
            "life_context": {
                "type": "string",
                "enum": ["work", "personal", "health", "education",
                         "finance", "civic", "hobby", "brand", "spiritual"],
                "description": "Optional atlas:lifeContext vocab tag to scope retrieval.",
            },
            "intent_hint": {
                "type": "string",
                "description": "Optional intent classifier shortcut (e.g. 'lookup', 'timeline', 'commitment').",
            },
            "max_chunks": {
                "type": "integer",
                "description": "Optional max chunks to retrieve (default 5).",
            },
        },
        "required": ["question"],
    },
}

CONTACT_SCHEMA = {
    "name": "atlas_contact",
    "description": (
        "Look up structured context about a person Blake knows — canonical "
        "name, organization, recent meetings, emails involving them, open "
        "commitments, stable preferences, and any unresolved contradictions. "
        "Routes through Atlas's /v1/contact/{iri}/context SPARQL aggregator "
        "over the identity / events / emails named graphs. Use when Blake "
        "asks 'brief me on <person>', 'what's the latest with <person>', or "
        "before a meeting with someone in his network. Always returns a "
        "well-formed schema even for unknown contacts (empty arrays = cold "
        "corpus, not error). Person IRIs typically look like "
        "'https://atlas.blakeaber.dev/person/<slug>' or "
        "'https://atlas.blakeaber.dev/person/email:<address>'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "person_iri": {
                "type": "string",
                "description": (
                    "Atlas IRI of the contact, e.g. "
                    "'https://atlas.blakeaber.dev/person/jane-doe' or "
                    "'https://atlas.blakeaber.dev/person/email:greg@apex.com'."
                ),
            },
            "recency": {
                "type": "string",
                "description": (
                    "Time window for interaction filtering. Accepts shorthand "
                    "('90d', '12w') or natural phrases ('last 30 days'). Default '90d'."
                ),
            },
        },
        "required": ["person_iri"],
    },
}

OPEN_CONTRADICTIONS_SCHEMA = {
    "name": "atlas_open_contradictions",
    "description": (
        "List currently-open contradictions in Blake's Atlas — facts the "
        "scanner has flagged as conflicting but Blake has NOT yet "
        "annotated. Each row carries the LLM adjudicator's advisory "
        "verdict + confidence so you can decide whether to surface it. "
        "Use when Blake asks 'what's contradictory in my memory?', "
        "'anything I need to reconcile?', before stating something "
        "potentially-stale, or when an answer hinges on a fact you "
        "suspect Blake has revised. Routes through GET "
        "/v1/contradictions?status=open with a confidence floor (default "
        "0.6) so low-signal rows stay out of the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "confidence_min": {
                "type": "number",
                "description": (
                    "Minimum LLM-adjudication confidence (0.0–1.0) "
                    "required for a row to surface. Default 0.6."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 25, max 500).",
            },
        },
        "required": [],
    },
}

INGEST_STATUS_SCHEMA = {
    "name": "atlas_ingest_status",
    "description": (
        "Report what data (email/calendar/contacts/etc.) is ingested into "
        "Atlas and when it was last refreshed. Use this when Blake asks "
        "whether his data is up to date or when atlas_ask returns nothing "
        "for a recent item."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

REMEMBER_SCHEMA = {
    "name": "atlas_remember",
    "description": (
        "Store a durable fact in Atlas. Stored verbatim with provenance and "
        "confidence bootstrapping. Use for explicit preferences, corrections, "
        "relationships, or decisions worth recalling across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
            "target": {
                "type": "string",
                "enum": ["user", "memory"],
                "description": "'user' for facts about Blake; 'memory' for agent self-notes. Default 'user'.",
            },
            "life_context": {
                "type": "string",
                "enum": ["work", "personal", "health", "education",
                         "finance", "civic", "hobby", "brand", "spiritual"],
                "description": "Optional life-domain tag.",
            },
        },
        "required": ["content"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class AtlasMemoryProvider(MemoryProvider):
    """Atlas RDF-grounded long-term memory provider."""

    def __init__(self):
        self._config = None
        self._base_url = ""
        self._token = ""
        self._agent_name = "hermes"
        self._max_age_days = 90
        self._session_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "atlas"

    def is_available(self) -> bool:
        # No network call here — just config presence (per ABC contract).
        cfg = _load_config()
        return bool(cfg.get("base_url"))

    def save_config(self, values, hermes_home):
        """Write non-secret config to $HERMES_HOME/atlas.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "atlas.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "base_url", "description": "Atlas API base URL", "required": True,
             "default": "http://localhost:8000", "env_var": "ATLAS_BASE_URL"},
            {"key": "token", "description": "Atlas bearer token (required for non-localhost)",
             "secret": True, "env_var": "ATLAS_BEARER_TOKEN"},
            {"key": "agent_name", "description": "Agent identifier for fact attribution",
             "default": "hermes"},
            {"key": "max_age_days", "description": "Exclude facts older than N days",
             "default": "90"},
        ]

    # -- HTTP helpers --------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Atlas circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _fetch_facts(self) -> list:
        """GET /v1/memory/hermes/read — returns list of fact dicts. Raises on error."""
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/memory/hermes/read"
        resp = httpx.get(
            url,
            params={"agent": self._agent_name, "max_age_days": self._max_age_days},
            headers=self._headers(),
            timeout=_READ_TIMEOUT_SECS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    @staticmethod
    def _format_facts(facts: list) -> str:
        """Render Atlas facts as a compact bullet list for prompt injection."""
        lines = []
        for f in facts:
            key = f.get("key", "")
            value = f.get("value", "")
            if not value:
                continue
            ctx = f.get("life_context")
            tag = f" [{ctx}]" if ctx else ""
            if key and key != value:
                lines.append(f"- {key}: {value}{tag}")
            else:
                lines.append(f"- {value}{tag}")
        return "\n".join(lines)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._base_url = self._config.get("base_url", "")
        self._token = self._config.get("token", "")
        self._agent_name = self._config.get("agent_name", "hermes")
        self._max_age_days = int(self._config.get("max_age_days", 90))
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        return (
            "# Atlas Memory\n"
            "Active — RDF-grounded long-term memory across Blake's life domains.\n"
            "Use atlas_recall to fetch stored facts, atlas_remember to store a "
            "durable fact worth recalling across sessions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=_READ_TIMEOUT_SECS + 0.5)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Atlas Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                facts = self._fetch_facts()
                if facts:
                    formatted = self._format_facts(facts)
                    with self._prefetch_lock:
                        self._prefetch_result = formatted
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Atlas prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="atlas-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """No-op: Atlas does NOT do turn-level fact extraction. Facts are written
        explicitly via atlas_remember or mirrored from built-in memory via
        on_memory_write. This avoids polluting the RDF store with raw turns."""
        return

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            RECALL_SCHEMA,
            REMEMBER_SCHEMA,
            ASK_SCHEMA,
            CONTACT_SCHEMA,
            OPEN_CONTRADICTIONS_SCHEMA,
            INGEST_STATUS_SCHEMA,
        ]

    def _ask(self, *, question: str, life_context: str | None = None,
             intent_hint: str | None = None, max_chunks: int | None = None) -> dict:
        """POST /v1/ask — full retrieval+rerank+synthesis pipeline. Raises on error.

        Contract mirror of army-of-one atlas/api/ask_routes.py:AskRequest /
        AskResponse. Returns the parsed JSON response verbatim so the caller
        can pass citation markers ([cite:<chunk_id>]) through unmodified.

        Atlas's current AskRequest accepts {question, intent_hint?} with
        extra="forbid". life_context and max_chunks are accepted at the tool
        boundary for forward-compatibility; we fold any provided hints into
        intent_hint as a single composite signal so the strict server-side
        Pydantic model accepts the payload.
        """
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/ask"
        body: dict[str, Any] = {"question": question}
        hint_parts: list[str] = []
        if intent_hint:
            hint_parts.append(intent_hint)
        if life_context:
            hint_parts.append(f"life_context:{life_context}")
        if max_chunks is not None:
            hint_parts.append(f"max_chunks:{int(max_chunks)}")
        if hint_parts:
            body["intent_hint"] = ";".join(hint_parts)
        resp = httpx.post(
            url, json=body, headers=self._headers(), timeout=_ASK_TIMEOUT_SECS,
        )
        resp.raise_for_status()
        return resp.json()

    def _contact(self, *, person_iri: str, recency: str | None = None) -> dict:
        """GET /v1/contact/{iri}/context — Plan 025-C SPARQL aggregator.

        Returns the structured contact context: canonical_name, org, recent
        interactions, events, commitments, preferences, contradictions, and
        last_touch. Empty arrays signal cold-corpus rather than missing-person.
        """
        import urllib.parse

        import httpx
        # Encode the IRI as a path segment. FastAPI's `{iri:path}` accepts
        # raw slashes, but we URL-quote the colon/at-sign in the email:<addr>
        # form so the bearer-protected route receives a clean path.
        encoded = urllib.parse.quote(person_iri, safe="/:")
        url = f"{self._base_url.rstrip('/')}/v1/contact/{encoded}/context"
        params: dict[str, Any] = {}
        if recency:
            params["recency"] = recency
        resp = httpx.get(
            url, params=params, headers=self._headers(), timeout=_READ_TIMEOUT_SECS + 5.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _open_contradictions(
        self, *, confidence_min: float = 0.6, limit: int = 25
    ) -> list[dict]:
        """GET /v1/contradictions?status=open — Plan 025-E.

        Returns the list verbatim (each row is the Atlas ContradictionItem
        shape), client-side filtered by `llm_confidence >= confidence_min`
        so rows the Haiku adjudicator was uncertain about don't bubble up.
        [cite:...] markers (when present in rationale) are preserved.
        """
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/contradictions"
        capped_limit = max(1, min(int(limit), 500))
        resp = httpx.get(
            url,
            params={"status": "open", "limit": capped_limit},
            headers=self._headers(),
            timeout=_READ_TIMEOUT_SECS + 3.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        floor = float(confidence_min)
        filtered: list[dict] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            conf = r.get("llm_confidence")
            try:
                conf_f = float(conf) if conf is not None else 0.0
            except (TypeError, ValueError):
                conf_f = 0.0
            if conf_f >= floor:
                filtered.append(r)
        return filtered

    def _ingest_status(self) -> dict:
        """GET /v1/stats — corpus + job + ingest aggregates (Atlas stats_routes).

        Returns the parsed StatsResponse JSON verbatim. We read three slices:
        - corpus.top_sources  → per-source chunk counts (what's in memory)
        - jobs.last_7_days     → per-source ingest job activity (recent refresh)
        - ingest.last_ingest_at / ingest_rate_last_7d_per_day → recency

        This reuses the existing ATLAS_BASE_URL + bearer-token client surface;
        no new Atlas endpoint is introduced. Raises on HTTP error.
        """
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/stats"
        resp = httpx.get(
            url, headers=self._headers(), timeout=_READ_TIMEOUT_SECS + 3.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _today(self) -> dict:
        """GET /v1/today — the Atlas "cockpit" (army-of-one ask_routes:get_today).

        Returns the parsed JSON verbatim: ``upcoming_meetings`` (today-onward
        CalendarEvents) and ``overdue_emails`` (most-recent EmailMessages),
        each a list of TimelineEntry dicts ``{iri, kind, canonical_name,
        event_time, summary}`` straight from the graph (SPARQL, no LLM cost).
        Reuses the existing ATLAS_BASE_URL + bearer client surface; no new
        Atlas endpoint is introduced. Raises on HTTP error — the /daily
        fetcher catches and degrades the calendar/inbox sections.
        """
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/today"
        resp = httpx.get(
            url, headers=self._headers(), timeout=_READ_TIMEOUT_SECS + 3.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _format_ingest_status(stats: dict) -> str:
        """Render /v1/stats into a one-paragraph human summary of what's ingested."""
        corpus = stats.get("corpus") or {}
        jobs = stats.get("jobs") or {}
        ingest = stats.get("ingest") or {}

        chunks_total = corpus.get("chunks_total") or 0
        sources_total = corpus.get("sources_total") or 0
        top_sources = corpus.get("top_sources") or []
        last_7 = jobs.get("last_7_days") or {}
        last_ingest_at = ingest.get("last_ingest_at")
        rate = ingest.get("ingest_rate_last_7d_per_day") or 0.0

        if not chunks_total and not top_sources and not last_7:
            return (
                "Atlas has nothing ingested yet — 0 chunks across 0 sources, and "
                "no ingest jobs ran in the last 7 days. If you expect email or "
                "calendar to be here, ingestion hasn't run."
            )

        parts: list[str] = [
            f"Atlas holds {chunks_total} chunks across {sources_total} source(s)."
        ]

        if top_sources:
            src_bits = ", ".join(
                f"{s.get('source_iri', '?')} ({s.get('chunks', 0)} chunks)"
                for s in top_sources[:6]
            )
            parts.append(f"Top sources: {src_bits}.")

        if last_7:
            recent_bits = ", ".join(
                f"{src}: {cnt}" for src, cnt in sorted(
                    last_7.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            parts.append(f"Ingest jobs in the last 7 days by source — {recent_bits}.")
        else:
            parts.append("No ingest jobs ran in the last 7 days.")

        if last_ingest_at:
            parts.append(
                f"Most recent ingest: {last_ingest_at} "
                f"(~{rate} items/day over the last week)."
            )
        else:
            parts.append("No recorded last-ingest timestamp.")

        return " ".join(parts)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Atlas temporarily unavailable (consecutive failures). Will retry automatically."
            })

        if tool_name == "atlas_recall":
            try:
                facts = self._fetch_facts()
                self._record_success()
                if not facts:
                    return json.dumps({"result": "No facts stored in Atlas yet."})
                return json.dumps({"result": self._format_facts(facts), "count": len(facts)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas recall failed: {e}")

        elif tool_name == "atlas_ask":
            question = args.get("question", "")
            if not question or not str(question).strip():
                return tool_error("Missing required parameter: question")
            try:
                payload = self._ask(
                    question=str(question).strip(),
                    life_context=args.get("life_context"),
                    intent_hint=args.get("intent_hint"),
                    max_chunks=args.get("max_chunks"),
                )
                self._record_success()
                # Return the Atlas response verbatim so [cite:<chunk_id>]
                # markers in `answer` and the structured `citations` list
                # are preserved for the model to surface to Blake.
                return json.dumps(payload)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas ask failed: {e}")

        elif tool_name == "atlas_contact":
            person_iri = args.get("person_iri", "")
            if not person_iri or not str(person_iri).strip():
                return tool_error("Missing required parameter: person_iri")
            try:
                payload = self._contact(
                    person_iri=str(person_iri).strip(),
                    recency=args.get("recency"),
                )
                self._record_success()
                return json.dumps(payload)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas contact lookup failed: {e}")

        elif tool_name == "atlas_open_contradictions":
            try:
                conf_min = args.get("confidence_min")
                conf_min_f = float(conf_min) if conf_min is not None else 0.6
                limit = args.get("limit")
                limit_i = int(limit) if limit is not None else 25
                rows = self._open_contradictions(
                    confidence_min=conf_min_f, limit=limit_i,
                )
                self._record_success()
                return json.dumps({
                    "result": rows,
                    "count": len(rows),
                    "confidence_min": conf_min_f,
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas open-contradictions lookup failed: {e}")

        elif tool_name == "atlas_ingest_status":
            try:
                stats = self._ingest_status()
                self._record_success()
                return json.dumps({"result": self._format_ingest_status(stats)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas ingest-status lookup failed: {e}")

        elif tool_name == "atlas_remember":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            target = args.get("target", "user")
            life_context = args.get("life_context")
            # Plan 056-D: optional run_id join key (backward-compatible — absent
            # for callers that don't stamp one).
            run_id = args.get("run_id")
            try:
                self._write_fact(
                    target=target, action="add", content=content,
                    life_context=life_context, run_id=run_id,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored in Atlas."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Atlas write failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def _write_fact(self, *, target: str, action: str, content: str,
                    old_text: str | None = None, life_context: str | None = None,
                    run_id: str | None = None) -> None:
        """POST /v1/memory/hermes/write. Raises on error.

        Plan 056-D: ``run_id`` is an OPTIONAL annotation (default ``None``,
        backward-compatible). When a Hermes-initiated action (a cron tick, a
        Slack-triggered task) stamps a ``run_id`` and threads it here, the memory
        write carries that join key so a Hermes-initiated → orchestrator-executed
        chain is one queryable Atlas subgraph (the shared-``run_id`` spine). When
        omitted, the request body is byte-identical to pre-056-D.
        """
        import httpx
        url = f"{self._base_url.rstrip('/')}/v1/memory/hermes/write"
        body = {
            "target": target,
            "action": action,
            "content": content,
            "agent": self._agent_name,
            "session_id": self._session_id or None,
        }
        if old_text:
            body["old_text"] = old_text
        if life_context:
            body["life_context"] = life_context
        if run_id:
            body["run_id"] = run_id
        resp = httpx.post(url, json=body, headers=self._headers(), timeout=_WRITE_TIMEOUT_SECS)
        resp.raise_for_status()

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror built-in memory writes into Atlas (non-blocking, best-effort).

        When Hermes's built-in memory tool writes a fact, echo it to Atlas so
        the RDF store stays in sync with the flat memory.md. Atlas-side
        failures are swallowed — the built-in write already succeeded.
        """
        if self._is_breaker_open():
            return
        # Atlas targets are 'user' | 'memory'; map unknown targets to 'memory'.
        atlas_target = target if target in ("user", "memory") else "memory"
        # Atlas actions are 'add' | 'replace' | 'remove'.
        atlas_action = action if action in ("add", "replace", "remove") else "add"
        old_text = (metadata or {}).get("old_text") if metadata else None
        if atlas_action in ("replace", "remove") and not old_text:
            # Can't satisfy Atlas's contract without old_text — downgrade to add.
            atlas_action = "add"

        def _mirror():
            try:
                self._write_fact(
                    target=atlas_target, action=atlas_action,
                    content=content, old_text=old_text,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Atlas memory-write mirror failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=_WRITE_TIMEOUT_SECS + 1.0)
        self._sync_thread = threading.Thread(target=_mirror, daemon=True, name="atlas-mirror")
        self._sync_thread.start()

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=_WRITE_TIMEOUT_SECS + 1.0)


def register(ctx) -> None:
    """Register Atlas as a memory provider plugin."""
    ctx.register_memory_provider(AtlasMemoryProvider())
