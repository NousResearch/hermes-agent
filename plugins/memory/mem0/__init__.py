"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_HOST          — Self-hosted OSS server URL (direct REST mode)
  MEM0_ADMIN_API_KEY — Self-hosted OSS server admin API key
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import concurrent.futures.thread as _threadpool
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
import weakref

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .temporal_parse import created_at_in_window, parse_temporal_window
from . import qmd_recall
from . import gbrain_recall

logger = logging.getLogger(__name__)


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose workers preserve the old daemon-thread behavior."""

    def _adjust_thread_count(self):
        # Copy of concurrent.futures.thread.ThreadPoolExecutor with daemon=True.
        # Do not register in _threads_queues: those atexit joins would undo the
        # previous daemon-thread semantics for a stuck network call.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)  # type: ignore[arg-type]

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_threadpool._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)  # type: ignore[attr-defined]

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Destructive-tools (mem0_forget / mem0_delete) — gated, Apollo+Aegis only.
# All overridable via $HERMES_HOME/mem0.json. See the spec:
# ~/.hermes/plans/2026-06-10_mem0-destructive-tools-spec.md
_FORGOTTEN_PREFIX = "[FORGOTTEN]"
_DEFAULT_DESTRUCTIVE = {
    "max_bulk": 25,               # soft review cap (both verbs, by-id batch + filter)
    "max_bulk_hard_force": 100,   # absolute hard-delete ceiling; force can't breach
    "max_bulk_forget": 100,       # forget review cap
    "max_bulk_forget_force": 500, # forget ceiling
    "max_delete_per_hour": 50,    # per-agent hard-delete velocity (rows/hour)
    "max_forget_per_hour": 200,   # per-agent forget velocity (rows/hour)
    "unscoped_ratio": 0.9,        # C8a: matched/total ≥ this → mass op, refused
    "absolute_mass_floor": 200,   # C8a: matched ≥ this → mass op, refused
    "token_ttl_seconds": 300,     # dry-run confirm-token lifetime
}

# Capture-mode values that mean "auto-capture ON" (per-turn sync_turn write).
# Anything else (notably "off") = recall-only.
_CAPTURE_ON = ("auto", "on", "true", "1")

# Upper bound on rows the post-write scrub fetches for one capture_idem. Must be high enough that a
# long turn extracting many memories can't hide a secret-bearing row beyond the fetched page and then
# complete unscanned (Greptile P1). A single turn realistically yields a handful of facts; 500 is a
# safe ceiling well above any real extraction.
_CAPTURE_SCRUB_MAX_ROWS = 500

# Max combined user+assistant characters an auto-captured turn may have. Above this a turn is a
# system/review/tool-dump prompt (no durable user facts) that the extraction provider rejects as
# malformed (HTTP 502) — capturing it just poison-loops the queue. ~16k chars is far above any real
# conversational exchange but below the multi-KB review prompts that trip the provider.
_CAPTURE_MAX_TURN_CHARS = 16000

# W3-TEMPORAL defaults (all overridable via $HERMES_HOME/mem0.json).
# tz = the calendar-day reference zone for "the 20th"/"yesterday" (PT, matching the
# digest's DST-correct PT-day bounds). overfetch = how many candidates to pull from
# /search before the in-window boost re-rank (the window can favour rows below the
# top_k semantic cut, so we need a deeper pool to surface them — spec §2.1).
_TEMPORAL_DEFAULT_TZ = "America/Los_Angeles"
_TEMPORAL_DEFAULT_OVERFETCH = 50


def resolve_capture(env_value: Optional[str], config_value: Optional[str]) -> tuple:
    """Resolve the effective capture mode + its source. SINGLE SOURCE OF TRUTH.

    F1 (2026-06-21, P2-2): the plugin AND the capture-flip-lag drift detector
    cron MUST agree on intent, so the env→config→default precedence lives here
    and both import it (never replicate it — a replicated resolver drifts the
    instant precedence changes, reintroducing the very drift the detector guards).

    Precedence (matches the historical plugin line 544): MEM0_CAPTURE env var
    wins over the profile config `capture` key, which wins over the "auto" default.
    An env override makes a disk edit a NO-OP even after a restart — that is the
    flip-lag footgun, so the source is returned too (the operator/detector must
    know WHICH source is live).

    Returns (value, source) where value is lower-cased/stripped and source is one
    of "env" | "config" | "default".
    """
    if env_value is not None and str(env_value).strip() != "":
        return str(env_value).strip().lower(), "env"
    if config_value is not None and str(config_value).strip() != "":
        return str(config_value).strip().lower(), "config"
    return "auto", "default"


def capture_is_on(value: str) -> bool:
    """True if a resolved capture value means auto-capture is active."""
    return str(value).strip().lower() in _CAPTURE_ON


def _trunc(s: str, n: int = 200) -> str:
    """Truncate resolved memory text for ledger/preview (never store unbounded)."""
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "admin_api_key": os.environ.get("MEM0_ADMIN_API_KEY", ""),
        "ca_bundle": os.environ.get("MEM0_CA_BUNDLE", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        # Default-off safety gate for single-user fleets that want one shared
        # user memory scope across Discord/Telegram/CLI sender ids. When false,
        # gateway-provided user_id continues to win exactly as today.
        "pin_user_id": False,
        # Prefetch join ceiling (seconds): how long a turn waits at start for the
        # background mem0 search (now incl. rerank) before proceeding. Hit = the turn
        # proceeds with whatever memory finished. Behavioral config, not a secret.
        "prefetch_join_timeout_s": float(os.environ.get("MEM0_PREFETCH_JOIN_TIMEOUT_S", "10") or "10"),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


class _DirectRestMem0Client:
    """Small stdlib client for the self-hosted OSS Mem0 server."""

    def __init__(self, host: str, admin_api_key: str, agent_id: str, user_id: str = "",
                 ca_bundle: str = ""):
        self._host = (host or "").strip().rstrip("/")
        self._admin_api_key = (admin_api_key or "").strip()
        self._agent_id = (agent_id or "").strip()
        self._user_id = (user_id or "").strip()
        # Optional CA bundle for verifying a self-hosted HTTPS endpoint signed by a
        # private CA (e.g. the LAN "Ace Local Root CA" in front of mem0.ace). urllib
        # uses the system trust store by default, which fleet hosts don't carry the
        # private CA in — so without this, https://<name>.ace fails
        # CERTIFICATE_VERIFY_FAILED. Building a context from the bundle keeps TLS ON
        # (no plaintext fallback) and is per-host config, not a system-trust mutation.
        self._ssl_context = None
        bundle = (ca_bundle or "").strip()
        if bundle and self._host.startswith("https://"):
            import ssl
            self._ssl_context = ssl.create_default_context(cafile=bundle)

    def _scope(self, filters: Optional[Dict[str, Any]] = None,
               *, scope_agent: bool = True) -> Dict[str, Any]:
        """Inject the client's configured user_id (always) and agent_id (writes only).

        Defense-in-depth on a multi-agent shared store: a caller that omits a scope
        must still be constrained to this client's user_id, never querying or writing
        across the whole store (B3/B4).

        scope_agent=True (writes/add): also inject agent_id for attribution.
        scope_agent=False (reads/search/get_all): user_id floor ONLY — do NOT add
        agent_id. The provider deliberately reads user-scoped (see
        Mem0MemoryProvider._read_filters) so cross-session recall works; and most
        historical memories were stored agent-scoped WITHOUT a user_id, so ANDing
        agent_id onto a read silently drops them (the live-cutover 0-results bug:
        agent-only=20 hits vs agent+user=2). A caller that passes an explicit
        agent_id in `filters` is always respected.
        """
        scope = dict(filters or {})
        if self._user_id and not scope.get("user_id"):
            scope["user_id"] = self._user_id
        if scope_agent and self._agent_id and not scope.get("agent_id"):
            scope["agent_id"] = self._agent_id
        return {k: v for k, v in scope.items() if v is not None and v != ""}

    def _request(self, method: str, path: str, *, body: Optional[dict] = None,
                 params: Optional[dict] = None) -> Any:
        clean_params = {k: v for k, v in (params or {}).items()
                        if v is not None and v != ""}
        query = urllib.parse.urlencode(clean_params)
        url = f"{self._host}{path}"
        if query:
            url = f"{url}?{query}"

        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if self._admin_api_key:
            req.add_header("X-API-Key", self._admin_api_key)

        try:
            with urllib.request.urlopen(req, timeout=30, context=self._ssl_context) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"Mem0 self-host REST {method} {path} failed: HTTP {e.code} {e.reason}{suffix}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Mem0 self-host REST {method} {path} failed: {e.reason}"
            ) from e

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise RuntimeError(
                f"Mem0 self-host REST {method} {path} returned invalid JSON"
            ) from e

    def add(self, messages, **kwargs):
        body = {"messages": messages}
        # scope user_id/agent_id via _scope so an add with no explicit scope still
        # lands under this client's user/agent, never globally on the shared store (B4).
        scoped = self._scope({"user_id": kwargs.get("user_id"), "agent_id": kwargs.get("agent_id")})
        body.update(scoped)
        # prompt = the salience gate (server-side extraction custom_instructions); model_chain =
        # A-full per-call model fallback. Both MUST reach the wire or auto-capture extracts with the
        # server default + no gate. (They were previously dropped — the server ignores unknown keys,
        # so an old server simply falls back to its default, which is safe.)
        for key in ("metadata", "infer", "run_id", "prompt", "model_chain"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return self._request("POST", "/memories", body=body)

    def search(self, query=None, filters=None, rerank=None, top_k=None,
               keyword_search=None, reference_date=None, **kwargs):
        # reads scope to user_id only (scope_agent=False): cross-session recall +
        # match historical agent-scoped-without-user memories. Explicit agent_id in
        # filters is still honored by _scope.
        body = {"query": query or "", **self._scope(filters, scope_agent=False)}
        # Retrieval flags MUST reach the wire. The param-drop bug: this body was built
        # with only query+scope, so rerank/keyword_search/top_k were silently discarded
        # and rerank was a no-op on the wire for months. Put every caller-set flag in the
        # POST body. A None flag is OMITTED so the server resolves it from its settings
        # default (INV-8(i)/INV-10); only the two enumerated call-sites (queue_prefetch,
        # mem0_search) send explicit values — every other caller passes None.
        for _flag_key, _flag_val in (
            ("rerank", rerank),
            ("keyword_search", keyword_search),
            ("top_k", top_k),
            ("reference_date", reference_date),
        ):
            if _flag_val is not None:
                body[_flag_key] = _flag_val
        response = self._request("POST", "/search", body=body)
        try:
            limit = int(top_k)
        except (TypeError, ValueError):
            limit = 0
        if limit > 0:
            if isinstance(response, dict) and isinstance(response.get("results"), list):
                response = dict(response)
                response["results"] = response["results"][:limit]
            elif isinstance(response, list):
                response = response[:limit]
        return response

    def search_meta_filtered(self, query, meta_filters, top_k=5):
        """Search with a TRUE nested ``filters`` clause (metadata equality).

        The regular ``search()`` spreads scope keys to the body top-level, which the
        self-host server IGNORES for metadata equality (probe 2026-06-27: top-level
        dedup_hash -> 20 unfiltered rows; nested filters -> exact 1). The dedup ladder
        needs real server-side metadata filtering, so this builds the nested shape:
        ``{"query":..., "user_id":..., "filters": {<meta equality>}, "top_k":...}``.

        Deliberately does NOT send ``agent_id``: the self-host server treats a top-level
        ``agent_id`` as a RESULT filter (probe 2026-06-28: searching agent_id=apollo for
        a row written by another agent_id returns 0 even on a matching dedup_hash). So
        adding it would narrow dedup to same-agent rows — the "live-cutover 0-results"
        class _read_filters guards against. The accepted cost is that these dedup lookups
        log as agent_id=null in recall_events (digest "unattributed"); correct recall
        scope is user-only, and attribution must never narrow results.
        """
        # The self-host server REJECTS an empty/whitespace-only query with HTTP 400 ("Invalid
        # query: cannot be empty or whitespace-only") even when a metadata `filters` clause is the
        # real selector. For a pure metadata-equality lookup (e.g. capture_idem reconcile) there is
        # no semantic query, so send a minimal non-empty placeholder; the `filters` clause does the
        # actual selection. (Probed 2026-07-03: ""→400, "."→200 with the filter honored.)
        # FAIL-CLOSED (Greptile): the "." placeholder is only safe BECAUSE a real metadata filter
        # narrows the result. With NO filter, "." would degrade into a broad match over all memories
        # and callers could treat unrelated rows as hits. This method is metadata-equality ONLY, so
        # an empty filter is a caller bug — refuse it rather than silently broad-search.
        filters = dict(meta_filters or {})
        if not filters:
            raise ValueError("search_meta_filtered requires a non-empty metadata filter "
                             "(it is metadata-equality only; use search() for a semantic query)")
        q = query if (query and query.strip()) else "."
        body = {"query": q, "user_id": self._user_id, "filters": filters, "top_k": top_k}
        return self._request("POST", "/search", body=body)

    def get_all(self, filters=None, **kwargs):
        # reads scope to user_id only (see search) — agent-scoped recall would drop
        # historical agent-without-user memories.
        return self._request("GET", "/memories", params=self._scope(filters, scope_agent=False))

    def get(self, memory_id):
        mid = urllib.parse.quote(str(memory_id), safe="")
        return self._request("GET", f"/memories/{mid}")

    def update(self, memory_id, text=None, metadata=None, timestamp=None, **kwargs):
        mid = urllib.parse.quote(str(memory_id), safe="")
        body = {}
        if text is not None:
            body["text"] = text
        if metadata is not None:
            body["metadata"] = metadata
        return self._request("PUT", f"/memories/{mid}", body=body)

    def delete(self, memory_id, delete_linked=False):
        mid = urllib.parse.quote(str(memory_id), safe="")
        return self._request("DELETE", f"/memories/{mid}")

    def history(self, memory_id):
        return []


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Reranking follows the configured profile by default; set rerank=false to "
        "skip it for a faster, lower-precision search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Override reranking for this query (default: the configured rerank profile)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Deliberately store ONE durable fact about the user or their stable environment, "
        "verbatim (no LLM extraction). This is the PRIMARY way memories get saved — auto-capture "
        "is off, so a fact is only remembered across sessions if you call this. Save proactively "
        "the moment you learn something durable; do not wait to be asked.\n"
        "SAVE when you learn: a preference or taste; a standing decision or directive; a correction "
        "to something previously believed; an account / device / service / credential pointer "
        "(not the secret itself); durable environment or topology (hosts, IPs, paths, tools, how "
        "things are wired); a long-lived plan, goal, or constraint. One fact per call; phrase it as "
        "a standalone declarative fact that will still make sense months from now.\n"
        "Do NOT save: work-narration or what you did this turn (built, ran, tested, committed, "
        "pushed, deployed, verified, reviewed); status / progress / ETA / cost / token counts; "
        "PR / issue / commit / SHA / phase-done / task-state; transient state that will be stale in "
        "a week; anything already obvious from a stable doc. When in doubt between a durable user "
        "fact and session exhaust, save the fact and skip the exhaust."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The single durable fact to store, as a standalone declarative sentence."},
        },
        "required": ["conclusion"],
    },
}

# --- mem0_remember: the background-review write helper (registry tool, manager-free) ---
# Registered via ctx.register_tool(toolset="memory_write", ...) so it dispatches through
# the regular tool registry (handle_function_call), NOT the memory-provider path that
# requires _memory_manager (None in the skip_memory=True review fork). See spec §5A
# Phase-0 correction. Inherits the mem0_conclude salience rubric verbatim.
REMEMBER_SCHEMA = {
    "name": "mem0_remember",
    "description": (
        "Deliberately store ONE durable fact about the user or their stable environment into "
        "long-term memory (mem0), verbatim (no LLM extraction). Use this from the background "
        "self-improvement review when you find a fact worth keeping across sessions; it is the "
        "manager-free counterpart to mem0_conclude. Save the FACT, never the conversation or "
        "this prompt.\n"
        "SAVE when you learn: a preference or taste; a standing decision or directive; a correction "
        "to something previously believed; an account / device / service / credential pointer "
        "(not the secret itself); durable environment or topology (hosts, IPs, paths, tools, how "
        "things are wired); a long-lived plan, goal, or constraint. One fact per call; phrase it as "
        "a standalone declarative fact that will still make sense months from now.\n"
        "Do NOT save: work-narration or what you did this turn (built, ran, tested, committed, "
        "pushed, deployed, verified, reviewed); status / progress / ETA / cost / token counts; "
        "PR / issue / commit / SHA / phase-done / task-state; transient state that will be stale in "
        "a week; anything already obvious from a stable doc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "The single durable fact to store, as a standalone declarative sentence."},
        },
        "required": ["fact"],
    },
}


def _dedup_norm_hash(text: str) -> str:
    """Normalize (lowercase + collapse-whitespace + strip) then MD5.

    DD-5: raw-text MD5 is trivially defeated by a trailing space / case; normalize first
    so Tier-1 catches byte-and-whitespace-equal dupes. Tier-2 (cosine) catches the rest.
    """
    import hashlib
    norm = " ".join((text or "").lower().split()).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# Dedup Tier-2 thresholds (D-7, calibrated by eval/dedup_threshold_sweep.py 2026-06-27).
# CALIBRATION FINDING: on this store, reworded-same-fact cosines (0.58–0.92) and
# contradiction cosines (0.61–0.99) OVERLAP — there is NO cosine threshold that catches
# paraphrase-dupes without ALSO swallowing contradictions (value-flips like "weight 0.02"
# vs "0.10" embed at ~0.99). So Tier-2 cosine CANNOT safely auto-skip on a fidelity-first
# store. Resolution (matches DD-1): IDENTICAL is set to 0.995 — a near-verbatim safety belt
# that Tier-1 exact-hash already covers — so Tier-2 effectively NEVER auto-skips; the
# ambiguous band always WRITES. Real semantic dedup is deferred to Tier-4 (LLM reconcile).
_DEDUP_COSINE_THRESHOLD = 0.95
_DEDUP_COSINE_IDENTICAL = 0.995


def _dedup_cosine_band(top_score, threshold=_DEDUP_COSINE_THRESHOLD, identical=_DEDUP_COSINE_IDENTICAL):
    """Map a top-hit similarity to a band: 'skip_identical' | 'write_ambiguous' | 'write'."""
    try:
        s = float(top_score)
    except (TypeError, ValueError):
        return "write"
    if s >= identical:
        return "skip_identical"
    if s >= threshold:
        return "write_ambiguous"
    return "write"


# --- Destructive tools (gated; appended only when destructive_tools_enabled) ---

FORGET_SCHEMA = {
    "name": "mem0_forget",
    "description": (
        "SOFT, REVERSIBLE REDACTION. Mark a memory obsolete so it stops surfacing in recall, "
        "WITHOUT destroying it (it can be restored). This is the SAFE DEFAULT for "
        "'this is no longer true / superseded'. Prefer this over mem0_delete unless "
        "data must genuinely not exist. By-id (read-before-forget); a gated by-filter "
        "path requires a dry-run then a confirm_token. restore=true un-forgets.\n"
        "CAVEAT (self-host): forget = reversible REDACTION, not erasure. The original "
        "content is retained verbatim in the row's plaintext metadata (original_text) in "
        "the shared store, and recall-hiding is client-side. For anything that must "
        "genuinely NOT EXIST (secrets, wrong/sensitive data), use mem0_delete — it is the "
        "only true removal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Single memory id to forget (or restore)."},
            "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "Batch of memory ids to forget."},
            "filter": {"type": "string", "description": "Natural-language/structured filter for bulk forget (dry-run first, returns a confirm_token)."},
            "confirm_token": {"type": "string", "description": "Token from a prior dry-run; required to execute a by-filter forget."},
            "reason": {"type": "string", "description": "Short, NON-sensitive reason (shown in the tombstone). Do not echo sensitive content."},
            "restore": {"type": "boolean", "description": "Un-forget: restore a previously-forgotten memory_id to its original text."},
            "force": {"type": "boolean", "description": "Override the soft review cap (still bounded by hard ceilings/floors)."},
        },
        "required": [],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "HARD, IRREVERSIBLE. Physically remove a memory from the shared store — it "
        "CANNOT be restored. Use ONLY for wrong/sensitive data that must not exist; "
        "for 'no longer true' prefer mem0_forget (reversible). By-id (read-before-"
        "delete); a gated by-filter path requires a dry-run then a confirm_token and "
        "is capped. delete_linked also removes the superseded chain (counted)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Single memory id to hard-delete."},
            "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "Batch of memory ids to hard-delete."},
            "filter": {"type": "string", "description": "Natural-language/structured filter for bulk delete (dry-run first, returns a confirm_token)."},
            "confirm_token": {"type": "string", "description": "Token from a prior dry-run; required to execute a by-filter delete."},
            "delete_linked": {"type": "boolean", "description": "Also delete the superseded/linked chain (default false; full chain counted against caps)."},
            "force": {"type": "boolean", "description": "Override the soft review cap up to the hard ceiling (never beyond)."},
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._host = ""
        self._admin_api_key = ""
        self._ca_bundle = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._temporal_search = False
        self._temporal_tz = _TEMPORAL_DEFAULT_TZ
        self._temporal_overfetch = _TEMPORAL_DEFAULT_OVERFETCH
        self._capture = "auto"
        self._prefetch_result = ""
        self._prefetch_qmd = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future: Optional[Future] = None
        self._prefetch_submit_lock = threading.Lock()
        self._prefetch_pending: Optional[str] = None
        self._prefetch_timed_out = False
        # Rotation epoch: bumped whenever the prefetch executor is rotated on timeout.
        # A zombie worker (the abandoned in-flight thread shutdown() can't kill) carries
        # its submit-time epoch and must not write results or drain pending queries once
        # the epoch has moved on (Greptile P1: zombie overwrites fresh results).
        self._prefetch_epoch = 0
        self._prefetch_join_timeout_s = 10.0
        self._qmd_cfg = qmd_recall.load_qmd_config(None)
        self._qmd_enabled = False
        # Phase 2b: gbrain document-leg state (flag-gated QMD replacement, default off).
        self._gbrain_cfg = gbrain_recall.load_gbrain_config(None)
        self._gbrain_enabled = False
        self._gbrain_prefetch_enabled = False
        self._gbrain_search_enabled = False
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        # Destructive-tools state (gated; Apollo+Aegis only)
        self._destructive_enabled = False
        self._destructive_cfg = dict(_DEFAULT_DESTRUCTIVE)
        self._ledger_lock = threading.Lock()
        self._mint_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        host = str(cfg.get("host", "") or "").strip()
        if host:
            return bool(str(cfg.get("admin_api_key", "") or "").strip())
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform (cloud) API key. Required for cloud mode; leave empty for a self-hosted server (set host + admin_api_key instead).", "secret": True, "required": False, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 OSS server URL (direct REST mode)", "default": "", "env_var": "MEM0_HOST"},
            {"key": "admin_api_key", "description": "Self-hosted Mem0 OSS server admin API key", "secret": True, "required": False, "env_var": "MEM0_ADMIN_API_KEY"},
            {"key": "ca_bundle", "description": "Path to a CA bundle PEM for verifying a self-hosted HTTPS endpoint signed by a private CA (e.g. mem0.ace). Empty = system trust store.", "default": "", "env_var": "MEM0_CA_BUNDLE"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
            {"key": "capture", "description": "Per-turn auto-capture mode (auto=capture every turn; off=recall-only, no per-turn writes)", "default": "auto", "choices": ["auto", "off"], "env_var": "MEM0_CAPTURE"},
            {"key": "destructive_tools_enabled", "description": "Expose mem0_forget/mem0_delete (Apollo+Aegis ONLY — fail-closed; default off)", "default": "false", "choices": ["true", "false"], "env_var": "MEM0_DESTRUCTIVE_TOOLS"},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._host:
                self._client = _DirectRestMem0Client(
                    self._host, self._admin_api_key, self._agent_id, self._user_id,
                    ca_bundle=self._ca_bundle,
                )
                return self._client
            try:
                from mem0 import MemoryClient
                # Bound the SDK's httpx pool. The mem0 SDK builds httpx.Client with
                # no limits + no keepalive expiry, so idle keepalive sockets the edge
                # half-closes rot into CLOSE_WAIT and leak fds in a long-lived gateway
                # (the incident in HANDOFF-fd-leak-client-pool.md). A bounded client
                # with keepalive_expiry actively reaps idle conns. The SDK's client=
                # path sets base_url + auth on the client we pass (mem0/client/main.py
                # v2.0.4). Self-hosted mode uses _DirectRestMem0Client (stdlib urllib,
                # per-call close — leak-proof) so this only guards the cloud fallback.
                try:
                    import httpx
                    bounded = httpx.Client(
                        timeout=300,
                        limits=httpx.Limits(
                            max_connections=10,
                            max_keepalive_connections=5,
                            keepalive_expiry=30.0,
                        ),
                    )
                    self._client = MemoryClient(api_key=self._api_key, client=bounded)
                except (ImportError, TypeError):
                    # httpx absent, or a future SDK that no longer accepts client=:
                    # fall back to the default client (latent leak returns — pin/verify
                    # the SDK client= contract on upgrade, per the handoff).
                    self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
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
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    @staticmethod
    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    # Exact-token query detector (W2-RERANK gate). The cross-encoder reranker is a
    # SEMANTIC relevance model — on exact-identifier lookups (IP / port / email /
    # long hex / dotted-or-hyphenated ID) it DEMOTES the exact-match row that RRF +
    # the exact-token arm already rank #1 (measured on the clone TEST split: ip 10→3,
    # email 2→0). So when a query is dominated by an exact token, force rerank OFF and
    # let RRF own it. Mirrors the server's `_EXACT_TOKEN_RE` shape so the gate matches
    # the arm that wins these queries.
    _EXACT_TOKEN_RE = re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"                 # IPv4
        r"|\bport\s+\d{2,5}\b"                          # "port 8443"
        r"|\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b" # email
        r"|\b[0-9a-f]{8,}\b"                            # long hex (commit/firmware/id)
        r"|\b[a-z0-9]+(?:[.\-_][a-z0-9]+){2,}\b",       # dotted/hyphen/underscore compound id
        re.I,
    )

    @classmethod
    def _is_exact_token_query(cls, query: Optional[str]) -> bool:
        """True when the query is an exact-identifier lookup → rerank should stay OFF
        (RRF + the exact-token arm already nail these; the cross-encoder regresses them)."""
        if not query:
            return False
        return bool(cls._EXACT_TOKEN_RE.search(query))

    def _rerank_killed(self) -> bool:
        """Runtime kill-switch for rerank, the correctness canary's auto-revert surface
        (INV-8). Read FRESH from mem0.json each call (NOT cached at init) so the canary
        can flip `retrieval_kill.rerank: true` and have the live plugin honor it on the
        very next turn — no gateway restart. Targeted read (just the one key) rather than
        a full _load_config() env-default reconstruction, because this is on the per-turn
        hot path (queue_prefetch + every mem0_search). Fail-open: a missing file / read /
        parse error never forces rerank off and never crashes recall."""
        try:
            from hermes_constants import get_hermes_home
            cfg_path = get_hermes_home() / "mem0.json"
            kill = (json.loads(cfg_path.read_text(encoding="utf-8")).get("retrieval_kill") or {})
            return self._truthy(kill.get("rerank", False))
        except Exception:
            return False

    # -- Prefetch relevance floor (INV-1..7; spec 2026-07-06 prefetch-relevance-floor v0.6) --
    #
    # The mem0 /search score is RRF-fused + reranked → saturates ~1.0 for ANY query, so it
    # CANNOT gate relevance. A live benchmark (§0) ALSO proved client-side cosine can't
    # separate vague-ack junk (0.32-0.46) from on-topic memories (0.41-0.51) — they overlap.
    # So the fix is TWO layers:
    #   GATE A (primary): query-specificity. A pure-acknowledgment turn ("yes please","ok")
    #     has ZERO content tokens after stripping a filler wordlist → inject NOTHING (there
    #     is no recall intent for a memory to be relevant to). This is what kills the bug.
    #   GATE B (weak secondary): a LOW cosine floor (0.25, below the on-topic band) that only
    #     trims a truly-orthogonal candidate slipping into a substantive query's top-5.
    # Both fail OPEN to today's behavior on any error; both stamp a greppable outcome.

    _PREFETCH_FLOOR_DEFAULT_COSINE = 0.10    # Gate B fallback (benchmarked 2026-07-06: 0.10=recall .959/prec .983; operational value in mem0.json, D-6)
    _PREFETCH_FLOOR_COSINE_CLAMP = 0.95      # D-7: an impossible threshold can never fully starve recall
    _PREFETCH_FLOOR_DEFAULT_MIN_CONTENT = 1  # Gate A fallback; PINNED at 1 (INV-7 — a 1-content-token command must pass)
    _PREFETCH_FLOOR_MAX_EMBED_TIMEOUT_S = 3.0
    _PREFETCH_FLOOR_MIN_EMBED_TIMEOUT_S = 1.0

    # -- L2 rerank-score gate (spec 2026-07-07) --------------------------------------------
    # The mem0 /search response already returns a per-row `rerank_score` (the cross-encoder's
    # raw logit) SEPARATE from the RRF-fused `score`. Unlike `score` (saturates ~1.0) and the
    # client cosine (flat 0.40-0.46 on this store), rerank_score SEPARATES on-topic (+2..+5)
    # from off-topic junk (-6..-11) — measured live 2026-07-07. This gate drops any candidate
    # below `min_rerank`. Default threshold 0.0 (keep only what the reranker rates positively).
    # Config-gated (default OFF), fails OPEN (a missing rerank_score → keep all), exact-token
    # queries bypass (rerank unreliable for IP/email/port, same carve-out as Gate B).
    _PREFETCH_RERANK_DEFAULT_MIN = 0.0

    # Pure acknowledgment / politeness / filler tokens ONLY (INV-7: never a domain verb/noun,
    # never a ≤2-char filter — "ac","tv","ip" are real content). A message that reduces to
    # zero non-wordlist tokens carries no recall intent.
    _PREFETCH_ACK_WORDLIST = frozenset({
        "yes", "yeah", "yep", "yup", "ya", "yah", "yes'm", "aye",
        "no", "nope", "nah",
        "ok", "okay", "k", "kk", "okey", "okie",
        "sure", "surely", "fine", "alright", "aight", "right", "correct",
        "please", "pls", "plz", "thanks", "thank", "thx", "ty", "cheers",
        "you", "u", "do", "it", "that", "this", "the", "a", "an",
        "go", "ahead", "proceed", "continue", "carry", "on",
        "sounds", "sound", "good", "great", "perfect", "nice", "cool", "sweet",
        "awesome", "excellent", "lovely", "brilliant", "wonderful",
        "done", "got", "gotcha", "roger", "copy", "noted", "understood",
        "and", "then", "also", "too", "let", "lets", "let's", "us",
        "to", "for", "of", "is", "are", "am", "be", "can", "could", "would",
        "should", "will", "i", "my", "me", "we", "our", "your",
        "bet", "word", "facts", "true", "indeed", "absolutely",
        "definitely", "certainly", "totally", "exactly", "yea",
        "hmm", "hm", "mhm", "mmhm", "uh", "um", "oh", "ah", "ha", "haha", "lol",
        "so", "well", "just", "now", "here", "there", "yet", "still",
    })

    def _prefetch_floor_cfg(self) -> dict:
        """Fresh read of the ``prefetch_relevance_floor`` block from mem0.json each call
        (canary pattern, like _rerank_killed) so enable/threshold flips take effect on the
        next turn with no gateway restart. Fail-safe: a missing file / read / PARSE error
        (incl. a mid-write truncated mem0.json) → {} → floor stays ENABLED at the default
        threshold (the safer behavior), never crashes recall."""
        try:
            from hermes_constants import get_hermes_home
            cfg_path = get_hermes_home() / "mem0.json"
            blk = json.loads(cfg_path.read_text(encoding="utf-8")).get("prefetch_relevance_floor")
            return blk if isinstance(blk, dict) else {}
        except Exception:
            return {}

    def _prefetch_floor_enabled(self) -> bool:
        """Default-ON, fail-safe-ON (INV-3): unset/garbage → enabled."""
        cfg = self._prefetch_floor_cfg()
        if "enabled" not in cfg:
            return True
        return self._truthy(cfg.get("enabled"))

    def _prefetch_min_content_tokens(self) -> int:
        """Gate A threshold (default 1). A garbage value falls to the default."""
        cfg = self._prefetch_floor_cfg()
        try:
            v = int(cfg.get("min_content_tokens", self._PREFETCH_FLOOR_DEFAULT_MIN_CONTENT))
            return v if v >= 1 else self._PREFETCH_FLOOR_DEFAULT_MIN_CONTENT
        except (TypeError, ValueError):
            return self._PREFETCH_FLOOR_DEFAULT_MIN_CONTENT

    @classmethod
    def _content_token_count(cls, text) -> int:
        """Count tokens NOT in the acknowledgment wordlist (Gate A). No min-length filter
        (INV-7: 'ac','tv','ip' are real content). A non-string / empty / error input → a high
        count (fail-open: treat the turn as substantive so we never wrongly blank recall on a
        parsing hiccup or a missing turn string)."""
        try:
            if not isinstance(text, str) or not text.strip():
                return 999  # fail-open: no usable turn string → treat as substantive
            toks = re.findall(r"[a-z0-9']+", text.lower())
            return sum(1 for t in toks if t not in cls._PREFETCH_ACK_WORDLIST)
        except Exception:
            return 999  # fail-open: treat as substantive, never gate on an error

    @staticmethod
    def _prefetch_query_hash(text) -> str:
        """Short non-reversible hash of the gated query for FIELD AUDIT (INV-6/RC pass-4):
        lets us tell WHICH query was over-gated without logging its content."""
        try:
            import hashlib
            return hashlib.sha256((text or "").encode("utf-8", "ignore")).hexdigest()[:10]
        except Exception:
            return "-"

    def _prefetch_specificity_gated(self, query) -> bool:
        """GATE A (INV-7): True → this turn is pure acknowledgment/filler, inject nothing.
        Wrapped fail-open: any error → NOT gated (treat as substantive)."""
        try:
            if not self._prefetch_floor_enabled():
                return False
            return self._content_token_count(query) < self._prefetch_min_content_tokens()
        except Exception:
            return False

    def _prefetch_embed_timeout(self, budget_s: float) -> float:
        """INV-5/RC-2: give the floor embed at most min(3s, real time left before the join
        ceiling), floored at 1s. Below the floor, the caller skips the embed and fails open."""
        try:
            b = float(budget_s)
        except (TypeError, ValueError):
            b = self._PREFETCH_FLOOR_MAX_EMBED_TIMEOUT_S
        return max(self._PREFETCH_FLOOR_MIN_EMBED_TIMEOUT_S,
                   min(self._PREFETCH_FLOOR_MAX_EMBED_TIMEOUT_S, b))

    def _prefetch_floor_cosine(self) -> float:
        """Gate B threshold, clamped to [0, 0.95] with warn-on-out-of-range (D-7). A garbage
        or out-of-range value never fully starves recall."""
        cfg = self._prefetch_floor_cfg()
        raw = cfg.get("min_cosine", self._PREFETCH_FLOOR_DEFAULT_COSINE)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            logger.warning("mem0.prefetch_floor min_cosine=%r not a number; using default %.2f",
                           raw, self._PREFETCH_FLOOR_DEFAULT_COSINE)
            return self._PREFETCH_FLOOR_DEFAULT_COSINE
        clamped = max(0.0, min(self._PREFETCH_FLOOR_COSINE_CLAMP, val))
        if clamped != val:
            logger.warning("mem0.prefetch_floor min_cosine=%s out of range; clamped to %.2f",
                           val, clamped)
        return clamped

    def _apply_gate_b_cosine(self, query, results, *, budget_s: float):
        """GATE B (weak secondary): drop candidates whose real cosine-to-query is below the
        low floor (default 0.25). Returns (kept_results, outcome) where outcome ∈ {disabled,
        ran_kept_N_of_M, failed_open}. ALWAYS fails OPEN (returns the full ``results``) on any
        path that can't produce a trustworthy cosine — only ever REMOVES a proven-low-cosine
        candidate, never loses one to its own infra failure (INV-2)."""
        try:
            if not self._prefetch_floor_enabled():
                return results, "disabled"
            # Exact-token queries (IP/port/email) BYPASS Gate B (D-5): cosine is
            # known-unreliable for that class, so flooring risks a false-drop.
            if self._is_exact_token_query(query):
                return results, "disabled"
            texts = [r.get("memory", "") for r in results]
            if not any(texts):
                return results, "disabled"
            # Below the min timeout there is no time to embed before the join ceiling —
            # fail open fast rather than issue a doomed call (INV-5).
            if budget_s is not None and float(budget_s) < self._PREFETCH_FLOOR_MIN_EMBED_TIMEOUT_S:
                logger.info("mem0.prefetch_floor outcome=failed_open reason=no_budget of=%d", len(results))
                return results, "failed_open"
            budget = self._prefetch_embed_timeout(budget_s)
            vecs = self._dedup_embed([query] + texts, timeout=budget)
            if not vecs or len(vecs) != len(texts) + 1:
                logger.info("mem0.prefetch_floor outcome=failed_open reason=embed of=%d", len(results))
                return results, "failed_open"
            floor = self._prefetch_floor_cosine()
            qv = vecs[0]
            # Score every candidate so the telemetry can show the actual relevance
            # DISTRIBUTION (not just a kept/total count). This is the signal that answers
            # "is recall injecting junk?" — a `ran_kept_5_of_5` with a top cosine of 0.12 is
            # junk that slipped the floor; the same count with a 0.55 top is genuinely on-topic.
            scored = [(r, self._dedup_cos(qv, cv)) for r, cv in zip(results, vecs[1:])]
            kept = [r for r, c in scored if c >= floor]
            cosines = sorted((round(c, 3) for _, c in scored), reverse=True)
            # min/median/max over ALL candidates + how many cleared the floor. No memory TEXT
            # is logged (privacy) — only the scalar cosines and counts.
            cmax = cosines[0] if cosines else 0.0
            cmin = cosines[-1] if cosines else 0.0
            cmed = cosines[len(cosines) // 2] if cosines else 0.0
            logger.info(
                "mem0.prefetch_floor outcome=ran_kept_%d_of_%d floor=%.2f "
                "cos_max=%.3f cos_med=%.3f cos_min=%.3f cos=%s q=%s",
                len(kept), len(results), floor, cmax, cmed, cmin,
                ",".join(f"{c:.3f}" for c in cosines),
                self._prefetch_query_hash(query),
            )
            return kept, f"ran_kept_{len(kept)}_of_{len(results)}"
        except Exception as e:
            # Any unforeseen error → fail open (INV-2), never break recall.
            logger.info("mem0.prefetch_floor outcome=failed_open reason=exc:%s of=%d",
                        type(e).__name__, len(results) if results else 0)
            return results, "failed_open"

    # -- L2 rerank-score gate (spec 2026-07-07) ------------------------------------------------
    def _prefetch_cfg_block(self, key: str) -> dict:
        """Fresh read of a named config block from mem0.json each call (canary pattern), so
        enable/threshold flips take effect on the next turn with no restart. Fail-safe: a
        missing file / read / parse error → {} (the layer falls back to its default = OFF)."""
        try:
            from hermes_constants import get_hermes_home
            cfg_path = get_hermes_home() / "mem0.json"
            blk = json.loads(cfg_path.read_text(encoding="utf-8")).get(key)
            return blk if isinstance(blk, dict) else {}
        except Exception:
            return {}

    def _rerank_gate_enabled(self) -> bool:
        """Default-OFF (unlike the cosine floor which is default-on): the rerank gate ships inert
        and is enabled by data. Read fresh each call (canary pattern) so a flip takes effect on
        the next turn with no restart."""
        cfg = self._prefetch_cfg_block("prefetch_rerank_gate")
        return self._truthy(cfg.get("enabled", False))

    def _rerank_gate_min(self) -> float:
        """Threshold; default 0.0 (keep only what the cross-encoder rates positively). A garbage
        value falls back to the default."""
        cfg = self._prefetch_cfg_block("prefetch_rerank_gate")
        try:
            return float(cfg.get("min_rerank", self._PREFETCH_RERANK_DEFAULT_MIN))
        except (TypeError, ValueError):
            return self._PREFETCH_RERANK_DEFAULT_MIN

    def _apply_rerank_gate(self, query, results):
        """Drop candidates whose server-provided `rerank_score` is below `min_rerank`. Returns
        (kept, outcome). ALWAYS fails OPEN (returns full `results`) when disabled, on an
        exact-token query (rerank unreliable there), or if ANY row lacks a numeric rerank_score
        (rerank was off / server variant) — only ever removes a proven-low-rerank candidate,
        never loses one to a missing field."""
        try:
            if not self._rerank_gate_enabled():
                return results, "rr_disabled"
            if not results:
                return results, "rr_disabled"
            if self._is_exact_token_query(query):
                return results, "rr_bypass_exact"
            scores = []
            for r in results:
                rs = r.get("rerank_score") if isinstance(r, dict) else None
                if not isinstance(rs, (int, float)):
                    # missing/non-numeric on any row → can't trust the gate → fail open
                    logger.info("mem0.prefetch_rerank outcome=failed_open reason=no_score of=%d",
                                len(results))
                    return results, "rr_failed_open"
                scores.append(float(rs))
            floor = self._rerank_gate_min()
            kept = [r for r, s in zip(results, scores) if s >= floor]
            ordered = sorted(scores, reverse=True)
            logger.info(
                "mem0.prefetch_rerank outcome=kept_%d_of_%d min=%.2f "
                "rr_max=%.2f rr_min=%.2f rr=%s q=%s",
                len(kept), len(results), floor, ordered[0], ordered[-1],
                ",".join(f"{s:.2f}" for s in ordered),
                self._prefetch_query_hash(query),
            )
            return kept, f"rr_kept_{len(kept)}_of_{len(results)}"
        except Exception as e:
            logger.info("mem0.prefetch_rerank outcome=failed_open reason=exc:%s of=%d",
                        type(e).__name__, len(results) if results else 0)
            return results, "rr_failed_open"

    # -- L3 dynamic top-k by rerank gap (spec 2026-07-07) --------------------------------------
    def _rerank_gap_cfg(self):
        cfg = self._prefetch_cfg_block("prefetch_rerank_gap")
        enabled = self._truthy(cfg.get("enabled", False))
        try:
            gap = float(cfg.get("max_gap", 6.0))
        except (TypeError, ValueError):
            gap = 6.0
        # A negative gap would make `(top - s) <= gap` unmatchable and silently
        # clear every L2 survivor — a config typo must not fail closed. Clamp.
        if gap < 0:
            gap = 6.0
        return enabled, gap

    def _apply_rerank_gap(self, query, results):
        """Even above the rerank threshold, drop the long tail: keep only candidates within
        `max_gap` of the TOP rerank_score. Prevents a '1 great + 4 mediocre' turn from injecting
        all 5. Runs AFTER L2. Default-OFF, fails OPEN, needs numeric rerank_scores (else keep)."""
        try:
            enabled, gap = self._rerank_gap_cfg()
            if not enabled or not results or self._is_exact_token_query(query):
                return results, "gap_disabled"
            scores = []
            for r in results:
                rs = r.get("rerank_score") if isinstance(r, dict) else None
                if not isinstance(rs, (int, float)):
                    return results, "gap_failed_open"
                scores.append(float(rs))
            top = max(scores)
            kept = [r for r, s in zip(results, scores) if (top - s) <= gap]
            logger.info("mem0.prefetch_gap outcome=kept_%d_of_%d gap=%.1f top=%.2f q=%s",
                        len(kept), len(results), gap, top, self._prefetch_query_hash(query))
            return kept, f"gap_kept_{len(kept)}_of_{len(results)}"
        except Exception as e:
            logger.info("mem0.prefetch_gap outcome=failed_open reason=exc:%s of=%d",
                        type(e).__name__, len(results) if results else 0)
            return results, "gap_failed_open"



    @staticmethod
    def _resolve_orig_lane(kwargs: Dict[str, Any]) -> str:
        """Best-effort lane/source label for pin-window provenance.

        Gateway sessions pass platform (discord/telegram/etc.). CLI/cron/subagent
        paths may not have a raw sender id, so lane is required for reversible
        provenance of pin-on writes where orig_sender_id is absent.
        """
        for key in ("platform", "source", "agent_context"):
            value = str(kwargs.get(key) or "").strip()
            if value:
                return value
        return "unknown"

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        self._host = str(self._config.get("host", "") or "").strip().rstrip("/")
        self._admin_api_key = str(self._config.get("admin_api_key", "") or "").strip()
        self._ca_bundle = str(self._config.get("ca_bundle", "") or "").strip()
        self._pin_user_id = self._truthy(self._config.get("pin_user_id", False))
        try:
            self._prefetch_join_timeout_s = float(self._config.get("prefetch_join_timeout_s", 10.0) or 10.0)
        except (TypeError, ValueError):
            self._prefetch_join_timeout_s = 10.0
        raw_gateway_user_id = kwargs.get("user_id")
        configured_user_id = str(self._config.get("user_id") or "").strip()
        if self._pin_user_id:
            # Fail closed: pin mode exists to prevent platform-id fragmentation, so
            # silently falling back to the gateway sender id would defeat the gate.
            if not configured_user_id:
                raise RuntimeError("mem0 pin_user_id=true but no canonical user_id is configured")
            self._user_id = configured_user_id
            self._orig_sender_id = raw_gateway_user_id
            self._orig_lane = self._resolve_orig_lane(kwargs)
        else:
            # Existing behavior: gateway-provided user_id wins; config/env is only
            # a fallback for CLI/single-user sessions.
            self._user_id = raw_gateway_user_id or configured_user_id or "hermes-user"
            self._orig_sender_id = None
            self._orig_lane = None
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)
        # keyword_search: hybrid BM25+semantic toggle. Resolved here at init (config/env
        # floor) and threaded to the client search call at BOTH enumerated call-sites so
        # setting `keyword_search: true` in mem0.json actually reaches the wire (the
        # param-drop fix listed it but initialize never read it -> it was a silent no-op).
        # None/unset -> omitted from the body so the server resolves its own default
        # (INV-8(i)).
        self._keyword_search = self._config.get("keyword_search", None)
        # QMD unified-recall fold-in (spec v0.3). Default-off; loaded from the `qmd`
        # sub-block of mem0.json. Stores stay separate — this only adds a read-only
        # local-document SEARCH leg to prefetch + mem0_search (INV-1, never writes).
        # Config block renamed `qmd` -> `mem0_qmd` (clearer: it's the mem0<->QMD integration,
        # not QMD itself). Read the new key, fall back to the legacy `qmd` block so a
        # config/code skew on a running gateway can never break recall.
        self._qmd_cfg = qmd_recall.load_qmd_config(
            self._config.get("mem0_qmd", self._config.get("qmd"))
        )
        self._qmd_enabled = self._truthy(self._qmd_cfg.get("enabled", False))
        # Sub-lane gates: each requires the master `enabled` AND its own toggle (default true, so
        # flipping only `enabled` behaves exactly as before). Lets an operator kill just the
        # every-turn PREFETCH lane (cost + noise) while keeping the explicit mem0_search fan-out.
        self._qmd_prefetch_enabled = self._qmd_enabled and self._truthy(
            self._qmd_cfg.get("prefetch_enabled", True)
        )
        self._qmd_search_enabled = self._qmd_enabled and self._truthy(
            self._qmd_cfg.get("search_enabled", True)
        )
        # Phase 2b: gbrain document leg — a flag-gated ALTERNATIVE backend for the
        # document lanes (prefetch + mem0_search `docs`). Read from the `mem0_gbrain`
        # (fallback `gbrain`) block of the SAME config surface the qmd block lives in.
        # Default OFF (deploy is inert). When enabled, it REPLACES the QMD leg —
        # one retrieval leg per turn, never both — via the effective-gate derivation
        # below; when disabled every gate is False and behavior is byte-identical
        # to today.
        self._gbrain_cfg = gbrain_recall.load_gbrain_config(
            self._config.get("mem0_gbrain", self._config.get("gbrain"))
        )
        self._gbrain_enabled = self._truthy(self._gbrain_cfg.get("enabled", False))
        self._gbrain_prefetch_enabled = self._gbrain_enabled and self._truthy(
            self._gbrain_cfg.get("prefetch_enabled", True)
        )
        self._gbrain_search_enabled = self._gbrain_enabled and self._truthy(
            self._gbrain_cfg.get("search_enabled", True)
        )
        # One-leg-per-lane rule: gbrain, when on for a lane, SUPERSEDES QMD for that
        # lane (QMD stays configured + untouched for instant rollback).
        if self._gbrain_prefetch_enabled:
            self._qmd_prefetch_enabled = False
        if self._gbrain_search_enabled:
            self._qmd_search_enabled = False
        # W3-TEMPORAL (tau_m created_at window) — plugin-side, config-gated, reversible
        # (INV-4). Off by default so deploy is inert until the flag flips. When on,
        # mem0_search detects a temporal expression, resolves it to a created_at
        # [start,end) UTC window (DST-correct PT-day bounds), over-fetches, and
        # boosts in-window candidates to the top (recency as a BOOST, never a hard
        # filter — the best in-window match is never dropped, spec 2.1). This
        # changes only WHICH rows rank, not the injected payload size/prefix (INV-3).
        self._temporal_search = self._truthy(self._config.get("temporal_search", False))
        self._temporal_tz = str(
            self._config.get("temporal_tz", _TEMPORAL_DEFAULT_TZ) or _TEMPORAL_DEFAULT_TZ
        ).strip()
        try:
            self._temporal_overfetch = max(
                1, int(self._config.get("temporal_overfetch", _TEMPORAL_DEFAULT_OVERFETCH))
            )
        except (ValueError, TypeError):
            self._temporal_overfetch = _TEMPORAL_DEFAULT_OVERFETCH
        # Capture mode: "auto" (default) syncs every completed turn to Mem0 for
        # server-side extraction. "off"/"manual" keeps recall (prefetch + search)
        # and explicit mem0_conclude writes, but skips per-turn auto-capture —
        # used for latency-sensitive / high-traffic agents (e.g. voice agents)
        # where we want shared recall without paying a write on every turn.
        # Resolution + the env>config>default precedence live in the shared
        # resolve_capture() (P2-2) so the flip-lag drift cron resolves intent
        # IDENTICALLY (import, never replicate).
        self._capture, self._capture_source = resolve_capture(
            os.environ.get("MEM0_CAPTURE"),
            self._config.get("capture"),
        )
        # F1-L1: make the in-effect capture state an explicit, timestamped,
        # greppable fact in agent.log (turns the silent flip-lag — disk says off
        # but the running process still captures — into something verifiable by
        # pid). "verify it took" = grep this line for the CURRENT pid (P2-6).
        logger.info(
            "mem0: capture=%s (source=%s); auto-capture %s for this process (pid %s)",
            self._capture, self._capture_source,
            "ON" if capture_is_on(self._capture) else "OFF", os.getpid(),
        )

        # Destructive tools gate (C1, fail-closed): only enabled when the
        # profile's mem0.json sets destructive_tools_enabled true (Apollo+Aegis).
        # Env override MEM0_DESTRUCTIVE_TOOLS for tests/ops.
        flag = os.environ.get("MEM0_DESTRUCTIVE_TOOLS")
        if flag is None:
            flag = self._config.get("destructive_tools_enabled", False)
        self._destructive_enabled = str(flag).strip().lower() in ("1", "true", "yes", "on")
        # Per-key config overrides (all caps/limits tunable via mem0.json)
        self._destructive_cfg = dict(_DEFAULT_DESTRUCTIVE)
        for k in self._destructive_cfg:
            if k in self._config and self._config[k] is not None:
                try:
                    self._destructive_cfg[k] = type(self._destructive_cfg[k])(self._config[k])
                except (ValueError, TypeError):
                    pass

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self, write_kind: str = "auto") -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution.

        When pin_user_id is enabled, new writes also carry audit-only provenance
        so rows written during the pin window can be restored to their platform
        bucket if the feature is rolled back. Historical backfill MUST NOT stamp
        these keys; it rewrites only user_id and is guarded by payload hashes.

        write_kind (F1, 2026-06-21): provenance of HOW this row was written —
        "auto" (per-turn sync_turn extraction, infer=True) vs "deliberate"
        (mem0_conclude / seeders / backfills, verbatim). Stamped on EVERY write
        regardless of pin_user_id, so the capture-flip-lag drift detector can
        count auto-writes exactly. Rows WITHOUT this key are legacy/pre-deploy =
        unknown, NEVER inferred as auto (absence-of-stamp != auto).
        """
        filters: Dict[str, Any] = {"user_id": self._user_id, "agent_id": self._agent_id}
        metadata: Dict[str, Any] = {"write_kind": write_kind}
        if getattr(self, "_pin_user_id", False):
            metadata["orig_lane"] = getattr(self, "_orig_lane", "unknown") or "unknown"
            orig_sender = getattr(self, "_orig_sender_id", None)
            if orig_sender is not None and str(orig_sender) != "":
                metadata["orig_sender_id"] = str(orig_sender)
        filters["metadata"] = metadata
        return filters

    def _dedup_then_write(self, client, fact: str) -> Dict[str, Any]:
        """Write a background-review fact through the dedup ladder (D-5).

        Tier 1: exact-hash skip (normalized MD5, server-side `filters` lookup).
        Tier 2: two-band cosine (>= IDENTICAL skip; ambiguous band WRITES — DD-1:
                cosine is sign-blind, dropping the newer fact is unrecoverable).
        Stamps write_origin=background_review + dedup_hash on the write.
        Returns {"result": ..., "dedup": <tag>} so the digest can split outcomes.
        """
        norm_hash = _dedup_norm_hash(fact)

        # Resolve thresholds (config-overridable, D-7).
        try:
            threshold = float(self._config.get("dedup_cosine_threshold", _DEDUP_COSINE_THRESHOLD))
        except (TypeError, ValueError, AttributeError):
            threshold = _DEDUP_COSINE_THRESHOLD
        try:
            identical = float(self._config.get("dedup_cosine_identical", _DEDUP_COSINE_IDENTICAL))
        except (TypeError, ValueError, AttributeError):
            identical = _DEDUP_COSINE_IDENTICAL

        # --- Tier 1: exact-hash skip (TRUE nested-filters lookup, server-side) ---
        try:
            hit = client.search_meta_filtered(fact, {"dedup_hash": norm_hash}, top_k=1)
            if self._unwrap_results(hit):
                return {"result": "Already stored (exact dup).", "dedup": "skipped_exacthash"}
        except Exception:
            # Fail-open: a dedup-check failure must never block a real write.
            pass

        # --- Tier 2: two-band cosine pre-write check ---
        # The live /search score is RRF-fused + reranked (top hit ~1.0 for ANY query),
        # NOT a cosine — useless as a dedup signal (probe 2026-06-27). So we use /search
        # only for CANDIDATE RETRIEVAL, then compute REAL cosine client-side against the
        # candidate texts via the same embedder the store uses (text-embedding-3-small).
        band = "write"
        try:
            sem = client.search(query=fact, top_k=self._dedup_candidate_k())
            rows = self._drop_forgotten(self._unwrap_results(sem))
            cand_texts = []
            for r in rows:
                if isinstance(r, dict):
                    t = r.get("memory") or r.get("data") or ""
                    if t:
                        cand_texts.append(t)
            if cand_texts:
                vecs = self._dedup_embed([fact] + cand_texts)
                if vecs and len(vecs) == len(cand_texts) + 1:
                    qv = vecs[0]
                    top_cos = max(self._dedup_cos(qv, cv) for cv in vecs[1:])
                    band = _dedup_cosine_band(top_cos, threshold, identical)
        except Exception:
            band = "write"

        if band == "skip_identical":
            return {"result": "Already stored (near-identical).", "dedup": "skipped_identical"}

        # band in ("write", "write_ambiguous") -> WRITE (ambiguous never skips, DD-1)
        write_filters = self._write_filters(write_kind="deliberate")
        meta = write_filters.get("metadata", {})
        meta["write_origin"] = "background_review"
        meta["dedup_hash"] = norm_hash
        write_filters["metadata"] = meta
        client.add([{"role": "user", "content": fact}], **write_filters, infer=False)
        tag = "wrote_ambiguous" if band == "write_ambiguous" else "wrote"
        return {"result": "Fact stored.", "dedup": tag}

    def _dedup_candidate_k(self) -> int:
        """How many retrieval candidates to cosine-check (config-overridable)."""
        try:
            return int(self._config.get("dedup_candidate_k", 5))
        except (TypeError, ValueError, AttributeError):
            return 5

    def _dedup_embed(self, texts, *, timeout: float = 15):
        """Embed texts with the SAME model the store uses (text-embedding-3-small, 1536d).

        Returns a list of vectors aligned to ``texts``, or None on any failure (the
        caller fails-open to WRITE). Uses the OpenAI embeddings REST API directly with
        the key already in the runtime env — no new dependency, no SDK.

        ``timeout`` (keyword-only, default 15s = legacy dedup behavior) bounds the socket
        read. The per-turn prefetch relevance floor passes a SMALLER budget-derived value
        so a slow embed can never blow the prefetch join ceiling (INV-5). Dedup callers
        keep the 15s default → byte-identical to before this widening.
        """
        import urllib.request as _u
        key = os.environ.get("OPENAI_API_KEY", "") or (self._config.get("openai_api_key", "") if self._config else "")
        if not key or not texts:
            return None
        model = (self._config.get("dedup_embed_model") if self._config else None) or "text-embedding-3-small"
        body = json.dumps({"model": model, "input": list(texts)}).encode("utf-8")
        req = _u.Request("https://api.openai.com/v1/embeddings", data=body, method="POST",
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with _u.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [d["embedding"] for d in data.get("data", [])]
        except Exception:
            return None

    @staticmethod
    def _dedup_cos(a, b) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    @staticmethod
    def _is_forgotten(m: Any) -> bool:
        """True if a memory carries the forget tombstone (metadata flag or text sentinel)."""
        if not isinstance(m, dict):
            return False
        meta = m.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("forgotten") is True:
            return True
        text = m.get("memory") or ""
        return isinstance(text, str) and text.lstrip().startswith(_FORGOTTEN_PREFIX)

    @classmethod
    def _drop_forgotten(cls, memories: list) -> list:
        """C9 single choke point: strip forgotten memories from any read result.

        Every read surface (search, profile, prefetch) MUST route through this so a
        soft-forgotten memory genuinely stops influencing the agent. Authoritative
        client-side filter (get_all has no server-side metadata exclude).
        """
        return [m for m in (memories or []) if not cls._is_forgotten(m)]

    @staticmethod
    def _apply_temporal_boost(results: list, window) -> list:
        """W3-TEMPORAL: stable-partition results so in-window candidates rank first.

        Recency is a BOOST, not a hard filter (spec §2.1): rows whose created_at
        falls in the [start,end) UTC window move ahead of out-of-window rows, but
        every row is retained and relative order WITHIN each group is preserved
        (Python's sort is stable) — so the best in-window match is never dropped,
        and an out-of-window semantic hit still surfaces if nothing in-window beats
        it. Rows missing/unparseable created_at sort as out-of-window (never
        promoted on a bad timestamp).
        """
        if not results:
            return results
        return sorted(
            results,
            key=lambda r: 0 if created_at_in_window(r.get("created_at"), window) else 1,
        )

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Recall with mem0_search; mem0_profile for an overview. Auto-capture is OFF, so "
            "memories persist across sessions ONLY when you deliberately save them with "
            "mem0_conclude. When you learn something durable about the user or their environment "
            "— a preference, a standing decision or correction, an account/device/topology fact, a "
            "long-lived plan or constraint — save it with mem0_conclude in that turn, proactively, "
            "without being asked. Do NOT save work-narration, status, or transient state."
        )

    def _qmd_pointers(self, query: str, *, limit: int, deadline_s: float) -> list:
        """Run the read-only QMD document leg. Degraded-safe: any failure -> []."""
        if not self._qmd_enabled:
            return []
        cfg = self._qmd_cfg
        try:
            return qmd_recall.qmd_query(
                query,
                limit=int(limit),
                min_score=float(cfg.get("min_score", 0.5)),
                collections=cfg.get("collections") or None,
                rerank=self._truthy(cfg.get("prefetch_rerank", True)),
                deadline_s=float(deadline_s),
                url=str(cfg.get("url") or "http://[::1]:8181/mcp"),
                exclude_globs=cfg.get("exclude_path_globs") or None,
                use_rerank_score_floor=self._truthy(cfg.get("use_rerank_score_floor", False)),
                rerank_score_min=float(cfg.get("rerank_score_min", cfg.get("min_score", 0.5))),
            )
        except Exception as e:  # belt-and-suspenders; qmd_query already swallows
            logger.debug("QMD prefetch leg failed: %s", e)
            return []

    def _gbrain_pointers(self, query: str, *, limit: int, deadline_s: float) -> list:
        """Run the read-only gbrain document leg (Phase 2b). Same contract as
        _qmd_pointers: returns [{file,title,score,line,docid}] pointers, and is
        degraded-safe — ANY failure (serve down, auth broken, deadline hit) -> []
        so a down gbrain never breaks a turn and never falls back to blocking."""
        if not self._gbrain_enabled:
            return []
        cfg = self._gbrain_cfg
        try:
            return gbrain_recall.gbrain_search(
                query,
                limit=int(limit),
                min_score=float(cfg.get("min_score", 0.5)),
                deadline_s=float(deadline_s),
                url=str(cfg.get("url") or "http://127.0.0.1:8199"),
                creds_path=str(cfg.get("creds_path") or "~/gbrain/.gbrain/rail-client.env"),
            )
        except Exception as e:  # belt-and-suspenders; gbrain_search already swallows
            logger.debug("gbrain prefetch leg failed: %s", e)
            return []

    def _prefetch_executor_for_submit(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is None:
            self._prefetch_executor = _DaemonThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mem0-prefetch",
            )
        return self._prefetch_executor

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        future = self._prefetch_future
        if future and not future.done():
            try:
                future.result(timeout=self._prefetch_join_timeout_s)
            except FutureTimeoutError:
                with self._prefetch_submit_lock:
                    if self._prefetch_future is future:
                        self._prefetch_timed_out = True
            except Exception as e:
                logger.debug("Mem0 prefetch worker failed: %s", e)
        with self._prefetch_lock:
            result = self._prefetch_result
            qmd_block = self._prefetch_qmd
            self._prefetch_result = ""
            self._prefetch_qmd = ""
        # mem0 block is rendered EXACTLY as before (byte-identical when QMD off, INV-6/AC1);
        # the QMD block is strictly additive and only joined when present (INV-3a/m2).
        mem0_block = f"## Mem0 Memory\n{result}" if result else ""
        return qmd_recall.join_blocks(mem0_block, qmd_block)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run_once(run_query: str, epoch: int):
            t_start = time.monotonic()
            try:
                # GATE A (specificity, PRIMARY — spec v0.6): a pure-acknowledgment turn
                # ("yes please","ok do it") carries no recall intent, so inject NOTHING and
                # SKIP the /search round-trip entirely. This is the actual fix for the
                # "yes please → random memory" bug (a live benchmark proved cosine alone
                # can't separate the junk from on-topic; query specificity can). Fail-open:
                # _prefetch_specificity_gated swallows any error → not gated.
                if self._prefetch_specificity_gated(run_query):
                    logger.info("mem0.prefetch_floor outcome=gated_specificity n_content=0 q=%s",
                                self._prefetch_query_hash(run_query))
                    with self._prefetch_lock:
                        if epoch == self._prefetch_epoch:
                            self._prefetch_result = ""   # inject nothing (D-4)
                    self._record_success()
                else:
                    client = self._get_client()
                    # INV-8(ii) PREFETCH profile (the every-turn hot path). Ace's call
                    # (2026-06-24): rerank RIDES prefetch too (the cross-encoder's +40
                    # semantic / +11 temporal win is worth it), bounded by the prefetch
                    # join ceiling (now 10s, config prefetch_join_timeout_s). Two gates keep
                    # it honest: (1) exact-token queries skip rerank — the cross-encoder
                    # REGRESSES IP/email/port that RRF already nails; (2) a runtime kill-flag
                    # (mem0.json retrieval_kill.rerank) the correctness canary can flip to
                    # auto-revert without a redeploy (INV-8 control surface). An EXPLICIT flag
                    # — one of exactly two enumerated call-sites that send retrieval flags.
                    _pf_rerank = (
                        self._truthy(self._rerank)
                        and not self._is_exact_token_query(run_query)
                        and not self._rerank_killed()
                    )
                    results = self._drop_forgotten(self._unwrap_results(client.search(
                        query=run_query,
                        filters=self._read_filters(),
                        rerank=_pf_rerank,
                        keyword_search=self._keyword_search,
                        top_k=5,
                    )))
                    _floor_outcome = "no_results"
                    _injected = 0
                    # L2 RERANK GATE (PRIMARY, spec 2026-07-07): the cross-encoder's per-row
                    # rerank_score SEPARATES on-topic (+2..+5) from off-topic junk (-6..-11)
                    # where the RRF score (saturates ~1.0) and cosine (flat) cannot. Runs FIRST
                    # (cheapest — no embed, just reads the score already in the response) and
                    # drops sub-threshold candidates. Default-OFF (config-gated); fails OPEN.
                    _rr_outcome = "rr_disabled"
                    if results:
                        results, _rr_outcome = self._apply_rerank_gate(run_query, results)
                    # L3 GAP (dynamic top-k): trim the long tail beyond `max_gap` of the top
                    # rerank_score. Runs after L2, config-gated (default OFF), fails OPEN.
                    _gap_outcome = "gap_disabled"
                    if results:
                        results, _gap_outcome = self._apply_rerank_gap(run_query, results)
                    # GATE B (cosine, weak SECONDARY): a substantive query passed Gate A, so
                    # trim any truly-orthogonal candidate (cos < low floor) that slipped into
                    # the top-5. Budget = time left before the 10s join ceiling (−0.5s margin);
                    # a slow/failed embed fails OPEN to the un-floored results. Empty→inject
                    # nothing (D-4).
                    if results:
                        _join_to = (self._config.get("prefetch_join_timeout_s", 10) if self._config else 10) or 10
                        _floor_budget = float(_join_to) - (time.monotonic() - t_start) - 0.5
                        results, _floor_outcome = self._apply_gate_b_cosine(
                            run_query, results, budget_s=_floor_budget
                        )
                    # INV-3a: commit the mem0 block FIRST — it is never dropped because the
                    # QMD leg is slow. QMD is strictly additive and runs after.
                    # Gate B may drain ALL candidates (empty results); make "inject nothing"
                    # EXPLICIT like Gate A does — never rely on _prefetch_result being pre-cleared
                    # by a preceding prefetch() (a double queue_prefetch could otherwise leave a
                    # prior query's stale block; Greptile #212 P2).
                    with self._prefetch_lock:
                        if epoch == self._prefetch_epoch:
                            if results:
                                lines = [r.get("memory", "") for r in results if r.get("memory")]
                                self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                                _injected = len(lines)
                            else:
                                self._prefetch_result = ""
                                _injected = 0
                    # Final "what recall actually injected this turn" line — fires on EVERY
                    # substantive turn (past Gate A), regardless of whether Gate B ran, was
                    # disabled, or exact-token-bypassed. `injected` is the number of memory lines
                    # placed in front of the model; `floor_outcome` says which path decided it.
                    # No memory text (privacy). This is the top-level recall observability row.
                    logger.info(
                        "mem0.prefetch injected=%d floor_outcome=%s rr_outcome=%s gap_outcome=%s rerank=%s exact=%s q=%s",
                        _injected, _floor_outcome, _rr_outcome, _gap_outcome, _pf_rerank,
                        self._is_exact_token_query(run_query),
                        self._prefetch_query_hash(run_query),
                    )
                    self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

            # INV-4a/AC12: the mem0 leg has its OWN budget. If mem0 already overran it,
            # the join ceiling is at risk — skip QMD entirely rather than stack a second
            # multi-second leg on top of a slow mem0 (a slow mem0 must not be made worse
            # by QMD). When mem0 was fast, the QMD leg gets the smaller of its own
            # deadline and the time actually remaining before the join ceiling, so the
            # two legs combined can never blow prefetch_join_timeout_s (INV-7).
            try:
                mem0_budget = float(self._qmd_cfg.get("mem0_budget_s", 6.0))
                mem0_elapsed = time.monotonic() - t_start
                qmd_deadline = float(self._qmd_cfg.get("qmd_total_deadline_s", 4.0))
                remaining = float(self._prefetch_join_timeout_s) - mem0_elapsed - 0.25
                eff_deadline = min(qmd_deadline, remaining)
                if (
                    self._qmd_prefetch_enabled
                    and mem0_elapsed <= mem0_budget
                    and eff_deadline >= 0.5
                    and qmd_recall.is_lookup_intent(
                        run_query, int(self._qmd_cfg.get("intent_min_tokens", 4))
                    )
                ):
                    hits = self._qmd_pointers(
                        run_query,
                        limit=int(self._qmd_cfg.get("prefetch_limit", 3)),
                        deadline_s=eff_deadline,
                    )
                    block = qmd_recall.render_qmd_block(hits)
                    if block:
                        with self._prefetch_lock:
                            if epoch == self._prefetch_epoch:
                                self._prefetch_qmd = block
            except Exception as e:
                logger.debug("QMD prefetch leg failed: %s", e)

            # Phase 2b: gbrain document leg — the flag-gated REPLACEMENT for the QMD
            # leg above (mutually exclusive: enabling gbrain forces
            # _qmd_prefetch_enabled False at init, so exactly one leg runs per turn).
            # Same budget discipline (INV-4a/INV-7): skipped when mem0 overran its
            # budget; deadline is min(own deadline, join-ceiling remainder); same
            # intent gate. Degraded-safe end to end.
            try:
                if self._gbrain_prefetch_enabled:
                    gb_budget = float(self._gbrain_cfg.get("mem0_budget_s", 6.0))
                    gb_elapsed = time.monotonic() - t_start
                    gb_deadline = float(self._gbrain_cfg.get("total_deadline_s", 4.0))
                    gb_remaining = float(self._prefetch_join_timeout_s) - gb_elapsed - 0.25
                    gb_eff_deadline = min(gb_deadline, gb_remaining)
                    if (
                        gb_elapsed <= gb_budget
                        and gb_eff_deadline >= 0.5
                        and qmd_recall.is_lookup_intent(
                            run_query, int(self._gbrain_cfg.get("intent_min_tokens", 1))
                        )
                    ):
                        gb_hits = self._gbrain_pointers(
                            run_query,
                            limit=int(self._gbrain_cfg.get("prefetch_limit", 3)),
                            deadline_s=gb_eff_deadline,
                        )
                        gb_block = gbrain_recall.render_gbrain_block(gb_hits)
                        if gb_block:
                            with self._prefetch_lock:
                                if epoch == self._prefetch_epoch:
                                    self._prefetch_qmd = gb_block
            except Exception as e:
                logger.debug("gbrain prefetch leg failed: %s", e)

        def _run_latest(first_query: str, epoch: int):
            run_query = first_query
            while True:
                _run_once(run_query, epoch)
                with self._prefetch_submit_lock:
                    if epoch != self._prefetch_epoch or self._prefetch_pending is None:
                        return
                    run_query = self._prefetch_pending
                    self._prefetch_pending = None

        with self._prefetch_submit_lock:
            future = self._prefetch_future
            if future and not future.done():
                if self._prefetch_timed_out:
                    if self._prefetch_executor is not None:
                        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
                    self._prefetch_executor = None
                    self._prefetch_future = None
                    self._prefetch_pending = None
                    self._prefetch_timed_out = False
                    # Invalidate the abandoned worker: it may still be running inside a
                    # stuck network call and must not write results or drain pending
                    # queries once we hand the lane to a fresh executor.
                    self._prefetch_epoch += 1
                else:
                    # Do not let the single-worker executor queue unbounded stale
                    # prefetches. Keep only the latest request; the worker drains it
                    # after the in-flight call completes.
                    self._prefetch_pending = query
                    return
            self._prefetch_pending = None
            self._prefetch_timed_out = False
            self._prefetch_thread = None
            self._prefetch_future = self._prefetch_executor_for_submit().submit(
                _run_latest, query, self._prefetch_epoch
            )

    def _live_capture(self) -> str:
        """Re-resolve the capture flag from the LIVE source (env > mem0.json `capture` > default),
        so a flip is honored WITHOUT a provider restart and WITHOUT lag (fixes the flip-lag footgun).

        Cache invalidation is by mem0.json MTIME + the env var, not a time window: the hot path only
        does a cheap stat(), and re-reads/parses the file only when it actually changed. So an
        operator's auto<->off flip is picked up on the very next capture decision — no rollback lag
        (Greptile P1) — while steady-state calls stay cheap. On any error, falls back to the
        init-time self._capture. Also updates self._capture so interlock + logs stay consistent."""
        try:
            from hermes_constants import get_hermes_home
            cfg_path = get_hermes_home() / "mem0.json"
            try:
                mtime = cfg_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            env_val = os.environ.get("MEM0_CAPTURE")
            sig = (mtime, env_val)
            if sig == getattr(self, "_live_capture_sig", None) and getattr(self, "_live_capture_val", None) is not None:
                return self._live_capture_val
            cfg_capture = None
            if mtime and cfg_path.exists():
                cfg_capture = json.loads(cfg_path.read_text(encoding="utf-8")).get("capture")
            value, source = resolve_capture(env_val, cfg_capture)
            self._live_capture_val = value
            self._live_capture_sig = sig
            # keep the frozen fields in sync so the interlock / logs reflect the live decision
            self._capture, self._capture_source = value, source
            return value
        except Exception:
            return self._capture

    def _build_capture_router(self):
        """Build the Arm-B two-pass capture router from the `mem0_capture_router` sub-block of
        mem0.json, or return None if the flag is absent/off (default OFF -> byte-identical to today).
        Degrade-safe: any import/construction error -> None (router disabled, drain worker unchanged).

        The router runs ADDITIVELY on top of the unchanged mem0 write path and NEVER writes to mem0;
        it stages world/event facts to disk while `staging_mode` is true (default). No new env var:
        the flag + all knobs live in mem0.json; the bridge secrets come from 1Password via `op read`
        exactly as the codex-/gemini-bridge launchers resolve them."""
        try:
            from . import capture_router
        except ImportError:  # flat import (unit tests with PYTHONPATH=<dir>)
            try:
                import capture_router  # type: ignore
            except Exception:
                return None
        try:
            return capture_router.build_router_from_config(self._config or {})
        except Exception as e:
            logger.warning("mem0 capture-router build failed (router disabled): %s", e)
            return None

    def _get_capture_pipeline(self):
        """Lazy-build the A-lite capture pipeline (queue + drain worker + gate-version guard +
        cross-process bgr interlock). Composed, not inlined — see capture_pipeline.py. Built once,
        degrade-safe: on any construction error, returns None and capture stays off (INV-3)."""
        pipe = getattr(self, "_capture_pipeline", None)
        if pipe is not None:
            # If capture was OFF at build time (worker not started) and has since been flipped ON,
            # an idle agent may have inherited un-drained rows that never got a start signal
            # (Greptile P1). Re-check on every access so a live enable activates the drain+reaper.
            try:
                pipe.maybe_start_pending()
            except Exception:
                pass
            return pipe
        try:
            from . import capture_pipeline, capture_scrub

            def _add(messages, kwargs) -> int:
                # Returns the number of memories the server extracted+wrote for this turn, so the
                # drainer knows whether to EXPECT rows in the post-write scrub read. FAIL-CLOSED
                # (Greptile P1): if the response shape is anything we can't confidently parse as an
                # explicit empty result, return 1 (assume >=1 written) so the scrub REQUIRES rows —
                # an unknown success shape must never let an empty read be read as "nothing written".
                resp = self._get_client().add(messages, **kwargs)
                self._record_success()
                try:
                    if isinstance(resp, dict) and "results" in resp:
                        return len(resp["results"] or [])
                    if isinstance(resp, list):
                        return len(resp)
                except Exception:
                    pass
                return 1   # unknown/opaque success shape -> assume a write happened (require_rows)

            def _recall_idem(key: str) -> int:
                # NOTE: must RAISE on transient failure (do NOT swallow to 0). The drain worker
                # treats an idem-check error as "unknown" and requeues fail-closed, so a transient
                # search fault never causes a duplicate re-add (Greptile P1).
                resp = self._get_client().search_meta_filtered(
                    "", {"capture_idem": key}, top_k=_CAPTURE_SCRUB_MAX_ROWS)
                rows = self._unwrap_results(resp)
                return len(rows)

            def _get_written(key: str):
                # Must RAISE on transient failure: the drainer's post-write scrub fails CLOSED
                # (requeues) if it cannot read the rows it just wrote, so a secret is never left
                # recallable behind a completed queue row (Greptile P1). Fetch a HIGH ceiling of
                # rows for the idem key — NOT one page.
                # COMPLETENESS GUARD (Greptile P1): if the result HITS the cap, we cannot prove we
                # saw every row (mem0 may have extracted more, or the backend may cap top_k). Rather
                # than scan a partial set and complete the item — leaving a possible secret beyond
                # the window — RAISE, which routes into the fail-closed scrub-requeue. In practice a
                # single turn yields a handful of facts, so this never trips; it just makes "assumed
                # complete" impossible.
                resp = self._get_client().search_meta_filtered(
                    "", {"capture_idem": key}, top_k=_CAPTURE_SCRUB_MAX_ROWS)
                rows = self._unwrap_results(resp)
                if len(rows) >= _CAPTURE_SCRUB_MAX_ROWS:
                    raise RuntimeError(
                        f"capture scrub read hit the {_CAPTURE_SCRUB_MAX_ROWS}-row cap for "
                        f"capture_idem={key!r}; cannot prove the page is complete — failing closed")
                return [{"id": r.get("id", ""), "memory": r.get("memory", "")}
                        for r in rows]

            def _forget(mid: str):
                # Must RAISE on failure (Greptile P1): the drain worker's scrub fails CLOSED and
                # requeues if a forget of a secret-bearing memory doesn't land. Swallowing here would
                # let the row be marked done with the secret still recallable.
                self._get_client().update(mid, text=f"{_FORGOTTEN_PREFIX} [secret-scrubbed]",
                                          metadata={"forgotten": True, "capture_scrubbed": True})

            pipe = capture_pipeline.CapturePipeline(
                # LIVE capture source (Greptile P1 — flip-lag footgun): resolve the capture flag at
                # DECISION TIME from the same precedence resolver, not the value frozen at init, so a
                # live capture flip (off<->on) is honored without a provider restart.
                capture_on_fn=lambda: capture_is_on(self._live_capture()),
                add_fn=_add,
                recall_idem_fn=_recall_idem,
                scrub_fn=lambda facts: capture_scrub.filter_facts(facts),
                forget_fn=_forget,
                get_written_fn=_get_written,
                write_filters=self._write_filters(write_kind="auto"),
                model=str(self._config.get("capture_model", "gpt-5.4-mini")),
                breaker_open_fn=self._is_breaker_open,
                alert_fn=lambda m: logger.warning("MEM0-CAPTURE-ALERT %s", m),
                # Arm-B two-pass capture router (Phase 2.5), flag-gated via mem0.json
                # `mem0_capture_router.enabled` (default OFF). None => byte-identical to today.
                router=self._build_capture_router(),
            )
            self._capture_pipeline = pipe
            return pipe
        except Exception as e:
            logger.warning("mem0 capture pipeline unavailable (capture disabled): %s", e)
            self._capture_pipeline = None
            return None

    def _auto_capture_active(self) -> bool:
        """True only when the FOREGROUND per-turn capture path would ACTUALLY write — i.e. capture is
        configured on AND a certified capture pipeline is available (gate assets present + version
        matches). This must mirror EXACTLY the condition under which sync_turn enqueues, so the D-7
        interlock never suppresses the background writer (mem0_remember) in a state where the
        foreground path is ALSO not writing (Greptile P1): capture=auto but gate missing/mismatched
        would otherwise drop the fact on the floor from BOTH paths.

        The interlock is cross-process (foreground session vs background-review fork are separate
        processes); the shared, decision-time signals are the persisted capture flag and the shipped,
        version-pinned gate assets, both resolved identically in either process. Degrade-safe: any
        error -> False (interlock does not block the background writer)."""
        try:
            if not capture_is_on(self._live_capture()):
                return False
            pipe = self._get_capture_pipeline()
            return bool(pipe is not None and pipe._certified)
        except Exception:
            return False

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Enqueue the completed turn for durable, salience-gated, server-side auto-capture.

        A-lite: the ONLY synchronous step is a tiny durable enqueue (INV-3, never blocks the turn);
        the background drain worker does the slow server-side extraction (mem0 runs the salience gate
        as custom_instructions) + retry + secret-scrub. Capture only fires when capture_is_on AND the
        certified gate version matches (D-11). Degrade-safe: any failure leaves the turn untouched.
        Reads the LIVE capture flag (self._live_capture) so an off->on/on->off flip takes effect
        without a restart (flip-lag footgun).
        """
        if not capture_is_on(self._live_capture()):
            return
        # SIZE CAP (Greptile P1: don't drop a user fact just because the ASSISTANT side is large).
        # The poison case the provider rejects (HTTP 502 "provider rejected as malformed") is a giant
        # *input* — a pasted multi-KB review/system/tool-dump prompt with no durable user facts. That
        # lives in user_content. A normal turn with a small user fact ("my DNS is AdGuard") but a huge
        # assistant/tool response is legitimate and MUST still be captured. So:
        #   - if the USER side alone exceeds the cap -> skip (true poison prompt, can't succeed), else
        #   - keep the turn but TRUNCATE the assistant side to the remaining budget, preserving the
        #     user fact while staying under the provider's malformed-payload ceiling.
        user_content = user_content or ""
        assistant_content = assistant_content or ""
        if len(user_content) > _CAPTURE_MAX_TURN_CHARS:
            logger.debug("mem0 sync_turn: user content exceeds capture size cap (%d chars) — skipped",
                         len(user_content))
            return
        budget = _CAPTURE_MAX_TURN_CHARS - len(user_content)
        if len(assistant_content) > budget:
            logger.debug("mem0 sync_turn: assistant content truncated %d->%d chars to fit size cap "
                         "(user fact preserved)", len(assistant_content), budget)
            assistant_content = assistant_content[:budget]
        pipe = self._get_capture_pipeline()
        if pipe is None:
            return
        try:
            # a stable per-turn ordinal so the idempotency key is deterministic across retries
            self._capture_turn_ordinal = getattr(self, "_capture_turn_ordinal", 0) + 1
            pipe.enqueue_turn(user_content, assistant_content,
                              session_id=session_id or "default",
                              turn_ordinal=self._capture_turn_ordinal)
        except Exception as e:
            logger.warning("mem0 sync_turn enqueue failed (turn not captured): %s", e)

    def capture_stats(self) -> Dict[str, Any]:
        """Observability for the daily digest: certified?, gate version, queue depth, drain counters."""
        pipe = getattr(self, "_capture_pipeline", None)
        if pipe is None:
            return {"capture": self._capture, "pipeline": "not-built"}
        try:
            return {"capture": self._capture, **pipe.stats()}
        except Exception as e:
            return {"capture": self._capture, "error": str(e)[:120]}

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]
        if self._destructive_enabled:
            schemas += [FORGET_SCHEMA, DELETE_SCHEMA]
        return schemas

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._drop_forgotten(
                    self._unwrap_results(client.get_all(filters=self._read_filters())))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            # INV-8(ii) rerank profile (the deliberate precision path): when the model
            # does not pass rerank, fall back to the configured rerank profile
            # (self._rerank, config-driven + reversible per INV-4), NOT the server
            # default. The model may still force it per-call. Coerced because the config
            # value can be a JSON string ("true"/"false") loaded from mem0.json. This is
            # the SECOND of exactly two enumerated flag-passing call-sites (call-site lint).
            rerank = args.get("rerank")
            rerank = self._truthy(self._rerank if rerank is None else rerank)
            # W2-RERANK gate: an exact-identifier lookup (IP/port/email/long-id) skips
            # rerank — the cross-encoder demotes the exact match RRF already ranks #1.
            # A model-explicit rerank=true overrides the exact-token gate (the gate only
            # suppresses the PROFILE default), but the canary runtime kill-flag below is a
            # HARD override that wins even over a model-explicit rerank=true (safety > the
            # model's per-call preference; the canary only trips on a measured regression).
            if rerank and args.get("rerank") is None and self._is_exact_token_query(query):
                rerank = False
            if rerank and self._rerank_killed():
                rerank = False
            top_k = min(int(args.get("top_k", 10)), 50)
            # W3-TEMPORAL: if the query carries a temporal expression and the feature
            # is on, resolve it to a created_at [start,end) UTC window and over-fetch a
            # deeper candidate pool so the in-window boost can surface a correctly-dated
            # row that sits below the top_k semantic cut. No window / feature off → the
            # fetch is byte-identical to before (fetch_k == top_k). This is NOT a third
            # flag-passing call-site (INV-8): it only varies top_k, an already-enumerated
            # flag on this call-site.
            window = None
            if self._temporal_search:
                try:
                    window = parse_temporal_window(query, tz_name=self._temporal_tz)
                except Exception as e:  # never let a parse bug break recall
                    logger.debug("Mem0 temporal parse failed (%s); ignoring window", e)
                    window = None
            fetch_k = max(top_k, self._temporal_overfetch) if window else top_k
            try:
                results = self._drop_forgotten(self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    keyword_search=self._keyword_search,
                    top_k=fetch_k,
                )))
                self._record_success()
                if window is not None:
                    results = self._apply_temporal_boost(results, window)
                results = results[:top_k]
                # INV-7/AC4: explicit search fans out to QMD regardless of the intent
                # gate (the user chose to search). Additive `docs` key only; mem0 result
                # is computed exactly as before. QMD off or empty -> no `docs` key, so the
                # return is byte-identical to pre-change (INV-6/AC1, INV-8).
                qmd_docs = self._qmd_pointers(
                    query,
                    limit=int(self._qmd_cfg.get("search_limit", 5)),
                    deadline_s=float(self._qmd_cfg.get("qmd_total_deadline_s", 4.0)),
                ) if self._qmd_search_enabled else []
                # Phase 2b: gbrain replacement for the docs fan-out. Mutually
                # exclusive with QMD (gate derivation at init forces
                # _qmd_search_enabled False when gbrain owns this lane), so at
                # most ONE document backend is queried per call. Same additive
                # `docs` contract: off/empty -> no key, byte-identical reply.
                if self._gbrain_search_enabled:
                    qmd_docs = self._gbrain_pointers(
                        query,
                        limit=int(self._gbrain_cfg.get("search_limit", 5)),
                        deadline_s=float(self._gbrain_cfg.get("total_deadline_s", 4.0)),
                    )
                if not results:
                    if qmd_docs:
                        return json.dumps({"result": "No relevant memories found.", "docs": qmd_docs})
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                out = {"results": items, "count": len(items)}
                if qmd_docs:
                    out["docs"] = qmd_docs
                return json.dumps(out)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(write_kind="deliberate"),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        elif tool_name == "mem0_remember":
            fact = args.get("fact", "")
            if not fact:
                return tool_error("Missing required parameter: fact")
            # D-7 CROSS-PROCESS INTERLOCK (Greptile P1 — now ENFORCED, not just defined):
            # when foreground per-turn auto-capture is genuinely ACTIVE, the background-review writer
            # must NOT also write, or the two writers race overlapping facts. Key on the pipeline's
            # real activity (certified gate AND capture on), NOT merely the configured capture value —
            # so mem0_remember still writes when auto-capture isn't actually running (no certified
            # pipeline). Read at DECISION TIME so a live capture flip is honored without a restart.
            if self._auto_capture_active():
                logger.info("mem0_remember suppressed: auto-capture is ACTIVE (D-7 interlock)")
                return json.dumps({"status": "skipped",
                                   "reason": "auto-capture active; background write suppressed (D-7 interlock)"})
            try:
                result = self._dedup_then_write(client, fact)
                self._record_success()
                return json.dumps(result)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to remember: {e}")

        elif tool_name in ("mem0_forget", "mem0_delete"):
            # C1 fail-closed: if the gate is off these tools were never registered,
            # so reaching here means "unknown tool" (never a silent enabled-looking no-op).
            if not self._destructive_enabled:
                return tool_error(f"Unknown tool: {tool_name}")
            try:
                if tool_name == "mem0_forget":
                    return self._handle_forget(client, args)
                return self._handle_delete(client, args)
            except Exception as e:
                self._record_failure()
                return tool_error(f"{tool_name} failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    # -----------------------------------------------------------------------
    # Destructive tools (mem0_forget / mem0_delete) — gated, Apollo+Aegis only.
    # Spec: ~/.hermes/plans/2026-06-10_mem0-destructive-tools-spec.md
    # -----------------------------------------------------------------------

    def _hermes_home(self):
        from hermes_constants import get_hermes_home
        return get_hermes_home()

    def _ledger_path(self):
        return self._hermes_home() / "mem0-destructive-ledger.jsonl"

    def _mint_store_path(self):
        return self._hermes_home() / "mem0-mint-store.json"

    # --- C4 ledger: append-only, 0o600, write-before-act ---------------------

    def _ledger_append(self, record: dict) -> None:
        """Append one JSONL record (never mutate a prior line) and fsync.

        Caller appends a `pending` record (fsync'd) BEFORE the irreversible op,
        then a terminal `ok`/`fail` record after. Two rows, never a rewrite.
        """
        record = {"ts": time.time(), "agent_id": self._agent_id, **record}
        path = self._ledger_path()
        with self._ledger_lock:
            newfile = not path.exists()
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                if newfile:
                    try:
                        os.fchmod(fd, 0o600)
                    except OSError:
                        pass
                os.write(fd, (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def _velocity_count(self, op: str) -> int:
        """C8d: count trailing-hour terminal `ok` records of an op-type from the
        DURABLE ledger (not process memory — a restart can't reset the guard)."""
        path = self._ledger_path()
        if not path.exists():
            return 0
        cutoff = time.time() - 3600.0
        n = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if (rec.get("op") == op and rec.get("outcome") == "ok"
                            and rec.get("agent_id") == self._agent_id
                            and float(rec.get("ts", 0)) >= cutoff):
                        n += int(rec.get("rows", 1))
        except OSError:
            return 0
        return n

    # --- C3 confirm-token mint-store: random, single-use, TTL, set-hash ------

    @staticmethod
    def _set_hash(ids) -> str:
        import hashlib
        joined = "\n".join(sorted(str(i) for i in ids))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _load_mint_store(self) -> dict:
        path = self._mint_store_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def _save_mint_store(self, store: dict) -> None:
        path = self._mint_store_path()
        ttl = self._destructive_cfg["token_ttl_seconds"]
        now = time.time()
        pruned = {t: r for t, r in store.items()
                  if not r.get("consumed") and (now - r.get("issued_at", 0)) < ttl}
        tmp = str(path) + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(pruned).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, str(path))
        try:
            os.chmod(str(path), 0o600)
        except OSError:
            pass

    def _mint_token(self, ids) -> str:
        import secrets
        token = secrets.token_urlsafe(32)
        with self._mint_lock:
            store = self._load_mint_store()
            store[token] = {"set_hash": self._set_hash(ids),
                            "issued_at": time.time(), "consumed": False}
            self._save_mint_store(store)
        return token

    def _consume_token(self, token: str, ids):
        """Validate exist+unexpired+unconsumed+set-hash, then mark consumed.

        Returns (ok: bool, reason: str). Single-use: success marks consumed.
        """
        if not token:
            return False, "no confirm_token supplied"
        ttl = self._destructive_cfg["token_ttl_seconds"]
        with self._mint_lock:
            store = self._load_mint_store()
            rec = store.get(token)
            if not rec:
                return False, "no valid dry-run token (not minted / expired / already used)"
            if rec.get("consumed"):
                return False, "token already used"
            if (time.time() - rec.get("issued_at", 0)) >= ttl:
                return False, "token expired — re-run the dry-run"
            if rec.get("set_hash") != self._set_hash(ids):
                return False, "matched set changed since dry-run (TOCTOU) — re-run the dry-run"
            rec["consumed"] = True
            store[token] = rec
            self._save_mint_store(store)
        return True, "ok"

    # --- scope + filter resolution (C7) -------------------------------------

    def _resolve_filter(self, client, filter_arg: str) -> list:
        """Resolve a filter to an in-scope memory list (C7 scope-lock).

        The filter is ALWAYS constrained to the configured user_id; a caller can
        never reach another tenant's store. We fetch the full in-scope set and
        apply a best-effort substring match for a NL/text filter (the catastrophe
        floor C8a is the real guard, so over-matching is safe — it just refuses).
        An empty/whitespace filter matches the WHOLE in-scope set (→ C8a refuses).
        """
        allm = self._unwrap_results(client.get_all(filters=self._read_filters()))
        f = (filter_arg or "").strip().lower()
        if not f or f in ("*", "all", "everything"):
            return list(allm)
        return [m for m in allm if f in str(m.get("memory", "")).lower()]

    def _in_scope_total(self, client) -> int:
        """C8a denominator: ALWAYS the full configured user-wide scope count."""
        return len(self._unwrap_results(client.get_all(filters=self._read_filters())))

    def _mass_check(self, matched: int, total: int):
        """C8a: refuse (no token) if matched is a mass fraction of the store.
        Returns (is_mass: bool, reason: str)."""
        ratio = self._destructive_cfg["unscoped_ratio"]
        floor = self._destructive_cfg["absolute_mass_floor"]
        if matched >= floor:
            return True, (f"refused: {matched} memories ≥ absolute mass floor ({floor}). "
                          "Whole-store/mass deletion is an operator-CLI action, not an agent tool.")
        if total > 0 and (matched / total) >= ratio:
            return True, (f"refused: {matched}/{total} ({matched/total:.0%}) ≥ mass ratio "
                          f"({ratio:.0%}). Mass deletion is an operator-CLI action, not an agent tool.")
        return False, ""

    @staticmethod
    def _mem_id(m: dict):
        return m.get("id") or m.get("memory_id")

    def _read_before(self, client, mid: str):
        """C2: get(mid) before any destroy. Returns (mem_dict | None)."""
        try:
            m = client.get(mid)
            return m if isinstance(m, dict) and m else None
        except Exception:
            return None

    # --- forget handler ------------------------------------------------------

    def _handle_forget(self, client, args: dict) -> str:
        # restore (un-forget) path (C5)
        if args.get("restore"):
            return self._forget_restore(client, args.get("memory_id"))

        ids = self._collect_ids(args)
        if ids:
            return self._forget_ids(client, ids, args.get("reason", ""))
        if args.get("filter") is not None:
            return self._bulk("forget", client, args)
        return tool_error("mem0_forget needs memory_id, memory_ids, filter, or restore.")

    def _forget_ids(self, client, ids: list, reason: str) -> str:
        # velocity guard (C8d)
        used = self._velocity_count("forget")
        cap = self._destructive_cfg["max_forget_per_hour"]
        if used + len(ids) > cap:
            return tool_error(f"forget velocity exceeded: {used}/{cap} in the last hour; "
                              f"cooldown. (requested {len(ids)} more)")
        results = []
        for mid in ids:
            m = self._read_before(client, mid)
            if not m:
                results.append({"id": mid, "outcome": "not_found"})
                continue
            original = m.get("memory", "")
            if self._is_forgotten(m):
                results.append({"id": mid, "outcome": "already_forgotten"})
                continue
            meta = dict(m.get("metadata") or {})
            safe_reason = (reason or "superseded").replace("\n", " ")[:120]
            meta.update({"forgotten": True, "forgotten_at": time.time(),
                         "forgotten_by": self._agent_id, "original_text": original})
            self._ledger_append({"op": "forget", "mode": "id", "id": mid,
                                 "was": _trunc(original), "outcome": "pending"})
            try:
                client.update(mid, text=f"{_FORGOTTEN_PREFIX} {safe_reason}", metadata=meta)
                self._ledger_append({"op": "forget", "mode": "id", "id": mid,
                                     "rows": 1, "outcome": "ok"})
                results.append({"id": mid, "outcome": "forgotten", "was": _trunc(original)})
            except Exception as e:
                self._ledger_append({"op": "forget", "mode": "id", "id": mid,
                                     "outcome": "fail", "error": str(e)[:120]})
                results.append({"id": mid, "outcome": "fail", "error": str(e)[:120]})
        self._record_success()
        return json.dumps({"tool": "mem0_forget", "results": results,
                           "forgotten": sum(1 for r in results if r["outcome"] == "forgotten")})

    def _forget_restore(self, client, mid) -> str:
        if not mid:
            return tool_error("restore needs a memory_id.")
        m = self._read_before(client, mid)
        if not m:
            return json.dumps({"tool": "mem0_forget", "result": "not found, nothing restored", "id": mid})
        meta = dict(m.get("metadata") or {})
        if not self._is_forgotten(m):
            return json.dumps({"tool": "mem0_forget", "result": "not forgotten — no-op", "id": mid})
        original = meta.get("original_text")
        if not original:
            # fall back to history()
            try:
                hist = client.history(mid)
                hist = hist if isinstance(hist, list) else self._unwrap_results(hist)
                for h in hist or []:
                    cand = h.get("old_memory") or h.get("memory")
                    if cand and not str(cand).lstrip().startswith(_FORGOTTEN_PREFIX):
                        original = cand
                        break
            except Exception:
                pass
        if not original:
            return tool_error(f"cannot restore {mid}: no original_text in metadata or history.")
        meta.update({"forgotten": False, "restored_at": time.time(), "restored_by": self._agent_id})
        meta.pop("original_text", None)
        client.update(mid, text=original, metadata=meta)
        self._ledger_append({"op": "restore", "mode": "id", "id": mid, "rows": 1, "outcome": "ok"})
        self._record_success()
        return json.dumps({"tool": "mem0_forget", "result": "restored", "id": mid, "text": _trunc(original)})

    # --- delete handler ------------------------------------------------------

    def _handle_delete(self, client, args: dict) -> str:
        ids = self._collect_ids(args)
        if ids:
            return self._delete_ids(client, ids, bool(args.get("delete_linked", False)))
        if args.get("filter") is not None:
            return self._bulk("delete", client, args)
        return tool_error("mem0_delete needs memory_id, memory_ids, or filter.")

    def _delete_ids(self, client, ids: list, delete_linked: bool) -> str:
        # D3: when delete_linked, count the full superseded chain. Hosted SDK does
        # not expose the chain pre-delete, so we conservatively count provided ids;
        # the linked rows are surfaced post-hoc in the ledger response count.
        used = self._velocity_count("delete")
        cap = self._destructive_cfg["max_delete_per_hour"]
        if used + len(ids) > cap:
            return tool_error(f"delete velocity exceeded: {used}/{cap} in the last hour; "
                              f"cooldown. (requested {len(ids)} more)")
        # by-id batch also honors the soft review cap / hard ceiling
        over = self._cap_check("delete", len(ids), force=False)
        if over:
            return tool_error(over)
        results = []
        for mid in ids:
            m = self._read_before(client, mid)
            if not m:
                results.append({"id": mid, "outcome": "not_found"})
                continue
            original = m.get("memory", "")
            self._ledger_append({"op": "delete", "mode": "id", "id": mid,
                                 "was": _trunc(original), "delete_linked": delete_linked,
                                 "outcome": "pending"})
            try:
                client.delete(mid, delete_linked=delete_linked)
                self._ledger_append({"op": "delete", "mode": "id", "id": mid,
                                     "rows": 1, "outcome": "ok"})
                results.append({"id": mid, "outcome": "deleted", "was": _trunc(original)})
            except Exception as e:
                # abort-on-first-error for irreversible delete (ladder the rest)
                self._ledger_append({"op": "delete", "mode": "id", "id": mid,
                                     "outcome": "fail", "error": str(e)[:120]})
                results.append({"id": mid, "outcome": "fail", "error": str(e)[:120]})
                remaining = ids[ids.index(mid) + 1:]
                for r in remaining:
                    results.append({"id": r, "outcome": "skipped"})
                break
        self._record_success()
        return json.dumps({"tool": "mem0_delete",
                           "results": results,
                           "deleted": sum(1 for r in results if r["outcome"] == "deleted"),
                           "irreversible": True})

    # --- shared bulk by-filter path (C3 dry-run/token + C8 floor/cap) --------

    def _collect_ids(self, args: dict) -> list:
        ids = []
        if args.get("memory_id"):
            ids.append(args["memory_id"])
        if args.get("memory_ids"):
            ids.extend([i for i in args["memory_ids"] if i])
        # de-dupe, preserve order
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _cap_check(self, op: str, count: int, force: bool) -> str:
        """Return an error string if count breaches caps for op, else ''. C8b/C8c."""
        if op == "delete":
            soft = self._destructive_cfg["max_bulk"]
            hard = self._destructive_cfg["max_bulk_hard_force"]
            if count > hard:
                return (f"refused: {count} hard-deletes exceeds the absolute ceiling ({hard}); "
                        "force cannot breach it. Split into smaller ops or use mem0_forget.")
            if count > soft and not force:
                return (f"{count} hard-deletes exceeds the soft review cap ({soft}). "
                        "Re-run with force=true to proceed (still ≤ {} ceiling), or prefer "
                        "mem0_forget (reversible).".format(hard))
        else:  # forget
            soft = self._destructive_cfg["max_bulk_forget"]
            hard = self._destructive_cfg["max_bulk_forget_force"]
            if count > hard:
                return f"refused: {count} forgets exceeds the ceiling ({hard})."
            if count > soft and not force:
                return (f"{count} forgets exceeds the soft review cap ({soft}). "
                        "Re-run with force=true to proceed.")
        return ""

    def _bulk(self, op: str, client, args: dict) -> str:
        token = args.get("confirm_token")
        force = bool(args.get("force", False))
        matches = self._resolve_filter(client, args.get("filter", ""))
        ids = [self._mem_id(m) for m in matches if self._mem_id(m)]
        total = self._in_scope_total(client)

        # C8a catastrophe floor — refuse with NO token, for BOTH verbs.
        is_mass, why = self._mass_check(len(ids), total)
        if is_mass:
            return tool_error(why)

        # Dry-run: no token supplied → preview + mint token, delete nothing.
        if not token:
            cap_note = self._cap_check(op, len(ids), force=False)
            preview = [{"id": self._mem_id(m), "memory": _trunc(m.get("memory", ""))}
                       for m in matches[:25]]
            minted = self._mint_token(ids) if ids else None
            resp = {
                "tool": f"mem0_{op}", "dry_run": True, "count": len(ids),
                "total_in_scope": total, "preview": preview, "confirm_token": minted,
                "note": cap_note,
            }
            if op == "delete":
                resp["hint"] = ("IRREVERSIBLE. For bulk cleanup of 'no longer true' memories, "
                                "prefer mem0_forget (reversible). Re-call with confirm_token to execute.")
            else:
                resp["hint"] = "Re-call with confirm_token to execute the forget."
            return json.dumps(resp)

        # Execute path: validate token (exist+unexpired+unconsumed+set-hash).
        ok, reason = self._consume_token(token, ids)
        if not ok:
            return tool_error(reason)
        # Re-check mass + cap at execute time (defence in depth).
        is_mass, why = self._mass_check(len(ids), total)
        if is_mass:
            return tool_error(why)
        cap_err = self._cap_check(op, len(ids), force=force)
        if cap_err:
            return tool_error(cap_err)
        # velocity
        used = self._velocity_count(op)
        vcap = self._destructive_cfg["max_delete_per_hour" if op == "delete" else "max_forget_per_hour"]
        if used + len(ids) > vcap:
            return tool_error(f"{op} velocity exceeded: {used}/{vcap} in the last hour; cooldown.")

        if op == "forget":
            return self._forget_ids(client, ids, args.get("reason", "bulk forget"))
        return self._delete_ids_bulk(client, ids)

    def _delete_ids_bulk(self, client, ids: list) -> str:
        """Bulk hard-delete after token+cap+floor already validated.
        Caps already enforced by caller; here we just laddered-execute."""
        results = []
        for mid in ids:
            m = self._read_before(client, mid)
            if not m:
                results.append({"id": mid, "outcome": "not_found"})
                continue
            original = m.get("memory", "")
            self._ledger_append({"op": "delete", "mode": "filter", "id": mid,
                                 "was": _trunc(original), "outcome": "pending"})
            try:
                client.delete(mid)
                self._ledger_append({"op": "delete", "mode": "filter", "id": mid,
                                     "rows": 1, "outcome": "ok"})
                results.append({"id": mid, "outcome": "deleted", "was": _trunc(original)})
            except Exception as e:
                self._ledger_append({"op": "delete", "mode": "filter", "id": mid,
                                     "outcome": "fail", "error": str(e)[:120]})
                results.append({"id": mid, "outcome": "fail", "error": str(e)[:120]})
                for r in ids[ids.index(mid) + 1:]:
                    results.append({"id": r, "outcome": "skipped"})
                break
        self._record_success()
        return json.dumps({"tool": "mem0_delete", "mode": "filter", "results": results,
                           "deleted": sum(1 for r in results if r["outcome"] == "deleted"),
                           "irreversible": True})

    def shutdown(self) -> None:
        future = self._prefetch_future
        if future and not future.done():
            try:
                future.result(timeout=5.0)
            except FutureTimeoutError:
                pass
            except Exception as e:
                logger.debug("Mem0 background worker failed during shutdown: %s", e)
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        self._prefetch_executor = None
        self._prefetch_future = None
        self._prefetch_thread = None
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
