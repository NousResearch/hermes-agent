"""palace-discipline plugin — mechanical palace context injection.

Fires on ``pre_gateway_dispatch`` (a single hook fired per incoming
``MessageEvent`` BEFORE auth/pairing and BEFORE the model sees the
user message). For every fresh user-originated message the plugin:

1. Calls ``memory.init`` (HTTP ``POST /sessions/init``) on the local
   memory-palace MCP service to refresh active session state.
2. Infers the session type from the user's message keywords using the
   table in ``CLAUDE.md`` § Session Inference.
3. Loads the type-specific prompt via ``GET /prompts/<type>``.
4. If the platform is Discord, additionally loads
   ``GET /prompts/discord-delivery`` (delivery rules: MEDIA:/path
   syntax, no third-party uploads, send-once).
5. Runs two semantic searches in parallel (``GET /search/semantic``
   with corpus=skills and corpus=palace) on the user's message.
6. Prepends a single combined ``<palace_context>...</palace_context>``
   block to ``event.text`` and returns ``{"action": "rewrite",
   "text": ...}`` so the dispatcher carries the enriched message
   through normal processing.

Quality bars (mandated by the Session 18 brief):

* **Idempotent** per ``(platform, chat_id, conversation_id)``. A
  duplicate ``pre_gateway_dispatch`` for the same conversation in a
  short window does NOT re-inject (re-injection would double-load
  context and confuse the model).
* **Fast** — total wall-clock budget 2 s under healthy MCP, hard
  timeout 5 s (per request 1.5 s; capped via the executor).
* **Failure-tolerant** — any MCP failure (network, 5xx, missing file)
  is logged at WARNING and the plugin returns ``None`` so the
  dispatcher proceeds with the original event.text. A broken MCP must
  NEVER block Hermes.
* **Observable** — every fire writes one line to
  ``$HERMES_HOME/logs/palace-discipline.log`` describing what was
  injected (or why it was skipped/degraded).

Disable-able via ``plugins.disabled: [palace-discipline]`` in
``$HERMES_HOME/config.yaml`` — the plugin loader honors that and the
hook never fires. Rollback to pre-Session-18 behavior is one line.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MCP_BASE_URL = os.environ.get("PALACE_MCP_URL", "http://127.0.0.1:7411")
BEARER_TOKEN_PATH = Path(
    os.environ.get(
        "PALACE_MCP_BEARER_PATH",
        str(Path.home() / ".config" / "memory-palace" / "bearer-token"),
    )
)

PER_REQUEST_TIMEOUT_SEC = 1.5
TOTAL_BUDGET_SEC = 5.0

SKILLS_K = 5
CANON_K = 3

# Idempotency: cache "already-injected" conversation keys for this many
# seconds. Within the window, a duplicate pre_gateway_dispatch returns
# None (no rewrite). 30 minutes covers a typical multi-turn conversation
# while still letting a long pause re-inject fresh context.
IDEMPOTENCY_WINDOW_SEC = 30 * 60

LOG_DIR = Path.home() / ".hermes" / "logs"
LOG_FILE = LOG_DIR / "palace-discipline.log"

# ---------------------------------------------------------------------------
# Session-type inference
# ---------------------------------------------------------------------------

# Lower-case, whole-word matching. First match wins. Order matters:
# system-correction is checked BEFORE implementation because phrases like
# "fix the protocol" should route to system-correction, not implementation.
TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    # System-correction first: "tighten this rule" must beat
    # implementation's generic "fix " / "update".
    ("system-correction", [
        "tighten", "update the rule", "fix the protocol",
        "fix the rule", "correct the protocol", "tighten this",
        "update claude.md",
    ]),
    # Research before triage: "research X options" should be research,
    # not triage. The leading verb "research" / "look into" etc. is the
    # strong signal.
    ("research", [
        "research", "look into", "evaluate", "compare", "best ",
        "deep dive", "investigate",
    ]),
    ("triage", [
        "what's next", "whats next", "what should i", "priorities",
        "what's on the docket", "whats on the docket", "options",
        "what to work on",
    ]),
    ("design", [
        "design", "spec ", "spec the", "brainstorm", "figure out how",
        "plan how", "architecture for",
    ]),
    ("verification", [
        "verify", "check ", "confirm", "is it live", "is it up",
        "make sure", "validate",
    ]),
    ("communication", [
        "draft ", "draft an", "draft a", "email", "write to",
        "prep for", "compose",
    ]),
    ("ingestion", [
        "ingest", "file this", "capture this", "extract from",
        "process this", "save this",
    ]),
    ("implementation", [
        "build ", "implement", "fix ", "add ", "wire up",
        "ship ", "patch ", "refactor", "create the",
    ]),
]

DEFAULT_TYPE = "triage"


def _infer_session_type(text: str) -> str:
    """Return the inferred session type for a user message.

    Lower-cased substring match against TYPE_KEYWORDS in order. Default
    is `triage` per CLAUDE.md ("If the request is empty or ambiguous,
    default to triage").
    """
    if not text:
        return DEFAULT_TYPE
    lo = text.lower()
    for stype, keywords in TYPE_KEYWORDS:
        for kw in keywords:
            if kw in lo:
                return stype
    return DEFAULT_TYPE


# ---------------------------------------------------------------------------
# MCP HTTP client — minimal, urllib-only, no external deps
# ---------------------------------------------------------------------------

_bearer_cache: Dict[str, str] = {}


def _bearer_token() -> Optional[str]:
    """Read the bearer token from disk; cached after first read."""
    key = str(BEARER_TOKEN_PATH)
    if key in _bearer_cache:
        return _bearer_cache[key]
    try:
        token = BEARER_TOKEN_PATH.read_text().strip()
        if token:
            _bearer_cache[key] = token
            return token
    except OSError as exc:
        logger.warning("palace-discipline: bearer token read failed: %s", exc)
    return None


def _http_request(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = PER_REQUEST_TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    """Issue a single HTTP request to the local MCP service.

    Returns the decoded JSON dict on success, or ``None`` on any
    failure (network error, non-2xx status, malformed body). Never
    raises — the caller treats a ``None`` as a graceful skip.
    """
    url = f"{MCP_BASE_URL.rstrip('/')}{path}"
    data = None
    headers = {"Accept": "application/json"}
    token = _bearer_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - localhost only
            payload = resp.read()
            if not payload:
                return {}
            try:
                return json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning(
                    "palace-discipline: %s %s — bad JSON: %s", method, path, exc
                )
                return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(
            "palace-discipline: %s %s failed: %s", method, path, exc
        )
        return None


# ---------------------------------------------------------------------------
# Idempotency cache
# ---------------------------------------------------------------------------

_inject_lock = Lock()
_inject_seen: Dict[str, float] = {}  # conversation_key -> last-fired ts


def _conversation_key(event: Any) -> str:
    """Stable per-conversation key for idempotency.

    Falls back through whatever the event exposes — message_id is too
    granular (changes every turn), so we key on platform+chat_id which
    represents one conversation thread.
    """
    source = getattr(event, "source", None)
    platform = "unknown"
    chat = "unknown"
    if source is not None:
        platform = getattr(source.platform, "value", str(source.platform))
        chat = getattr(source, "chat_id", "") or "unknown"
    return f"{platform}:{chat}"


def _already_injected(key: str) -> bool:
    """Return True if ``key`` was injected within IDEMPOTENCY_WINDOW_SEC."""
    now = time.time()
    with _inject_lock:
        last = _inject_seen.get(key)
        if last is not None and (now - last) < IDEMPOTENCY_WINDOW_SEC:
            return True
        _inject_seen[key] = now
        # Opportunistic GC: drop entries older than 4x the window.
        cutoff = now - (4 * IDEMPOTENCY_WINDOW_SEC)
        stale = [k for k, ts in _inject_seen.items() if ts < cutoff]
        for k in stale:
            _inject_seen.pop(k, None)
    return False


def _reset_idempotency_for_tests() -> None:
    """Clear the idempotency cache. Used by the test suite only."""
    with _inject_lock:
        _inject_seen.clear()
    _bearer_cache.clear()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def _audit(line: str) -> None:
    """Append a single line to the plugin's audit log.

    Best-effort — log failures must not affect the message pipeline.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts} {line}\n")
    except OSError:
        pass  # never block on logging


# ---------------------------------------------------------------------------
# Hook implementation
# ---------------------------------------------------------------------------

def _platform_value(event: Any) -> str:
    src = getattr(event, "source", None)
    if src is None:
        return ""
    plat = getattr(src, "platform", None)
    return getattr(plat, "value", str(plat or ""))


def _is_discord(event: Any) -> bool:
    return _platform_value(event).lower() == "discord"


def _format_skills_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    lines = ["<relevant_skills>"]
    for h in hits:
        path = h.get("path", "?")
        snippet = (h.get("snippet") or "").strip()
        lines.append(f"- {path}: {snippet}")
    lines.append("</relevant_skills>")
    return "\n".join(lines)


def _format_canon_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    lines = ["<relevant_canon>"]
    for h in hits:
        path = h.get("path", "?")
        snippet = (h.get("snippet") or "").strip()
        lines.append(f"- {path}: {snippet}")
    lines.append("</relevant_canon>")
    return "\n".join(lines)


def _build_context_block(
    init_payload: Optional[Dict[str, Any]],
    type_prompt: Optional[Dict[str, Any]],
    discord_prompt: Optional[Dict[str, Any]],
    skills_hits: List[Dict[str, Any]],
    canon_hits: List[Dict[str, Any]],
    inferred_type: str,
) -> str:
    sections: List[str] = ["<palace_context>"]
    sections.append(f"<inferred_session_type>{inferred_type}</inferred_session_type>")
    if init_payload:
        sections.append("<memory_init>")
        # init payload is sessions package output; truncate defensively
        s = json.dumps(init_payload, ensure_ascii=False)
        sections.append(s[:4000])
        sections.append("</memory_init>")
    if type_prompt and isinstance(type_prompt.get("body"), str):
        sections.append(f"<session_type_prompt type='{inferred_type}'>")
        sections.append(type_prompt["body"].strip())
        sections.append("</session_type_prompt>")
    if discord_prompt and isinstance(discord_prompt.get("body"), str):
        sections.append("<discord_delivery_rules>")
        sections.append(discord_prompt["body"].strip())
        sections.append("</discord_delivery_rules>")
    skills_block = _format_skills_block(skills_hits)
    if skills_block:
        sections.append(skills_block)
    canon_block = _format_canon_block(canon_hits)
    if canon_block:
        sections.append(canon_block)
    sections.append("</palace_context>")
    return "\n".join(sections)


def _gather_context(
    user_text: str,
    inferred_type: str,
    want_discord: bool,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Run the four MCP calls in parallel, respecting TOTAL_BUDGET_SEC."""
    deadline = time.time() + TOTAL_BUDGET_SEC

    def _init() -> Optional[Dict[str, Any]]:
        return _http_request("POST", "/sessions/init", body={})

    def _prompt(t: str) -> Optional[Dict[str, Any]]:
        return _http_request("GET", f"/prompts/{t}")

    def _search_skills() -> Optional[Dict[str, Any]]:
        from urllib.parse import quote
        return _http_request(
            "GET",
            f"/search/semantic?q={quote(user_text)}&corpus=skills&k={SKILLS_K}",
        )

    def _search_canon() -> Optional[Dict[str, Any]]:
        from urllib.parse import quote
        return _http_request(
            "GET",
            f"/search/semantic?q={quote(user_text)}&corpus=palace&k={CANON_K}",
        )

    init_payload: Optional[Dict[str, Any]] = None
    type_prompt: Optional[Dict[str, Any]] = None
    discord_prompt: Optional[Dict[str, Any]] = None
    skills_hits: List[Dict[str, Any]] = []
    canon_hits: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_init = pool.submit(_init)
        f_type = pool.submit(_prompt, inferred_type)
        f_discord = pool.submit(_prompt, "discord-delivery") if want_discord else None
        f_skills = pool.submit(_search_skills)
        f_canon = pool.submit(_search_canon)

        def _await(fut, label: str) -> Optional[Dict[str, Any]]:
            if fut is None:
                return None
            remaining = max(0.05, deadline - time.time())
            try:
                return fut.result(timeout=remaining)
            except FutureTimeout:
                logger.warning("palace-discipline: %s timed out", label)
                return None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("palace-discipline: %s raised: %s", label, exc)
                return None

        init_payload = _await(f_init, "memory.init")
        type_prompt = _await(f_type, f"prompts/{inferred_type}")
        discord_prompt = _await(f_discord, "prompts/discord-delivery")
        skills_resp = _await(f_skills, "search/semantic skills")
        canon_resp = _await(f_canon, "search/semantic palace")

    if isinstance(skills_resp, dict):
        skills_hits = list(skills_resp.get("results") or [])
    if isinstance(canon_resp, dict):
        canon_hits = list(canon_resp.get("results") or [])

    return init_payload, type_prompt, discord_prompt, skills_hits, canon_hits


def on_pre_gateway_dispatch(*, event: Any, gateway: Any = None,
                            session_store: Any = None) -> Optional[Dict[str, Any]]:
    """Hook entry point.

    Returns a dict ``{"action": "rewrite", "text": <enriched_text>}`` to
    have the dispatcher continue with the augmented message, or ``None``
    to leave the event unchanged (degraded path / idempotent skip /
    empty message / disabled).
    """
    text = (getattr(event, "text", None) or "").strip()
    if not text:
        return None  # nothing to enrich

    key = _conversation_key(event)
    if _already_injected(key):
        _audit(f"skip-idempotent key={key}")
        return None

    inferred_type = _infer_session_type(text)
    want_discord = _is_discord(event)

    t0 = time.time()
    try:
        init_payload, type_prompt, discord_prompt, skills_hits, canon_hits = (
            _gather_context(text, inferred_type, want_discord)
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("palace-discipline: gather_context crashed: %s", exc)
        _audit(f"degraded key={key} reason=gather_crash err={exc!r}")
        return None
    elapsed_ms = int((time.time() - t0) * 1000)

    # If literally everything failed, don't inject an empty wrapper —
    # that's pure noise. Degrade silently.
    if (
        init_payload is None
        and type_prompt is None
        and discord_prompt is None
        and not skills_hits
        and not canon_hits
    ):
        _audit(
            f"degraded key={key} type={inferred_type} elapsed_ms={elapsed_ms} "
            f"reason=all-mcp-failed"
        )
        return None

    block = _build_context_block(
        init_payload, type_prompt, discord_prompt,
        skills_hits, canon_hits, inferred_type,
    )

    new_text = f"{block}\n\n{text}"
    _audit(
        f"injected key={key} type={inferred_type} discord={want_discord} "
        f"init={'y' if init_payload else 'n'} "
        f"type_prompt={'y' if type_prompt else 'n'} "
        f"discord_prompt={'y' if discord_prompt else 'n'} "
        f"skills_hits={len(skills_hits)} canon_hits={len(canon_hits)} "
        f"elapsed_ms={elapsed_ms} block_chars={len(block)}"
    )
    return {"action": "rewrite", "text": new_text}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the pre_gateway_dispatch callback."""
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
    logger.info(
        "palace-discipline: registered pre_gateway_dispatch hook "
        "(MCP=%s, idempotency=%ds)",
        MCP_BASE_URL, IDEMPOTENCY_WINDOW_SEC,
    )
