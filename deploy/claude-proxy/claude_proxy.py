#!/usr/bin/env python3
"""
Claude Code Proxy — turns claude -p into an OpenAI-compatible /v1/chat/completions endpoint.
Hermes → this proxy → claude -p (OAuth) → Anthropic Claude (Sonnet/Opus).
"""

import glob as _glob
import hashlib
import json
import signal
import shutil
import subprocess
import sys
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path


def _resolve_claude_bin() -> str:
    """Locate the claude binary robustly, computed once at import time:
    env override → ``shutil.which`` → the historical node_modules path (glob)."""
    override = os.environ.get("CLAUDE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found:
        return found
    # Fall back to the historical hardcoded install location. Glob the
    # platform-specific dir so a node/version bump doesn't break resolution.
    patterns = [
        "/opt/data/home/.local/lib/node_modules/@anthropic-ai/claude-code/"
        "node_modules/@anthropic-ai/claude-code-*/claude",
        "/opt/data/home/.local/lib/node_modules/@anthropic-ai/claude-code/cli.js",
    ]
    for pat in patterns:
        matches = sorted(_glob.glob(pat))
        if matches:
            return matches[0]
    # Last-resort: the original literal path (kept so /health can report it
    # missing rather than crashing at import).
    return (
        "/opt/data/home/.local/lib/node_modules/@anthropic-ai/claude-code/"
        "node_modules/@anthropic-ai/claude-code-linux-x64/claude"
    )


CLAUDE_BIN = _resolve_claude_bin()
HOME = "/opt/data/home"
PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "11435"))
# Move the proxy's working directory off the shared data volume so a sandboxed
# claude -p has no incidental cwd access to tenant data.
WORKDIR = os.environ.get("CLAUDE_PROXY_WORKDIR", "/opt/data/proxy/workdir")

# Backend selection. DEFAULT "cli" keeps prod behavior unchanged; "anthropic"
# is opt-in (direct Anthropic API). See ClaudeCliBackend / AnthropicApiBackend.
BACKEND_NAME = os.environ.get("CLAUDE_PROXY_BACKEND", "cli").strip().lower() or "cli"

# Tenant auth. ALLOW_ANON defaults ON so a deploy does NOT take existing prod
# bots dark: requests without a valid bearer key are still served (tenant
# "anon"). Tightening to "0" is a DELIBERATE step taken AFTER per-tenant keys
# have been rolled out — then a missing/invalid key returns 401.
ALLOW_ANON = os.environ.get("CLAUDE_PROXY_ALLOW_ANON", "1").lower() not in ("0", "false", "no", "")
KEYS_FILE = os.environ.get("CLAUDE_PROXY_KEYS_FILE", "/opt/data/proxy/keys.json")

# Per-tenant daily token budget (input+output). Generous default; over budget
# → 429 so the gateway's existing fallback chain takes over. 0 disables.
DAILY_TOKEN_BUDGET = int(os.environ.get("CLAUDE_PROXY_DAILY_TOKEN_BUDGET", "5000000"))
# Durable metering: persist per-tenant counters + the daily-budget window so a
# proxy restart (deploy, crash) doesn't reset usage data or silently refund
# every tenant's spent budget. Atomic write, debounced like the cache, flushed
# on SIGTERM. Empty path disables persistence (in-memory only).
METERS_FILE = os.environ.get("CLAUDE_PROXY_METERS_FILE", "/opt/data/proxy/meters.json")

# Request hardening. The proxy is loopback-only, but a buggy/compromised
# co-tenant gateway shouldn't be able to OOM it with a giant body or exhaust
# threads. Cap the request body and bound concurrent work; over the cap → 413,
# saturated → 503 (both signals the gateway's fallback chain understands).
MAX_BODY_BYTES = int(os.environ.get("CLAUDE_PROXY_MAX_BODY_BYTES", "0"))
MAX_WORKERS = int(os.environ.get("CLAUDE_PROXY_MAX_WORKERS", "64"))  # 0 disables the cap
# Seconds a request will wait for a worker slot before shedding with 503.
QUEUE_TIMEOUT = float(os.environ.get("CLAUDE_PROXY_QUEUE_TIMEOUT", "10"))

# Anthropic API backend config (only used when CLAUDE_PROXY_BACKEND=anthropic).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
TIER_MODEL_MAP = {
    "haiku": os.environ.get("ANTHROPIC_MODEL_HAIKU", "claude-haiku-4-5"),
    "sonnet": os.environ.get("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6"),
    "opus": os.environ.get("ANTHROPIC_MODEL_OPUS", "claude-opus-4-8"),
}

# Cache config
CACHE_TTL = int(os.environ.get("CLAUDE_CACHE_TTL", "86400"))  # seconds; 0 = disabled
CACHE_MAX_SIZE = int(os.environ.get("CLAUDE_CACHE_MAX_SIZE", "1000"))
CACHE_FILE = os.environ.get("CLAUDE_CACHE_FILE", "/opt/data/proxy/.cache.json")

# Negative-cache config.  Empty/error results must NEVER enter the 24h cache
# (the original bug: an empty `result` got cached for CACHE_TTL, so every
# conversation_loop retry replayed the cached empty and failed over to the
# fallback provider for a full day).  Bad results instead go into a short-TTL
# negative cache: it absorbs conversation_loop's in-turn burst retries (so we
# don't respawn claude 3x within seconds) while letting a fresh attempt run
# again moments later.
NEG_CACHE_TTL = int(os.environ.get("CLAUDE_NEG_CACHE_TTL", "60"))  # seconds; 0 = disabled

# In-proxy retries when claude returns an empty/error result before giving up.
# Recovers transient empties so Claude stays primary instead of failing over.
EMPTY_RETRIES = int(os.environ.get("CLAUDE_EMPTY_RETRIES", "1"))

# Circuit breaker.  Trips only when MANY DISTINCT prompts return bad within a
# window — i.e. Claude is broadly unhealthy (auth/quota/outage), not just one
# poisoned prompt (the negative cache handles single-prompt cases).  While open,
# claude calls are short-circuited so traffic fails over fast instead of
# hammering a dead backend.
BREAKER_ENABLED = os.environ.get("CLAUDE_BREAKER_ENABLED", "1").lower() not in ("0", "false", "no", "")
BREAKER_THRESHOLD = int(os.environ.get("CLAUDE_BREAKER_THRESHOLD", "8"))  # distinct bad prompts
BREAKER_WINDOW = int(os.environ.get("CLAUDE_BREAKER_WINDOW", "120"))      # seconds
BREAKER_COOLDOWN = int(os.environ.get("CLAUDE_BREAKER_COOLDOWN", "90"))   # seconds

# Cache state: key -> {"response": dict, "ts": float}
_cache: dict = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

# Debounced cache persistence: write to disk at most once per ~30s instead of
# spawning a thread per cache_set.
_CACHE_SAVE_INTERVAL = 30.0
_cache_last_save = 0.0
_cache_save_lock = threading.Lock()

# Negative cache: key -> {"response": dict, "ts": float}
_neg_cache: dict = {}

# Classification cache: prompt_hash -> tier.  A repeated prompt must not
# re-spawn the Haiku classifier subprocess (the classifier-cost bug).  Bounded;
# no TTL needed (a prompt's tier is stable).
_classify_cache: dict = {}
_classify_lock = threading.Lock()
_CLASSIFY_CACHE_MAX = 2000

# Circuit-breaker state (guarded by _breaker_lock)
_breaker_lock = threading.Lock()
_breaker_bad: dict = {}          # prompt_key -> ts of last bad result (distinct prompts)
_breaker_open_until = 0.0        # epoch seconds; calls short-circuit while now < this
_breaker_trips = 0               # total times the breaker has opened (for /health)

# Track for health checks
request_count = 0

# Bounds concurrent completion work so a burst (or a misbehaving co-tenant
# gateway) can't spawn unbounded threads/subprocesses. None = unbounded.
_worker_sem = threading.BoundedSemaphore(MAX_WORKERS) if MAX_WORKERS > 0 else None

# ── Tenant identity (keys.json) ──
#
# keys.json maps an opaque per-client proxy key → client name. The gateway
# sends it as ``Authorization: Bearer <key>`` (the existing custom-provider
# auth mechanism), so per-tenant identity rides that with no new protocol.
# Loaded lazily and reloaded on mtime change — rolling a new client key never
# needs a proxy restart.
_keys_lock = threading.Lock()
_keys_map: dict = {}             # key -> tenant name
_keys_mtime = 0.0
ANON_TENANT = "anon"


def _load_keys_if_changed() -> None:
    """(Re)load keys.json into _keys_map if the file's mtime changed."""
    global _keys_map, _keys_mtime
    try:
        st = os.stat(KEYS_FILE)
    except OSError:
        # Missing file → no keys. Leave any previously-loaded map cleared so a
        # deleted file revokes keys.
        with _keys_lock:
            if _keys_map:
                _keys_map = {}
            _keys_mtime = 0.0
        return
    if st.st_mtime == _keys_mtime and _keys_map:
        return
    try:
        data = json.loads(Path(KEYS_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception as e:
        print(f"[keys] failed to load {KEYS_FILE}: {e}", flush=True)
        return
    with _keys_lock:
        _keys_map = {str(k): str(v) for k, v in data.items()}
        _keys_mtime = st.st_mtime
    print(f"[keys] loaded {len(_keys_map)} tenant key(s) from {KEYS_FILE}", flush=True)


def resolve_tenant(auth_header: str | None) -> tuple[str | None, bool]:
    """Resolve the tenant from an Authorization header.

    Returns ``(tenant, authorized)``. A valid bearer key → that tenant.
    Missing/invalid key → (ANON_TENANT, ALLOW_ANON): authorized only when
    ALLOW_ANON is on (the deploy-safe default)."""
    _load_keys_if_changed()
    token = ""
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = auth_header.strip()
    if token:
        with _keys_lock:
            tenant = _keys_map.get(token)
        if tenant:
            return tenant, True
    # No key, or an unknown key.
    return ANON_TENANT, ALLOW_ANON


# ── Per-tenant metering + daily token budget ──

_meter_lock = threading.Lock()
# tenant -> {"requests": int, "input_tokens": int, "output_tokens": int}
_tenant_meters: dict = {}
# tenant -> {"day": "YYYY-MM-DD", "tokens": int} for the daily budget window.
_tenant_budgets: dict = {}

# Debounced meter persistence (mirrors the cache's writer): at most one disk
# write per ~30s instead of one per request.
_METERS_SAVE_INTERVAL = 30.0
_meters_last_save = 0.0
_meters_save_lock = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def meter_record(tenant: str, input_tokens: int, output_tokens: int) -> None:
    """Record a completed request's token usage for /health metering and the
    daily budget window."""
    with _meter_lock:
        m = _tenant_meters.setdefault(
            tenant, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        )
        m["requests"] += 1
        m["input_tokens"] += int(input_tokens or 0)
        m["output_tokens"] += int(output_tokens or 0)
        b = _tenant_budgets.get(tenant)
        today = _today()
        if b is None or b["day"] != today:
            b = {"day": today, "tokens": 0}
            _tenant_budgets[tenant] = b
        b["tokens"] += int(input_tokens or 0) + int(output_tokens or 0)
    _meters_save_debounced()


def _meters_save_debounced() -> None:
    """Persist meters at most once per ~30s, in a background thread."""
    global _meters_last_save
    if not METERS_FILE:
        return
    now = time.time()
    with _meters_save_lock:
        if now - _meters_last_save < _METERS_SAVE_INTERVAL:
            return
        _meters_last_save = now
    threading.Thread(target=_meters_save, daemon=True).start()


def _meters_save() -> None:
    """Atomically snapshot per-tenant meters + budgets to disk (temp + replace),
    matching the cache writer so a crash mid-write can't truncate the file."""
    if not METERS_FILE:
        return
    try:
        with _meter_lock:
            snapshot = {
                "tenants": {t: dict(m) for t, m in _tenant_meters.items()},
                "budgets": {t: dict(b) for t, b in _tenant_budgets.items()},
            }
        path = Path(METERS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(snapshot), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _meters_load() -> None:
    """Restore meters + budgets from disk on boot. Stale budget windows (a
    different UTC day) are dropped on load so a restart never resurrects
    yesterday's spend against today's allowance."""
    if not METERS_FILE or not Path(METERS_FILE).exists():
        return
    try:
        data = json.loads(Path(METERS_FILE).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        tenants = data.get("tenants", {}) or {}
        budgets = data.get("budgets", {}) or {}
        today = _today()
        with _meter_lock:
            for t, m in tenants.items():
                if isinstance(m, dict):
                    _tenant_meters[str(t)] = {
                        "requests": int(m.get("requests", 0) or 0),
                        "input_tokens": int(m.get("input_tokens", 0) or 0),
                        "output_tokens": int(m.get("output_tokens", 0) or 0),
                    }
            for t, b in budgets.items():
                if isinstance(b, dict) and b.get("day") == today:
                    _tenant_budgets[str(t)] = {"day": today, "tokens": int(b.get("tokens", 0) or 0)}
        print(f"[meters] loaded {len(_tenant_meters)} tenant meter(s) from {METERS_FILE}")
    except Exception as e:
        print(f"[meters] failed to load {METERS_FILE}: {e}")


def _effective_no_cache(cache_control: str, tenant: str) -> bool:
    """Honor an explicit ``X-Cache-Control: no-cache`` bypass only for an
    AUTHENTICATED tenant. An anonymous caller must not be able to force backend
    spend (and skip the cost-saving cache) — that's a cheap cost-amplification
    lever against the shared Claude account/budget."""
    return cache_control.lower() == "no-cache" and tenant != ANON_TENANT


def budget_exceeded(tenant: str) -> bool:
    """True if the tenant has already spent its daily token budget. 0 disables."""
    if DAILY_TOKEN_BUDGET <= 0:
        return False
    with _meter_lock:
        b = _tenant_budgets.get(tenant)
        if b is None or b["day"] != _today():
            return False
        return b["tokens"] >= DAILY_TOKEN_BUDGET


# ── Cache ──

def _cache_key(tenant: str, system: str, prompt: str, tier: str) -> str:
    # Tenant-scoped so one tenant's cached result can never be served to
    # another (the good cache AND the negative cache key off this).
    payload = f"{tenant}\x00{tier}\x00{system}\x00{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_evict_expired() -> None:
    """Remove expired entries. Must be called with _cache_lock held."""
    if CACHE_TTL == 0:
        return
    cutoff = time.time() - CACHE_TTL
    expired = [k for k, v in _cache.items() if v["ts"] < cutoff]
    for k in expired:
        del _cache[k]


def cache_get(tenant: str, system: str, prompt: str, tier: str) -> dict | None:
    global _cache_hits, _cache_misses
    if CACHE_TTL == 0:
        _cache_misses += 1
        return None
    key = _cache_key(tenant, system, prompt, tier)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None or (time.time() - entry["ts"] > CACHE_TTL):
            if entry is not None:
                del _cache[key]
            _cache_misses += 1
            return None
        _cache_hits += 1
        return entry["response"]


def cache_set(tenant: str, system: str, prompt: str, tier: str, response: dict) -> None:
    if CACHE_TTL == 0:
        return
    # Guard: empty/error results must never enter the 24h cache (poison-cache
    # bug).  They belong in the short negative cache instead.
    if _is_bad_result(response):
        return
    key = _cache_key(tenant, system, prompt, tier)
    with _cache_lock:
        _cache_evict_expired()
        # Evict oldest entries if over max size
        if len(_cache) >= CACHE_MAX_SIZE:
            oldest = sorted(_cache.items(), key=lambda kv: kv[1]["ts"])
            for k, _ in oldest[:max(1, CACHE_MAX_SIZE // 10)]:
                del _cache[k]
        _cache[key] = {"response": response, "ts": time.time()}
    _cache_save_debounced()


def _cache_save_debounced() -> None:
    """Persist the cache at most once per ~30s, in a background thread.
    Avoids spawning a writer thread on every cache_set."""
    global _cache_last_save
    if not CACHE_FILE:
        return
    now = time.time()
    with _cache_save_lock:
        if now - _cache_last_save < _CACHE_SAVE_INTERVAL:
            return
        _cache_last_save = now
    threading.Thread(target=_cache_save, daemon=True).start()


def _cache_save() -> None:
    if not CACHE_FILE:
        return
    try:
        with _cache_lock:
            snapshot = dict(_cache)
        # Atomic write: temp file in the same dir + os.replace (matches
        # hermes_cli.add_client.write_doc) so a crash mid-write can't truncate
        # the live cache file.
        path = Path(CACHE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(snapshot), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _cache_load() -> None:
    if not CACHE_FILE or not Path(CACHE_FILE).exists():
        return
    try:
        data = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
        cutoff = time.time() - CACHE_TTL if CACHE_TTL > 0 else 0
        skipped_bad = 0
        with _cache_lock:
            for k, v in data.items():
                if isinstance(v, dict) and "ts" in v and "response" in v:
                    if CACHE_TTL == 0 or v["ts"] >= cutoff:
                        # Drop any pre-fix poisoned (empty/error) entries.
                        if _is_bad_result(v["response"]):
                            skipped_bad += 1
                            continue
                        _cache[k] = v
        print(f"[cache] loaded {len(_cache)} entries from {CACHE_FILE}"
              + (f" (skipped {skipped_bad} bad)" if skipped_bad else ""))
    except Exception as e:
        print(f"[cache] failed to load {CACHE_FILE}: {e}")


# ── Bad-result detection ──

def _result_text(result: dict) -> str:
    """Final assistant text from a parsed claude -p result, or ''."""
    if not isinstance(result, dict):
        return ""
    txt = result.get("result", "")
    return txt if isinstance(txt, str) else ""


def _is_bad_result(result: dict) -> bool:
    """True if a claude result is empty or an error and must NOT be cached
    in the 24h cache nor returned as a valid completion.

    Covers: proxy-level error envelopes (``error``), claude-reported errors
    (``is_error``), non-success ``subtype`` (e.g. error_max_turns), and empty
    / whitespace-only ``result`` text (the tool_use-only / silent-turn case
    that originally got poison-cached)."""
    if not isinstance(result, dict):
        return True
    if result.get("error"):
        return True
    if result.get("is_error"):
        return True
    subtype = result.get("subtype")
    if subtype is not None and subtype != "success":
        return True
    if not _result_text(result).strip():
        return True
    return False


# ── Negative cache (short TTL for bad results) ──

def _neg_cache_get(key: str) -> dict | None:
    if NEG_CACHE_TTL <= 0:
        return None
    with _cache_lock:
        entry = _neg_cache.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > NEG_CACHE_TTL:
            del _neg_cache[key]
            return None
        return entry["response"]


def _neg_cache_set(key: str, response: dict) -> None:
    if NEG_CACHE_TTL <= 0:
        return
    with _cache_lock:
        now = time.time()
        # Drop expired and bound size so a long-running proxy can't grow it
        # without limit.
        for k in [k for k, v in _neg_cache.items() if now - v["ts"] > NEG_CACHE_TTL]:
            del _neg_cache[k]
        if len(_neg_cache) >= 500:
            for k, _ in sorted(_neg_cache.items(), key=lambda kv: kv[1]["ts"])[:50]:
                del _neg_cache[k]
        _neg_cache[key] = {"response": response, "ts": now}


# ── Circuit breaker ──

def _breaker_is_open() -> bool:
    if not BREAKER_ENABLED:
        return False
    return time.time() < _breaker_open_until


def _breaker_record_bad(key: str) -> None:
    """Record a bad result for a distinct prompt; open the breaker if many
    distinct prompts have failed within the window (= broad Claude outage)."""
    global _breaker_open_until, _breaker_trips
    if not BREAKER_ENABLED:
        return
    now = time.time()
    with _breaker_lock:
        _breaker_bad[key] = now
        for k in [k for k, t in _breaker_bad.items() if now - t > BREAKER_WINDOW]:
            del _breaker_bad[k]
        if len(_breaker_bad) >= BREAKER_THRESHOLD and now >= _breaker_open_until:
            _breaker_open_until = now + BREAKER_COOLDOWN
            _breaker_trips += 1
            print(
                f"[breaker] OPEN: {len(_breaker_bad)} distinct prompts returned "
                f"bad within {BREAKER_WINDOW}s — short-circuiting Claude to "
                f"fallback for {BREAKER_COOLDOWN}s (trip #{_breaker_trips})",
                flush=True,
            )


def _breaker_record_good() -> None:
    """A healthy result clears the failure tally (and closes the breaker if its
    cooldown has elapsed)."""
    global _breaker_open_until
    if not BREAKER_ENABLED:
        return
    with _breaker_lock:
        if _breaker_bad:
            _breaker_bad.clear()
        if _breaker_open_until and time.time() >= _breaker_open_until:
            print("[breaker] CLOSED: Claude returning healthy results again", flush=True)
            _breaker_open_until = 0.0


# ── Sandbox flags for claude -p ──

# Every dangerous tool name, dropped explicitly (belt). --tools "" removes the
# whole tool surface (braces). We also do NOT pass --permission-mode
# bypassPermissions: non-interactive -p denies tool calls by default, so
# default-deny is the safety net even if a flag is ignored by some binary.
_DANGEROUS_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
]


def _sandbox_flags() -> list:
    """Flags that strip all tool access from a claude -p subprocess."""
    return [
        "--tools", "",
        "--disallowedTools", "*", *_DANGEROUS_TOOLS,
    ]


# ── Model Tier Classification (adapted from claude-model-router-hook) ──

OPUS_KEYWORDS = [
    "architect", "architecture", "evaluate", "tradeoff", "trade-off",
    "strategy", "strategic", "compare approaches", "why does", "deep dive",
    "redesign", "across the codebase", "investor", "multi-system",
    "complex refactor", "analyze", "analysis", "plan mode", "rethink",
    "high-stakes", "critical decision", "research", "design pattern",
    "system design", "code review", "security audit", "migration",
]

HAIKU_PATTERNS = [
    r"\bgit\s+(commit|push|pull|status|log|diff|add|stash|branch|merge|rebase|checkout)\b",
    r"\bcommit\b.*\b(change|push|all)\b", r"\bpush\s+(to|the|remote|origin)\b",
    r"\brename\b", r"\bre-?order\b", r"\bmove\s+file\b", r"\bdelete\s+file\b",
    r"\badd\s+(import|route|link)\b", r"\bformat\b", r"\blint\b",
    r"\bprettier\b", r"\beslint\b", r"\bremove\s+(unused|dead)\b",
    r"\bupdate\s+(version|package)\b", r"\b(good\s+)?morning\b", r"\bhi\b", r"\bhey\b",
    r"\bthanks?\b", r"\bok\b", r"\bgot it\b", r"\bsimple\s+question\b",
]

SONNET_PATTERNS = [
    r"\bbuild\b", r"\bimplement\b", r"\bcreate\b", r"\bfix\b", r"\bdebug\b",
    r"\badd\s+feature\b", r"\bwrite\b", r"\bcomponent\b", r"\bservice\b",
    r"\bpage\b", r"\bdeploy\b", r"\btest\b", r"\bupdate\b", r"\brefactor\b",
    r"\bstyle\b", r"\bcss\b", r"\broute\b", r"\bapi\b", r"\bfunction\b",
    r"\bscript\b", r"\bconfig\b", r"\bproxy\b", r"\bsetup\b", r"\binstall\b",
]


def _classify_cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def classify_model(prompt: str) -> str:
    """Classify prompt complexity and return a recommended model tier.

    Cost-aware order (the classifier-cost fix): explicit ``~tier`` override →
    per-prompt classification cache → KEYWORDS FIRST (free, no subprocess) →
    Haiku subprocess ONLY for genuinely ambiguous prompts → keyword default.
    The result is cached by prompt hash so a repeated prompt never re-spawns
    the classifier."""
    # Override: prefix with ~model forces a specific model (never cached, cheap).
    stripped = prompt.lower().strip()
    for override_tier in ["opus", "sonnet", "haiku"]:
        if stripped.startswith(f"~{override_tier}"):
            return override_tier

    ckey = _classify_cache_key(prompt)
    with _classify_lock:
        hit = _classify_cache.get(ckey)
    if hit is not None:
        return hit

    # Keyword-first: a confident keyword match avoids the subprocess entirely.
    tier = _classify_by_keywords(prompt)
    if _classify_is_ambiguous(prompt):
        # Only genuinely ambiguous prompts are worth a Haiku call.
        haiku_tier = classify_via_haiku(prompt)
        if haiku_tier:
            tier = haiku_tier

    with _classify_lock:
        if len(_classify_cache) >= _CLASSIFY_CACHE_MAX:
            # Cheap bound: drop an arbitrary chunk rather than track LRU.
            for k in list(_classify_cache)[: _CLASSIFY_CACHE_MAX // 10]:
                del _classify_cache[k]
        _classify_cache[ckey] = tier
    return tier


def _classify_is_ambiguous(prompt: str) -> bool:
    """True when keywords give no confident signal, so a Haiku call is worth
    its cost. Confident = any tier keyword/pattern matched, or the prompt is
    long enough that the keyword length heuristics already decide it."""
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())
    if any(kw in prompt_lower for kw in OPUS_KEYWORDS):
        return False
    if word_count > 100:           # length heuristics in _classify_by_keywords decide
        return False
    if any(re.search(p, prompt_lower) for p in HAIKU_PATTERNS):
        return False
    if any(re.search(p, prompt_lower) for p in SONNET_PATTERNS):
        return False
    return True


def classify_via_haiku(prompt: str) -> str | None:
    """Use Haiku to classify task complexity. Returns None on failure.

    Sandboxed: no tool access (``_sandbox_flags``) and NOT bypassPermissions —
    the classifier never needs tools, so default-deny is the safety net."""
    classifier_prompt = (
        "Rate the complexity of this task. Reply with EXACTLY one word: "
        "haiku (for simple/mechanical tasks like greetings, git ops, formatting, quick lookups), "
        "sonnet (for standard work like implementation, debugging, writing code, features), or "
        "opus (for complex tasks like architecture, strategy, deep analysis, multi-system reasoning).\n\n"
        f"Task:\n{prompt[:1500]}\n\n"
        "Reply with one word only:"
    )

    try:
        proc = subprocess.run(
            [
                CLAUDE_BIN, "-p",
                "--output-format", "json",
                "--no-session-persistence",
                "--model", "haiku",
                *_sandbox_flags(),
            ],
            input=classifier_prompt,
            capture_output=True,
            text=True,
            timeout=15,  # Haiku should be fast
            cwd=WORKDIR,
            env={**os.environ, "HOME": HOME},
        )

        # Parse response
        stdout = proc.stdout.strip()
        for line in reversed(stdout.splitlines()):
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    result_text = data.get("result", "").strip().lower()
                    for tier in ["haiku", "sonnet", "opus"]:
                        if tier in result_text:
                            print(f"[classify] Haiku → {tier} | prompt: {prompt[:60]}...")
                            return tier
                except json.JSONDecodeError:
                    pass

    except subprocess.TimeoutExpired:
        print(f"[classify] Haiku timeout, falling back to keywords")
    except Exception as e:
        print(f"[classify] Haiku error: {e}, falling back to keywords")

    return None


def _classify_by_keywords(prompt: str) -> str:
    """Keyword-based fallback classifier."""
    word_count = len(prompt.split())
    prompt_lower = prompt.lower()

    # Opus: complex tasks
    if any(kw in prompt_lower for kw in OPUS_KEYWORDS):
        return "opus"
    if word_count > 100 and "?" in prompt:
        return "opus"
    if word_count > 200:
        return "opus"

    # Haiku: simple mechanical tasks
    if word_count < 60:
        if any(re.search(p, prompt_lower) for p in HAIKU_PATTERNS):
            return "haiku"

    # Sonnet: everything else (default)
    return "sonnet"


def _extract_data_url_image(part: dict) -> dict | None:
    """Parse an OpenAI ``image_url`` content part into a base64 image dict.

    Returns ``{"data": <base64>, "media_type": <mime>}`` for ``data:`` URLs, or
    ``None`` for remote URLs (the sandboxed CLI has no network to fetch them)
    and malformed parts."""
    url = (part.get("image_url") or {}).get("url", "")
    if not url.startswith("data:"):
        return None  # remote images unsupported in the sandboxed path
    try:
        header, b64data = url.split(",", 1)
    except ValueError:
        return None
    media_type = "image/jpeg"
    if ":" in header and ";" in header:
        media_type = header.split(":", 1)[1].split(";", 1)[0] or media_type
    if not b64data:
        return None
    return {"data": b64data, "media_type": media_type}


def build_prompt(messages: list) -> tuple[str, str, list]:
    """Extract system prompt and build conversation text from OpenAI-format messages.

    Returns ``(system, prompt, images)`` where ``images`` is a list of
    ``{"data": <base64>, "media_type": <mime>}`` dicts pulled from any
    multimodal ``image_url`` parts (data: URLs only)."""
    system = ""
    conversation = []
    images = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Multimodal — extract text parts and any inline images
            text_parts = []
            for p in content:
                if p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    img = _extract_data_url_image(p)
                    if img is not None:
                        images.append(img)
            content = "\n".join(text_parts)

        if role == "system":
            system = content
        elif role == "user":
            conversation.append(f"User: {content}")
        elif role == "assistant":
            # Could be text or tool_call
            if isinstance(content, str):
                conversation.append(f"Assistant: {content}")
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        conversation.append(f"Assistant: {part['text']}")
                    elif isinstance(part, dict) and part.get("type") == "tool_use":
                        conversation.append(f"[Assistant used tool: {part.get('name', 'unknown')}]")
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            conversation.append(f"[Tool result from {tool_name}: {str(content)[:500]}]")
    
    prompt = "\n\n".join(conversation)
    return system, prompt, images


def build_stream_json_input(system_prompt: str, prompt: str, images: list) -> str:
    """Build ``--input-format stream-json`` stdin for a vision request.

    Emits JSON-lines: an optional system-as-user primer line, then one user
    message whose ``content`` interleaves the text prompt with base64 image
    blocks. (System is passed via ``--system-prompt`` on the CLI; the primer
    line is only used when a backend cannot take a separate system flag.)"""
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["data"],
            },
        })
    message = {"type": "user", "message": {"role": "user", "content": content}}
    return json.dumps(message) + "\n"


def _run_claude_once(cmd: list, prompt: str, env: dict, timeout: int = 300,
                     stream_mode: bool = False) -> dict:
    """Spawn ``claude -p`` once and return the parsed JSON result dict, or a
    proxy error dict (``{"error": True, ...}``) on no-output / parse-fail /
    timeout / exception.  ``timeout`` is the per-call hang watchdog.  When
    ``stream_mode`` is True the output is stream-json (one JSON object per
    line) and we return the last ``{"type": "result", ...}`` object."""
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKDIR,
            env=env,
        )

        # Find the last JSON line (there might be stderr noise)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if not stdout:
            print(f"[proxy] Claude returned NO OUTPUT. returncode={proc.returncode} stderr={stderr[:300]}", flush=True)
            return {
                "error": True,
                "message": f"Claude returned no output. Stderr: {stderr[:500]}",
                "code": proc.returncode,
            }

        # Parse output — mode-dependent
        if stream_mode:
            # Stream-json output: one JSON object per line. Find the last
            # line with {"type":"result",...} — this is Claude's final answer.
            result = None
            for line in stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "result":
                    result = obj
            if result is None:
                print(f"[proxy] Claude STREAM_PARSE_FAIL (no result object). "
                      f"returncode={proc.returncode} lines={len(stdout.splitlines())} "
                      f"stderr={stderr[:200]}", flush=True)
                return {
                    "error": True,
                    "message": f"Stream-json output missing result object. Stderr: {stderr[:500]}",
                    "code": proc.returncode,
                }
        else:
            # Text mode: might be multiple lines, take the last valid JSON
            result = None
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        result = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if result is None:
                print(f"[proxy] Claude PARSE FAIL. returncode={proc.returncode} "
                      f"stdout_preview={stdout[:200]} stderr={stderr[:200]}", flush=True)
                return {
                    "error": True,
                    "message": f"Failed to parse Claude output: {stdout[:500]}",
                    "code": proc.returncode,
                }

        return result

    except subprocess.TimeoutExpired:
        print(f"[proxy] Claude TIMEOUT after {timeout}s. prompt={prompt[:100]}...", flush=True)
        return {"error": True, "message": f"Claude call timed out after {timeout}s", "timeout": True}
    except FileNotFoundError:
        print(f"[proxy] Claude binary MISSING: {CLAUDE_BIN}", flush=True)
        return {"error": True, "message": f"Claude binary not found at {CLAUDE_BIN}"}
    except Exception as e:
        print(f"[proxy] Claude EXCEPTION: {type(e).__name__}: {e}", flush=True)
        return {"error": True, "message": f"Claude call failed: {str(e)}"}


def _log_claude_call(meta: dict) -> None:
    """Emit one structured observability line per proxy call."""
    try:
        print("claude_call: " + json.dumps(meta, ensure_ascii=False, default=str), flush=True)
    except Exception:
        pass


# ── Pluggable generation backends ──
#
# A backend's ``complete(system, prompt, tier, tenant, images=None) -> result_dict`` returns
# the SAME claude -p-shaped dict the rest of the pipeline consumes
# (``result``/``is_error``/``subtype``/``usage``/...), so cache, negative cache,
# breaker, _is_bad_result and claude_to_openai all keep working unchanged.
# DEFAULT backend is "cli" — prod behavior is unchanged. "anthropic" is opt-in.

class ClaudeCliBackend:
    """The historical behavior: spawn the OAuth ``claude -p`` subprocess, now
    sandboxed (no tools, no bypassPermissions)."""

    name = "cli"

    def complete(self, system: str, prompt: str, tier: str, tenant: str,
                 images: list = None) -> dict:
        model_flag = []
        if "opus" in tier:
            model_flag = ["--model", "opus"]
        elif "haiku" in tier:
            model_flag = ["--model", "haiku"]
        # sonnet: no flag needed (default)

        images = images or []
        stream_mode = bool(images)

        if stream_mode:
            cmd = [
                CLAUDE_BIN,
                "-p",
                "--input-format", "stream-json",
                "--output-format", "stream-json",
                "--verbose",
                "--no-session-persistence",
                *_sandbox_flags(),
            ]
            stdin_input = build_stream_json_input(system, prompt, images)
        else:
            cmd = [
                CLAUDE_BIN,
                "-p",
                "--output-format", "json",
                "--no-session-persistence",
                *_sandbox_flags(),
            ]
            stdin_input = prompt

        if system:
            cmd.extend(["--system-prompt", system])
        cmd.extend(model_flag)

        env = os.environ.copy()
        env["HOME"] = HOME
        return _run_claude_once(cmd, stdin_input, env, stream_mode=stream_mode)


class AnthropicApiBackend:
    """Opt-in: call the Anthropic Messages API directly over HTTPS (stdlib
    urllib only). Converts the API response into the claude -p result shape so
    the rest of the pipeline is agnostic to the backend."""

    name = "anthropic"

    def complete(self, system: str, prompt: str, tier: str, tenant: str,
                 images: list = None) -> dict:
        if not ANTHROPIC_API_KEY:
            return {"error": True, "message": "ANTHROPIC_API_KEY not set for anthropic backend", "code": 500}
        model = TIER_MODEL_MAP.get(
            "opus" if "opus" in tier else "haiku" if "haiku" in tier else "sonnet",
            TIER_MODEL_MAP["sonnet"],
        )
        images = images or []
        # Build content: text + optional images in Anthropic format
        content = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"],
                },
            })
        payload = {
            "model": model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            print(f"[anthropic] HTTP {e.code}: {body}", flush=True)
            return {"error": True, "message": f"anthropic api http {e.code}: {body}", "code": e.code}
        except Exception as e:
            print(f"[anthropic] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            return {"error": True, "message": f"anthropic api call failed: {e}"}
        return self._to_claude_shape(data)

    @staticmethod
    def _to_claude_shape(data: dict) -> dict:
        """Map an Anthropic /v1/messages response onto the claude -p result
        dict the pipeline expects."""
        text = ""
        for block in data.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        usage = data.get("usage", {}) or {}
        stop = data.get("stop_reason")
        # Anthropic stop reasons: end_turn/max_tokens/stop_sequence/tool_use.
        return {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": text,
            "stop_reason": "end_turn" if stop in (None, "end_turn", "stop_sequence") else stop,
            "session_id": data.get("id", "anthropic"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        }


def _make_backend(name: str):
    if name == "anthropic":
        return AnthropicApiBackend()
    return ClaudeCliBackend()


# Selected once at import; DEFAULT "cli" keeps prod unchanged.
BACKEND = _make_backend(BACKEND_NAME)


def call_claude(system_prompt: str, prompt: str, model: str = None,
                no_cache: bool = False, tenant: str = ANON_TENANT,
                images: list = None) -> dict:
    """Call Claude Code in -p mode and return a parsed JSON result.

    Pipeline: daily budget → 24h cache (good results only) → negative cache
    (recent bad) → circuit breaker → backend with one retry on empty/error.
    Bad results are NEVER written to the 24h cache (the original poison-cache
    bug); they go to the short negative cache so conversation_loop's in-turn
    burst retries don't respawn the backend, while a fresh attempt can still run
    seconds later.  Cache keys are tenant-scoped; the breaker stays GLOBAL (it
    measures backend health, not tenant behavior).  Emits a structured
    ``claude_call:`` log line (with the tenant) on every path.

    When ``images`` is non-empty, the text-based cache is skipped (image content
    is not reflected in the cache key), and the backend is called with the
    images for multimodal processing."""
    t0 = time.time()

    # Auto-classify if no explicit model
    if model is None or model == "auto":
        tier = classify_model(prompt)
    else:
        tier = model.lower()

    images = images or []
    no_cache = no_cache or bool(images)  # text-based cache key can't reflect images

    key = _cache_key(tenant, system_prompt, prompt, tier)
    meta = {
        "tenant": tenant,
        "tier": tier,
        "prompt_chars": len(prompt),
        "system_chars": len(system_prompt or ""),
        "no_cache": no_cache,
    }

    # 0. Daily token budget — over budget → 429 so the gateway fails over.
    if budget_exceeded(tenant):
        _log_claude_call({**meta, "cache": "miss", "decision": "budget_exceeded",
                          "elapsed_ms": int((time.time() - t0) * 1000)})
        return {"error": True, "message": f"tenant '{tenant}' daily token budget exceeded", "code": 429}

    # 1. Good cache (only non-empty, non-error results are ever stored here)
    if not no_cache:
        cached = cache_get(tenant, system_prompt, prompt, tier)
        if cached is not None:
            print(f"[cache] HIT tenant={tenant} tier={tier} prompt={prompt[:60]}...")
            _log_claude_call({**meta, "cache": "good_hit", "decision": "ok",
                              "has_text": True, "text_len": len(_result_text(cached)),
                              "elapsed_ms": int((time.time() - t0) * 1000)})
            return cached

        # 2. Negative cache (recent bad result for this exact prompt)
        neg = _neg_cache_get(key)
        if neg is not None:
            _log_claude_call({**meta, "cache": "neg_hit", "decision": "bad_cached",
                              "is_error": bool(neg.get("is_error") or neg.get("error")),
                              "subtype": neg.get("subtype"),
                              "elapsed_ms": int((time.time() - t0) * 1000)})
            return neg

    # 3. Circuit breaker — broad backend outage, short-circuit to fallback
    if _breaker_is_open():
        _log_claude_call({**meta, "cache": "miss", "decision": "breaker_open",
                          "elapsed_ms": int((time.time() - t0) * 1000)})
        return {"error": True, "message": "claude circuit breaker open — failing over", "code": 503}

    # 4. Run via the selected backend with one retry on empty/error (recovers
    #    transient empties so Claude stays primary instead of failing over).
    attempts = 0
    result: dict = {}
    for attempt in range(EMPTY_RETRIES + 1):
        attempts += 1
        result = BACKEND.complete(system_prompt, prompt, tier, tenant, images=images)
        if not _is_bad_result(result):
            break
        if attempt < EMPTY_RETRIES:
            print(f"[proxy] empty/error result (attempt {attempts}/{EMPTY_RETRIES + 1}) "
                  f"tier={tier} — retrying", flush=True)

    bad = _is_bad_result(result)
    if not bad:
        if not no_cache:
            cache_set(tenant, system_prompt, prompt, tier, result)
        _breaker_record_good()
        # Meter only successful generations (cache/neg/breaker/budget paths
        # return early without spending backend tokens).
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        meter_record(tenant, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    else:
        # Never poison the 24h cache; use the short negative cache instead.
        if not no_cache:
            _neg_cache_set(key, result)
        _breaker_record_bad(key)

    _log_claude_call({
        **meta,
        "cache": "miss",
        "attempts": attempts,
        "has_text": bool(_result_text(result).strip()),
        "text_len": len(_result_text(result)),
        "is_error": bool(result.get("is_error") or result.get("error")),
        "subtype": result.get("subtype"),
        "stop_reason": result.get("stop_reason"),
        "num_turns": result.get("num_turns"),
        "claude_duration_ms": result.get("duration_ms"),
        "decision": "empty_or_error" if bad else ("recovered" if attempts > 1 else "ok"),
        "elapsed_ms": int((time.time() - t0) * 1000),
    })
    return result


def claude_to_openai(claude_result: dict, model_name: str) -> dict:
    """Convert Claude Code JSON output to OpenAI chat completion format."""
    
    if claude_result.get("error"):
        return {
            "error": {
                "message": claude_result["message"],
                "type": "proxy_error",
                "code": claude_result.get("code", 500),
            }
        }

    # Claude-reported error (is_error / non-success subtype) — surface as an
    # error envelope so the gateway fails over rather than treating it as a
    # valid empty completion.
    if claude_result.get("is_error") or (
        claude_result.get("subtype") not in (None, "success")
    ):
        return {
            "error": {
                "message": claude_result.get("result")
                or f"claude error (subtype={claude_result.get('subtype')}, "
                   f"status={claude_result.get('api_error_status')})",
                "type": "claude_error",
                "code": 502,
            }
        }

    text = claude_result.get("result", "")
    stop_reason = claude_result.get("stop_reason", "end_turn")
    usage = claude_result.get("usage", {})
    
    # Map Claude usage to OpenAI-style tokens
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    return {
        "id": f"claude-proxy-{claude_result.get('session_id', 'unknown')}",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop" if stop_reason == "end_turn" else "length",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Claude Code proxy."""
    
    def log_message(self, format, *args):
        """Suppress default logging to stderr."""
        pass
    
    def do_GET(self):
        """Health check + root info."""
        global request_count
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            _load_keys_if_changed()
            with _meter_lock:
                tenants_snapshot = {t: dict(m) for t, m in _tenant_meters.items()}
                budgets_snapshot = {t: dict(b) for t, b in _tenant_budgets.items()}
            with _keys_lock:
                key_count = len(_keys_map)
            self.wfile.write(json.dumps({
                "status": "ok",
                "claude_bin_exists": os.path.exists(CLAUDE_BIN),
                "workdir": WORKDIR,
                "backend": BACKEND_NAME,
                "requests": request_count,
                "auth": {
                    "allow_anon": ALLOW_ANON,
                    "keys_loaded": key_count,
                },
                "tenants": tenants_snapshot,            # usage-metering seed
                "meters_persisted": bool(METERS_FILE),
                "budget": {
                    "daily_token_budget": DAILY_TOKEN_BUDGET,
                    "spent_today": budgets_snapshot,
                },
                "limits": {
                    "max_body_bytes": MAX_BODY_BYTES,
                    "max_workers": MAX_WORKERS,
                    "queue_timeout_s": QUEUE_TIMEOUT,
                },
                "cache": {
                    "size": len(_cache),
                    "hits": _cache_hits,
                    "misses": _cache_misses,
                    "ttl": CACHE_TTL,
                    "max_size": CACHE_MAX_SIZE,
                    "enabled": CACHE_TTL > 0,
                    "neg_size": len(_neg_cache),
                    "neg_ttl": NEG_CACHE_TTL,
                    "classify_size": len(_classify_cache),
                },
                "breaker": {
                    "enabled": BREAKER_ENABLED,
                    "open": _breaker_is_open(),
                    "recent_bad_prompts": len(_breaker_bad),
                    "threshold": BREAKER_THRESHOLD,
                    "trips": _breaker_trips,
                },
            }).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "service": "Claude Code Proxy",
                "endpoints": ["POST /v1/chat/completions", "GET /health"],
                "port": PORT,
            }).encode())
    
    def do_POST(self):
        """Handle chat completion requests."""
        global request_count
        
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return

        # Read body — cap it so an oversized request can't exhaust memory.
        try:
            content_length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            content_length = -1
        if content_length < 0:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Invalid Content-Length"}}).encode())
            return
        if MAX_BODY_BYTES and content_length > MAX_BODY_BYTES:
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"error": {"message": f"request body exceeds {MAX_BODY_BYTES} bytes", "code": 413}}
            ).encode())
            return
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Invalid JSON"}}).encode())
            return
        
        # Tenant identity from the Bearer key. When ALLOW_ANON is off and the
        # key is missing/invalid → 401. (Default-on so a deploy can't take prod
        # bots dark before keys are rolled out.)
        tenant, authorized = resolve_tenant(self.headers.get("Authorization"))
        if not authorized:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"error": {"message": "missing or invalid proxy key", "type": "auth_error", "code": 401}}
            ).encode())
            return

        messages = data.get("messages", [])
        model_name = data.get("model", "claude-sonnet-4-6")
        # Honor an explicit cache bypass only for an authenticated tenant
        # (anon can't force backend spend). See _effective_no_cache.
        no_cache = _effective_no_cache(self.headers.get("X-Cache-Control", ""), tenant)

        if not messages:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "No messages provided"}}).encode())
            return

        # Build prompt
        system, prompt, images = build_prompt(messages)

        # Determine model: if explicit tier in name, use it; otherwise auto-classify.
        # When images are present, default to sonnet (or explicit model) — haiku
        # lacks vision and would fail.
        if any(t in model_name.lower() for t in ["opus", "haiku", "sonnet"]):
            tier = model_name.lower()
        elif images:
            tier = "sonnet"
        else:
            tier = classify_model(prompt)
        print(f"[proxy] {'explicit' if any(t in model_name.lower() for t in ['opus', 'haiku', 'sonnet']) else 'images' if images else 'auto'}: {tier} | tenant={tenant} | prompt: {prompt[:80]}...")

        request_count += 1

        # Bound concurrent completion work: shed load with 503 (a signal the
        # gateway's fallback chain already handles) rather than spawning
        # unbounded subprocesses when saturated.
        if _worker_sem is not None and not _worker_sem.acquire(timeout=QUEUE_TIMEOUT):
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"error": {"message": "proxy saturated — retry", "type": "overloaded", "code": 503}}
            ).encode())
            return
        try:
            # Call Claude with selected tier
            claude_result = call_claude(system, prompt, tier, no_cache=no_cache, tenant=tenant,
                                        images=images)
        finally:
            if _worker_sem is not None:
                _worker_sem.release()

        # Convert to OpenAI format — report actual tier used
        openai_response = claude_to_openai(claude_result, f"claude-{tier}")

        if "error" in openai_response:
            # Pass the envelope's status through (429 budget / 503 breaker /
            # 502 claude error / 500 default) so the gateway's fallback chain
            # sees the right signal.
            self.send_response(int(openai_response["error"].get("code", 500)))
        else:
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(openai_response).encode())


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server — handles concurrent Claude calls."""
    daemon_threads = True


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # Log all unhandled exceptions before dying
    def _excepthook(exc_type, exc_value, exc_tb):
        print(f"[proxy] FATAL unhandled exception: {exc_type.__name__}: {exc_value}", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.exit(1)
    sys.excepthook = _excepthook

    # Log SIGTERM so the finish script can see it was a signal death. Flush the
    # meters first so a deploy/restart doesn't lose the last <=30s of usage.
    def _sigterm_handler(signum, frame):
        print(f"[proxy] Received SIGTERM — shutting down gracefully", flush=True)
        _meters_save()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Create the sandbox workdir 0700 (off the shared data volume) so the
    # sandboxed claude -p has its own private, non-readable cwd.
    try:
        Path(WORKDIR).mkdir(parents=True, exist_ok=True)
        os.chmod(WORKDIR, 0o700)
    except Exception as e:
        print(f"[proxy] WARN: could not prepare workdir {WORKDIR}: {e}", flush=True)

    _cache_load()
    _meters_load()
    _load_keys_if_changed()
    print(f"Starting Claude Code Proxy on port {PORT}...")
    print(f"Claude binary: {CLAUDE_BIN} (exists: {os.path.exists(CLAUDE_BIN)})")
    print(f"Backend: {BACKEND_NAME}")
    print(f"HOME: {HOME}")
    print(f"Workdir: {WORKDIR}")
    print(f"Auth: allow_anon={ALLOW_ANON} keys_file={KEYS_FILE} "
          f"(loaded {len(_keys_map)} key(s))")
    print(f"Budget: daily_token_budget={DAILY_TOKEN_BUDGET} (0=disabled)")
    print(f"Cache: {'enabled' if CACHE_TTL > 0 else 'disabled'} ttl={CACHE_TTL}s max={CACHE_MAX_SIZE} file={CACHE_FILE or 'none'}")
    print(f"NegCache: ttl={NEG_CACHE_TTL}s | EmptyRetries: {EMPTY_RETRIES} | "
          f"Breaker: {'on' if BREAKER_ENABLED else 'off'} "
          f"(threshold={BREAKER_THRESHOLD} window={BREAKER_WINDOW}s cooldown={BREAKER_COOLDOWN}s)")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler)
    except OSError as e:
        print(f"[proxy] FATAL: cannot bind to port {PORT}: {e}", flush=True)
        sys.exit(1)

    print(f"Proxy ready (threaded): http://127.0.0.1:{PORT}/v1/chat/completions")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down (SIGINT)...", flush=True)
        server.shutdown()
    except Exception as e:
        print(f"[proxy] FATAL: serve_forever crashed: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
