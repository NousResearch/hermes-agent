#!/usr/bin/env python3
"""
Claude Code Proxy — turns claude -p into an OpenAI-compatible /v1/chat/completions endpoint.
Hermes → this proxy → claude -p (OAuth) → Anthropic Claude (Sonnet/Opus).
"""

import hashlib
import json
import signal
import subprocess
import sys
import os
import re
import threading
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

CLAUDE_BIN = "/opt/data/home/.local/lib/node_modules/@anthropic-ai/claude-code/node_modules/@anthropic-ai/claude-code-linux-x64/claude"
HOME = "/opt/data/home"
PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "11435"))
WORKDIR = os.environ.get("CLAUDE_PROXY_WORKDIR", "/opt/data")

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

# Negative cache: key -> {"response": dict, "ts": float}
_neg_cache: dict = {}

# Circuit-breaker state (guarded by _breaker_lock)
_breaker_lock = threading.Lock()
_breaker_bad: dict = {}          # prompt_key -> ts of last bad result (distinct prompts)
_breaker_open_until = 0.0        # epoch seconds; calls short-circuit while now < this
_breaker_trips = 0               # total times the breaker has opened (for /health)

# Track for health checks
request_count = 0

# ── Cache ──

def _cache_key(system: str, prompt: str, tier: str) -> str:
    payload = f"{tier}\x00{system}\x00{prompt}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_evict_expired() -> None:
    """Remove expired entries. Must be called with _cache_lock held."""
    if CACHE_TTL == 0:
        return
    cutoff = time.time() - CACHE_TTL
    expired = [k for k, v in _cache.items() if v["ts"] < cutoff]
    for k in expired:
        del _cache[k]


def cache_get(system: str, prompt: str, tier: str) -> dict | None:
    global _cache_hits, _cache_misses
    if CACHE_TTL == 0:
        _cache_misses += 1
        return None
    key = _cache_key(system, prompt, tier)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None or (time.time() - entry["ts"] > CACHE_TTL):
            if entry is not None:
                del _cache[key]
            _cache_misses += 1
            return None
        _cache_hits += 1
        return entry["response"]


def cache_set(system: str, prompt: str, tier: str, response: dict) -> None:
    if CACHE_TTL == 0:
        return
    # Guard: empty/error results must never enter the 24h cache (poison-cache
    # bug).  They belong in the short negative cache instead.
    if _is_bad_result(response):
        return
    key = _cache_key(system, prompt, tier)
    with _cache_lock:
        _cache_evict_expired()
        # Evict oldest entries if over max size
        if len(_cache) >= CACHE_MAX_SIZE:
            oldest = sorted(_cache.items(), key=lambda kv: kv[1]["ts"])
            for k, _ in oldest[:max(1, CACHE_MAX_SIZE // 10)]:
                del _cache[k]
        _cache[key] = {"response": response, "ts": time.time()}
    threading.Thread(target=_cache_save, daemon=True).start()


def _cache_save() -> None:
    if not CACHE_FILE:
        return
    try:
        with _cache_lock:
            snapshot = dict(_cache)
        Path(CACHE_FILE).write_text(json.dumps(snapshot), encoding="utf-8")
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


def classify_model(prompt: str) -> str:
    """Classify prompt complexity and return recommended model tier.
    Uses Haiku as lightweight classifier, falls back to keyword matching."""
    
    # Override: prefix with ~model forces specific model
    prompt_lower = prompt.lower()
    stripped = prompt_lower.strip()
    for override_tier in ["opus", "sonnet", "haiku"]:
        if stripped.startswith(f"~{override_tier}"):
            return override_tier
    
    # Try Haiku-based classification first
    result = classify_via_haiku(prompt)
    if result:
        return result
    
    # Fallback: keyword matching
    return _classify_by_keywords(prompt)


def classify_via_haiku(prompt: str) -> str | None:
    """Use Haiku to classify task complexity. Returns None on failure."""
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
                "--permission-mode", "bypassPermissions",
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


def build_prompt(messages: list) -> tuple[str, str]:
    """Extract system prompt and build conversation text from OpenAI-format messages."""
    system = ""
    conversation = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if isinstance(content, list):
            # Multimodal — extract text parts
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
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
    return system, prompt


def _run_claude_once(cmd: list, prompt: str, env: dict, timeout: int = 300) -> dict:
    """Spawn ``claude -p`` once and return the parsed JSON result dict, or a
    proxy error dict (``{"error": True, ...}``) on no-output / parse-fail /
    timeout / exception.  ``timeout`` is the per-call hang watchdog."""
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

        # Parse JSON — might be multiple lines, take the last valid JSON
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
            print(f"[proxy] Claude PARSE FAIL. returncode={proc.returncode} stdout_preview={stdout[:200]} stderr={stderr[:200]}", flush=True)
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


def call_claude(system_prompt: str, prompt: str, model: str = None, no_cache: bool = False) -> dict:
    """Call Claude Code in -p mode and return a parsed JSON result.

    Pipeline: 24h cache (good results only) → negative cache (recent bad) →
    circuit breaker → subprocess with one retry on empty/error.  Bad results
    are NEVER written to the 24h cache (the original poison-cache bug); they go
    to the short negative cache so conversation_loop's in-turn burst retries
    don't respawn claude, while a fresh attempt can still run seconds later.
    Emits a structured ``claude_call:`` log line on every path."""
    t0 = time.time()

    # Auto-classify if no explicit model
    if model is None or model == "auto":
        tier = classify_model(prompt)
    else:
        tier = model.lower()

    key = _cache_key(system_prompt, prompt, tier)
    meta = {
        "tier": tier,
        "prompt_chars": len(prompt),
        "system_chars": len(system_prompt or ""),
        "no_cache": no_cache,
    }

    # 1. Good cache (only non-empty, non-error results are ever stored here)
    if not no_cache:
        cached = cache_get(system_prompt, prompt, tier)
        if cached is not None:
            print(f"[cache] HIT tier={tier} prompt={prompt[:60]}...")
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

    # 3. Circuit breaker — broad Claude outage, short-circuit to fallback
    if _breaker_is_open():
        _log_claude_call({**meta, "cache": "miss", "decision": "breaker_open",
                          "elapsed_ms": int((time.time() - t0) * 1000)})
        return {"error": True, "message": "claude circuit breaker open — failing over", "code": 503}

    # Map model names to Claude flags
    model_flag = []
    if "opus" in tier:
        model_flag = ["--model", "opus"]
    elif "haiku" in tier:
        model_flag = ["--model", "haiku"]
    # sonnet: no flag needed (default)

    cmd = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "json",
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    cmd.extend(model_flag)

    env = os.environ.copy()
    env["HOME"] = HOME

    # 4. Run with one retry on empty/error (recovers transient empties so
    #    Claude stays primary instead of failing over on a one-off).
    attempts = 0
    result: dict = {}
    for attempt in range(EMPTY_RETRIES + 1):
        attempts += 1
        result = _run_claude_once(cmd, prompt, env)
        if not _is_bad_result(result):
            break
        if attempt < EMPTY_RETRIES:
            print(f"[proxy] empty/error result (attempt {attempts}/{EMPTY_RETRIES + 1}) "
                  f"tier={tier} — retrying", flush=True)

    bad = _is_bad_result(result)
    if not bad:
        if not no_cache:
            cache_set(system_prompt, prompt, tier, result)
        _breaker_record_good()
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
            self.wfile.write(json.dumps({
                "status": "ok",
                "claude_bin_exists": os.path.exists(CLAUDE_BIN),
                "workdir": WORKDIR,
                "requests": request_count,
                "cache": {
                    "size": len(_cache),
                    "hits": _cache_hits,
                    "misses": _cache_misses,
                    "ttl": CACHE_TTL,
                    "max_size": CACHE_MAX_SIZE,
                    "enabled": CACHE_TTL > 0,
                    "neg_size": len(_neg_cache),
                    "neg_ttl": NEG_CACHE_TTL,
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
        
        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Invalid JSON"}}).encode())
            return
        
        messages = data.get("messages", [])
        model_name = data.get("model", "claude-sonnet-4-6")
        no_cache = self.headers.get("X-Cache-Control", "").lower() == "no-cache"

        if not messages:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "No messages provided"}}).encode())
            return
        
        # Build prompt
        system, prompt = build_prompt(messages)
        
        # Determine model: if explicit tier in name, use it; otherwise auto-classify
        if any(t in model_name.lower() for t in ["opus", "haiku", "sonnet"]):
            tier = model_name.lower()
            print(f"[proxy] explicit: {tier} | prompt: {prompt[:80]}...")
        else:
            tier = classify_model(prompt)
            print(f"[proxy] auto: {tier} | prompt: {prompt[:80]}...")
        
        request_count += 1
        
        # Call Claude with selected tier
        claude_result = call_claude(system, prompt, tier, no_cache=no_cache)
        
        # Convert to OpenAI format — report actual tier used
        openai_response = claude_to_openai(claude_result, f"claude-{tier}")
        
        if "error" in openai_response:
            self.send_response(500)
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

    # Log SIGTERM so the finish script can see it was a signal death
    def _sigterm_handler(signum, frame):
        print(f"[proxy] Received SIGTERM — shutting down gracefully", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    _cache_load()
    print(f"Starting Claude Code Proxy on port {PORT}...")
    print(f"Claude binary: {CLAUDE_BIN} (exists: {os.path.exists(CLAUDE_BIN)})")
    print(f"HOME: {HOME}")
    print(f"Workdir: {WORKDIR}")
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
