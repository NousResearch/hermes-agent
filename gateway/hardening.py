"""
Hardening layer for extreme breach scenarios.

Provides:
  RateLimiter   — sliding-window per-key rate limiter with exponential lockout
                  and a final hard block after N repeated violations.
  InputGuard    — strips null bytes / C0 control chars, enforces a message
                  length cap, and detects + redacts prompt-injection preambles.
  hardening_gate — pre_gateway_dispatch hook wiring both together.

Activated when HERMES_HARDENING=true is set. All tunables are overridable
via environment variables (see _intenv() calls below).
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable configuration (env-overridable)
# ---------------------------------------------------------------------------


def _intenv(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


_WINDOW_SECS       = _intenv("HERMES_RATE_WINDOW_SECS",    60)   # sliding window
_WINDOW_MAX        = _intenv("HERMES_RATE_MAX_MSGS",        30)   # max msgs per window
_LOCKOUT_BASE_SECS = _intenv("HERMES_RATE_LOCKOUT_BASE",    30)   # first lockout
_LOCKOUT_MAX_SECS  = _intenv("HERMES_RATE_LOCKOUT_MAX",    900)   # max lockout (15 min)
_HARD_BLOCK_AFTER  = _intenv("HERMES_RATE_HARD_BLOCK",       5)   # strikes before hard block
_MAX_MSG_LEN       = _intenv("HERMES_MAX_MSG_LEN",       32_000)  # chars before truncation

# ---------------------------------------------------------------------------
# Prompt-injection detection patterns
# ---------------------------------------------------------------------------
# Each pattern targets a well-known jailbreak / indirect-injection class.
# Matches are *redacted* (not hard-dropped) so benign coincidental matches
# are recoverable via the operator log; only the preamble fragment is removed.

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Ignore all previous instructions" variants
    re.compile(
        r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|your)\s+"
        r"(?:instructions?|prompts?|commands?|context|rules?|constraints?)",
        re.IGNORECASE,
    ),
    # Injected header blocks: "NEW INSTRUCTIONS:", "SYSTEM:", "OVERRIDE:"
    re.compile(
        r"(?:\n|^)\s*(?:NEW\s+INSTRUCTIONS?|SYSTEM|OVERRIDE|JAILBREAK)\s*[:\-]",
        re.IGNORECASE | re.MULTILINE,
    ),
    # DAN / character-swap preambles
    re.compile(
        r"\b(?:DAN|jailbreak|pretend\s+you\s+are|act\s+as\s+if\s+you\s+(?:are|were|have\s+no))",
        re.IGNORECASE,
    ),
    # "You are now X with no restrictions"
    re.compile(
        r"\byou\s+(?:are\s+now|must\s+now)\b.{0,80}\b"
        r"(?:no\s+(?:restrictions|limits|filters|rules|guidelines))",
        re.IGNORECASE | re.DOTALL,
    ),
    # LLM prompt-delimiter smuggling ("<|im_start|>", "[INST]", "###System")
    re.compile(
        r"(?:<\|(?:im_start|system|user|assistant)\|>|\[/?INST\]|###\s*System)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Per-key sliding-window rate limiter.

    State machine:
      NORMAL  → messages accepted while window count < _WINDOW_MAX
      LOCKED  → reject until lockout expires; each new violation doubles duration
      BLOCKED → permanent hard block (cleared only by reset())

    Thread-safe via an internal Lock. Uses monotonic time.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # key → deque of monotonic timestamps within the current window
        self._windows: dict[str, deque[float]] = {}
        # key → (lockout_until_mono, strike_count, last_lockout_secs)
        # lockout_until_mono == inf signals a hard block
        self._lockouts: dict[str, tuple[float, int, int]] = {}

    # ------------------------------------------------------------------
    def check(self, key: str) -> tuple[bool, str]:
        """Return (allowed, reason).  Caller should skip the message when not allowed."""
        now = time.monotonic()
        with self._lock:
            entry = self._lockouts.get(key)

            # Hard block — permanent until reset()
            if entry is not None:
                until, strikes, _ = entry
                if until == float("inf"):
                    return False, f"hard_blocked_after_{strikes}_strikes"

            # Active lockout
            if entry is not None:
                until, strikes, _ = entry
                if now < until:
                    remaining = int(until - now)
                    return False, f"rate_limited_{remaining}s_strike_{strikes}"

            # Advance sliding window (drop entries older than the window)
            window = self._windows.setdefault(key, deque())
            cutoff = now - _WINDOW_SECS
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= _WINDOW_MAX:
                # Compute next lockout duration (doubling, capped)
                if entry is None:
                    strikes, lockout_secs = 1, _LOCKOUT_BASE_SECS
                else:
                    _, prev_strikes, prev_secs = entry
                    strikes = prev_strikes + 1
                    lockout_secs = min(prev_secs * 2, _LOCKOUT_MAX_SECS)

                if strikes >= _HARD_BLOCK_AFTER:
                    self._lockouts[key] = (float("inf"), strikes, lockout_secs)
                    return False, f"hard_blocked_after_{strikes}_strikes"

                self._lockouts[key] = (now + lockout_secs, strikes, lockout_secs)
                return False, f"rate_limited_{lockout_secs}s_strike_{strikes}"

            window.append(now)
            return True, "ok"

    def reset(self, key: str) -> None:
        """Admin reset — clears all rate-limit state for a key."""
        with self._lock:
            self._windows.pop(key, None)
            self._lockouts.pop(key, None)

    def stats(self, key: str) -> dict:
        """Return diagnostic stats for a key (for admin tooling / tests)."""
        now = time.monotonic()
        with self._lock:
            window = self._windows.get(key, deque())
            cutoff = now - _WINDOW_SECS
            active = sum(1 for t in window if t >= cutoff)
            entry = self._lockouts.get(key)
            if entry is None:
                return {"messages_in_window": active, "state": "normal"}
            until, strikes, secs = entry
            if until == float("inf"):
                return {"messages_in_window": active, "state": "hard_blocked", "strikes": strikes}
            remaining = max(0.0, until - now)
            return {
                "messages_in_window": active,
                "state": "locked",
                "strikes": strikes,
                "remaining_secs": round(remaining, 1),
            }


# ---------------------------------------------------------------------------
# InputGuard
# ---------------------------------------------------------------------------

class InputGuard:
    """
    Input sanitizer.

    sanitize(text) → (clean_text, was_modified, injection_fragments)

    Operations (in order):
      1. Strip null bytes
      2. Strip dangerous C0/C1 control characters (keeps \\t, \\n, \\r)
      3. Hard-truncate at _MAX_MSG_LEN characters
      4. Strip prompt-injection preambles (replaced with [REDACTED])
    """

    def sanitize(self, text: str) -> tuple[str, bool, list[str]]:
        """
        Returns (sanitized_text, was_modified, list_of_injection_fragments).
        injection_fragments are truncated to 120 chars each for logging.
        """
        original = text

        # 1. Null bytes — many injection payloads embed these to confuse parsers
        text = text.replace("\x00", "")

        # 2. Dangerous control chars (keep \t=0x09, \n=0x0a, \r=0x0d)
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 3. Length cap
        if len(text) > _MAX_MSG_LEN:
            text = text[:_MAX_MSG_LEN] + " [TRUNCATED]"

        # 4. Prompt injection scrubbing
        fragments: list[str] = []
        for pat in _INJECTION_PATTERNS:
            m = pat.search(text)
            if m:
                fragments.append(m.group(0)[:120])
                text = pat.sub("[REDACTED]", text)

        return text, text != original, fragments


# ---------------------------------------------------------------------------
# Module-level singletons (shared across all sessions in the same process)
# ---------------------------------------------------------------------------

_rate_limiter = RateLimiter()
_input_guard  = InputGuard()


def _is_enabled() -> bool:
    return os.environ.get("HERMES_HARDENING", "").lower() in ("1", "true", "yes")


def _key_for(event: Any) -> str:
    """Derive a stable, platform-qualified rate-limit key from a MessageEvent."""
    try:
        src = event.source
        platform = getattr(src.platform, "value", str(src.platform)) if src.platform else "unknown"
        user     = str(getattr(src, "user_id",  "") or "")
        chat     = str(getattr(src, "chat_id",  "") or "")
        # Prefer user_id (per-user) over chat_id (per-conversation)
        identity = user or chat or "anon"
        return f"{platform}:{identity}"
    except Exception:
        return "anon"


# ---------------------------------------------------------------------------
# pre_gateway_dispatch hook entry point
# ---------------------------------------------------------------------------

def hardening_gate(event_type: str, context: dict) -> dict | None:
    """
    pre_gateway_dispatch hook — enforces rate limiting and input sanitization.

    Return contract (same as other pre_gateway_dispatch plugins):
      {"action": "skip",    "reason": ...}  → drop message silently
      {"action": "rewrite", "text":  ...}   → replace message text, continue
      {"action": "allow"}                   → pass through unchanged
      None                                  → equivalent to "allow"
    """
    if not _is_enabled():
        return {"action": "allow"}

    event = context.get("event")
    if event is None:
        return {"action": "allow"}

    key  = _key_for(event)
    text = getattr(event, "text", None) or ""

    # --- Rate limit ----------------------------------------------------------
    allowed, reason = _rate_limiter.check(key)
    if not allowed:
        logger.warning("hardening_gate: rate_limit key=%s reason=%s", key, reason)
        return {"action": "skip", "reason": f"rate_limit:{reason}"}

    # --- Input sanitation ----------------------------------------------------
    if text:
        clean, was_modified, fragments = _input_guard.sanitize(text)
        if fragments:
            logger.warning(
                "hardening_gate: injection_detected key=%s fragments=%r",
                key,
                fragments,
            )
        if was_modified:
            return {"action": "rewrite", "text": clean}

    return {"action": "allow"}
