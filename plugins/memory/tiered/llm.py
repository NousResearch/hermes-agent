"""Shared LLM helper for tiered memory composition.

Composer: deepseek-v4-pro via DeepSeek API.
Uses the same DEEPSEEK_API_KEY as the rest of Hermes — no separate
OpenRouter or Ollama dependency.

P169/MOL-560: dropped the local Ollama qwen3:1.7b primary path entirely.
Post-MOL-546 the embedder is in-process via fastembed, so the composer
was the only remaining Ollama consumer. Removing it lets us stop the
Ollama daemon permanently and reclaim ~220 MB resident + remove a
launchd-managed background process from the stack.

Quality direction is UP (Kimi K2.6 ≫ qwen3:1.7b on instruction following
and long-form summarization). Cost trade is ~$0.01-0.05 per nightly
consolidation run — accepted as worth the operational simplification.

Public API `llm_compose(prompt, context)` signature is unchanged.

P169 review fix-pass (silent-failure-hunter findings on PR #200):
- Distinct `ComposerKeyMissing` / `ComposerAuthFailure` exceptions separate
  permanent-failure modes (missing key, expired key) from transient ones
  (rate limit, 5xx, network blip). Both write tripwires to
  `~/.hermes/state/composer-<reason>-<ts>.json` so cron failures surface
  via the existing maintenance-pending scanner instead of waiting for
  MEMORY.md staleness to become user-visible.
- `COMPOSER_BASE_URL` resolved-value logged once per process so a stale
  `HERMES_MOCK_LLM_URL` env var doesn't silently redirect production to
  the mock (`Mock-First Dev Loops` memory pattern).
- Per-process `MAX_COMPOSER_CALLS_PER_RUN` budget cap guards against
  runaway loops in multi-session ingest paths (memory_ingest_external.py
  calls llm_compose once per session × N sessions × cron tick).
"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# P169/MOL-560: Kimi K2.6 was the sole composer (was fallback).
# MOL-602 follow-up: switched to deepseek-v4-pro (primary model) — no separate
# OpenRouter dependency, no Ollama dependency. Uses same key as rest of Hermes.
COMPOSER_MODEL = "deepseek-v4-pro"
# Base URL matches the deepseek provider plugin (plugins/model-providers/deepseek/).
COMPOSER_BASE_URL = os.environ.get("HERMES_MOCK_LLM_URL", "").strip() or "https://api.deepseek.com/v1"
COMPOSER_API_KEY_ENV = "DEEPSEEK_API_KEY"

# P169 review fix-pass I1: canonical endpoint — anything else triggers the
# "stale mock URL silent redirect" guard log.
_CANONICAL_COMPOSER_URL = "https://api.deepseek.com/v1"

MAX_TOKENS = 4096
TEMPERATURE = 0.3

# Hard char budget — see MOL-168. Kimi K2.6 has 200k context; the 300k cap
# is a cost guard, not an overflow guard.
MAX_PROMPT_CHARS = 300_000

# P169 review fix-pass I3: per-process budget cap. Resets on gateway restart.
# 500 covers a normal 4 AM cron (1 consolidation + up to ~100 sessions × 1
# memory_ingest call each + headroom). Runaway loops trip it visibly.
MAX_COMPOSER_CALLS_PER_RUN = int(os.environ.get("MEMORY_COMPOSER_MAX_CALLS_PER_RUN", "500"))
_call_count = 0

# P169 review fix-pass I1: one-shot announce gate for resolved BASE_URL.
_base_url_announced = False

# P169 review fix-pass C2/I2: tripwire dir — matches P79/P168 patterns.
_TRIPWIRE_DIR = os.path.expanduser("~/.hermes/state")


class ComposerKeyMissing(RuntimeError):
    """OPENROUTER_API_KEY not set. Permanent failure until operator intervenes."""


class ComposerAuthFailure(RuntimeError):
    """OpenRouter returned 401/403. Key expired or revoked — permanent until rotated."""


def _write_tripwire(reason: str, detail: dict) -> None:
    """Best-effort tripwire write to ~/.hermes/state/. Fails open (primary error
    path already logged at ERROR; tripwire is the secondary audit trail)."""
    try:
        os.makedirs(_TRIPWIRE_DIR, exist_ok=True)
        path = os.path.join(_TRIPWIRE_DIR, f"composer-{reason}-{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "reason": reason, **detail}, f)
    except OSError:
        pass


def _announce_base_url_once() -> None:
    """Log resolved COMPOSER_BASE_URL on first composer call.

    P169 I1: guards against stale HERMES_MOCK_LLM_URL silently redirecting
    production traffic to the mock. Non-canonical URLs log at WARNING so a
    grep of gateway.log surfaces the override.
    """
    global _base_url_announced
    if _base_url_announced:
        return
    _base_url_announced = True
    if COMPOSER_BASE_URL == _CANONICAL_COMPOSER_URL:
        logger.info("memory LLM composer: base_url=%s (canonical)", COMPOSER_BASE_URL)
    else:
        logger.warning(
            "memory LLM composer: base_url=%s (non-canonical — HERMES_MOCK_LLM_URL active). "
            "Confirm this is intentional; stale env var silently redirects production to mock.",
            COMPOSER_BASE_URL,
        )


def _call_composer(prompt: str) -> str:
    """Call deepseek-v4-pro via DeepSeek API. Returns content text.

    Raises:
        ComposerKeyMissing: DEEPSEEK_API_KEY not set in env. Logged at ERROR
            + tripwire written before raise.
        ComposerAuthFailure: 401/403 from DeepSeek (expired/revoked key).
            Logged at ERROR + tripwire before raise.
        Other openai.* exceptions: transient API errors (rate limit, 5xx,
            network) — caller's `llm_compose` outer except catches at WARNING.
    """
    _announce_base_url_once()

    from openai import OpenAI, AuthenticationError, PermissionDeniedError

    api_key = os.environ.get(COMPOSER_API_KEY_ENV, "").strip()
    if not api_key:
        logger.error(
            "Memory LLM composer: %s not set — composer is permanently degraded. "
            "Re-launch gateway under envchain or check ~/.hermes/.env loader.",
            COMPOSER_API_KEY_ENV,
        )
        _write_tripwire("key-missing", {"env_var": COMPOSER_API_KEY_ENV})
        raise ComposerKeyMissing(f"{COMPOSER_API_KEY_ENV} not set")

    client = OpenAI(base_url=COMPOSER_BASE_URL, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=COMPOSER_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
    except (AuthenticationError, PermissionDeniedError) as e:
        # 401/403 — expired or revoked key. Permanent failure until operator
        # rotates the credential. Distinct from transient API errors so the
        # tripwire scanner can escalate it specifically.
        logger.error(
            "Memory LLM composer: AUTH FAILURE (%s: %s). Key likely expired or revoked. "
            "Composer fails every subsequent call until rotated via envchain.",
            type(e).__name__, str(e)[:200],
        )
        _write_tripwire("auth-failure", {
            "error_class": type(e).__name__,
            "detail": str(e)[:200],
        })
        raise ComposerAuthFailure(str(e)) from e
    return response.choices[0].message.content


def llm_compose(prompt: str, context: str) -> Optional[str]:
    """Compose memory via deepseek-v4-pro composer. Returns text or None on failure.

    P169 contract change: pre-P169 the Ollama-then-Kimi chain meant `None`
    was a rare both-failed case. Post-P169 `None` covers any single failure
    — callers should treat `None` as a loud event and surface it (see
    consolidation.py and hot_cache.py for the canonical patterns).

    Permanent-failure paths (missing key, auth failure, budget exceeded)
    log at ERROR + write tripwires to `~/.hermes/state/composer-*.json`.
    Transient paths (rate limit, 5xx, empty response) log at WARNING.
    Either way the return is `None`; distinguish via the audit trail.
    """
    global _call_count

    # I3 budget cap — runaway-loop guard. Resets on process restart.
    if _call_count >= MAX_COMPOSER_CALLS_PER_RUN:
        logger.error(
            "Memory LLM composer: budget cap hit (%d calls this process). "
            "Increase MEMORY_COMPOSER_MAX_CALLS_PER_RUN or restart gateway.",
            MAX_COMPOSER_CALLS_PER_RUN,
        )
        _write_tripwire("budget-exceeded", {
            "cap": MAX_COMPOSER_CALLS_PER_RUN,
            "called": _call_count,
        })
        return None

    full = f"{prompt}\n\n{context}"
    if len(full) > MAX_PROMPT_CHARS:
        logger.error(
            "llm_compose input too large (%d chars, cap %d); truncating tail. "
            "See MOL-168.",
            len(full),
            MAX_PROMPT_CHARS,
        )
        full = full[:MAX_PROMPT_CHARS] + "\n\n[...truncated for size cap — see MOL-168...]"

    _call_count += 1
    try:
        result = _call_composer(full)
        if result:
            logger.info("memory LLM: %s (composer)", COMPOSER_MODEL)
            return result
        logger.warning("memory LLM composer returned empty content")
        return None
    except (ComposerKeyMissing, ComposerAuthFailure):
        # Already logged at ERROR + tripwire written inside _call_composer.
        # Don't double-log; return None per public-API contract.
        return None
    except Exception as e:
        # Transient API errors (rate limit, 5xx, network blip). Caller still
        # treats None as a failure, but log severity reflects that this class
        # of failure typically self-resolves on the next cron tick.
        logger.warning(
            "memory LLM composer failed transient (%s: %s)",
            type(e).__name__, str(e)[:200],
        )
        return None
