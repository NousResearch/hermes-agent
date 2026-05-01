"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import ast
import hashlib
import hmac
import json
import logging
import os
import socket as _socket
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
)
logger = logging.getLogger(__name__)

# Dynamic role aliases exposed by Hermes Gateway. These are virtual models
# from the client's point of view, but they route through hermes-swarm using
# role-specific routing hints instead of a fixed backend mapping.
ROLE_ALIAS_CONFIG = {
    "hermes-gateway/hermes-translator": {
        "mode": "swarm",
        "hint": {
            "role": "translator",
            "task_type": "translation",
            "recommended_tier": "premium",
        },
    },
    "hermes-gateway/hermes-triage": {
        "mode": "swarm",
        "hint": {
            "role": "triage",
            "task_type": "triage",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-duplicate-pr": {
        "mode": "swarm",
        "hint": {
            "role": "duplicate-pr",
            "task_type": "repo_review",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-fast": {
        "mode": "swarm",
        "hint": {
            "role": "fast",
            "task_type": "general",
            "recommended_tier": "cheap",
        },
    },
    "hermes-gateway/hermes-balanced": {
        "mode": "swarm",
        "hint": {
            "role": "balanced",
            "task_type": "general",
            "recommended_tier": "balanced",
        },
    },
}


def _get_role_alias_config(model: str) -> Optional[Dict[str, Any]]:
    raw = str(model or "").strip().lower()
    if not raw:
        return None
    for alias, cfg in ROLE_ALIAS_CONFIG.items():
        if alias.lower() == raw:
            return cfg
    return None


def _resolve_role_alias(model: str) -> str | None:
    """Legacy fixed mapping lookup.

    Dynamic aliases defined in ROLE_ALIAS_CONFIG are handled by swarm routing
    and intentionally do not resolve to a single concrete backend here.
    """
    if _get_role_alias_config(model):
        return None
    return None



_SWARM_PREMIUM_MODEL_HINTS = (
    "github-copilot/gpt-5.4",
    "zai/glm-4.7",
    "minimax/MiniMax-M2.7",
)

_SWARM_BALANCED_MODEL_HINTS = (
    "github-copilot/gpt-5-mini",
    "openai/gpt-5-mini",
    "opencode-zen/gpt-5-nano",
)

_SWARM_CHEAP_MODEL_HINTS = (
    "opencode-zen/gpt-5-nano",
    "opencode-zen/ling-2.6-flash-free",
    "ollama/qwen3-coder-next",
)

_OPENROUTER_FREE_CACHE: Dict[str, tuple[float, bool]] = {}
_OPENROUTER_FREE_CACHE_TTL_SECONDS = 300.0


def _openrouter_model_free_cached(model_id: str) -> bool:
    """Best-effort free-tier check for OpenRouter models.

    Fast path: models explicitly suffixed with ``:free`` are treated as free.
    Otherwise, query OpenRouter model metadata and only allow models whose
    prompt and completion pricing are both zero. Results are cached briefly so
    repeated swarm checks don't hammer the models endpoint.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return False
    if ":free" in raw.lower():
        return True

    now = time.time()
    cached = _OPENROUTER_FREE_CACHE.get(raw)
    if cached and (now - cached[0]) < _OPENROUTER_FREE_CACHE_TTL_SECONDS:
        return bool(cached[1])

    is_free = False
    try:
        import httpx

        headers = {}
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = httpx.get(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/models",
            headers=headers,
            timeout=5.0,
        )
        if response.status_code == 200:
            payload = response.json()
            for item in payload.get("data", []):
                if str(item.get("id") or "").strip() != raw:
                    continue
                pricing = item.get("pricing") or {}

                def _as_float(value: Any) -> float:
                    try:
                        return float(value)
                    except Exception:
                        return 0.0

                prompt_cost = _as_float(pricing.get("prompt", 0))
                completion_cost = _as_float(pricing.get("completion", 0))
                is_free = (prompt_cost == 0.0 and completion_cost == 0.0)
                break
    except Exception:
        is_free = False

    _OPENROUTER_FREE_CACHE[raw] = (now, is_free)
    return is_free


# Default settings


def _resolve_swarm_model(pool, *, context_overflow: bool = False, estimated_tokens: int = 0):
    """Resolve a model from the swarm pool based on selection policy.

    When context_overflow is True and large_context_fallbacks are available,
    switch to the largest available model (no compression needed).

    When estimated_tokens > 0, models whose context window cannot safely
    hold the request are excluded from consideration, preventing 413 errors.
    """
    import random, os
    primary = pool.get("primary", "")
    fallbacks = pool.get("fallbacks", [])
    large_context_fallbacks = pool.get("large_context_fallbacks", [])
    policy = pool.get("selection_policy", "cost-balanced")
    # Use _swarm_model_is_available to check both credentials AND cooldown status
    available_fallbacks = [m for m in fallbacks if _swarm_model_is_available(m)]
    available_large_context = [m for m in large_context_fallbacks if _swarm_model_is_available(m)]
    primary_available = bool(primary) and _swarm_model_is_available(primary)

    def _model_safe_for_tokens(model: str, tokens: int) -> bool:
        """Return True if model's context window can safely hold `tokens` tokens."""
        if tokens <= 0:
            return True
        ctx_len = _model_context_length(model)
        if ctx_len <= 0:
            return True  # Unknown context — skip filtering
        # Use 85% of context window as safe limit (leaves room for output + buffer)
        safe_limit = int(ctx_len * 0.85)
        return tokens <= safe_limit

    # Filter candidates by context fit when we have an estimate
    if estimated_tokens > 0:
        _ctx_filtered_fallbacks = [m for m in available_fallbacks if _model_safe_for_tokens(m, estimated_tokens)]
        _ctx_filtered_large = [m for m in available_large_context if _model_safe_for_tokens(m, estimated_tokens)]
        _ctx_filtered_primary = _model_safe_for_tokens(primary, estimated_tokens)
        if _ctx_filtered_fallbacks:
            available_fallbacks = _ctx_filtered_fallbacks
        if _ctx_filtered_large:
            available_large_context = _ctx_filtered_large
        if not _ctx_filtered_primary:
            primary_available = False
        logger.info(
            "[api_server] context filter: estimated_tokens=%d, primary_fit=%s, "
            "fallbacks_context_fit=%s, large_context_fit=%s",
            estimated_tokens, _ctx_filtered_primary,
            [m for m in available_fallbacks], [m for m in available_large_context],
        )

    # Context overflow path: use a large-context model instead of compressing
    if context_overflow and available_large_context:
        from agent.model_metadata import get_model_context_length_quick
        # Sort large-context models by context length descending, pick the largest
        _sorted = sorted(
            available_large_context,
            key=lambda m: get_model_context_length_quick(m),
            reverse=True,
        )
        if _sorted:
            logger.info(f"[api_server] context overflow — switching to large-context model: {_sorted[0]}")
            return _sorted[0]
        # Fall through to normal selection if large-context list is empty
    
    if not fallbacks:
        return "openrouter/free"
    if primary and not primary_available:
        logger.warning("[api_server] primary swarm model unavailable (missing credentials or in cooldown): %s", primary)
    candidates = available_fallbacks or fallbacks
    if policy == "complexity-aware":
        routing_hint = pool.get("routing_hint") or {}
        logger.info("[api_server] complexity-aware selection: estimated_tokens=%s routing_hint=%s", estimated_tokens, routing_hint)
        recommended_tier = str(routing_hint.get("recommended_tier") or "").strip().lower()
        task_type = str(routing_hint.get("task_type") or "").strip().lower()
        needs_instruction_following = bool(routing_hint.get("needs_instruction_following"))
        needs_repo_reasoning = bool(routing_hint.get("needs_repo_reasoning"))
        needs_bug_judgement = bool(routing_hint.get("needs_bug_judgement"))

        def _pick(preferred: tuple[str, ...]) -> Optional[str]:
            for preferred_model in preferred:
                for candidate in candidates:
                    if candidate == preferred_model:
                        return candidate
            return None

        def _pick_stronger_than_primary() -> Optional[str]:
            for preferred_model in (
                "google/gemini-2.5-flash",
                "openai/gpt-5-mini",
                "github-copilot/gpt-5-mini",
            ):
                for candidate in candidates:
                    if candidate == preferred_model and candidate != primary:
                        return candidate
            for candidate in candidates:
                if candidate != primary:
                    return candidate
            return None

        if recommended_tier == "premium":
            premium_choice = _pick(_SWARM_PREMIUM_MODEL_HINTS)
            if premium_choice:
                logger.info("[api_server] scout escalation → premium model: %s", premium_choice)
                return premium_choice
            stronger_choice = _pick_stronger_than_primary()
            if stronger_choice:
                logger.info("[api_server] premium requested but unavailable; using strongest available non-primary model: %s", stronger_choice)
                return stronger_choice
        elif recommended_tier == "balanced":
            balanced_choice = _pick(_SWARM_BALANCED_MODEL_HINTS)
            if balanced_choice:
                logger.info("[api_server] scout routing → balanced model: %s", balanced_choice)
                return balanced_choice
        elif recommended_tier == "cheap":
            cheap_choice = _pick(_SWARM_CHEAP_MODEL_HINTS)
            if cheap_choice:
                logger.info("[api_server] scout routing → cheap model: %s", cheap_choice)
                return cheap_choice

        if task_type in {"repo_review", "debugging", "implementation", "architecture"} or (
            needs_instruction_following and (needs_repo_reasoning or needs_bug_judgement)
        ):
            premium_choice = _pick(_SWARM_PREMIUM_MODEL_HINTS)
            if premium_choice:
                logger.info("[api_server] heuristic escalation → premium model: %s", premium_choice)
                return premium_choice

        if estimated_tokens > 8000:
            premium = [m for m in candidates if "gpt-5.3-codex" in m or "gpt-5.2-codex" in m]
            if premium:
                logger.info("[api_server] HIGH complexity (%s tokens) — using premium: %s", estimated_tokens, premium[0])
                return premium[0]
        if estimated_tokens > 2000:
            balanced = [m for m in candidates if m in {"github-copilot/gpt-5-mini", "openai/gpt-5-mini", "google/gemini-2.5-flash"}]
            if balanced:
                logger.info("[api_server] MEDIUM complexity (%s tokens) — using: %s", estimated_tokens, balanced[0])
                return balanced[0]
        if primary_available:
            logger.info("[api_server] SIMPLE task — using primary: %s", primary)
            return primary
        return candidates[0]
    if primary_available:
        return primary
    if policy == "round-robin":
        return candidates[0]
    elif policy == "cost-balanced":
        cheap = [m for m in candidates if any(x in m for x in ["gemma", "free", "nemotron"])]
        return random.choice(cheap) if cheap else candidates[0]
    return candidates[0]


# Pattern for detecting context overflow errors
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length", "context size", "maximum context", "token limit",
    "too many tokens", "context window", "prompt is too long",
    "context length exceeded", "context overflow", "too many input tokens",
    "maximum tokens", "output tokens", "context_exceeded",
)


def _is_context_overflow_error(error_msg: str) -> bool:
    """Return True if error_msg indicates a context-length overflow."""
    if not error_msg:
        return False
    msg_lower = error_msg.lower()
    return any(pat in msg_lower for pat in _CONTEXT_OVERFLOW_PATTERNS)


_EXPLICIT_MODEL_PROVIDER_ALIASES = {
    "openrouter": "openrouter",
    "openai": "openai-codex",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
    "anthropic": "anthropic",
    "copilot": "copilot",
    "github-copilot": "copilot",
    "copilot-acp": "copilot-acp",
    "opencode": "opencode-zen",
    "opencode-zen": "opencode-zen",
    "zen": "opencode-zen",
    "opencode-go": "opencode-go",
    "go": "opencode-go",
    "nous": "nous",
    "custom": "custom",
    "xai": "xai",
    "zai": "zai",
    "kimi-coding": "kimi-coding",
    "kimi-coding-cn": "kimi-coding-cn",
    "minimax": "minimax",
    "minimax-cn": "minimax-cn",
    "ai-gateway": "ai-gateway",
    "kilocode": "kilocode",
    "alibaba": "alibaba",
    "arcee": "arcee",
    "huggingface": "huggingface",
    "xiaomi": "xiaomi",
    "bedrock": "bedrock",
    "qwen-oauth": "qwen-oauth",
    "ollama-cloud": "ollama-cloud",
    "ollama": "ollama-cloud",
}


def _explicit_provider_from_model(model: str) -> str:
    raw = str(model or "").strip()
    if "/" not in raw:
        return ""
    prefix = raw.split("/", 1)[0].strip().lower()
    return _EXPLICIT_MODEL_PROVIDER_ALIASES.get(prefix, "")


def _align_runtime_with_explicit_model(runtime_kwargs: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Honor provider-prefixed model IDs before the first upstream call.

    This prevents auto provider resolution from selecting an unrelated backend
    (for example Codex OAuth) when the model string itself already names the
    intended provider (for example ``opencode-zen/big-pickle``).
    """
    explicit_provider = _explicit_provider_from_model(model)
    if not explicit_provider:
        return runtime_kwargs

    current_provider = str(runtime_kwargs.get("provider") or "").strip().lower()
    if current_provider == explicit_provider and runtime_kwargs.get("api_key"):
        return runtime_kwargs

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolved = resolve_runtime_provider(requested=explicit_provider)
    except Exception as exc:
        logger.warning(
            "[api_server] failed to resolve provider %s for explicit model %s: %s",
            explicit_provider,
            model,
            exc,
        )
        return runtime_kwargs

    merged = dict(runtime_kwargs)
    for key in ("api_key", "base_url", "provider", "api_mode"):
        value = resolved.get(key)
        if value is not None:
            merged[key] = value

    logger.info(
        "[api_server] aligned runtime provider to %s for explicit model %s",
        explicit_provider,
        model,
    )
    return merged


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 1_000_000  # 1 MB default limit for POST bodies
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array
MAX_HISTORY_MESSAGES = 200
MAX_HISTORY_TEXT_LENGTH = 240_000


def _model_context_length(model_name: str) -> int:
    """Look up the context window size for a model, or 0 if unknown."""
    try:
        from agent.model_metadata import get_model_context_length
        return get_model_context_length(model_name) or 0
    except Exception:
        return 0


def _model_safe_for_tokens(model: str, tokens: int, margin_fraction: float = 0.85) -> bool:
    """Return True if model's context window can safely hold `tokens` tokens.

    This module-level helper mirrors the in-function implementation used by
    _resolve_swarm_model but ensures other code paths (like the swarm scout)
    can call it without raising NameError if the local inner function isn't in
    scope.
    """
    if tokens <= 0:
        return True
    ctx_len = _model_context_length(model)
    if ctx_len <= 0:
        return True  # Unknown context — skip filtering
    safe_limit = int(ctx_len * margin_fraction)
    return tokens <= safe_limit


def _messages_token_count(messages: List[Dict[str, Any]], system_prompt: str = "") -> int:
    """Estimate token count for messages + optional system prompt."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough
        total = estimate_messages_tokens_rough(messages)
        if system_prompt:
            total += estimate_messages_tokens_rough([{"role": "system", "content": system_prompt}])
        return total
    except Exception:
        # Fallback: rough char-based estimate
        total = len(system_prompt)
        for msg in messages:
            total += len(str(msg.get("content") or ""))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    total += len(str(tc.get("function", {}).get("arguments") or ""))
        return total // 4  # ~4 chars per token


def _model_can_handle_context(model_name: str, estimated_tokens: int, margin_fraction: float = 0.85) -> bool:
    """Return True if the model's context window can safely hold the estimated tokens."""
    ctx_len = _model_context_length(model_name)
    if ctx_len <= 0:
        return True  # Unknown context — assume it's fine
    safe_limit = int(ctx_len * margin_fraction)
    return estimated_tokens <= safe_limit


def _compact_message_history(
    messages: List[Dict[str, Any]],
    session_id: str = "unknown",
    *,
    system_prompt: str = "",
    target_model: str = "",
) -> List[Dict[str, Any]]:
    """Keep recent history bounded so provider request bodies stay sane.

    Applies two filters:
    1. Hard cap: last MAX_HISTORY_MESSAGES messages.
    2. Token-aware cap: if target_model is known, truncate to its safe context limit.
       Falls back to MAX_HISTORY_TEXT_LENGTH chars when target_model is unknown.

    This prevents 413 errors by ensuring the final payload fits the model's window
    before it reaches the LLM API call.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if not messages:
        return []

    original_count = len(messages)
    trimmed = list(messages[-MAX_HISTORY_MESSAGES:])
    trimmed_count = len(trimmed)

    # Determine the effective token budget from target model (or fallback to chars)
    estimated_tokens = _messages_token_count(trimmed, system_prompt)
    target_ctx = _model_context_length(target_model)
    if target_ctx > 0:
        # Use 85% of context window as the safe token budget (leaves room for output)
        effective_budget = int(target_ctx * 0.85)
        token_based = True
    else:
        # Fallback: use char-based budget scaled to ~4 chars/token
        effective_budget = MAX_HISTORY_TEXT_LENGTH // 4
        token_based = False

    def _msg_cost(msg: Dict[str, Any]) -> int:
        cost = len(str(msg.get("content") or ""))
        reasoning_content = msg.get("reasoning_content")
        if isinstance(reasoning_content, str):
            cost += len(reasoning_content)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        cost += len(str(fn.get("arguments") or ""))
        return cost // 4  # convert chars to rough token estimate

    total_cost = sum(_msg_cost(msg) for msg in trimmed)

    if original_count > MAX_HISTORY_MESSAGES:
        _logger.warning(
            "[CONTEXT] Session %s compacting: %d messages -> %d (limit: %d, total ~%d tokens)",
            session_id, original_count, trimmed_count, MAX_HISTORY_MESSAGES, total_cost,
        )

    if total_cost <= effective_budget:
        return trimmed

    kept: List[Dict[str, Any]] = []
    running = 0
    dropped_count = 0
    for msg in reversed(trimmed):
        cost = _msg_cost(msg)
        if kept and running + cost > effective_budget:
            dropped_count += 1
            continue
        kept.append(msg)
        running += cost
        if running >= effective_budget:
            break

    kept.reverse()

    if dropped_count > 0:
        budget_desc = f"~{effective_budget:,} tokens ({target_model})" if token_based else f"{MAX_HISTORY_TEXT_LENGTH:,} chars"
        _logger.warning(
            "[CONTEXT] Session %s TRUNCATED: dropped %d messages (budget: %s, ~%d total tokens). "
            "Original messages: %d, Kept: %d",
            session_id, dropped_count, budget_desc, running,
            original_count, len(kept),
        )

    return kept


def _extract_openai_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI-style tool_calls from an incoming chat request."""
    normalized: List[Dict[str, Any]] = []
    if not isinstance(raw_tool_calls, list):
        return normalized
    for tc in raw_tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        call_id = tc.get("id") or tc.get("call_id")
        item: Dict[str, Any] = {
            "type": tc.get("type", "function"),
            "function": {
                "name": name.strip(),
                "arguments": arguments,
            },
        }
        if isinstance(call_id, str) and call_id.strip():
            item["id"] = call_id.strip()
            item["call_id"] = call_id.strip()
        normalized.append(item)
    return normalized


def _enrich_client_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure client-visible tool calls satisfy strict downstream schemas."""
    if not isinstance(tc, dict):
        return tc
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    name = fn.get("name")
    arguments = fn.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return tc
    elif isinstance(arguments, dict):
        parsed = dict(arguments)
    else:
        return tc

    if name in {"bash", "terminal"} and "description" not in parsed:
        cmd = parsed.get("command", "")
        parsed["description"] = f"Execute command: {str(cmd)[:100]}"

    fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
    tc["function"] = fn
    return tc


def _enrich_client_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    return [_enrich_client_tool_call(dict(tc)) for tc in tool_calls if isinstance(tc, dict)]


def _fallback_provider_for_model(model_id: str) -> tuple[str, str]:
    raw = str(model_id or "").strip()
    if not raw:
        return "", ""
    if "/" not in raw:
        return "openrouter", raw

    prefix, rest = raw.split("/", 1)
    prefix = prefix.strip().lower()
    rest = rest.strip()
    if prefix == "openai":
        return "openai-codex", raw
    if prefix in {"github-copilot", "opencode-go", "opencode-zen", "zai", "minimax"}:
        # CRITICAL FIX: Return bare model name (rest) without provider prefix for OpenCode providers
        # and direct API providers (zai, minimax). These APIs only accept bare model names
        # (e.g., "glm-4.7", not "zai/glm-4.7")
        return prefix, rest
    if prefix == "openrouter":
        return "openrouter", rest
    return "openrouter", raw


def _build_env_fallback_chain(prefix: str) -> List[Dict[str, Any]]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    chain: List[Dict[str, Any]] = []
    for idx in range(1, 33):
        raw_model = os.getenv(f"{prefix}_{idx}", "").strip()
        if not raw_model:
            continue
        provider, resolved_model = _fallback_provider_for_model(raw_model)
        if not provider or not resolved_model:
            continue
        try:
            runtime = resolve_runtime_provider(requested=provider)
        except Exception:
            runtime = {}
        # Use provider from runtime if available, otherwise fall back to requested_provider or raw provider.
        # runtime["provider"] is the resolved canonical name (e.g., "copilot").
        # runtime["requested_provider"] may be an alias (e.g., "github-copilot") which is not a valid provider key.
        resolved_provider = runtime.get("provider") or ""
        normalized_provider = str(resolved_provider or runtime.get("requested_provider") or provider).strip()
        chain.append({
            "provider": normalized_provider,
            "model": resolved_model,
            "base_url": str(runtime.get("base_url") or "").strip(),
            "api_key": str(runtime.get("api_key") or "").strip(),
        })
    return chain


def _build_swarm_model_pool(*, estimated_tokens: int = 0, routing_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build hermes-swarm routing config from environment variables."""
    primary = os.getenv("HERMES_SWARM_PRIMARY_MODEL", "google/gemma-4-26b-a4b-it:free").strip()
    fallback_chain = [primary] if primary else []
    for idx in range(1, 33):
        fb = os.getenv(f"HERMES_SWARM_FALLBACK_{idx}", "").strip()
        if fb:
            fallback_chain.append(fb)

    seen = set()
    deduped: List[str] = []
    for model in fallback_chain:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    fallback_chain = deduped

    large_context_fallbacks: List[str] = []
    for idx in range(1, 17):
        fb = os.getenv(f"HERMES_SWARM_LARGE_CONTEXT_FALLBACK_{idx}", "").strip()
        if fb and fb not in large_context_fallbacks:
            large_context_fallbacks.append(fb)

    if not large_context_fallbacks:
        from agent.model_metadata import get_model_context_length_quick
        primary_ctx = get_model_context_length_quick(primary) if primary else 0
        large_context_fallbacks = [
            model for model in fallback_chain
            if get_model_context_length_quick(model) > primary_ctx
        ]

    scout_fallbacks: List[str] = []
    for idx in range(1, 17):
        fb = os.getenv(f"HERMES_SWARM_SCOUT_FALLBACK_{idx}", "").strip()
        if fb and fb not in scout_fallbacks:
            scout_fallbacks.append(fb)

    if not scout_fallbacks:
        scout_fallbacks = [
            "github-copilot/gpt-5-mini",
            "github-copilot/gpt-5.4",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "minimax/MiniMax-M2.7",
            primary,
        ]

    scout_fallbacks = [m for m in scout_fallbacks if m]

    return {
        "primary": primary,
        "fallbacks": fallback_chain,
        "selection_policy": os.getenv("HERMES_SWARM_SELECTION_POLICY", "cost-balanced"),
        "large_context_fallbacks": large_context_fallbacks,
        "scout_fallbacks": scout_fallbacks,
        "estimated_tokens": estimated_tokens,
        "routing_hint": dict(routing_hint or {}),
    }


def _summarize_swarm_messages(
    *,
    system_prompt: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
) -> str:
    history = conversation_history or []
    parts: List[str] = []
    if system_prompt:
        parts.append(f"SYSTEM:\n{system_prompt}")
    if history:
        trimmed = history[-8:]
        rendered = []
        for msg in trimmed:
            role = str(msg.get("role") or "user").lower().strip()
            if role != "user":
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            rendered.append(f"USER_HISTORY: {content[:1200]}")
        if rendered:
            parts.append("HISTORY:\n" + "\n".join(rendered))
    if user_message:
        parts.append(f"USER:\n{user_message[:2000]}")
    return "\n\n".join(parts)


def _heuristic_swarm_routing_hint(
    *,
    system_prompt: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    estimated_tokens: int = 0,
) -> Dict[str, Any]:
    text = _summarize_swarm_messages(
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        user_message=user_message,
    ).lower()
    task_type = "general"
    recommended_tier = "primary"
    needs_instruction_following = False
    needs_repo_reasoning = False
    needs_bug_judgement = False

    inline_instruction_context = any(
        k in text for k in (
            "repo instructions say:",
            "agents excerpt:",
            "agants excerpt:",
            "use only the provided context",
            "use only the context below",
        )
    )

    if any(k in text for k in ("agents.md", "workspace", "repo", "repository", "codebase", "readme", "package", "custom_components/")):
        needs_instruction_following = True
        needs_repo_reasoning = True
    if any(k in text for k in ("review", "analy", "find bug", "likely bug", "correctness", "why is", "root cause", "debug", "fix this", "regression")):
        needs_bug_judgement = True
    if any(k in text for k in ("implement", "patch", "modify", "change code", "refactor")):
        task_type = "implementation"
        recommended_tier = "premium"
    elif any(k in text for k in ("review", "code review", "likely bug", "correctness")):
        task_type = "repo_review"
        recommended_tier = "premium"
    elif any(k in text for k in ("debug", "root cause", "why is", "broken")):
        task_type = "debugging"
        recommended_tier = "premium"
    elif any(k in text for k in ("architecture", "design", "best approach", "tradeoff")):
        task_type = "architecture"
        recommended_tier = "premium"

    if tools:
        needs_repo_reasoning = True
        if recommended_tier == "primary":
            recommended_tier = "balanced"

    if estimated_tokens > 6000 and recommended_tier == "primary":
        recommended_tier = "balanced"

    if needs_instruction_following and needs_repo_reasoning and recommended_tier == "primary":
        recommended_tier = "balanced"
    if inline_instruction_context and needs_instruction_following:
        recommended_tier = "premium"
    if needs_bug_judgement and recommended_tier != "premium":
        recommended_tier = "premium"

    return {
        "task_type": task_type,
        "recommended_tier": recommended_tier,
        "needs_instruction_following": needs_instruction_following,
        "needs_repo_reasoning": needs_repo_reasoning,
        "needs_bug_judgement": needs_bug_judgement,
        "provided_context_only": any(
            k in text for k in (
                "use only the context below",
                "use only the provided context",
                "provided snippets only",
                "do not assume file access",
            )
        ),
        "source": "heuristic",
        "confidence": 0.55,
    }


def _swarm_execution_system_prompt(routing_hint: Optional[Dict[str, Any]]) -> str:
    hint = routing_hint or {}
    parts = [
        "For hermes-swarm tasks: prioritize correctness over style, avoid hallucinating filesystem/tool access, and explicitly distinguish provided context from inferred assumptions.",
        "If AGENTS.md is not available on disk, treat that as missing optional repo guidance, not as a hard failure. Continue with standard assumptions unless the user explicitly required the physical file to be read.",
    ]
    if hint.get("needs_instruction_following"):
        parts.append(
            "If the user supplies AGENTS.md or workflow instructions in the prompt, treat those quoted instructions as authoritative even if you cannot access the repo directly."
        )
        parts.append(
            "When AGENTS.md instructions are supplied inline, never answer that you cannot proceed just because the real file was not found. Use the supplied instructions."
        )
    if hint.get("provided_context_only"):
        parts.append(
            "Use only the context provided in the request. Do not claim files are missing, unreadable, or present unless the prompt itself states that."
        )
        parts.append(
            "Do not say that AGENTS.md, helper scripts, or repo files are missing when their relevant contents are already quoted in the prompt."
        )
    if hint.get("needs_bug_judgement"):
        parts.append(
            "Rank real correctness issues above style or micro-optimizations, and prefer saying 'only 2 high-confidence issues' over inventing a weak third issue."
        )
    return "\n".join(parts)


def _extract_agent_result_text(result: Any) -> str:
    """Best-effort extraction of assistant text from agent results."""
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, dict):
        return ""
    final_response = str(result.get("final_response") or "").strip()
    if final_response:
        return final_response
    for msg in reversed(list(result.get("messages", []))):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    error_text = str(result.get("error") or "").strip()
    return error_text


def _parse_loose_json_object(response_text: str) -> Dict[str, Any]:
    """Parse a JSON-ish object from model output."""
    text = str(response_text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    parsed = ast.literal_eval(candidate)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"No JSON object found: {text[:200]}")


def _client_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(func, dict):
            name = str(func.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _agents_prefetch_tool_names(tool_names: set[str]) -> Optional[tuple[str, str]]:
    """Return the search/read tool pair available for AGENTS prefetch.

    Supports both Hermes-native API tool names (search_files/read_file) and the
    OpenCode client tool names exposed through hermes-swarm (glob/read).
    """
    if {"search_files", "read_file"}.issubset(tool_names):
        return ("search_files", "read_file")
    if {"glob", "read"}.issubset(tool_names):
        return ("glob", "read")
    return None


def _extract_tool_result_by_call_id(messages: List[Dict[str, Any]], call_id: str) -> Optional[str]:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and str(msg.get("tool_call_id") or "") == call_id:
            return str(msg.get("content") or "")
    return None


def _extract_first_path_from_search_result(raw: str) -> Optional[str]:
    try:
        data = _parse_loose_json_object(raw)
    except Exception:
        return None

    def _walk(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            for key in ("path", "file", "filepath"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            paths = value.get("paths")
            if isinstance(paths, list):
                for candidate in paths:
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
            for nested in value.values():
                found = _walk(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = _walk(item)
                if found:
                    return found
        return None

    return _walk(data)


def _needs_agents_prefetch(user_message: str, system_prompt: Optional[str], tools: Any, messages: Optional[List[Dict[str, Any]]] = None) -> bool:
    tool_names = _client_tool_names(tools)
    if _agents_prefetch_tool_names(tool_names) is None:
        return False
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls", []) or []:
            if str(tc.get("id") or "").startswith("agents_prefetch_"):
                return True
        if str(msg.get("tool_call_id") or "").startswith("agents_prefetch_"):
            return True
    text = f"{system_prompt or ''}\n{user_message or ''}".lower()
    return any(marker in text for marker in (
        "agents.md",
        "read agents",
        "repo instructions",
        "workspace",
        "codebase",
    ))


def _build_agents_prefetch_tool_call(name: str, call_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _determine_agents_prefetch_action(messages: List[Dict[str, Any]], tools: Any = None) -> Dict[str, Any]:
    search_call_id = "agents_prefetch_search"
    read_call_id = "agents_prefetch_read"
    search_result = _extract_tool_result_by_call_id(messages, search_call_id)
    read_result = _extract_tool_result_by_call_id(messages, read_call_id)
    tool_names = _client_tool_names(tools)
    tool_pair = _agents_prefetch_tool_names(tool_names) or ("search_files", "read_file")
    search_tool, read_tool = tool_pair
    if read_result:
        return {"status": "done", "agents_text": read_result}
    if search_result:
        path = _extract_first_path_from_search_result(search_result)
        if path:
            read_args: Dict[str, Any] = {"offset": 1, "limit": 260}
            if read_tool == "read_file":
                read_args["path"] = path
            else:
                read_args["filePath"] = path
            return {
                "status": "need_read",
                "tool_call": _build_agents_prefetch_tool_call(
                    read_tool,
                    read_call_id,
                    read_args,
                ),
            }
        return {"status": "done", "agents_text": "No AGENTS.md found in client workspace; proceed with standard assumptions."}
    search_args: Dict[str, Any]
    if search_tool == "search_files":
        search_args = {"pattern": "AGENTS.md", "target": "files", "path": ".", "limit": 5}
    else:
        search_args = {"pattern": "**/AGENTS.md", "path": "."}
    return {
        "status": "need_search",
        "tool_call": _build_agents_prefetch_tool_call(
            search_tool,
            search_call_id,
            search_args,
        ),
    }


def _swarm_model_has_credentials(model: str) -> bool:
    raw = str(model or "").strip()
    if not raw:
        return False
    if "/" not in raw:
        return True
    prefix = raw.split("/", 1)[0].strip().lower()
    if prefix == "github-copilot":
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token
            token, _source = resolve_copilot_token()
            return bool(token)
        except Exception:
            return bool(os.getenv("GITHUB_COPILOT_API_KEY", "").strip())
    if prefix == "opencode-go":
        return bool(os.getenv("OPENCODE_GO_API_KEY", "").strip())
    if prefix == "opencode-zen":
        return bool(os.getenv("OPENCODE_ZEN_API_KEY", "").strip())
    if prefix == "zai":
        return bool(os.getenv("ZAI_API_KEY", "").strip())
    if prefix == "minimax":
        return bool(os.getenv("MINIMAX_API_KEY", "").strip())
    if prefix == "synthetic":
        return bool(os.getenv("SYNTHETIC_API_KEY", "").strip())
    if prefix == "google":
        # Direct Google API key takes priority; otherwise we fall back to OpenRouter
        if (
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
        ):
            return True
        # No direct key — will route via OpenRouter, so check for that key
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if prefix == "nvidia":
        if os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVCLOUD_API_KEY", "").strip():
            return True
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if prefix == "local":
        return False
    if prefix == "openai":
        # Check env vars first, then fall back to Hermes auth store for Codex OAuth tokens
        if (
            os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("OPENAI_CODEX_API_KEY", "").strip()
            or os.getenv("OPENAI_CODEX_TOKEN", "").strip()
            or os.getenv("CODEX_ACCESS_TOKEN", "").strip()
            or os.getenv("OPENAI_OAUTH_TOKEN", "").strip()
        ):
            return True
        # Check auth store for OpenAI Codex OAuth tokens
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            resolved = resolve_runtime_provider(requested="openai-codex")
            return bool(resolved.get("api_key"))
        except Exception:
            pass
        return False
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _runtime_kwargs_for_model_id(model: str) -> tuple[Dict[str, Any], str]:
    runtime_kwargs: Dict[str, Any] = {}
    provider_prefix = ""
    normalized_model = str(model or "").strip()

    if "/" in normalized_model:
        provider_prefix = normalized_model.split("/", 1)[0].strip().lower()
        if provider_prefix == "opencode-zen":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_ZEN_API_KEY", "")
            runtime_kwargs["provider"] = "opencode-zen"
        elif provider_prefix == "opencode-go":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_GO_API_KEY", "")
            runtime_kwargs["provider"] = "opencode-go"
        elif provider_prefix == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
            openai_base = os.getenv("OPENAI_BASE_URL", "").strip()
            codex_key = (
                os.getenv("OPENAI_CODEX_API_KEY", "").strip()
                or os.getenv("OPENAI_OAUTH_TOKEN", "").strip()
                or os.getenv("OPENAI_CODEX_TOKEN", "").strip()
                or os.getenv("CODEX_ACCESS_TOKEN", "").strip()
            )
            if openai_api_key:
                runtime_kwargs["base_url"] = openai_base or "https://api.openai.com/v1"
                runtime_kwargs["provider"] = "openai"
            elif openai_base:
                runtime_kwargs["base_url"] = openai_base
                runtime_kwargs["provider"] = "openai"
            else:
                # Try auth store for Codex OAuth tokens before defaulting
                _codex_resolved = False
                try:
                    from hermes_cli.runtime_provider import resolve_runtime_provider
                    resolved = resolve_runtime_provider(requested="openai-codex")
                    _codex_api_key = resolved.get("api_key", "")
                    _codex_base_url = resolved.get("base_url", "")
                    if _codex_api_key:
                        runtime_kwargs["base_url"] = _codex_base_url or os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                        runtime_kwargs["api_key"] = _codex_api_key
                        runtime_kwargs["provider"] = "openai-codex"
                        runtime_kwargs["api_mode"] = "codex_responses"
                        _codex_resolved = True
                except Exception:
                    pass
                if not _codex_resolved:
                    runtime_kwargs["base_url"] = os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                    runtime_kwargs["provider"] = "openai-codex"
            if not runtime_kwargs.get("api_key"):
                runtime_kwargs["api_key"] = openai_api_key or codex_key
        elif provider_prefix == "github-copilot":
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider

                resolved = resolve_runtime_provider(requested="copilot")
                runtime_kwargs["base_url"] = resolved.get("base_url") or os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com")
                runtime_kwargs["api_key"] = resolved.get("api_key") or ""
                runtime_kwargs["provider"] = resolved.get("provider") or "copilot"
                runtime_kwargs["api_mode"] = resolved.get("api_mode")
                runtime_kwargs["credential_pool"] = resolved.get("credential_pool")
            except Exception as exc:
                logging.warning(f"[API_SERVER] Swarm: failed to resolve Copilot runtime provider: {exc}")
                _copilot_token = os.getenv("GITHUB_COPILOT_API_KEY", "")
                if not _copilot_token:
                    try:
                        from hermes_cli.copilot_auth import resolve_copilot_token
                        _copilot_token, _copilot_source = resolve_copilot_token()
                        if _copilot_token:
                            logging.warning(f"[API_SERVER] Swarm: resolved Copilot token from {_copilot_source}")
                    except Exception as inner_exc:
                        logging.warning(f"[API_SERVER] Swarm: failed to resolve Copilot token: {inner_exc}")
                runtime_kwargs["base_url"] = os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com")
                runtime_kwargs["api_key"] = _copilot_token
                runtime_kwargs["provider"] = "copilot"
        elif provider_prefix == "minimax":
            runtime_kwargs["base_url"] = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
            runtime_kwargs["api_key"] = os.getenv("MINIMAX_API_KEY", "")
            runtime_kwargs["provider"] = "minimax"
        elif provider_prefix == "synthetic":
            runtime_kwargs["base_url"] = os.getenv("SYNTHETIC_BASE_URL", "https://api.synthetic.new/openai/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("SYNTHETIC_API_KEY", "")
            runtime_kwargs["provider"] = "synthetic"
        elif provider_prefix == "google":
            _google_key = (
                os.getenv("GOOGLE_API_KEY", "").strip()
                or os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
            )
            if _google_key:
                runtime_kwargs["base_url"] = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
                runtime_kwargs["api_key"] = _google_key
                runtime_kwargs["provider"] = "google"
            else:
                # No Google API key — enforce strict guard when forcing free OpenRouter
                if os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1", "true", "yes"):
                    runtime_kwargs["base_url"] = ""
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "blocked"
                else:
                    # Route to OpenRouter conservatively (requires :free if guard active)
                    runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "openrouter"
        elif provider_prefix == "nvidia":
            _nvidia_key = (
                os.getenv("NVIDIA_API_KEY", "").strip()
                or os.getenv("NVCLOUD_API_KEY", "").strip()
            )
            if _nvidia_key:
                runtime_kwargs["base_url"] = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
                runtime_kwargs["api_key"] = _nvidia_key
                runtime_kwargs["provider"] = "nvidia"
            else:
                # No NVIDIA API key — enforce openrouter free gate if configured
                if os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1","true","yes"):
                    runtime_kwargs["base_url"] = ""
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "blocked"
                else:
                    runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "openrouter"
        elif provider_prefix == "zai":
            runtime_kwargs["base_url"] = "https://api.z.ai/api/coding/paas/v4"
            runtime_kwargs["api_key"] = os.getenv("ZAI_API_KEY", "")
            runtime_kwargs["provider"] = "zai"
        elif provider_prefix == "qwen":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_ZEN_API_KEY", "")
            runtime_kwargs["provider"] = "alibaba"
        elif provider_prefix in ("ollama-cloud", "ollama"):
            runtime_kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
            runtime_kwargs["api_key"] = os.getenv("OLLAMA_API_KEY", "")
            runtime_kwargs["provider"] = "ollama-cloud"
        elif provider_prefix not in ("openrouter",):
            # Unknown non-openrouter prefix — still route via OpenRouter as last resort
            runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
            runtime_kwargs["provider"] = "openrouter"
        # NOTE: provider_prefix == "openrouter" is handled below
        else:
            runtime_kwargs["provider"] = provider_prefix
    else:
        from hermes_cli.model_normalize import detect_vendor
        detected_provider = detect_vendor(normalized_model)
        if detected_provider:
            runtime_kwargs["provider"] = detected_provider

    if "/" in normalized_model:
        normalized_model = normalized_model.split("/", 1)[1].strip()
    return runtime_kwargs, normalized_model


def _swarm_model_is_available(model: str) -> bool:
    """Check if a model is available (has credentials AND is not in cooldown/sin-bin).

    This function extends _swarm_model_has_credentials to also check the cooldown DB,
    preventing the swarm from repeatedly selecting rate-limited providers.
    """
    raw = str(model or "").strip()
    if not raw:
        return False

    # Block google/nvidia without keys early
    if raw.startswith("google/") or raw.startswith("nvidia/"):
        if not (os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip() or os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVCLOUD_API_KEY", "").strip()):
            logger.info("[api_server] blocking google/nvidia model %s due to missing provider keys", raw)
            return False

    if not _swarm_model_has_credentials(raw):
        return False

    # ── OpenRouter cost guard ──────────────────────────────────────────────
    # When HERMES_SWARM_FORCE_FREE_OPENROUTER is enabled, reject any model
    # routed through OpenRouter that is not explicitly marked with the ":free"
    # suffix.  This prevents accidental spending on paid OpenRouter models
    # (e.g. google/gemini-2.5-flash costs $0.15/M completion tokens).
    runtime_kwargs, normalized_model_result = _runtime_kwargs_for_model_id(raw)
    provider = str(runtime_kwargs.get("provider") or "").strip().lower()
    if provider == "openrouter" and os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1", "true", "yes"):
        if ":free" not in raw.lower():
            logger.info(
                "[api_server] model %s blocked — FORCE_FREE_OPENROUTER is enabled and model is not :free",
                raw,
            )
            return False

    runtime_kwargs_out, model_name = runtime_kwargs, normalized_model_result
    provider = str(runtime_kwargs_out.get("provider") or "").strip().lower()
    base_url = str(runtime_kwargs_out.get("base_url") or "").strip()
    if not provider or not model_name:
        return True

    try:
        from agent.model_cooldown_db import model_cooldown_remaining
        remaining = model_cooldown_remaining(provider, model_name, base_url=base_url)
        if remaining and remaining > 0:
            logger.info(
                "[api_server] model %s in cooldown (%.0fs remaining) — skipping",
                raw, remaining,
            )
            return False
    except Exception:
        pass

    if provider == "zai" and _zai_is_peak_hours():
        logger.info(
            "[api_server] model %s skipped during peak hours (14:00-18:00 UTC+8) — quota costs 3x",
            raw,
        )
        return False

    return True


def _zai_is_peak_hours() -> bool:
    """Check if current time is within ZAI peak hours (14:00-18:00 UTC+8).

    During peak hours, GLM-5.1 and GLM-5-Turbo consume quota at 3x normal rate.
    Off-peak usage is currently 1x through end of June (limited-time benefit).
    """
    import time
    utc_plus_8_offset = 8 * 3600
    utc_plus_8_seconds = time.time() + utc_plus_8_offset
    utc_plus_8_hour = (utc_plus_8_seconds % 86400) // 3600
    return 14 <= utc_plus_8_hour < 18


def _is_opencode_user_agent(user_agent: str) -> bool:
    return isinstance(user_agent, str) and "opencode/" in user_agent.lower()


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string.

    Some clients (Open WebUI, LobeChat, etc.) send content as an array of
    typed parts instead of a plain string::

        [{"type": "text", "text": "hello"}, {"type": "input_text", "text": "..."}]

    This function flattens those into a single string so the agent pipeline
    (which expects strings) doesn't choke.

    Defensive limits prevent abuse: recursion depth, list size, and output
    length are all bounded.
    """
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    parts.append(item[:MAX_NORMALIZED_TEXT_LENGTH])
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            parts.append(str(text)[:MAX_NORMALIZED_TEXT_LENGTH])
                        except Exception:
                            pass
                # Silently skip image_url / other non-text parts
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
            # Check accumulated size
            if sum(len(p) for p in parts) >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    # Fallback for unexpected types (int, float, bool, etc.)
    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                db_path = ":memory:"
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        if row is None:
            return None
        import time
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        return json.loads(row[0])

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        import time
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            self._conn.execute(
                "DELETE FROM responses WHERE response_id IN "
                "(SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?)",
                (count - self._max_size,),
            )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", (response_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            (name, response_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)

        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)

        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return web.json_response(_openai_error("Request body too large.", code="body_too_large"), status=413)
                except ValueError:
                    return web.json_response(_openai_error("Invalid Content-Length header.", code="invalid_content_length"), status=400)
        return await handler(request)
else:
    body_limit_middleware = None  # type: ignore[assignment]

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    security_headers_middleware = None  # type: ignore[assignment]


class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        import time as _t
        now = _t.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_or_set(self, key: str, fingerprint: str, compute_coro):
        self._purge()
        item = self._store.get(key)
        if item and item["fp"] == fingerprint:
            return item["resp"]
        resp = await compute_coro()
        import time as _t
        self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
        self._purge()
        return resp


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    return sha256(repr(subset).encode("utf-8")).hexdigest()


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
) -> str:
    """Derive a stable session ID from the conversation's first user message.

    OpenAI-compatible frontends (Open WebUI, LibreChat, etc.) send the full
    conversation history with every request.  The system prompt and first user
    message are constant across all turns of the same conversation, so hashing
    them produces a deterministic session ID that lets the API server reuse
    the same Hermes session (and therefore the same Docker container sandbox
    directory) across turns.
    """
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in ("default", "custom"):
                return profile
        except Exception:
            pass
        return "hermes-agent"

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed (only when API
        server is local).
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_gen_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        provider_mode: bool = False,
        swarm_mode: bool = False,
        swarm_model_pool: Optional[Dict[str, Any]] = None,
        toolset_mode: str = "auto",
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        external_tool_mode: str = "none",
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config
        from hermes_cli.tools_config import _get_platform_tools

        logging.warning(f"[API_SERVER] _create_agent called: swarm_mode={swarm_mode}, swarm_model_pool={swarm_model_pool}")

        # Swarm mode: select from free/cheap model pool FIRST
        # This must happen before _resolve_runtime_agent_kwargs() to avoid provider resolution errors
        if swarm_mode and swarm_model_pool:
            model = _resolve_swarm_model(
                swarm_model_pool,
                estimated_tokens=swarm_model_pool.get("estimated_tokens", 0),
            )
            logging.warning(f"[API_SERVER] Swarm mode: resolved model={model}")
            runtime_kwargs, model = self._runtime_kwargs_for_model(model)
            # Don't add model to runtime_kwargs since it's passed separately to AIAgent()
            logging.warning(f"[API_SERVER] Swarm: runtime_kwargs={runtime_kwargs}")

        # Only resolve credentials if NOT in swarm mode (swarm mode already has runtime_kwargs)
        # This prevents unnecessary Codex credential checks when using other providers
        if not swarm_mode:
            # Check if a specific provider is configured - if so, use it directly
            # without going through the default credential resolution (which checks Codex)
            requested_provider = os.getenv("HERMES_INFERENCE_PROVIDER")
            if not requested_provider:
                cfg = _load_gateway_config()
                model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
                if isinstance(model_cfg, dict):
                    requested_provider = str(model_cfg.get("provider") or "").strip() or None
            
            # When provider_mode=True (hermes-code), check if the HERMES_CODE_MODEL is set
            # and if its prefix matches the requested provider. If not, resolve provider
            # from the model prefix instead of using HERMES_INFERENCE_PROVIDER.
            code_model = os.getenv("HERMES_CODE_MODEL", "").strip()
            code_model_prefix = code_model.split("/")[0].lower() if code_model and "/" in code_model else ""
            
            if requested_provider and requested_provider.lower() in ("zai", "opencode-go", "opencode-zen"):
                # If provider_mode and model prefix doesn't match provider, resolve provider
                # from the model prefix instead of using HERMES_INFERENCE_PROVIDER.
                # e.g., HERMES_CODE_MODEL="openai/gpt-5.4" with HERMES_INFERENCE_PROVIDER="zai"
                # should route to openai-codex, not zai
                if provider_mode and code_model_prefix and code_model_prefix != requested_provider.lower():
                    # Resolve provider from model prefix, not from HERMES_INFERENCE_PROVIDER
                    # Map the prefix through _EXPLICIT_MODEL_PROVIDER_ALIASES (e.g., "openai" -> "openai-codex")
                    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
                    try:
                        # Map model prefix to canonical provider name
                        model_provider = _EXPLICIT_MODEL_PROVIDER_ALIASES.get(code_model_prefix, code_model_prefix)
                        runtime = resolve_runtime_provider(requested=model_provider)
                        runtime_kwargs = {
                            "api_key": runtime.get("api_key"),
                            "base_url": runtime.get("base_url"),
                            "provider": runtime.get("provider"),
                            "api_mode": runtime.get("api_mode"),
                            "command": runtime.get("command"),
                            "args": list(runtime.get("args") or []),
                            "credential_pool": runtime.get("credential_pool"),
                        }
                    except Exception as exc:
                        raise RuntimeError(format_runtime_provider_error(exc)) from exc
                elif requested_provider.lower() == "zai":
                    runtime_kwargs = {
                        "base_url": "https://api.z.ai/api/coding/paas/v4",
                        "api_key": os.getenv("ZAI_API_KEY", ""),
                        "provider": "zai",
                    }
                else:
                    runtime_kwargs = {
                        "base_url": os.getenv(f"{requested_provider.upper().replace('-', '_')}_BASE_URL", f"https://opencode.ai/zen/go/v1" if requested_provider.lower() == "opencode-go" else "https://opencode.ai/zen/v1"),
                        "api_key": os.getenv(f"{requested_provider.upper().replace('-', '_')}_API_KEY", ""),
                        "provider": requested_provider.lower(),
                    }
            else:
                runtime_kwargs = _resolve_runtime_agent_kwargs()
        
        # Non-swarm path continues here - model resolution (swarm already has model at this point)
        if not swarm_mode:
            if provider_mode:
                # OpenCode routes hermes-code through provider mode. Keep the
                # requested hermes-code model stable even when there is no gateway
                # config model.default configured.
                model = os.getenv("HERMES_CODE_MODEL", "").strip() or _resolve_gateway_model()
            else:
                model = _resolve_gateway_model()

            runtime_kwargs = _align_runtime_with_explicit_model(runtime_kwargs, model)
            # If the model had an explicit provider prefix (eg. "anthropic/..." or "ollama/...")
            # but the requested provider couldn't be resolved (missing creds), try to find a
            # viable provider for the bare model name. This prevents mismatched provider+model
            # calls (e.g., calling openai-codex with an anthropic/ model) which produce
            # non-retryable 400 errors. We only do this when the explicit provider was requested.
            try:
                from hermes_cli.models import detect_provider_for_model
                from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
                explicit_provider = _explicit_provider_from_model(model)
                if explicit_provider:
                    current_provider = str(runtime_kwargs.get("provider") or "").strip().lower()
                    # If runtime doesn't match explicit provider or lacks api_key, attempt resolution
                    if current_provider != explicit_provider or not runtime_kwargs.get("api_key"):
                        try:
                            # Try to resolve the explicit provider (may raise if creds missing)
                            resolved = resolve_runtime_provider(requested=explicit_provider)
                            runtime_kwargs = {
                                "api_key": resolved.get("api_key"),
                                "base_url": resolved.get("base_url"),
                                "provider": resolved.get("provider"),
                                "api_mode": resolved.get("api_mode"),
                                "command": resolved.get("command"),
                                "args": list(resolved.get("args") or []),
                                "credential_pool": resolved.get("credential_pool"),
                            }
                        except Exception:
                            # Couldn't resolve explicit provider (likely missing credentials).
                            # Try to detect an alternative provider for the bare model name.
                            try:
                                bare = model.split("/", 1)[1].strip() if "/" in model else model
                                detected = detect_provider_for_model(bare, current_provider)
                                if detected:
                                    alt_provider = detected[0]
                                    try:
                                        alt_runtime = resolve_runtime_provider(requested=alt_provider)
                                        runtime_kwargs = {
                                            "api_key": alt_runtime.get("api_key"),
                                            "base_url": alt_runtime.get("base_url"),
                                            "provider": alt_runtime.get("provider"),
                                            "api_mode": alt_runtime.get("api_mode"),
                                            "command": alt_runtime.get("command"),
                                            "args": list(alt_runtime.get("args") or []),
                                            "credential_pool": alt_runtime.get("credential_pool"),
                                        }
                                        logging.warning(
                                            "[API_SERVER] explicit provider %s unavailable; routed model %s to provider %s",
                                            explicit_provider,
                                            model,
                                            alt_provider,
                                        )
                                    except Exception:
                                        logging.warning(
                                            "[API_SERVER] explicit provider %s unavailable and alternative provider resolution failed; proceeding with default runtime kwargs",
                                            explicit_provider,
                                        )
                                else:
                                    logging.warning(
                                        "[API_SERVER] explicit provider %s unavailable and no alternative provider detected for model %s",
                                        explicit_provider,
                                        model,
                                    )
                            except Exception:
                                logging.exception("[API_SERVER] error while attempting alternative provider detection for model %s", model)
            except Exception:
                # If the import or detection fails, just continue with existing runtime_kwargs
                logging.debug("[API_SERVER] provider alignment/detection helper failed; continuing")

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))
        skip_memory = False
        skip_context_files = False
        if swarm_mode and swarm_model_pool:
            swarm_prompt = _swarm_execution_system_prompt(swarm_model_pool.get("routing_hint"))
            if swarm_prompt:
                ephemeral_system_prompt = (
                    f"{ephemeral_system_prompt}\n\n{swarm_prompt}" if ephemeral_system_prompt else swarm_prompt
                )
        if provider_mode:
            enabled_toolsets = []
            skip_memory = True
            skip_context_files = True
            if ephemeral_system_prompt is None:
                ephemeral_system_prompt = "You are a helpful AI assistant."
        elif toolset_mode == "local":
            enabled_toolsets = sorted(
                [t for t in enabled_toolsets if t in ("terminal", "hermes-cli")]
            )
        elif toolset_mode == "remote":
            enabled_toolsets = sorted(
                [t for t in enabled_toolsets if t in ("web", "skills")]
            )
        # "full", "auto" or anything else uses all configured toolsets

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        from gateway.run import GatewayRunner
        fallback_model = GatewayRunner._load_fallback_model()
        if not fallback_model:
            if provider_mode:
                fallback_model = _build_env_fallback_chain("HERMES_AGENT_FALLBACK")
            elif swarm_mode:
                fallback_model = _build_env_fallback_chain("HERMES_SWARM_FALLBACK")

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_gen_callback=tool_gen_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            skip_memory=skip_memory,
            skip_context_files=skip_context_files,
            tools=tools,
            tool_choice=tool_choice,
        )
        try:
            agent._provider_mode = provider_mode
            agent._tools_from_request = bool(tools)
            agent._toolset_mode = toolset_mode
            agent._external_tool_mode = external_tool_mode
        except Exception:
            pass
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  No authentication required.
        """
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
        return web.json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": runtime.get("gateway_state"),
            "platforms": runtime.get("platforms", {}),
            "active_agents": runtime.get("active_agents", 0),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_stats(self, request: "web.Request") -> "web.Response":
        """GET /stats — return SmartRouter and Deduplicator statistics.
        
        Returns aggregated stats from:
        - SmartRouter: routing decisions, cost savings
        - Deduplicator: cache hits, dedup rate
        - Combined: estimated total savings
        """
        try:
            from agent.deduplicator import get_global_deduplicator
            from agent.smart_router import get_global_router
            
            dedup = get_global_deduplicator()
            router = get_global_router()
            
            dedup_stats = dedup.get_stats().to_dict() if hasattr(dedup, 'get_stats') else {}
            router_stats = router.get_stats().to_dict() if hasattr(router, 'get_stats') else {}
            
            # Calculate combined savings
            dedup_savings = dedup_stats.get('cache_hits', 0) * 0.5  # Rough estimate
            routing_savings = router_stats.get('cost_savings_cents', 0)
            total_savings = dedup_savings + routing_savings
            
            return web.json_response({
                "status": "ok",
                "deduplicator": dedup_stats,
                "smart_router": router_stats,
                "combined": {
                    "estimated_cost_savings_cents": round(total_savings, 2),
                    "dedup_rate_pct": dedup_stats.get('dedup_rate_pct', 0),
                    "cache_hit_rate_pct": dedup_stats.get('cache_hit_rate_pct', 0),
                    "routing_decisions": router_stats.get('total_requests', 0),
                    "simple_routed_to_cheap": router_stats.get('simple_routed_to_cheap', 0),
                }
            })
        except Exception as e:
            logger.warning("Failed to get stats: %s", e)
            return web.json_response({
                "status": "ok",
                "deduplicator": {},
                "smart_router": {},
                "combined": {},
                "error": str(e)
            })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        now = int(time.time())
        data = [
            {
                "id": self._model_name,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": self._model_name,
                "parent": None,
            },
            {
                "id": "hermes-code",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-code",
                "parent": None,
                "description": "Routes all tools to client (OpenCode local). Use with tools from OpenCode.",
            },
            {
                "id": "hermes-agentic-remote",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-agentic-remote",
                "parent": None,
            },
            {
                "id": "hermes-agentic-full",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-agentic-full",
                "parent": None,
            },
            {
                "id": "hermes-swarm",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-swarm",
                "parent": None,
            },
            *[
                {
                    "id": alias,
                    "object": "model",
                    "created": now,
                    "owned_by": "hermes",
                    "permission": [],
                    "root": alias,
                    "parent": "hermes-swarm",
                    "description": "Role alias managed by Hermes Gateway. Routed dynamically through hermes-swarm.",
                }
                for alias in ROLE_ALIAS_CONFIG
            ],
        ]
        return web.json_response({
            "object": "list",
            "data": data,
        })

    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = body.get("stream", False)

        # Extract tools from request (passed by OpenCode client)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        user_agent = request.headers.get("User-Agent", "")
        force_connection_close = _is_opencode_user_agent(user_agent)
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool["_from_client"] = True

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = _normalize_chat_content(msg.get("content", ""))
            if role == "system":
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role == "assistant":
                assistant_entry: Dict[str, Any] = {"role": "assistant", "content": content}
                tool_calls = _extract_openai_tool_calls(msg.get("tool_calls"))
                if tool_calls:
                    assistant_entry["tool_calls"] = tool_calls
                # Preserve reasoning_content for providers that use it
                # (Moonshot/Kimi, GLM, Novita, OpenRouter) so multi-turn
                # conversations maintain reasoning context.
                reasoning_content = msg.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    assistant_entry["reasoning_content"] = reasoning_content
                conversation_messages.append(assistant_entry)
            elif role == "tool":
                tool_entry: Dict[str, Any] = {"role": "tool", "content": content}
                tool_call_id = msg.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id.strip():
                    tool_entry["tool_call_id"] = tool_call_id.strip()
                conversation_messages.append(tool_entry)
            elif role == "user":
                conversation_messages.append({"role": role, "content": content})

        user_message = ""
        history = []
        if conversation_messages:
            last_message = conversation_messages[-1]
            if last_message.get("role") == "user":
                user_message = last_message.get("content", "")
                history = conversation_messages[:-1]
            else:
                user_message = ""
                history = conversation_messages
        # NOTE: Message history is NOT pre-truncated here. The agent's own
        # context compressor (agent/context_compressor.py) handles context
        # overflow adaptively based on the actual model's context window.
        # Pre-truncation was destroying context for large conversations.
        # Dynamic model selection now picks a model with enough context
        # for the conversation size, so pre-truncation is unnecessary.

        # Tool loop prevention: allow legitimate OpenAI tool continuation
        # (assistant tool_calls -> tool results), but reject orphaned tool-only
        # continuations that have no preceding assistant tool call context.
        has_user_msg = bool(user_message and user_message.strip())
        is_tool_result_only = bool(
            conversation_messages and
            conversation_messages[-1].get("role") == "tool"
        )
        last_non_tool = None
        if is_tool_result_only:
            for msg in reversed(conversation_messages[:-1]):
                if isinstance(msg, dict) and msg.get("role") != "tool":
                    last_non_tool = msg
                    break
        has_assistant_tool_context = bool(
            isinstance(last_non_tool, dict)
            and last_non_tool.get("role") == "assistant"
            and isinstance(last_non_tool.get("tool_calls"), list)
            and last_non_tool.get("tool_calls")
        )
        if is_tool_result_only and not has_user_msg and not has_assistant_tool_context:
            return web.json_response(
                {"error": {"message": "Cannot continue with orphaned tool results. Include a user message, or send the preceding assistant tool_calls in the conversation.", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header injection.
            if re.search(r'[\r\n\x00]', provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = self._ensure_session_db()
                if db is not None:
                    history = db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a stable session ID from the conversation fingerprint so
            # that consecutive messages from the same Open WebUI (or similar)
            # conversation map to the same Hermes session.  The first user
            # message + system prompt are constant across all turns.
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = cm.get("content", "")
                    break
            session_id = _derive_chat_session_id(system_prompt, first_user)
            # history already set from request body above

        # NOTE: Message history is NOT pre-truncated. Agent's context compressor
        # handles overflow based on actual model's context window.

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
        role_cfg = _get_role_alias_config(model_name)
        role_hint = dict(role_cfg.get("hint") or {}) if role_cfg else None
        _toolset_mode = "auto"
        _provider_mode = False
        if model_name == "hermes-agentic-full":
            _toolset_mode = "full"
        elif model_name == "hermes-agentic-remote":
            _toolset_mode = "remote"
        elif model_name == "hermes-code":
            _provider_mode = True
        external_tool_mode = "none"
        if isinstance(tools, list) and tools:
            if model_name == "hermes-code":
                external_tool_mode = "inband"
            else:
                external_tool_mode = "inband" if force_connection_close else "broker"
        logger.info(
            "[api_server] chat request stream=%s tools=%s external_tool_mode=%s ua=%s model=%s",
            stream, bool(tools), external_tool_mode, user_agent[:120], model_name,
        )

        # Hermes-swarm: select free/cheap model pool
        swarm_mode = False
        swarm_model_pool = None
        if model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            swarm_mode = True
            from agent.model_metadata import estimate_request_tokens_rough
            _approx_tokens = 0
            try:
                _approx_tokens = estimate_request_tokens_rough(
                    history or [],
                    system_prompt=system_prompt or "",
                    tools=tools,
                )
            except Exception:
                pass
            swarm_model_pool = await self._prepare_swarm_model_pool(
                system_prompt=system_prompt or "",
                conversation_history=history,
                user_message=user_message,
                tools=tools,
                estimated_tokens=_approx_tokens,
                routing_hint=role_hint,
            )
            logger.info(
                "[api_server] swarm pool: primary=%s, fallbacks=%d models, "
                "large-context options: %s routing_hint=%s",
                swarm_model_pool["primary"], len(swarm_model_pool["fallbacks"]),
                swarm_model_pool["large_context_fallbacks"],
                swarm_model_pool.get("routing_hint"),
            )

            # Token-aware pre-truncation: ensure history fits the primary model's context.
            # Uses 85% of context window as safe budget. This prevents 413 errors before
            # they reach the LLM API. _resolve_swarm_model already filters candidate lists.
            _primary_model = swarm_model_pool.get("primary", "")
            if _primary_model and history:
                _history_tokens = _messages_token_count(history, system_prompt or "")
                _ctx_len = _model_context_length(_primary_model)
                if _ctx_len > 0 and _history_tokens > int(_ctx_len * 0.85):
                    history = _compact_message_history(
                        history,
                        session_id,
                        system_prompt=system_prompt or "",
                        target_model=_primary_model,
                    )
                    logger.info(
                        "[api_server] history pre-truncated: ~%d tokens -> ~%d, model=%s",
                        _history_tokens, _messages_token_count(history, system_prompt or ""), _primary_model,
                    )

        agents_prefetch_text = ""
        if swarm_mode and _needs_agents_prefetch(user_message, system_prompt, tools, conversation_messages):
            prefetch = _determine_agents_prefetch_action(conversation_messages, tools)
            status = prefetch.get("status")
            if status in {"need_search", "need_read"}:
                prefetch_call = prefetch.get("tool_call")
                response_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": _enrich_client_tool_calls([prefetch_call]) if prefetch_call else [],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                headers = {"X-Hermes-Session-Id": session_id}
                if not stream:
                    return web.json_response(response_data, headers=headers)
                sse_headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                response = web.StreamResponse(status=200, headers=sse_headers)
                await response.prepare(request)
                chunk = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": response_data["choices"][0]["message"]["tool_calls"]},
                        "finish_reason": None,
                    }],
                }
                finish = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await response.write(f"data: {json.dumps(finish)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
                return response
            agents_prefetch_text = str(prefetch.get("agents_text") or "").strip()

        if agents_prefetch_text:
            system_prompt = (
                f"{system_prompt}\n\n[Deterministic AGENTS preflight]\n{agents_prefetch_text}"
                if system_prompt else f"[Deterministic AGENTS preflight]\n{agents_prefetch_text}"
            )

        created = int(time.time())

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # Filter out None — the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Send tool progress as a separate SSE event.

                Previously, progress markers like ``⏰ list`` were injected
                directly into ``delta.content``.  OpenAI-compatible frontends
                (Open WebUI, LobeChat, …) store ``delta.content`` verbatim as
                the assistant message and send it back on subsequent requests.
                After enough turns the model learns to *emit* the markers as
                plain text instead of issuing real tool calls — silently
                hallucinating tool results.  See #6972.

                The fix: push a tagged tuple ``("__tool_progress__", payload)``
                onto the stream queue.  The SSE writer emits it as a custom
                ``event: hermes.tool.progress`` line that compliant frontends
                can render for UX but will *not* persist into conversation
                history.  Clients that don't understand the custom event type
                silently ignore it per the SSE specification.
                """
                if event_type != "tool.started":
                    return
                if name.startswith("_"):
                    return
                from agent.display import get_tool_emoji
                emoji = get_tool_emoji(name)
                label = preview or name
                _stream_q.put(("__tool_progress__", {
                    "tool": name,
                    "emoji": emoji,
                    "label": label,
                }))

            def _on_tool_gen(tool_name: str, call_id: Optional[str] = None, arguments: str = ""):
                """Emit function_call chunks when the model decides to use a tool."""
                if not isinstance(call_id, str) or not call_id.strip():
                    basis = f"{session_id}:{completion_id}:{tool_name}:{arguments or ''}"
                    call_id = f"call_{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:24]}"
                if external_tool_mode == "broker":
                    try:
                        from gateway.platforms import tool_call_hub
                        tool_call_hub.register_call(session_id, call_id, tool_name)
                        logger.info(
                            "[api_server] registered external tool call session=%s call_id=%s tool=%s",
                            session_id, call_id, tool_name,
                        )
                    except Exception as e:
                        logger.debug("tool_call_hub.register_call failed: %s", e)
                tool_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": arguments or "",
                                },
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                _stream_q.put(("__tool_call_start__", {
                    "session_id": session_id,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "register_with_hub": external_tool_mode == "broker",
                    "chunk": tool_chunk,
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_gen_callback=_on_tool_gen,
                agent_ref=agent_ref,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                force_connection_close=force_connection_close,
                swarm_model_pool=swarm_model_pool,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            # Smart routing: check dedup cache and route based on complexity
            from agent.deduplicator import get_global_deduplicator
            from agent.smart_router import get_global_router
            
            # Build the full prompt for routing analysis
            routing_prompt = user_message
            if system_prompt:
                routing_prompt = f"{system_prompt}\n\n{routing_prompt}"
            
            # Check dedup cache first
            dedup = get_global_deduplicator()
            dedup_cache_key = dedup.compute_key(routing_prompt, model_name) if hasattr(dedup, 'compute_key') else None
            
            if dedup_cache_key:
                cached_response, found = dedup.get(routing_prompt, model_name)
                if found and cached_response:
                    logger.info("[smart_router] dedup cache hit for model=%s", model_name)
                    # Return cached response - but we still need to run agent for usage tracking
                    # For now, just log and continue to actual execution
            
            # Route based on complexity if smart routing is enabled
            router = get_global_router()
            routing_result = router.route(routing_prompt)
            
            if routing_result.get("complexity") == "simple":
                logger.info(
                    "[smart_router] routing simple query to %s (tier: %s, savings: %.4f)",
                    routing_result.get("model"),
                    routing_result.get("tier"),
                    routing_result.get("savings_vs_primary", 0),
                )
            
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = result.get("final_response", "")
        if result.get("tool_calls_pending"):
            last_assistant = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_assistant = msg
                    break
            response_data = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            # For pending tool calls, return tool_calls only.
                            # Some clients mis-handle mixed assistant text +
                            # tool_calls and render them out of order.
                            "content": "",
                            "tool_calls": _enrich_client_tool_calls((last_assistant or {}).get("tool_calls", [])) if last_assistant else [],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            headers = {"X-Hermes-Session-Id": session_id}
            if force_connection_close:
                headers["Connection"] = "close"
            return web.json_response(response_data, headers=headers)
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        # Extract reasoning_content from the last assistant message if available
        # This is needed for Kimi/Moonshot/GLM reasoning models
        reasoning_content = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                reasoning_content = msg.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    break

        message_data = {
            "role": "assistant",
            "content": final_response,
        }
        # Include reasoning_content if present and non-empty
        if reasoning_content:
            message_data["reasoning_content"] = reasoning_content

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message_data,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        headers = {"X-Hermes-Session-Id": session_id}
        if force_connection_close:
            headers["Connection"] = "close"
        return web.json_response(response_data, headers=headers)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        force_connection_close: bool = False, swarm_model_pool: dict = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if force_connection_close:
            sse_headers["Connection"] = "close"
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
            last_activity = time.monotonic()

            buffered_text_deltas: List[str] = []
            tool_call_started = False

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972.
                """
                nonlocal tool_call_started
                if isinstance(item, tuple) and len(item) == 2:
                    tag, payload = item
                    if tag == "__tool_progress__":
                        event_data = json.dumps(payload)
                        await response.write(
                            f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                        )
                        return time.monotonic()
                    if tag == "__tool_call_start__":
                        tool_call_started = True
                        buffered_text_deltas.clear()
                        try:
                            logger.info(
                                "[api_server] emitting tool call SSE session=%s call_id=%s tool=%s",
                                payload.get("session_id", session_id), payload.get("call_id"), payload.get("tool_name"),
                            )
                            if payload.get("session_id") and payload.get("register_with_hub"):
                                from gateway.platforms import tool_call_hub
                                tool_call_hub.register_call(
                                    payload.get("session_id", session_id), payload.get("call_id"), payload.get("tool_name"),
                                )
                        except Exception:
                            pass
                        chunk = payload.get("chunk")
                        if isinstance(chunk, dict):
                            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            return time.monotonic()
                content_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                }
                if tool_call_started:
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                else:
                    buffered_text_deltas.append(item)
                return time.monotonic()

            # Stream content chunks as they arrive from the agent
            loop = asyncio.get_running_loop()
            while True:
                try:
                    delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            finish_reason = "stop"
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                if isinstance(result, dict) and result.get("tool_calls_pending"):
                    finish_reason = "tool_calls"
            except Exception:
                pass

            if finish_reason == "stop" and buffered_text_deltas:
                for item in buffered_text_deltas:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)

        return response

    async def _write_sse_responses(
        self,
        request: "web.Request",
        response_id: str,
        model: str,
        created_at: int,
        stream_q,
        agent_task,
        agent_ref,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        instructions: Optional[str],
        conversation: Optional[str],
        store: bool,
        session_id: str,
    ) -> "web.StreamResponse":
        """Write an SSE stream for POST /v1/responses (OpenAI Responses API).

        Emits spec-compliant event types as the agent runs:

        - ``response.created`` — initial envelope (status=in_progress)
        - ``response.output_text.delta`` / ``response.output_text.done`` —
          streamed assistant text
        - ``response.output_item.added`` / ``response.output_item.done``
          with ``item.type == "function_call"`` — when the agent invokes a
          tool (both events fire; the ``done`` event carries the finalized
          ``arguments`` string)
        - ``response.output_item.added`` with
          ``item.type == "function_call_output"`` — tool result with
          ``{call_id, output, status}``
        - ``response.completed`` — terminal event carrying the full
          response object with all output items + usage (same payload
          shape as the non-streaming path for parity)
        - ``response.failed`` — terminal event on agent error

        If the client disconnects mid-stream, ``agent.interrupt()`` is
        called so the agent stops issuing upstream LLM calls, then the
        asyncio task is cancelled.  When ``store=True`` the full response
        is persisted to the ResponseStore in a ``finally`` block so GET
        /v1/responses/{id} and ``previous_response_id`` chaining work the
        same as the batch path.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        # State accumulated during the stream
        final_text_parts: List[str] = []
        # Track open function_call items by name so we can emit a matching
        # ``done`` event when the tool completes.  Order preserved.
        pending_tool_calls: List[Dict[str, Any]] = []
        # Output items we've emitted so far (used to build the terminal
        # response.completed payload).  Kept in the order they appeared.
        emitted_items: List[Dict[str, Any]] = []
        # Monotonic counter for output_index (spec requires it).
        output_index = 0
        # Monotonic counter for call_id generation if the agent doesn't
        # provide one (it doesn't, from tool_progress_callback).
        call_counter = 0
        # Canonical Responses SSE events include a monotonically increasing
        # sequence_number. Add it server-side for every emitted event so
        # clients that validate the OpenAI event schema can parse our stream.
        sequence_number = 0
        # Track the assistant message item id + content index for text
        # delta events — the spec ties deltas to a specific item.
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_output_index: Optional[int] = None
        message_opened = False

        async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
            nonlocal sequence_number
            if "sequence_number" not in data:
                data["sequence_number"] = sequence_number
            sequence_number += 1
            payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            await response.write(payload.encode())

        def _envelope(status: str) -> Dict[str, Any]:
            env: Dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }
            return env

        final_response_text = ""
        agent_error: Optional[str] = None
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            # response.created — initial envelope, status=in_progress
            created_env = _envelope("in_progress")
            created_env["output"] = []
            await _write_event("response.created", {
                "type": "response.created",
                "response": created_env,
            })
            last_activity = time.monotonic()

            async def _open_message_item() -> None:
                """Emit response.output_item.added for the assistant message
                the first time any text delta arrives."""
                nonlocal message_opened, message_output_index, output_index
                if message_opened:
                    return
                message_opened = True
                message_output_index = output_index
                output_index += 1
                item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": item,
                })

            async def _emit_text_delta(delta_text: str) -> None:
                await _open_message_item()
                final_text_parts.append(delta_text)
                await _write_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta_text,
                    "logprobs": [],
                })

            async def _emit_tool_started(payload: Dict[str, Any]) -> str:
                """Emit response.output_item.added for a function_call.

                Returns the call_id so the matching completion event can
                reference it.  Prefer the real ``tool_call_id`` from the
                agent when available; fall back to a generated call id for
                safety in tests or older code paths.
                """
                nonlocal output_index, call_counter
                call_counter += 1
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
                args = payload.get("arguments", {})
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": payload.get("name", ""),
                    "call_id": call_id,
                    "arguments": arguments_str,
                }
                idx = output_index
                output_index += 1
                pending_tool_calls.append({
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "item_id": item["id"],
                    "output_index": idx,
                })
                emitted_items.append({
                    "type": "function_call",
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "call_id": call_id,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })
                return call_id

            async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
                """Emit response.output_item.done (function_call) followed
                by response.output_item.added (function_call_output)."""
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                result = payload.get("result", "")
                pending = None
                if call_id:
                    for i, p in enumerate(pending_tool_calls):
                        if p["call_id"] == call_id:
                            pending = pending_tool_calls.pop(i)
                            break
                if pending is None:
                    # Completion without a matching start — skip to avoid
                    # emitting orphaned done events.
                    return

                # function_call done
                done_item = {
                    "id": pending["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "arguments": pending["arguments"],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["output_index"],
                    "item": done_item,
                })

                # function_call_output added (result)
                result_str = result if isinstance(result, str) else json.dumps(result)
                output_parts = [{"type": "input_text", "text": result_str}]
                output_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": output_item,
                })
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": output_item,
                })

            # Main drain loop — thread-safe queue fed by agent callbacks.
            async def _dispatch(it) -> None:
                """Route a queue item to the correct SSE emitter.

                Plain strings are text deltas.  Tagged tuples with
                ``__tool_started__`` / ``__tool_completed__`` prefixes
                are tool lifecycle events.
                """
                if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                    tag, payload = it
                    if tag == "__tool_started__":
                        await _emit_tool_started(payload)
                    elif tag == "__tool_completed__":
                        await _emit_tool_completed(payload)
                    # Unknown tags are silently ignored (forward-compat).
                elif isinstance(it, str):
                    await _emit_text_delta(it)
                # Other types (non-string, non-tuple) are silently dropped.

            loop = asyncio.get_running_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain remaining
                        while True:
                            try:
                                item = stream_q.get_nowait()
                                if item is None:
                                    break
                                await _dispatch(item)
                                last_activity = time.monotonic()
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if item is None:  # EOS sentinel
                    break

                await _dispatch(item)
                last_activity = time.monotonic()

            # Pick up agent result + usage from the completed task
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                # If the agent produced a final_response but no text
                # deltas were streamed (e.g. some providers only emit
                # the full response at the end), emit a single fallback
                # delta so Responses clients still receive a live text part.
                agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
                if agent_final and not final_text_parts:
                    await _emit_text_delta(agent_final)
                if agent_final and not final_response_text:
                    final_response_text = agent_final
                if isinstance(result, dict) and result.get("error") and not final_response_text:
                    agent_error = result["error"]
            except Exception as e:  # noqa: BLE001
                logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
                agent_error = str(e)

            # Close the message item if it was opened
            final_response_text = "".join(final_text_parts) or final_response_text
            if message_opened:
                await _write_event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_response_text,
                    "logprobs": [],
                })
                msg_done_item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_response_text}
                    ],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": msg_done_item,
                })

            # Always append a final message item in the completed
            # response envelope so clients that only parse the terminal
            # payload still see the assistant text.  This mirrors the
            # shape produced by _extract_output_items in the batch path.
            final_items: List[Dict[str, Any]] = list(emitted_items)
            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text or (agent_error or "")}
                ],
            })

            if agent_error:
                failed_env = _envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": agent_error, "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            else:
                completed_env = _envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_env,
                })

                # Persist for future chaining / GET retrieval, mirroring
                # the batch path behavior.
                if store:
                    full_history = list(conversation_history)
                    full_history.append({"role": "user", "content": user_message})
                    if isinstance(result, dict) and result.get("messages"):
                        full_history.extend(result["messages"])
                    else:
                        full_history.append({"role": "assistant", "content": final_response_text})
                    self._response_store.put(response_id, {
                        "response": completed_env,
                        "conversation_history": full_history,
                        "instructions": instructions,
                        "session_id": session_id,
                    })
                    if conversation:
                        self._response_store.set_conversation(conversation, response_id)

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected — interrupt the agent so it stops
            # making upstream LLM calls, then cancel the task.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = body.get("store", True)
        tool_choice = body.get("tool_choice")

        # Extract tools from request and mark them as from client
        tools = body.get("tools")
        if tools:
            for tool in tools:
                tool["_from_client"] = True

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, str]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = _normalize_chat_content(item.get("content", ""))
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        # Accept explicit conversation_history from the request body.
        # This lets stateless clients supply their own history instead of
        # relying on server-side response chaining via previous_response_id.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message = input_messages[-1].get("content", "") if input_messages else ""
        if not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # NOTE: Message history is NOT pre-truncated. Agent's context compressor
        # handles overflow based on actual model's context window.

        # Reuse session from previous_response_id chain so the dashboard
        # groups the entire conversation under one session entry.
        session_id = stored_session_id or str(uuid.uuid4())

        model_name = body.get("model", self._model_name)
        role_cfg = _get_role_alias_config(model_name)
        role_hint = dict(role_cfg.get("hint") or {}) if role_cfg else None
        _toolset_mode = "auto"
        _provider_mode = False
        if model_name == "hermes-agentic-full":
            _toolset_mode = "full"
        elif model_name == "hermes-agentic-remote":
            _toolset_mode = "remote"
        elif model_name == "hermes-code":
            _provider_mode = True

        external_tool_mode = "none"
        if isinstance(tools, list) and tools:
            if model_name == "hermes-code":
                external_tool_mode = "inband"
            else:
                external_tool_mode = "broker"

        # Extract model_name FIRST - before any agent creation  
        _model_name = body.get("model", self._model_name)
        
        # Handle hermes-swarm mode - use _model_name from request body
        swarm_mode = False
        swarm_model_pool = None
        if _model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            swarm_mode = True
            from agent.model_metadata import estimate_request_tokens_rough
            _approx_tokens = 0
            try:
                _approx_tokens = estimate_request_tokens_rough(
                    conversation_history or [],
                    system_prompt=instructions or "",
                    tools=tools,
                )
            except Exception:
                pass
            swarm_model_pool = await self._prepare_swarm_model_pool(
                system_prompt=instructions or "",
                conversation_history=conversation_history,
                user_message=user_message,
                tools=tools,
                estimated_tokens=_approx_tokens,
                routing_hint=role_hint,
            )

        stream = bool(body.get("stream", False))
        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # None from the agent is a CLI box-close signal, not EOS.
                # Forwarding would kill the SSE stream prematurely; the
                # SSE writer detects completion via agent_task.done().
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Queue non-start tool progress events if needed in future.

                The structured Responses stream uses ``tool_start_callback``
                and ``tool_complete_callback`` for exact call-id correlation,
                so progress events are currently ignored here.
                """
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Queue a started tool for live function_call streaming."""
                _stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Queue a completed tool result for live function_call_output streaming."""
                _stream_q.put(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
            ))

            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            created_at = int(time.time())

            return await self._write_sse_responses(
                request=request,
                response_id=response_id,
                model=_model_name,
                created_at=created_at,
                stream_q=_stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=user_message,
                instructions=instructions,
                conversation=conversation,
                store=store,
                session_id=session_id,
            )

        async def _compute_response():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(
                body,
                keys=["input", "instructions", "previous_response_id", "conversation", "model", "tools"],
            )
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_response)
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_response()
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        if result.get("tool_calls_pending"):
            last_assistant = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_assistant = msg
                    break

            output_items: List[Dict[str, Any]] = []
            for tc in _enrich_client_tool_calls((last_assistant or {}).get("tool_calls", [])):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                output_items.append({
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                    "call_id": tc.get("id") or tc.get("call_id", ""),
                })

            response_data = {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "created_at": created_at,
                "model": body.get("model", self._model_name),
                "output": output_items,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

            if store:
                full_history = list(conversation_history)
                full_history.append({"role": "user", "content": user_message})
                full_history.extend(result.get("messages", []))
                self._response_store.put(response_id, {
                    "response": response_data,
                    "conversation_history": full_history,
                    "instructions": instructions,
                    "session_id": session_id,
                })
                if conversation:
                    self._response_store.set_conversation(conversation, response_id)

            return web.json_response(response_data)

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        # Add agent's internal messages if available
        agent_messages = result.get("messages", [])
        if agent_messages:
            full_history.extend(agent_messages)
        else:
            full_history.append({"role": "assistant", "content": final_response})

        # Build output items (includes tool calls + final message)
        output_items = self._extract_output_items(result)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", self._model_name),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
                "session_id": session_id,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        return web.json_response(response_data)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    # Check cron module availability once (not per-request)
    _CRON_AVAILABLE = False
    try:
        from cron.jobs import (
            list_jobs as _cron_list,
            get_job as _cron_get,
            create_job as _cron_create,
            update_job as _cron_update,
            remove_job as _cron_remove,
            pause_job as _cron_pause,
            resume_job as _cron_resume,
            trigger_job as _cron_trigger,
        )
        # Wrap as staticmethod to prevent descriptor binding — these are plain
        # module functions, not instance methods.  Without this, self._cron_*()
        # injects ``self`` as the first positional argument and every call
        # raises TypeError.
        _cron_list = staticmethod(_cron_list)
        _cron_get = staticmethod(_cron_get)
        _cron_create = staticmethod(_cron_create)
        _cron_update = staticmethod(_cron_update)
        _cron_remove = staticmethod(_cron_remove)
        _cron_pause = staticmethod(_cron_pause)
        _cron_resume = staticmethod(_cron_resume)
        _cron_trigger = staticmethod(_cron_trigger)
        _CRON_AVAILABLE = True
    except ImportError:
        pass

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    def _check_jobs_available(self) -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not self._CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in ("true", "1")
            jobs = self._cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs — create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = self._cron_create(**kwargs)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} — get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} — update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            job = self._cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = self._cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_gen_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        toolset_mode: str = "auto",
        provider_mode: bool = False,
        swarm_mode: bool = False,
        swarm_model_pool = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        external_tool_mode: str = "none",
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        loop = asyncio.get_running_loop()

        logging.warning(f"[API_SERVER] _run_agent called: swarm_mode={swarm_mode}, swarm_model_pool={swarm_model_pool}")

        def _run():
            generated_tool_calls: List[Dict[str, Any]] = []

            def _wrapped_tool_gen_callback(tool_name: str, call_id: Optional[str] = None, arguments: str = ""):
                generated_tool_calls.append(_enrich_client_tool_call({
                    "id": call_id or "",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments or "",
                    },
                }))
                if tool_gen_callback:
                    try:
                        tool_gen_callback(tool_name, call_id=call_id, arguments=arguments)
                    except Exception:
                        pass

            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                tool_gen_callback=_wrapped_tool_gen_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                toolset_mode=toolset_mode,
                provider_mode=provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
            )
            if agent_ref is not None:
                agent_ref[0] = agent
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                task_id="default",
            )
            if (
                swarm_mode
                and swarm_model_pool
                and not stream_delta_callback
                and isinstance(result, dict)
                and str(result.get("final_response") or "").strip()
            ):
                routing_hint = swarm_model_pool.get("routing_hint") or {}
                should_verify = os.getenv("HERMES_SWARM_ENABLE_VERIFIER", "true").strip().lower() not in {"0", "false", "no"}
                should_verify = should_verify and str(routing_hint.get("recommended_tier") or "") in {"balanced", "premium"}
                if should_verify:
                    try:
                        verification = self._run_swarm_verifier_sync(
                            system_prompt=ephemeral_system_prompt or "",
                            conversation_history=conversation_history,
                            user_message=user_message,
                            candidate_response=str(result.get("final_response") or ""),
                            swarm_model_pool=swarm_model_pool,
                        )
                        if str(verification.get("verdict") or "").strip().lower() == "revise":
                            revised = str(verification.get("revised_response") or "").strip()
                            if revised:
                                logger.info("[api_server] swarm verifier revised final response")
                                if not isinstance(result, dict):
                                    result = {"final_response": revised, "messages": []}
                                else:
                                    result["final_response"] = revised
                                if isinstance(result.get("messages"), list):
                                    for msg in reversed(result["messages"]):
                                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                                            msg["content"] = revised
                                            break
                                meta = result.get("meta")
                                if not isinstance(meta, dict):
                                    meta = {}
                                    result["meta"] = meta
                                meta["swarm_verifier"] = verification
                        else:
                            logger.info("[api_server] swarm verifier accepted final response")
                    except Exception as exc:
                        logger.warning("[api_server] swarm verifier failed: %s", exc)
            if (
                isinstance(result, dict)
                and generated_tool_calls
                and getattr(agent, "_external_tool_mode", "none") in ("broker", "inband")
            ):
                messages = list(result.get("messages", []))
                has_tool_calls = any(
                    isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
                    for msg in messages
                )
                if not has_tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": result.get("final_response", "") or "",
                        "tool_calls": _enrich_client_tool_calls(generated_tool_calls),
                    })
                    result["messages"] = messages
                result["tool_calls_pending"] = True
                result["finish_reason"] = "tool_calls"
                result["completed"] = False
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    def _runtime_kwargs_for_model(self, model: str) -> tuple[Dict[str, Any], str]:
        return _runtime_kwargs_for_model_id(model)

    def _run_swarm_scout_sync(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        swarm_model_pool: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = swarm_model_pool or {}
        # Scout model selection: prefer cheapest model with credentials that can
        # safely handle the estimated token count. When context is small, gpt-5-mini
        # (cheap/Copilot) is selected. When context is large, MiniMax-M2.7 (200K)
        # passes the context filter and is selected instead.
        estimated_tokens = swarm_model_pool.get("estimated_tokens", 0)
        preferred_models = [os.getenv("HERMES_SWARM_SCOUT_MODEL", "").strip()]
        preferred_models.extend(pool.get("scout_fallbacks", []))
        scout_model = ""
        for candidate in preferred_models:
            if candidate and _swarm_model_is_available(candidate):
                # Skip models that can't hold the estimated context (prevents 413 on scout itself)
                if not _model_safe_for_tokens(candidate, estimated_tokens):
                    logger.info("[api_server] scout skipping %s (context %d > safe limit)", candidate, estimated_tokens)
                    continue
                scout_model = candidate
                break
        if not scout_model:
            raise RuntimeError("No available scout model with credentials")

        logger.warning(
            "[API_SERVER] swarm scout selected model=%s primary=%s scout_fallbacks=%s estimated_tokens=%s",
            scout_model,
            pool.get("primary", ""),
            pool.get("scout_fallbacks", []),
            estimated_tokens,
        )

        runtime_kwargs, scout_model_name = self._runtime_kwargs_for_model(scout_model)
        from run_agent import AIAgent

        scout_prompt = (
            "Classify this task for routing. Return ONLY compact JSON with keys: "
            "task_type, recommended_tier, needs_instruction_following, needs_repo_reasoning, "
            "needs_bug_judgement, confidence, reason. "
            "recommended_tier must be one of cheap, primary, balanced, premium. "
            "Use premium for repo review, debugging, implementation, or architectural tasks. "
            "Use balanced for instruction-sensitive workspace analysis. "
            "If AGENTS.md instructions are quoted inline, treat them as authoritative and do not say the file is missing. "
            "Do not solve the task itself.\n\n"
            + _summarize_swarm_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                user_message=user_message,
            )
        )
        if tools:
            scout_prompt += f"\n\nTOOLS_PRESENT: {len(tools)}"

        agent = AIAgent(
            model=scout_model_name,
            **runtime_kwargs,
            max_iterations=2,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt="You are a routing classifier.",
            enabled_toolsets=[],
            session_id=f"swarm-scout-{uuid.uuid4().hex[:8]}",
            platform="api_server",
            session_db=None,
            skip_memory=True,
            skip_context_files=True,
            tools=[],
        )
        result = agent.run_conversation(
            user_message=scout_prompt,
            conversation_history=[],
            task_id="swarm_scout",
        )
        response_text = _extract_agent_result_text(result)
        if not response_text:
            raise RuntimeError("Empty scout response")
        logger.info("[api_server] swarm scout raw response: %s", response_text[:400])
        parsed = _parse_loose_json_object(response_text)
        parsed["source"] = "scout"
        parsed["model"] = scout_model
        return parsed

    def _run_swarm_verifier_sync(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        candidate_response: str = "",
        swarm_model_pool: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = swarm_model_pool or {}
        preferred_models = [
            os.getenv("HERMES_SWARM_VERIFY_MODEL", "").strip(),
            "openai/gpt-5.5",
            "openai/gpt-5.3-codex",
            "google/gemini-2.5-flash",
            "openai/gpt-5-mini",
        ]
        preferred_models.extend(pool.get("fallbacks", []))
        verify_model = ""
        primary = str(pool.get("primary") or "").strip()
        for candidate in preferred_models:
            if candidate and candidate != primary and _swarm_model_is_available(candidate):
                verify_model = candidate
                break
        if not verify_model:
            raise RuntimeError("No available verifier model with credentials")

        runtime_kwargs, verify_model_name = self._runtime_kwargs_for_model(verify_model)
        from run_agent import AIAgent

        verifier_prompt = (
            "Review the candidate answer for correctness and grounding. Return ONLY compact JSON with keys: "
            "verdict, issues, revised_response. verdict must be one of ok or revise. "
            "Revise if the answer hallucinates missing files/access, ignores supplied AGENTS instructions, "
            "or misses an obvious higher-severity issue that is visible from the provided context. "
            "Treat missing AGENTS.md on disk as non-fatal when equivalent instructions are quoted inline. "
            "issues must be an array of short strings. If verdict is ok, revised_response should be empty.\n\n"
            + _summarize_swarm_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                user_message=user_message,
            )
            + f"\n\nCANDIDATE_RESPONSE:\n{candidate_response[:12000]}"
        )

        agent = AIAgent(
            model=verify_model_name,
            **runtime_kwargs,
            max_iterations=2,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt="You are a strict answer verifier.",
            enabled_toolsets=[],
            session_id=f"swarm-verify-{uuid.uuid4().hex[:8]}",
            platform="api_server",
            session_db=None,
            skip_memory=True,
            skip_context_files=True,
            tools=[],
        )
        result = agent.run_conversation(
            user_message=verifier_prompt,
            conversation_history=[],
            task_id="swarm_verify",
        )
        response_text = _extract_agent_result_text(result)
        if not response_text:
            raise RuntimeError("Empty verifier response")
        logger.info("[api_server] swarm verifier raw response: %s", response_text[:400])
        if response_text.lower().startswith("invalid api response after"):
            return {"verdict": "ok", "issues": [response_text[:120]], "revised_response": "", "model": verify_model}
        parsed = _parse_loose_json_object(response_text)
        parsed["model"] = verify_model
        return parsed

    async def _prepare_swarm_model_pool(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        estimated_tokens: int = 0,
        routing_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = _build_swarm_model_pool(estimated_tokens=estimated_tokens, routing_hint=routing_hint)
        heuristic_hint = _heuristic_swarm_routing_hint(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_message=user_message,
            tools=tools,
            estimated_tokens=estimated_tokens,
        )
        merged_hint = dict(heuristic_hint)
        if routing_hint:
            merged_hint.update(routing_hint)
        pool["routing_hint"] = merged_hint

        should_scout = os.getenv("HERMES_SWARM_ENABLE_SCOUT", "true").strip().lower() not in {"0", "false", "no"}
        if should_scout and heuristic_hint.get("recommended_tier") in {"balanced", "premium"}:
            try:
                loop = asyncio.get_running_loop()
                scout_hint = await loop.run_in_executor(
                    None,
                    lambda: self._run_swarm_scout_sync(
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                        user_message=user_message,
                        tools=tools,
                        swarm_model_pool=pool,
                    ),
                )
                if isinstance(scout_hint, dict):
                    scout_merged_hint = dict(pool.get("routing_hint") or {})
                    scout_merged_hint.update(scout_hint)
                    pool["routing_hint"] = scout_merged_hint
            except Exception as exc:
                logger.warning("[api_server] swarm scout failed, falling back to heuristics: %s", exc)
        return pool

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _MAX_CONCURRENT_RUNS = 10  # Prevent unbounded resource allocation
    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            # _thinking and subagent_progress are intentionally not forwarded

        return _callback

    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs — start an agent run, return run_id immediately."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Enforce concurrency limit
        if len(self._run_streams) >= self._MAX_CONCURRENT_RUNS:
            return web.json_response(
                _openai_error(f"Too many concurrent runs (max {self._MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
                status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_input = body.get("input")
        if not raw_input:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
        if not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        run_id = f"run_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = time.time()

        event_cb = self._make_run_event_callback(run_id, loop)

        # Also wire stream_delta_callback so message.delta events flow through
        def _text_cb(delta: Optional[str]) -> None:
            if delta is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })
            except Exception:
                pass

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")

        # Accept explicit conversation_history from the request body.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored:
                conversation_history = list(stored.get("conversation_history", []))
                stored_session_id = stored.get("session_id")
                if instructions is None:
                    instructions = stored.get("instructions")

        # When input is a multi-message array, extract all but the last
        # message as conversation history (the last becomes user_message).
        # Only fires when no explicit history was provided.
        if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
            for msg in raw_input[:-1]:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, list):
                        # Flatten multi-part content blocks to text
                        content = " ".join(
                            part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    conversation_history.append({"role": msg["role"], "content": str(content)})

        session_id = body.get("session_id") or stored_session_id or run_id
        ephemeral_system_prompt = instructions

        async def _run_and_close():
            try:
                agent = self._create_agent(
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_text_cb,
                    tool_progress_callback=event_cb,
                    provider_mode=provider_mode,
                    swarm_mode=swarm_mode,
                    swarm_model_pool=swarm_model_pool,
                )
                def _run_sync():
                    r = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id="default",
                    )
                    u = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    return r, u

                result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                q.put_nowait({
                    "event": "run.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "output": final_response,
                    "usage": usage,
                })
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                try:
                    q.put_nowait({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": str(exc),
                    })
                except Exception:
                    pass
            finally:
                # Sentinel: signal SSE stream to close
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

        task = asyncio.create_task(_run_and_close())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        return web.json_response({"run_id": run_id, "status": "started"}, status=202)

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        q = self._run_streams[run_id]

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    # Run finished — send final SSE comment and close
                    await response.write(b": stream closed\n\n")
                    break
                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically clean up run streams that were never consumed."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                run_id
                for run_id, created_at in list(self._run_streams_created.items())
                if now - created_at > self._RUN_STREAM_TTL
            ]
            for run_id in stale:
                logger.debug("[api_server] sweeping orphaned run %s", run_id)
                self._run_streams.pop(run_id, None)
                self._run_streams_created.pop(run_id, None)

    async def _handle_tool_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/tool_responses — ingest client tool results."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = (request.match_info.get("session_id") or "").strip()
        if not session_id:
            return web.json_response({"error": "Missing session_id in path"}, status=400)

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        call_id = body.get("call_id")
        status = body.get("status")
        result = body.get("result")

        if not isinstance(call_id, str) or not call_id.strip():
            return web.json_response({"error": "Missing or invalid 'call_id'"}, status=400)
        if status not in ("ok", "error"):
            return web.json_response({"error": "'status' must be 'ok' or 'error'"}, status=400)

        try:
            from gateway.platforms import tool_call_hub
            tool_call_hub.set_response(session_id, call_id, status, result)
            logger.info(
                "[api_server] ingested tool response session=%s call_id=%s status=%s",
                session_id, call_id, status,
            )
        except Exception as e:
            logger.error("Failed to set tool response: %s", e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

        return web.json_response({"ok": True}, status=200)

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws)
            self._app["api_server_adapter"] = self
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/stats", self._handle_stats)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management API
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/sessions/{session_id}/tool_responses", self._handle_tool_responses)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            # Refuse to start network-accessible without authentication
            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires API_SERVER_KEY. "
                    "Set API_SERVER_KEY or use the default 127.0.0.1.",
                    self.name, self._host,
                )
                return False

            # Refuse to start network-accessible with a placeholder key.
            # Ported from openclaw/openclaw#64586.
            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error(
                            "[%s] Refusing to start: API_SERVER_KEY is set to a "
                            "placeholder value. Generate a real secret "
                            "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                            "before exposing the API server on %s.",
                            self.name, self._host,
                        )
                        return False
                except ImportError:
                    pass

            # Port conflict detection — fail fast if port is already in use
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use. Set a different port in config.yaml: platforms.api_server.port', self.name, self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._mark_connected()
            if not self._api_key:
                logger.warning(
                    "[%s] ⚠️  No API key configured (API_SERVER_KEY / platforms.api_server.key). "
                    "All requests will be accepted without authentication. "
                    "Set an API key for production deployments to prevent "
                    "unauthorized access to sessions, responses, and cron jobs.",
                    self.name,
                )
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
