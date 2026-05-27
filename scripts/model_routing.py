#!/usr/bin/env python3
"""
HERMES MODEL ROUTING ENGINE
Decisive model selection logic — the "when to use what" brain.

Replaces passive failover with deliberate, cost-aware, capability-matched routing.
"""

import json
import time
import os
from typing import Optional

# Resource guard — checks system readiness before heavy model launches
try:
    from resource_guard import (require_resources as _resource_check,
                            check_requirements as _check_req,
                            log_guard_result as _log_guard,
                            GUARDED_MODELS, _is_real_safari, find_processes)
    _HAS_RESOURCE_GUARD = True
except ImportError:
    _HAS_RESOURCE_GUARD = False

# ─── MODEL REGISTRY ────────────────────────────────────────────

MODELS = {
    "mac-ollama:qwen3:8b": {
        "provider": "mac-ollama",
        "model": "qwen3:8b",
        "context_length": 32768,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 50,
        "strengths": ["fast", "local", "cheap", "good_for_simple"],
        "max_tokens": 4096,
    },
    "mac-ollama:qwen3-coder:30b-a3b-q4_k_M": {
        "provider": "mac-ollama",
        "model": "qwen3-coder:30b-a3b-q4_k_M",
        "context_length": 16384,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 800,
        "strengths": ["code_generation", "reasoning", "review", "long_context", "best_local"],
        "max_tokens": 8192,
    },
    "linux-ollama:qwen3:8b": {
        "provider": "linux-ollama",
        "model": "qwen3:8b",
        "context_length": 32768,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 150,
        "strengths": ["fast", "efficient", "good_context_density", "local"],
        "max_tokens": 8192,
    },
    "deepseek:deepseek-v4-flash": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "context_length": 1000000,  # 1M token context on DeepSeek direct API
        "cost_per_1k_input": 0.14,
        "cost_per_1k_output": 0.28,
        "latency_estimate_ms": 800,
        "strengths": ["reasoning", "code_review", "debugging", "fast_cloud"],
        "max_tokens": 8192,
    },
    "deepseek:deepseek-v4-pro": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "context_length": 32768,
        "cost_per_1k_input": 0.28,
        "cost_per_1k_output": 0.56,
        "latency_estimate_ms": 2000,
        "strengths": ["deep_reasoning", "complex_analysis", "research"],
        "max_tokens": 4096,
    },
    "x-ai:grok-4.20-reasoning": {
        "provider": "x-ai",
        "model": "grok-4.20-reasoning",
        "context_length": 16384,
        "cost_per_1k_input": 1.25,
        "cost_per_1k_output": 10.0,
        "latency_estimate_ms": 1500,
        "strengths": ["creative", "synthesis", "architecture", "strategic"],
        "max_tokens": 4096,
    },
    "openrouter:grok-4.3": {
        "provider": "openrouter",
        "model": "grok-4.3",
        "context_length": 1000000,
        "cost_per_1k_input": 0.00000125,
        "cost_per_1k_output": 0.0000025,
        "latency_estimate_ms": 3000,
        "strengths": ["creative", "reasoning", "code_review", "long_context", "best_value"],
        "max_tokens": 32768,
    },
    "anthropic:claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "context_length": 200000,  # 200K token context on Claude 4 Sonnet
        "cost_per_1k_input": 3.0,
        "cost_per_1k_output": 15.0,
        "latency_estimate_ms": 4000,
        "strengths": ["deep_reasoning", "plot_analysis", "character_work", "long_context", "editorial_review"],
        "max_tokens": 8192,
    },
    "anthropic:claude-opus-4-7": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "context_length": 100000,
        "cost_per_1k_input": 15.0,
        "cost_per_1k_output": 75.0,
        "latency_estimate_ms": 6000,
        "strengths": ["deepest_reasoning", "complex_analysis", "research", "premium_editorial"],
        "max_tokens": 4096,
    },
    "openrouter:ring-2.6-1t": {
        "provider": "openrouter",
        "model": "ring-2.6-1t",
        "context_length": 1000000,  # 1M token context via OpenRouter
        "cost_per_1k_input": 0.88,
        "cost_per_1k_output": 0.88,
        "latency_estimate_ms": 2000,
        "strengths": ["quality_gate", "verification", "final_review"],
        "max_tokens": 4096,
    },
    "moonshot:kimi-v1-8k": {
        "provider": "moonshot",
        "model": "moonshot-v1-8k",
        "context_length": 8192,
        "cost_per_1k_input": 0.0,  # direct API, billed via moonshot.cn credits
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 3000,
        "strengths": ["aesthetic", "creative_judgment", "visual_reasoning", "multilingual"],
        "max_tokens": 2048,
        "direct": True,  # uses kimi_client.py, NOT openrouter proxy
    },
    "ollama-provider:qwen3-coder:30b-a3b-q4_k_M": {
        "provider": "ollama-provider",
        "model": "qwen3-coder:30b-a3b-q4_k_M",
        "context_length": 16384,
        "cost_per_1k_input": 0.0,  # free tier / included in subscription
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 2000,
        "strengths": ["code_generation", "reasoning", "review", "cloud_backup_local_30b"],
        "max_tokens": 8192,
    },
    "ollama-provider:llama3.1-70b": {
        "provider": "ollama-provider",
        "model": "llama3.1-70b",
        "context_length": 131072,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 3000,
        "strengths": ["reasoning", "long_context", "analysis", "strong_general"],
        "max_tokens": 4096,
    },
    "ollama-provider:llama3.1-8b": {
        "provider": "ollama-provider",
        "model": "llama3.1-8b",
        "context_length": 32768,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "latency_estimate_ms": 800,
        "strengths": ["fast", "efficient", "cheap", "good_context_density"],
        "max_tokens": 8192,
    },
}

# ─── HEALTH STATE ───────────────────────────────────────────────

HEALTH = {}  # provider/model -> {"ok": bool, "consecutive_failures": int, "last_check": float}

CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300  # 5 minutes


def check_health(provider: str, model: str) -> bool:
    """Get current health status for a model."""
    key = f"{provider}:{model}"
    if key not in HEALTH:
        return True
    h = HEALTH[key]
    if not h["ok"]:
        if time.time() - h["last_check"] > CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            # Cooldown expired, retry
            h["ok"] = True
            h["consecutive_failures"] = 0
            return True
    return h["ok"]


def report_health(provider: str, model: str, success: bool, latency_ms: int = 0):
    """Update health state after a call."""
    key = f"{provider}:{model}"
    if key not in HEALTH:
        HEALTH[key] = {"ok": True, "consecutive_failures": 0, "last_check": time.time()}

    h = HEALTH[key]
    h["last_check"] = time.time()

    if success:
        h["consecutive_failures"] = 0
        h["ok"] = True
        h["avg_latency_ms"] = latency_ms
    else:
        h["consecutive_failures"] += 1
        if h["consecutive_failures"] >= CIRCUIT_BREAKER_THRESHOLD:
            h["ok"] = False


# ─── TASK CLASSIFICATION ───────────────────────────────────────

TASK_PATTERNS = {
    # NOTE: Order matters — longer/more specific patterns first.
    # "write" alone → creative, but "write function"/"python"/"parse json" → code.
    "code_generation": [
        "write code", "implement", "create function", "build", "code",
        "program", "script", "generate", "refactor", "fix bug", "debug",
        "python", "javascript", "typescript", "java", "function", "method",
        "class", "api endpoint", "parse json", "parse xml", "regex",
        "algorithm", "data structure", "module", "package", "import",
    ],
    "reasoning": [
        "reason", "analyze", "explain why", "how does", "why",
        "compare", "evaluate", "assess", "think through",
    ],
    "research": [
        "research", "find", "search", "look up", "what is",
        "tell me about", "summarize", "overview", "background",
    ],
    "creative": [
        "write", "create", "design", "draft", "compose",
        "imagine", "brainstorm", "creative", "story", "poem",
    ],
    "review": [
        "review", "check", "verify", "audit", "critique",
        "improve", "feedback", "quality",
        "code review", "review code", "code for bugs", "looks correct",
    ],
    "tool_use": [
        "run", "execute", "terminal", "file", "filesystem",
        "command", "shell", "git", "search", "download",
    ],
}


def classify_task(user_message: str) -> str:
    """Classify what kind of task this is."""
    lower = user_message.lower()
    scores = {}

    for category, patterns in TASK_PATTERNS.items():
        # Weight longer/more-specific patterns higher than short generic ones
        score = 0
        for p in patterns:
            if p in lower:
                # Multi-word patterns get 2x weight vs single-word
                weight = 2 if " " in p else 1
                score += weight
        if score > 0:
            scores[category] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)


def get_task_size(user_message: str) -> str:
    """Estimate task size: quick (<2min), moderate (<15min), large (>15min)."""
    length = len(user_message)
    if length < 200:
        return "quick"
    elif length < 1000:
        return "moderate"
    else:
        return "large"


# ─── ROUTING DECISION ──────────────────────────────────────────

# Preference order: local first, then cloud by capability match
PREFERENCE_ORDER = [
    "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
    "mac-ollama:qwen3:8b",
    "linux-ollama:qwen3:8b",
    "ollama-provider:qwen3-coder:30b-a3b-q4_k_M",  # cloud backup for local 30B
    "ollama-provider:llama3.1-70b",
    "deepseek:deepseek-v4-flash",
    "deepseek:deepseek-v4-pro",
    "x-ai:grok-4.20-reasoning",
    "openrouter:grok-4.3",
    "moonshot:kimi-v1-8k",
    "openrouter:ring-2.6-1t",
]

# Internal-use preference: best models first for consult/merge/critique
INTERNAL_PREFERENCE_ORDER = [
    "openrouter:ring-2.6-1t",           # quality gate — always available for final review
    "x-ai:grok-4.20-reasoning",
    "openrouter:grok-4.3",         # board member — creative/lateral synthesis
    "anthropic:claude-opus-4-7",        # board member — deepest reasoning, premium editorial
    "anthropic:claude-sonnet-4-6",      # board member — plot/character analysis, long context
    "deepseek:deepseek-v4-pro",         # premium cloud deep reasoning
    "deepseek:deepseek-v4-flash",       # board member — fast cloud
    "moonshot:kimi-v1-8k",              # board member — aesthetic/creative judgment
    "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",  # local 30B coder (free, fastest turnaround)
    "ollama-provider:qwen3-coder:30b-a3b-q4_k_M",  # cloud 30B backup
    "ollama-provider:llama3.1-70b",     # cloud 70B generalist
    "linux-ollama:qwen3:8b",            # linux local fallback
    "mac-ollama:qwen3:8b",              # mac local fallback
    "ollama-provider:llama3.1-8b",      # cloud 8B fallback
]

# Best model for each task category
CATEGORY_BEST = {
    "code_generation": [
        "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
        "ollama-provider:qwen3-coder:30b-a3b-q4_k_M",  # cloud backup
        "deepseek:deepseek-v4-flash",
        "mac-ollama:qwen3:8b",
    ],
    "reasoning": [
        "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
        "ollama-provider:llama3.1-70b",
        "deepseek:deepseek-v4-pro",
        "x-ai:grok-4.20-reasoning",
    "openrouter:grok-4.3",
    ],
    "research": [
        "ollama-provider:llama3.1-70b",
        "linux-ollama:qwen3:8b",
        "deepseek:deepseek-v4-flash",
    ],
    "creative": [
        "moonshot:kimi-v1-8k",
        "x-ai:grok-4.20-reasoning",
        "openrouter:grok-4.3",
        "ollama-provider:llama3.1-8b",
        "mac-ollama:qwen3:8b",
    ],
    "review": [
        # Best coder (free local) — first choice
        "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
        "ollama-provider:qwen3-coder:30b-a3b-q4_k_M",  # cloud backup
        # Cloud coder alternatives
        "deepseek:deepseek-v4-pro",
        "ollama-provider:llama3.1-70b",
        # Board members — consult chain (Claude + Kimi + Grok + Ring)
        "anthropic:claude-sonnet-4-6",    # board — plot/character/editorial
        "moonshot:kimi-v1-8k",            # board — aesthetic/creative judgment
        "x-ai:grok-4.20-reasoning",
        "openrouter:grok-4.3",       # board — creative/lateral synthesis
        "openrouter:ring-2.6-1t",         # quality gate — ALWAYS LAST, mandatory
    ],
    "tool_use": [
        "mac-ollama:qwen3:8b",
        "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
    ],
    "general": [
        "mac-ollama:qwen3:8b",
        "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
    ],
}

# Reasoning models that get "think harder" treatment
REASONING_MODELS = {
    "deepseek:deepseek-v4-pro",
    "anthropic:claude-opus-4-7",
    "anthropic:claude-sonnet-4-6",
    "x-ai:grok-4.20-reasoning",
    "openrouter:grok-4.3",
    "openrouter:ring-2.6-1t",
    "mac-ollama:qwen3-coder:30b-a3b-q4_k_M",
}


def select_model(task_category: str, prompt_text: str,
                 history_length: int = 0, budget_usd: float = 1.0,
                 force_provider: str = None, force_model: str = None,
                 is_internal: bool = False) -> dict:
    """
    Main routing decision. Returns model config dict.

    Decision logic:
    1. If force_provider/force_model specified → bypass classification, use directly
    2. Task category → best candidates
    3. Filter by health (circuit breaker)
    4. Filter by budget (internal calls skip budget checks)
    5. Check context window fits history
    6. Select best available: internal→premium-first, external→cheap-first
    """
    # Forced override: bypass all classification and health checks
    if force_provider and force_model:
        key = f"{force_provider}:{force_model}"
        if key in MODELS:
            return MODELS[key]
        # Fall back to provider matching
        for mk, cfg in MODELS.items():
            if cfg["provider"] == force_provider:
                return cfg

    # Internal calls: skip budget constraints, prefer premium models
    effective_budget = 9999.0 if is_internal else budget_usd

    candidates = list(CATEGORY_BEST.get(task_category, CATEGORY_BEST["general"]))

    # Add fallback chain for resilience
    fallback_order = INTERNAL_PREFERENCE_ORDER if is_internal else PREFERENCE_ORDER
    candidates = candidates + [m for m in fallback_order if m not in candidates]

    suitable = []
    for model_key in candidates:
        if model_key not in MODELS:
            continue
        cfg = MODELS[model_key]

        # Health check
        if not check_health(cfg["provider"], cfg["model"]):
            continue

# Budget check (input cost only, we don't know output length yet)
        # Skip cost for local models (free), internal calls, or Ring (orchestrator — never budget-gated)
        ring_models = {"openrouter:ring-2.6-1t", "openrouter:ring-2"}
        is_ring = model_key in ring_models
        if cfg["cost_per_1k_input"] > 0 and effective_budget <= 0 and not is_ring:
            continue

        # Context window check
        if history_length > cfg["context_length"] * 0.7:
            continue  # Not enough room

        suitable.append(cfg)

    if not suitable:
        # Emergency fallback: use any local model
        for model_key in PREFERENCE_ORDER:
            cfg = MODELS.get(model_key)
            if cfg and "ollama" in cfg["provider"]:
                if check_health(cfg["provider"], cfg["model"]):
                    return cfg
        # Last resort: use whatever has the longest context
        return MODELS["mac-ollama:qwen3:8b"]

    suitable.sort(key=lambda c: (
        0 if c in [m for m in CATEGORY_BEST.get(task_category, [])
                   if m in MODELS] else 1,  # category-match first
        0 if (is_internal and "ollama" not in c["provider"])
             or (not is_internal and "ollama" in c["provider"]) else 1,
        # For internal: premium models first; for external: local/cheap first
        -(c["cost_per_1k_input"]) if is_internal else c["cost_per_1k_input"],
    ))

    # Resource guard: check top candidates for heavy-model readiness
    if _HAS_RESOURCE_GUARD:
        final = _apply_resource_guard(suitable)
        if final:
            return final
        # Guard rejected all — fall through to emergency fallback

    return suitable[0]


def _apply_resource_guard(candidates: list) -> Optional[dict]:
    """
    Check candidates from best to worst. Return the first one that
    either isn't guarded or passes the resource check.
    Uses check_requirements (silent) instead of require_resources (noisy).
    """
    for cfg in candidates:
        model_key = cfg["model"]
        if model_key in GUARDED_MODELS:
            result = _check_silent(cfg["provider"] + ":" + model_key)
            if result["allowed"]:
                return cfg
            # Guard failed — log to file, skip this model, try next
            _log_guard(result, cfg["provider"] + ":" + model_key)
        else:
            # Not a guarded model — safe to use
            return cfg
    return None  # all guarded and all failed


def _check_silent(model_name: str) -> dict:
    """Silent guard check — no stdout, just the result dict."""
    return _check_req(model_name)


def should_think_harder(model_key: str) -> bool:
    """Check if this model supports reasoning_effort adjustment."""
    return model_key in REASONING_MODELS


if __name__ == "__main__":
    print("Model routing engine self-test...")

    # Test classifications
    tests = [
        ("Write a Python function to parse JSON", "code_generation"),
        ("Analyze why this algorithm is O(n²)", "reasoning"),
        ("Research the history of machine learning", "research"),
        ("Write a poem about artificial intelligence", "creative"),
        ("Review this code for bugs", "review"),
        ("Run ls -la in /tmp", "tool_use"),
        ("What's the weather like?", "general"),
    ]

    all_passed = True
    for prompt, expected in tests:
        result = classify_task(prompt)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} '{prompt[:40]}...' → {result} (expected {expected})")

    # Test model selection
    print("\nModel selection tests:")
    for cat in ["code_generation", "reasoning", "research", "creative", "review"]:
        selected = select_model(cat, "test prompt", budget_usd=5.0)
        print(f"  {cat}: {selected['provider']}/{selected['model']} (${selected['cost_per_1k_input']}/1K in)")

    print(f"\nAll tests {'passed' if all_passed else 'FAILED'}")
    print("Model routing engine ready. ✅")