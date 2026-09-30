"""
Self Evolution Plugin — Strategy Injector
===========================================

Injects learned strategy hints into sessions via pre_llm_call hook.

Design reference: Claude Code plugins/learning-output-style/
  - SessionStart hook injects behavioral context automatically
  - Equivalent to CLAUDE.md but more flexible and distributable
  - No core modification needed
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from self_evolution.models import StrategyRule
from self_evolution.rule_engine import StrategyRuleEngine

logger = logging.getLogger(__name__)

_engine = StrategyRuleEngine()

# ── TTL-based cache to avoid reading strategies.json on every LLM call ────

_cached_strategies: list | None = None
_cache_ts: float = 0.0
_CACHE_TTL: float = 60.0  # seconds


def _load_active_strategies() -> list:
    """Load active strategies from strategy store (cached for _CACHE_TTL)."""
    global _cached_strategies, _cache_ts

    now = time.time()
    if _cached_strategies is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cached_strategies

    from self_evolution.strategy_store import StrategyStore

    store = StrategyStore()
    data = store.load()
    rules = data.get("rules", [])

    strategies = []
    for rule_data in rules:
        if not rule_data.get("enabled", True):
            continue
        strategy = StrategyRule.from_dict(rule_data)
        strategies.append(strategy)

    _cached_strategies = strategies
    _cache_ts = now
    return strategies


def invalidate_cache():
    """Invalidate the strategy cache (call after strategy updates)."""
    global _cached_strategies
    _cached_strategies = None


_MAX_INJECT_STRATEGIES = 3     # 最多注入策略数
_MAX_HINT_CHARS = 100          # 注入提示总字符预算
_MAX_SINGLE_HINT = 30          # 单条 hint_text 最大字符数

def inject_hints(kwargs: dict) -> Optional[str]:
    """Pre-llm-call hook: inject learned strategy hints.

    Rules:
      - Strategies without conditions are skipped (must be condition-based).
      - hint_text longer than _MAX_SINGLE_HINT chars are skipped.
      - At most _MAX_INJECT_STRATEGIES hints, total ≤ _MAX_HINT_CHARS.
    """
    strategies = _load_active_strategies()
    if not strategies:
        return None

    # Build context from current session
    context = _build_context(kwargs)

    # Match strategies
    matched = _engine.match_strategies(strategies, context)
    if not matched:
        return None

    # Filter: require conditions and enforce hint length
    eligible = []
    for s in matched:
        if not s.conditions:
            continue  # Skip unconditioned strategies
        if len(s.hint_text.strip()) > _MAX_SINGLE_HINT:
            continue  # Skip overly long hints
        eligible.append(s)

    if not eligible:
        return None

    # Deduplicate by hint_text content
    seen_hints: set[str] = set()
    unique: list = []
    for s in eligible:
        key = s.hint_text.strip().lower()
        if key not in seen_hints:
            seen_hints.add(key)
            unique.append(s)

    # Cap count
    selected = unique[:_MAX_INJECT_STRATEGIES]

    # Format hints within char budget
    return _engine.format_hints(selected, max_chars=_MAX_HINT_CHARS)


def _build_context(kwargs: dict) -> dict:
    """Build matching context from hook kwargs."""
    return {
        "platform": kwargs.get("platform", ""),
        "model": kwargs.get("model", ""),
        "task_type": kwargs.get("task_type", ""),
        "tool_name": kwargs.get("tool_name", ""),
    }
