"""
Self Evolution Plugin — Rule Engine (Strategy Matching)
========================================================

Conditional strategy matching engine.

Design reference: Claude Code plugins/hookify/core/rule_engine.py
  - LRU-cached regex compilation (max 128)
  - Multiple operators: regex_match, contains, equals, not_contains
  - All conditions must match (AND logic)
  - Severity levels: high, medium, low
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from self_evolution.models import StrategyRule, StrategyCondition


@lru_cache(maxsize=128)
def _compile_pattern(pattern: str) -> re.Pattern:
    """Compile and cache a regex pattern."""
    return re.compile(pattern, re.IGNORECASE)


class StrategyRuleEngine:
    """Evaluate strategy rules against session context."""

    def match_strategies(
        self,
        strategies: List[StrategyRule],
        context: Dict[str, Any],
    ) -> List[StrategyRule]:
        """Return strategies whose conditions match the context."""
        matched = []
        for strategy in strategies:
            if not strategy.enabled:
                continue
            if not strategy.conditions:
                # No conditions = always match
                matched.append(strategy)
                continue
            if self._conditions_match(strategy.conditions, context):
                matched.append(strategy)
        return matched

    def _conditions_match(
        self,
        conditions: List[StrategyCondition],
        context: Dict[str, Any],
    ) -> bool:
        """All conditions must match (AND logic)."""
        for cond in conditions:
            field_value = str(context.get(cond.field, ""))
            if not self._check_operator(cond.operator, cond.pattern, field_value):
                return False
        return True

    def _check_operator(self, op: str, pattern: str, value: str) -> bool:
        """Apply operator check."""
        try:
            if op == "regex_match":
                return bool(_compile_pattern(pattern).search(value))
            elif op == "contains":
                return pattern in value
            elif op == "equals":
                return pattern == value
            elif op == "not_contains":
                return pattern not in value
            elif op == "starts_with":
                return value.startswith(pattern)
            elif op == "ends_with":
                return value.endswith(pattern)
            else:
                return False
        except re.error:
            return False

    def format_hints(self, strategies: List[StrategyRule], max_chars: int = 0) -> str:
        """Format matched strategies into a system hint string.

        Args:
            max_chars: If > 0, truncate total output to this many characters.
        """
        if not strategies:
            return ""

        lines = ["[自我进化策略提示]"]
        for s in strategies:
            type_prefix = {"hint": "💡", "avoid": "⚠️", "prefer": "✅"}.get(
                s.strategy_type, "💡"
            )
            line = f"{type_prefix} {s.name}: {s.hint_text}"
            if max_chars and len("\n".join(lines)) + len(line) > max_chars:
                break
            lines.append(line)

        return "\n".join(lines)
