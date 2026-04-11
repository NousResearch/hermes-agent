"""
Intelligent Task Routing for Hermes-Agent.

Provides rule-based routing of tasks to specific toolsets based on
pattern matching against the goal and context.
"""
import re
import threading
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RoutingRule:
    """A routing rule that matches a pattern and suggests a toolset."""

    pattern: str
    toolset: str
    priority: int = 0

    def __post_init__(self):
        # Compile regex for efficient matching
        self._regex = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, goal: str, context: str) -> bool:
        """Check if this rule matches the goal or context."""
        text = f"{goal} {context}"
        return bool(self._regex.search(text))


class TaskRouter:
    """
    Routes tasks to appropriate toolsets based on rules.

    Thread-safe rule management and routing.
    """

    def __init__(self):
        self._rules: List[RoutingRule] = []
        self._lock = threading.Lock()
        self._enabled = True

    def add_rule(self, pattern: str, toolset: str, priority: int = 0) -> None:
        """
        Add a routing rule.

        Args:
            pattern: Regex pattern to match against goal+context
            toolset: Toolset to route to if pattern matches
            priority: Higher priority rules are evaluated first
        """
        with self._lock:
            rule = RoutingRule(pattern=pattern, toolset=toolset, priority=priority)
            self._rules.append(rule)
            # Sort by priority descending
            self._rules.sort(key=lambda r: r.priority, reverse=True)

    def route(self, goal: str, context: str = "") -> Optional[str]:
        """
        Route a goal+context to a toolset.

        Evaluates rules in priority order (highest first).
        Returns the toolset of the first matching rule, or None if no match.

        Args:
            goal: The task goal string
            context: Additional context string

        Returns:
            Toolset name if a rule matches, None otherwise
        """
        if not self._enabled:
            return None

        with self._lock:
            rules_snapshot = list(self._rules)

        for rule in rules_snapshot:
            if rule.matches(goal, context):
                return rule.toolset

        return None

    def is_enabled(self) -> bool:
        """Check if routing is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable routing."""
        self._enabled = enabled

    def clear_rules(self) -> None:
        """Remove all routing rules."""
        with self._lock:
            self._rules.clear()

    def get_rules(self) -> List[RoutingRule]:
        """Get a snapshot of current rules."""
        with self._lock:
            return list(self._rules)
