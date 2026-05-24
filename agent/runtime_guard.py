"""Runtime Guard — blocks forbidden patterns at tool/gateway/message level.

Intercepts:
1. Tool calls (in handle_function_call)
2. Terminal commands (in terminal_tool)
3. Gateway messages (in gateway/run.py)

Forbidden patterns are configurable via ``security.forbidden_patterns`` in
config.yaml.  Default set blocks references to legacy Hermes projects and
forbidden ports.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Default forbidden patterns ───────────────────────────────────────

_DEFAULT_PATTERNS: list[dict[str, str]] = [
    {"pattern": r"(?i)\bhermes\s*labs?\b",            "desc": "Hermes Labs reference (legacy)"},
    {"pattern": r"(?i)\bhermesnous\b",                "desc": "HermesNous reference (legacy)"},
    {"pattern": r"(?i)\bhermes[-_]nous\b",            "desc": "Hermes-Nous reference (legacy)"},
    {"pattern": r"(?i)\bhermes[-_]labs?\b",           "desc": "Hermes-Labs reference (legacy)"},
    {"pattern": r"127\.0\.0\.1:7421\b",              "desc": "Forbidden runtime port 7421"},
    {"pattern": r"127\.0\.0\.1:7422\b",              "desc": "Forbidden runtime port 7422"},
    {"pattern": r"localhost:7421\b",                 "desc": "Forbidden runtime port 7421 (localhost)"},
    {"pattern": r"localhost:7422\b",                 "desc": "Forbidden runtime port 7422 (localhost)"},
    {"pattern": r"(?i)\bhermes[-_]mcp\b",            "desc": "hermes-mcp MCP server reference (blocked)"},
    {"pattern": r"(?i)\bnous.*7422\b",               "desc": "Nous port 7422 reference"},
    {"pattern": r"(?i)\blabs?.*7421\b",              "desc": "Labs port 7421 reference"},
]


@dataclass
class GuardPattern:
    """A single forbidden pattern."""
    id: str
    pattern: str
    description: str
    enabled: bool = True
    created_at: str = ""
    source: str = "default"  # "default" | "user"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardResult:
    """Result of a guard check."""
    allowed: bool
    reason: Optional[str] = None
    matched_pattern_id: Optional[str] = None
    matched_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Pattern cache ────────────────────────────────────────────────────

_compiled_patterns: list[tuple[GuardPattern, re.Pattern]] = []
_user_patterns: list[GuardPattern] = []  # runtime-added patterns, survive cache invalidation
_cache_loaded = False


def _compile_pattern(pat_str: str) -> Optional[re.Pattern]:
    """Compile a regex pattern, returning None on failure."""
    try:
        return re.compile(pat_str)
    except re.error as e:
        logger.warning("RuntimeGuard: invalid pattern %r: %s", pat_str, e)
        return None


def _load_patterns() -> list[tuple[GuardPattern, re.Pattern]]:
    """Load and compile all patterns (default + user-config + runtime-added)."""
    global _compiled_patterns, _cache_loaded
    if _cache_loaded:
        return _compiled_patterns

    patterns: list[tuple[GuardPattern, re.Pattern]] = []

    # Default patterns
    for item in _DEFAULT_PATTERNS:
        compiled = _compile_pattern(item["pattern"])
        if compiled is not None:
            gp = GuardPattern(
                id=f"default-{uuid.uuid4().hex[:8]}",
                pattern=item["pattern"],
                description=item["desc"],
                enabled=True,
                source="default",
            )
            patterns.append((gp, compiled))

    # User-configured patterns from config.yaml
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        security_cfg = cfg.get("security", {}) if isinstance(cfg, dict) else {}
        user_patterns = security_cfg.get("forbidden_patterns", [])
        if isinstance(user_patterns, list):
            for entry in user_patterns:
                if isinstance(entry, dict):
                    pat_str = entry.get("pattern", "")
                    desc = entry.get("description", "user-configured forbidden pattern")
                elif isinstance(entry, str):
                    pat_str = entry
                    desc = "user-configured forbidden pattern"
                else:
                    continue
                if not pat_str:
                    continue
                compiled = _compile_pattern(pat_str)
                if compiled is not None:
                    gp = GuardPattern(
                        id=f"user-{uuid.uuid4().hex[:8]}",
                        pattern=pat_str,
                        description=desc,
                        enabled=True,
                        source="user",
                    )
                    patterns.append((gp, compiled))
    except Exception as e:
        logger.debug("RuntimeGuard: failed to load user patterns: %s", e)

    # Runtime-added patterns (survive cache invalidation)
    for gp in _user_patterns:
        compiled = _compile_pattern(gp.pattern)
        if compiled is not None:
            patterns.append((gp, compiled))

    _compiled_patterns = patterns
    _cache_loaded = True
    return patterns


def _invalidate_cache() -> None:
    """Clear the compiled pattern cache (call after config change).
    Runtime-added patterns (_user_patterns) are preserved."""
    global _compiled_patterns, _cache_loaded
    _compiled_patterns = []
    _cache_loaded = False


# ── Public API ───────────────────────────────────────────────────────

class RuntimeGuard:
    """Intercepts tool calls, terminal commands, and gateway messages
    to block forbidden patterns (legacy Hermes projects, forbidden ports)."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def check_tool_call(tool_name: str, args: Optional[Dict[str, Any]] = None) -> GuardResult:
        """Check a tool call against forbidden patterns.

        Checks:
        - tool name
        - all string values in args

        Returns:
            GuardResult with allowed=True/False and reason if blocked.
        """
        # Check tool name
        result = _check_text(tool_name)
        if not result.allowed:
            return result

        # Check args
        if args:
            for key, value in args.items():
                if isinstance(value, str):
                    result = _check_text(value)
                    if not result.allowed:
                        return result
                elif isinstance(value, dict):
                    for sub_val in value.values():
                        if isinstance(sub_val, str):
                            result = _check_text(sub_val)
                            if not result.allowed:
                                return result
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            result = _check_text(item)
                            if not result.allowed:
                                return result

        return GuardResult(allowed=True)

    @staticmethod
    def check_terminal_command(command: str) -> GuardResult:
        """Check a terminal command against forbidden patterns."""
        return _check_text(command)

    @staticmethod
    def check_message(text: str) -> GuardResult:
        """Check a gateway/user message against forbidden patterns."""
        return _check_text(text)

    @staticmethod
    def list_patterns() -> List[Dict[str, Any]]:
        """Return all compiled patterns (for diagnostics)."""
        patterns = _load_patterns()
        return [gp.to_dict() for gp, _ in patterns]

    @staticmethod
    def add_pattern(pattern: str, description: str) -> Optional[str]:
        """Add a user pattern at runtime. Returns pattern ID or None on failure."""
        compiled = _compile_pattern(pattern)
        if compiled is None:
            return None
        _invalidate_cache()
        gp = GuardPattern(
            id=f"user-{uuid.uuid4().hex[:8]}",
            pattern=pattern,
            description=description,
            enabled=True,
            source="user",
        )
        _user_patterns.append(gp)
        return gp.id

    @staticmethod
    def remove_pattern(pattern_id: str) -> bool:
        """Remove a pattern by ID. Returns True if removed."""
        global _compiled_patterns
        before = len(_user_patterns)
        _user_patterns[:] = [gp for gp in _user_patterns if gp.id != pattern_id]
        _compiled_patterns = []
        _cache_loaded = False
        return len(_user_patterns) < before


def _check_text(text: str) -> GuardResult:
    """Core check: scan text against all compiled patterns."""
    if not text:
        return GuardResult(allowed=True)

    patterns = _load_patterns()
    for gp, compiled in patterns:
        if not gp.enabled:
            continue
        if compiled.search(text):
            logger.info(
                "RuntimeGuard: blocked '%s' (pattern: %s — %s)",
                text[:200], gp.description, gp.pattern,
            )
            return GuardResult(
                allowed=False,
                reason=f"Blocked by runtime guard: {gp.description}",
                matched_pattern_id=gp.id,
                matched_pattern=gp.pattern,
            )

    return GuardResult(allowed=True)


def check_guard() -> Tuple[bool, Optional[str]]:
    """Quick guard check for the given text.

    Returns (allowed, reason).
    """
    result = _check_text("")
    return result.allowed, result.reason
