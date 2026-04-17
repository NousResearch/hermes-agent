#!/usr/bin/env python3
"""
Self-Learning Module - Failure Analysis & Skill Evolution

Analyzes tool call failures to extract lessons, detect recurring patterns,
and suggest skill improvements. Part of the Vibe Coding enhancements for
Hermes Agent.

Features:
- Tool call recording (success and failure)
- Failure pattern detection (same tool, similar error, 3+ occurrences)
- Lesson extraction with context preservation
- Skill update suggestions (create new or patch existing)
- Memory integration for persistent learning
- Persistent storage of failure patterns

Classes:
    SelfLearningTracker  -- Main tracker class (also available as SelfLearner alias)

Usage:
    from agent.self_learning import SelfLearningTracker

    tracker = SelfLearningTracker()
    tracker.record_tool_call("terminal", {"command": "npm install"}, result, 2.5, is_error=True)
    tracker.analyze_failure("terminal", {"command": "npm install"}, '{"error": "ENOENT..."}')
    suggestions = tracker.suggest_skill_update()
"""

import json
import hashlib
import logging
import os
import re
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level shared state (backward-compat for convenience functions)
# =============================================================================

# Failure tracking storage (per-session, in-memory)
# Key: failure_signature, Value: {count, first_seen, last_seen, examples, error_type}
_failure_history: Dict[str, Dict[str, Any]] = {}

# All tool call history (recent entries, capped)
_tool_call_history: List[Dict[str, Any]] = []
_MAX_CALL_HISTORY = 500

# Learned lessons (persisted via memory tool and disk)
_learned_lessons: List[Dict[str, Any]] = []

# Suggested skill updates
_skill_suggestions: List[Dict[str, Any]] = []

# Thread safety
_state_lock = threading.Lock()


# =============================================================================
# Helper functions
# =============================================================================

def _get_hermes_home() -> Path:
    """Get HERMES_HOME directory, respecting profile override."""
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _get_learning_dir() -> Path:
    """Get the learning data directory."""
    d = _get_hermes_home() / "learning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compute_failure_signature(tool_name: str, error_message: str) -> str:
    """
    Compute a signature for a failure to detect patterns.
    Uses tool name + normalized error message.
    """
    normalized = error_message.lower()
    # Remove file paths
    normalized = re.sub(r'/[^\s]+', '<path>', normalized)
    # Remove numbers
    normalized = re.sub(r'\b\d+\b', 'N', normalized)
    # Remove quoted strings
    normalized = re.sub(r"'[^']*'", '<string>', normalized)
    normalized = re.sub(r'"[^"]*"', '<string>', normalized)

    signature_input = f"{tool_name}:{normalized}"
    return hashlib.md5(signature_input.encode()).hexdigest()[:12]


def _extract_error_type(error_message: str) -> str:
    """Categorize error type from message."""
    error_lower = error_message.lower()

    error_patterns = [
        ("permission_denied", ["permission denied", "eacces", "access is denied"]),
        ("not_found", ["no such file", "not found", "does not exist", "enoent"]),
        ("timeout", ["timeout", "timed out", "deadline exceeded"]),
        ("network", ["network", "connection refused", "econnrefused", "socket"]),
        ("syntax", ["syntax error", "parse error", "invalid syntax"]),
        ("dependency", ["module not found", "import error", "no module named", "cannot find module"]),
        ("memory", ["out of memory", "memory error", "oom"]),
        ("auth", ["unauthorized", "authentication failed", "invalid api key", "forbidden"]),
        ("rate_limit", ["rate limit", "too many requests", "429"]),
        ("validation", ["validation error", "invalid argument", "type error", "value error"]),
        ("git", ["merge conflict", "not a git repository", "fatal:"]),
    ]

    for error_type, patterns in error_patterns:
        if any(p in error_lower for p in patterns):
            return error_type

    return "unknown"


def _detect_error_in_result(result: str) -> tuple[bool, str]:
    """
    Detect if a tool result string contains an error.
    Mirrors the logic from display._detect_tool_failure for the tracker.

    Returns: (is_error, error_summary)
    """
    if not result:
        return False, ""

    # Check for JSON error patterns
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            if "error" in data:
                return True, str(data["error"])
            if data.get("success") is False and "error" in data:
                return True, str(data["error"])
            # Terminal tool: check exit_code
            if data.get("exit_code") is not None and data.get("exit_code") != 0:
                return True, f"exit code {data['exit_code']}"
    except (json.JSONDecodeError, ValueError):
        pass

    # Heuristic: string starts with "Error"
    if result.startswith("Error"):
        return True, result[:200]

    # Heuristic: contains error indicators in first 500 chars
    lower_prefix = result[:500].lower()
    if '"error"' in lower_prefix or '"failed"' in lower_prefix:
        return True, result[:200]

    return False, ""


def _suggest_skill_action(tool_name: str, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate skill update suggestion based on failure pattern."""
    suggestions = {
        "permission_denied": {
            "pitfall": "Permission errors often occur when running commands without proper privileges",
            "fix": "Consider using sudo for system commands, or check file permissions with ls -la",
        },
        "not_found": {
            "pitfall": "File or command not found - verify paths exist before operating",
            "fix": "Use search_files or read_file to verify file exists before terminal commands",
        },
        "timeout": {
            "pitfall": "Long-running operations may timeout with default limits",
            "fix": "Increase timeout parameter for slow operations like builds or deployments",
        },
        "network": {
            "pitfall": "Network operations may fail due to connectivity issues",
            "fix": "Add retry logic or check connectivity before network-dependent operations",
        },
        "dependency": {
            "pitfall": "Missing dependencies can cause import/module errors",
            "fix": "Verify environment setup, check requirements.txt or package.json",
        },
        "auth": {
            "pitfall": "Authentication failures indicate missing or invalid credentials",
            "fix": "Check API keys in .env file, use environment variable validation",
        },
        "rate_limit": {
            "pitfall": "API rate limits can interrupt workflows",
            "fix": "Add exponential backoff, batch requests, or wait before retrying",
        },
    }

    pattern = suggestions.get(error_type, {
        "pitfall": f"Unexpected error type: {error_type}",
        "fix": "Investigate the error message and add appropriate error handling",
    })

    return {
        "tool": tool_name,
        "error_type": error_type,
        "pitfall": pattern["pitfall"],
        "suggested_fix": pattern["fix"],
        "context": context,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# SelfLearningTracker
# =============================================================================

class SelfLearningTracker:
    """
    Tracks tool call history, analyzes failures, detects recurring patterns,
    and generates skill update suggestions.

    Stored on the AIAgent instance as self.self_learning_tracker.

    Integration in run_agent.py:
      1. After every tool call, invoke:
         tracker.record_tool_call(tool_name, args, result, duration, is_error)
      2. If is_error is True, also invoke:
         tracker.analyze_failure(tool_name, args, result)
      3. Periodically (or after N failures), check:
         tracker.suggest_skill_update()
    """

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or _get_learning_dir()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_data()

    # -- Persistence ----------------------------------------------------------

    def _load_persisted_data(self):
        """Load previously persisted learning data from disk."""
        # Load lessons
        lessons_file = self.persistence_dir / "lessons.json"
        if lessons_file.exists():
            try:
                with open(lessons_file, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        _learned_lessons.clear()
                        _learned_lessons.extend(loaded)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug("Failed to load lessons: %s", e)

        # Load failure patterns (only entries not already in memory)
        patterns_file = self.persistence_dir / "failure_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for sig, data in loaded.items():
                            # Only load patterns not already tracked in memory
                            if sig not in _failure_history:
                                data.setdefault("examples", [])
                                data.setdefault("count", 0)
                                data.setdefault("first_seen", "")
                                data.setdefault("last_seen", "")
                                data.setdefault("error_type", "unknown")
                                data.setdefault("tool_name", "")
                                _failure_history[sig] = data
            except (json.JSONDecodeError, IOError) as e:
                logger.debug("Failed to load failure patterns: %s", e)

    def _persist_data(self):
        """Persist learning data to disk."""
        self._persist_lessons()
        self._persist_failure_patterns()

    def _persist_lessons(self):
        lessons_file = self.persistence_dir / "lessons.json"
        try:
            with open(lessons_file, 'w') as f:
                json.dump(_learned_lessons, f, indent=2)
        except IOError as e:
            logger.debug("Failed to persist lessons: %s", e)

    def _persist_failure_patterns(self):
        patterns_file = self.persistence_dir / "failure_patterns.json"
        try:
            # Strip large example data before persisting to keep file small
            compact = {}
            for sig, data in _failure_history.items():
                compact[sig] = {
                    "count": data["count"],
                    "first_seen": data.get("first_seen", ""),
                    "last_seen": data.get("last_seen", ""),
                    "error_type": data.get("error_type", "unknown"),
                    "tool_name": data.get("tool_name", ""),
                    "is_recurring": data["count"] >= 3,
                }
            with open(patterns_file, 'w') as f:
                json.dump(compact, f, indent=2)
        except IOError as e:
            logger.debug("Failed to persist failure patterns: %s", e)

    # -- Tool call recording --------------------------------------------------

    def record_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: str,
        duration: float,
        is_error: bool,
    ) -> Dict[str, Any]:
        """
        Record a tool call outcome (success or failure).

        This should be called after every tool call in run_agent.py, right
        after _detect_tool_failure() determines the result status.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: The result string returned by the tool
            duration: Execution time in seconds
            is_error: Whether the result indicates a failure

        Returns:
            Summary dict of what was recorded
        """
        with _state_lock:
            entry = {
                "tool_name": tool_name,
                "args": args,
                "duration": round(duration, 3),
                "is_error": is_error,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Extract error info if present
            if is_error:
                has_error, error_summary = _detect_error_in_result(result)
                entry["error_summary"] = error_summary if has_error else result[:200]
                entry["error_signature"] = _compute_failure_signature(
                    tool_name, entry.get("error_summary", result[:200])
                )
                entry["error_type"] = _extract_error_type(
                    entry.get("error_summary", "")
                )

            _tool_call_history.append(entry)

            # Cap history
            if len(_tool_call_history) > _MAX_CALL_HISTORY:
                del _tool_call_history[:len(_tool_call_history) - _MAX_CALL_HISTORY]

            return {
                "recorded": True,
                "tool_name": tool_name,
                "is_error": is_error,
                "history_length": len(_tool_call_history),
            }

    # -- Failure analysis -----------------------------------------------------

    def analyze_failure(
        self,
        tool_name: str,
        args: Dict[str, Any],
        error_result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a tool call failure, track patterns, and extract lessons.

        Should be called when record_tool_call reports is_error=True.

        Args:
            tool_name: Name of the tool that failed
            args: Arguments passed to the tool
            error_result: The error result string from the tool
            context: Additional context (cwd, session_id, etc.)

        Returns:
            Analysis result with pattern detection and suggestions
        """
        context = context or {}

        with _state_lock:
            signature = _compute_failure_signature(tool_name, error_result)
            error_type = _extract_error_type(error_result)

            # Track failure
            if signature not in _failure_history:
                _failure_history[signature] = {
                    "count": 0,
                    "first_seen": datetime.utcnow().isoformat(),
                    "examples": [],
                    "tool_name": tool_name,
                    "error_type": error_type,
                }

            _failure_history[signature]["count"] += 1
            _failure_history[signature]["last_seen"] = datetime.utcnow().isoformat()
            _failure_history[signature]["error_type"] = error_type
            _failure_history[signature]["tool_name"] = tool_name

            # Store example (keep last 3)
            example = {
                "args": args,
                "error_message": error_result[:500],
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
            }
            examples_list = _failure_history[signature].setdefault("examples", [])
            examples_list.append(example)
            if len(examples_list) > 3:
                examples_list.pop(0)

            # Detect pattern (3+ occurrences)
            is_pattern = _failure_history[signature]["count"] >= 3

            result = {
                "signature": signature,
                "tool_name": tool_name,
                "error_type": error_type,
                "occurrence_count": _failure_history[signature]["count"],
                "is_recurring_pattern": is_pattern,
                "first_seen": _failure_history[signature]["first_seen"],
                "last_seen": _failure_history[signature]["last_seen"],
            }

            # Generate skill suggestion
            suggestion = _suggest_skill_action(tool_name, error_type, context)
            result["suggestion"] = suggestion

            # If recurring pattern, create a lesson
            if is_pattern:
                lesson = self._create_lesson(tool_name, error_type, error_result, context)
                result["lesson"] = lesson

            # Persist updated data
            self._persist_data()

            return result

    def _create_lesson(
        self,
        tool_name: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a learned lesson from recurring failure."""
        lesson = {
            "id": f"lesson-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "tool": tool_name,
            "error_type": error_type,
            "summary": f"When using {tool_name}, avoid {error_type} errors by checking preconditions",
            "error_pattern": error_message[:200],
            "context": context,
            "learned_at": datetime.utcnow().isoformat(),
        }

        _learned_lessons.append(lesson)
        return lesson

    # -- Skill suggestions ----------------------------------------------------

    def suggest_skill_update(
        self,
        skill_name: Optional[str] = None,
        failure_pattern: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a skill update suggestion based on failure patterns.

        Args:
            skill_name: Existing skill to update (None for new skill)
            failure_pattern: Specific failure pattern to address (uses most frequent if omitted)

        Returns:
            Skill update suggestion dict
        """
        with _state_lock:
            if not failure_pattern:
                # Find most frequent recurring pattern
                recurring = {
                    sig: data for sig, data in _failure_history.items()
                    if data["count"] >= 3
                }
                if not recurring:
                    return {"status": "no_recurring_patterns", "message": "No recurring failure patterns detected (need 3+ occurrences)"}

                most_frequent_sig = max(recurring, key=lambda s: recurring[s]["count"])
                most_frequent = recurring[most_frequent_sig]

                examples = most_frequent.get("examples", [])
                if not examples:
                    return {"status": "no_examples", "message": "No failure examples available for most frequent pattern"}

                failure_pattern = examples[-1]
                failure_pattern["tool"] = most_frequent.get("tool_name", "unknown")

            tool_name = failure_pattern.get("tool", "unknown")
            error_message = failure_pattern.get("error_message", "")
            error_type = _extract_error_type(error_message)

            suggestion = {
                "action": "patch" if skill_name else "create",
                "skill_name": skill_name or f"{tool_name}-error-handling",
                "content": {
                    "pitfall": f"{error_type} errors can occur when using {tool_name}",
                    "resolution": _suggest_skill_action(tool_name, error_type, {}).get("suggested_fix", ""),
                    "example_error": error_message[:100],
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            _skill_suggestions.append(suggestion)
            return suggestion

    # -- Query methods --------------------------------------------------------

    def get_failure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all known failure patterns with their occurrence counts.

        Returns:
            Dict mapping signature -> pattern data
        """
        with _state_lock:
            return {
                sig: {
                    "count": data["count"],
                    "first_seen": data.get("first_seen", ""),
                    "last_seen": data.get("last_seen", ""),
                    "error_type": data.get("error_type", "unknown"),
                    "tool_name": data.get("tool_name", ""),
                    "is_recurring": data["count"] >= 3,
                    "example_count": len(data.get("examples", [])),
                }
                for sig, data in _failure_history.items()
            }

    def get_failure_history(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get failure history, optionally filtered by tool.

        Args:
            tool_name: Filter to specific tool (optional)

        Returns:
            List of failure records sorted by count descending
        """
        with _state_lock:
            results = []
            for signature, data in _failure_history.items():
                examples = data.get("examples", [])
                record = {
                    "signature": signature,
                    "count": data["count"],
                    "first_seen": data.get("first_seen", ""),
                    "last_seen": data.get("last_seen", ""),
                    "error_type": data.get("error_type", "unknown"),
                    "tool_name": data.get("tool_name", ""),
                }
                if examples:
                    record["last_example"] = examples[-1]

                if tool_name:
                    if data.get("tool_name") == tool_name:
                        results.append(record)
                else:
                    results.append(record)

            return sorted(results, key=lambda x: x["count"], reverse=True)

    def get_recurring_failures(self) -> List[Dict[str, Any]]:
        """Get failures that have occurred 3+ times."""
        return [f for f in self.get_failure_history() if f["count"] >= 3]

    def get_learned_lessons(self) -> List[Dict[str, Any]]:
        """Get all learned lessons."""
        with _state_lock:
            return _learned_lessons.copy()

    def get_skill_suggestions(self) -> List[Dict[str, Any]]:
        """Get pending skill update suggestions."""
        with _state_lock:
            return _skill_suggestions.copy()

    def get_tool_call_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics about tool calls.

        Returns:
            Dict with total_calls, error_count, error_rate, top_errors, etc.
        """
        with _state_lock:
            total = len(_tool_call_history)
            errors = sum(1 for c in _tool_call_history if c.get("is_error"))

            # Error breakdown by tool
            error_by_tool: Dict[str, int] = {}
            for c in _tool_call_history:
                if c.get("is_error"):
                    tool = c.get("tool_name", "unknown")
                    error_by_tool[tool] = error_by_tool.get(tool, 0) + 1

            # Top error types
            error_by_type: Dict[str, int] = {}
            for c in _tool_call_history:
                if c.get("is_error") and "error_type" in c:
                    et = c["error_type"]
                    error_by_type[et] = error_by_type.get(et, 0) + 1

            return {
                "total_calls": total,
                "error_count": errors,
                "error_rate": round(errors / total, 4) if total > 0 else 0,
                "errors_by_tool": dict(sorted(error_by_tool.items(), key=lambda x: x[1], reverse=True)),
                "errors_by_type": dict(sorted(error_by_type.items(), key=lambda x: x[1], reverse=True)),
                "failure_pattern_count": len(_failure_history),
                "recurring_pattern_count": sum(
                    1 for d in _failure_history.values() if d["count"] >= 3
                ),
                "lessons_learned": len(_learned_lessons),
            }

    # -- Memory integration ---------------------------------------------------

    def get_memory_entries(self) -> List[Dict[str, Any]]:
        """
        Get lessons formatted for memory tool persistence.

        Returns:
            List of memory entry suggestions
        """
        with _state_lock:
            entries = []
            for lesson in _learned_lessons:
                entries.append({
                    "action": "add",
                    "target": "memory",
                    "content": f"[LESSON] {lesson['tool']}: {lesson['summary']} (pattern: {lesson.get('error_pattern', '')[:50]})",
                })
            return entries

    def clear_history(self):
        """Clear in-memory failure history and call history (persisted lessons remain)."""
        global _failure_history, _tool_call_history, _skill_suggestions
        with _state_lock:
            _failure_history = {}
            _tool_call_history = []
            _skill_suggestions = []
            # Also clear persisted failure patterns to avoid reload on next instance
            patterns_file = self.persistence_dir / "failure_patterns.json"
            if patterns_file.exists():
                try:
                    patterns_file.unlink()
                except OSError:
                    pass


# Backward-compat alias -- tools/self_learning_tool.py imports SelfLearner
SelfLearner = SelfLearningTracker


# =============================================================================
# Convenience functions (module-level for backward compat)
# =============================================================================

def analyze_failure(
    tool_name: str,
    args: Dict[str, Any],
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function for one-off failure analysis."""
    learner = SelfLearningTracker()
    return learner.analyze_failure(tool_name, args, error_message, context)


def get_recurring_failures() -> List[Dict[str, Any]]:
    """Get failures that have occurred 3+ times."""
    learner = SelfLearningTracker()
    return learner.get_recurring_failures()


def get_pending_skill_suggestions() -> List[Dict[str, Any]]:
    """Get skill suggestions from failure patterns."""
    learner = SelfLearningTracker()
    return learner.get_skill_suggestions()
