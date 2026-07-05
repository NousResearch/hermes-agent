"""Conversation Observer — auto-discovers task patterns from agent usage.

Watches real agent conversations and detects recurring patterns that
can be turned into evolution task definitions. This closes the gap
between "engine works" and "users actually benefit."

Detected patterns:
  PATTERN_BUG_FIX: terminal → patch → terminal → user confirms
  PATTERN_FILE_WORK: write_file → read_file → user confirms
  PATTERN_RESEARCH: web_search → web_extract → write_file
  PATTERN_DEPLOY: terminal(docker/git) → terminal(curl/test) → user confirms
  PATTERN_VERIFY: agent declares done → user corrects → agent retries

Persistence: detected patterns are stored in
  ~/.hermes/evolution/observed_patterns.json

Usage:
  observer = ConversationObserver()
  observer.observe_turn(messages)          # Watch each turn
  observer.observe_user_correction(text)   # Note user corrections
  tasks = observer.suggest_tasks(min_occurrences=3)  # Get suggestions
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Minimum occurrences before a pattern is suggested as a task
DEFAULT_MIN_OCCURRENCES = 3
# Max patterns to store before pruning old ones
MAX_STORED_PATTERNS = 200
# How many sessions to look back for pattern detection
MAX_SESSION_LOOKBACK = 20


# ---------------------------------------------------------------------------
# Pattern types
# ---------------------------------------------------------------------------


class PatternType:
    BUG_FIX = "bug_fix"           # Agent fixes something, runs tests
    FILE_WORK = "file_work"       # Agent creates/modifies files
    RESEARCH = "research"         # Agent searches and synthesizes
    DEPLOY = "deploy"             # Agent deploys and verifies
    DATA_PIPELINE = "data_pipeline"  # Agent processes data
    VERIFY_FAIL = "verify_fail"   # Agent said done, user corrected
    RECURRING_CMD = "recurring_cmd"  # Same command run across sessions


PATTERN_LABELS = {
    PatternType.BUG_FIX: "Bug Fix",
    PatternType.FILE_WORK: "File Work",
    PatternType.RESEARCH: "Research & Synthesis",
    PatternType.DEPLOY: "Deploy & Verify",
    PatternType.DATA_PIPELINE: "Data Pipeline",
    PatternType.VERIFY_FAIL: "Verification Failure",
    PatternType.RECURRING_CMD: "Recurring Command",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ObservedPattern:
    """A detected pattern that could become a task definition."""
    pattern_type: str
    description: str
    tools_used: List[str] = field(default_factory=list)
    commands_seen: List[str] = field(default_factory=list)
    file_paths_seen: List[str] = field(default_factory=list)
    occurrences: int = 0
    sessions: List[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    confidence: float = 0.0
    suggested_task_name: str = ""
    suggested_criteria: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "tools_used": self.tools_used,
            "commands_seen": self.commands_seen[-20:],  # Keep recent
            "file_paths_seen": self.file_paths_seen[-10:],
            "occurrences": self.occurrences,
            "sessions": list(set(self.sessions))[-10:],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
            "suggested_task_name": self.suggested_task_name,
            "suggested_criteria": self.suggested_criteria,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObservedPattern":
        return cls(
            pattern_type=d.get("pattern_type", ""),
            description=d.get("description", ""),
            tools_used=d.get("tools_used", []),
            commands_seen=d.get("commands_seen", []),
            file_paths_seen=d.get("file_paths_seen", []),
            occurrences=d.get("occurrences", 0),
            sessions=d.get("sessions", []),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
            confidence=d.get("confidence", 0.0),
            suggested_task_name=d.get("suggested_task_name", ""),
            suggested_criteria=d.get("suggested_criteria", []),
        )


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class ConversationObserver:
    """Watches agent conversations and discovers recurring task patterns.

    Thread-safe. Persists observed patterns to disk. Designed to be
    called from evolution hooks during normal agent operation.
    """

    def __init__(self):
        self._patterns: Dict[str, ObservedPattern] = {}  # key = pattern signature
        self._current_session: str = ""
        self._session_turns: List[Dict[str, Any]] = []
        self._user_corrections: List[str] = []
        self._load()

    # -- Lifecycle -----------------------------------------------------------

    def start_session(self, session_id: str) -> None:
        """Begin observing a new session."""
        self._current_session = session_id
        self._session_turns = []
        self._user_corrections = []

    def end_session(self) -> None:
        """Finalize the current session's observations."""
        if not self._session_turns:
            return

        # Detect patterns from accumulated turns
        self._detect_bug_fix_pattern()
        self._detect_file_work_pattern()
        self._detect_deploy_pattern()
        self._detect_research_pattern()
        self._detect_recurring_commands()
        self._detect_verify_fail_pattern()

        self._prune_old_patterns()
        self._save()

    # -- Turn observation ----------------------------------------------------

    def observe_turn(self, messages: List[Dict[str, Any]]) -> None:
        """Record a conversation turn for pattern detection.

        Called from post_llm_call or post_tool_call hooks.
        """
        turn_summary = {
            "session": self._current_session,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role_counts": self._count_roles(messages),
            "tool_calls": self._extract_tool_names(messages),
            "commands": self._extract_commands(messages),
            "files_mentioned": self._extract_file_paths(messages),
        }
        self._session_turns.append(turn_summary)

    def observe_user_correction(self, text: str) -> None:
        """Record a user correction — strong signal for task definition."""
        self._user_corrections.append(text)

    # -- Pattern detection ---------------------------------------------------

    def _detect_bug_fix_pattern(self) -> None:
        """Detect: read → patch → terminal(test) → user confirms."""
        tools_sequence = []
        for turn in self._session_turns:
            tools_sequence.extend(turn["tool_calls"])

        has_read = any("read_file" in t or "search_files" in t for t in tools_sequence)
        has_patch = any("patch" in t for t in tools_sequence)
        has_terminal = any("terminal" in t for t in tools_sequence)
        has_test = any(
            "pytest" in c or "test" in c or "npm test" in c or "go test" in c
            for turn in self._session_turns for c in turn["commands"]
        )

        if has_read and has_patch and has_terminal:
            signature = self._make_signature(PatternType.BUG_FIX, "bug_fix")
            pattern = self._get_or_create(signature, PatternType.BUG_FIX,
                description="Fix bugs: read code → apply patch → run tests → verify")

            pattern.tools_used = list(set(pattern.tools_used + ["read_file", "patch", "terminal"]))
            pattern.occurrences += 1
            pattern.sessions.append(self._current_session)
            pattern.last_seen = datetime.now(timezone.utc).isoformat()

            # Stronger signal if tests were actually run
            if has_test:
                pattern.confidence = min(1.0, pattern.confidence + 0.25)
                pattern.commands_seen.extend(
                    c for turn in self._session_turns for c in turn["commands"]
                    if "test" in c.lower()
                )
            else:
                pattern.confidence = min(1.0, pattern.confidence + 0.15)

            # Suggest criteria
            test_cmds = [c for c in pattern.commands_seen if "test" in c.lower() or "pytest" in c.lower()]
            if test_cmds:
                pattern.suggested_criteria = [
                    {"type": "test_pass", "command": test_cmds[0], "weight": 0.5},
                    {"type": "content_match", "path": "CHANGES.md", "pattern": "Fixed:", "weight": 0.25},
                    {"type": "file_exists", "path": "fix_applied.patch", "weight": 0.25},
                ]
            pattern.suggested_task_name = "bug-fix-verify"

    def _detect_file_work_pattern(self) -> None:
        """Detect: write_file → read_file → user confirms."""
        tools_sequence = []
        for turn in self._session_turns:
            tools_sequence.extend(turn["tool_calls"])

        has_write = any("write_file" in t for t in tools_sequence)
        has_read = any("read_file" in t for t in tools_sequence)

        if has_write:
            signature = self._make_signature(PatternType.FILE_WORK, "file_work")
            pattern = self._get_or_create(signature, PatternType.FILE_WORK,
                description="Create/modify files and verify output")

            pattern.tools_used = list(set(pattern.tools_used + ["write_file", "read_file"]))
            pattern.occurrences += 1
            pattern.sessions.append(self._current_session)
            pattern.last_seen = datetime.now(timezone.utc).isoformat()

            # Track files mentioned
            files = []
            for turn in self._session_turns:
                files.extend(turn["files_mentioned"])
            pattern.file_paths_seen.extend(files[:5])

            if has_read:
                pattern.confidence = min(1.0, pattern.confidence + 0.12)
            else:
                pattern.confidence = min(1.0, pattern.confidence + 0.06)

            # Suggest criteria based on files seen
            if files:
                pattern.suggested_criteria = [
                    {"type": "file_exists", "path": files[0], "weight": 0.5},
                    {"type": "content_match", "path": files[0], "pattern": ".+", "weight": 0.5},
                ]
            pattern.suggested_task_name = "create-file-verify"

    def _detect_deploy_pattern(self) -> None:
        """Detect: docker/git → curl/healthcheck → user confirms."""
        commands = []
        for turn in self._session_turns:
            commands.extend(turn["commands"])

        has_deploy = any(
            "docker" in c or "kubectl" in c or "git push" in c or "deploy" in c.lower()
            for c in commands
        )
        has_verify = any(
            "curl" in c or "health" in c.lower() or "status" in c.lower()
            for c in commands
        )

        if has_deploy:
            signature = self._make_signature(PatternType.DEPLOY, "deploy")
            pattern = self._get_or_create(signature, PatternType.DEPLOY,
                description="Deploy application and verify it's healthy")

            pattern.tools_used = list(set(pattern.tools_used + ["terminal"]))
            pattern.commands_seen.extend(commands[:10])
            pattern.occurrences += 1
            pattern.sessions.append(self._current_session)
            pattern.last_seen = datetime.now(timezone.utc).isoformat()

            if has_verify:
                pattern.confidence = min(1.0, pattern.confidence + 0.15)
            else:
                pattern.confidence = min(1.0, pattern.confidence + 0.08)

            # Suggest criteria
            verify_cmds = [c for c in commands if "curl" in c or "health" in c.lower()]
            pattern.suggested_criteria = [
                {"type": "test_pass", "command": commands[0] if commands else "echo deploy", "weight": 0.3},
            ]
            if verify_cmds:
                pattern.suggested_criteria.append(
                    {"type": "command_output", "command": verify_cmds[0], "expected_output": "ok", "weight": 0.4}
                )
            pattern.suggested_criteria.append(
                {"type": "file_exists", "path": "/tmp/deploy.log", "weight": 0.3}
            )
            pattern.suggested_task_name = "deploy-and-verify"

    def _detect_research_pattern(self) -> None:
        """Detect: web_search → web_extract → write_file."""
        tools_sequence = []
        for turn in self._session_turns:
            tools_sequence.extend(turn["tool_calls"])

        has_search = any("web_search" in t for t in tools_sequence)
        has_write = any("write_file" in t for t in tools_sequence)

        if has_search and has_write:
            signature = self._make_signature(PatternType.RESEARCH, "research")
            pattern = self._get_or_create(signature, PatternType.RESEARCH,
                description="Research topics and produce structured reports")

            pattern.tools_used = list(set(pattern.tools_used + ["web_search", "write_file"]))
            pattern.occurrences += 1
            pattern.sessions.append(self._current_session)
            pattern.last_seen = datetime.now(timezone.utc).isoformat()
            pattern.confidence = min(1.0, pattern.confidence + 0.10)

            files = []
            for turn in self._session_turns:
                files.extend(turn["files_mentioned"])
            pattern.file_paths_seen.extend(files[:5])

            pattern.suggested_task_name = "research-and-report"
            if files:
                pattern.suggested_criteria = [
                    {"type": "file_exists", "path": files[0], "weight": 0.5},
                    {"type": "content_match", "path": files[0], "pattern": "(?i)summary|conclusion|finding", "weight": 0.5},
                ]

    def _detect_recurring_commands(self) -> None:
        """Detect commands that appear across multiple sessions."""
        commands = []
        for turn in self._session_turns:
            commands.extend(turn["commands"])

        # Find commands that match existing patterns
        for cmd in commands:
            if len(cmd) < 5 or cmd.startswith("echo") or cmd.startswith("cd "):
                continue
            # Check if this command has been seen in other sessions
            for sig, pattern in self._patterns.items():
                if pattern.pattern_type == PatternType.RECURRING_CMD:
                    if cmd in pattern.commands_seen:
                        pattern.occurrences += 1
                        pattern.sessions.append(self._current_session)
                        pattern.last_seen = datetime.now(timezone.utc).isoformat()
                        pattern.confidence = min(1.0, pattern.confidence + 0.12)
                        break
            else:
                # New recurring command candidate
                if len(commands) >= 2:  # Need at least 2 commands to be interesting
                    sig = self._make_signature(PatternType.RECURRING_CMD, cmd[:40])
                    pattern = self._get_or_create(sig, PatternType.RECURRING_CMD,
                        description=f"Run command: {cmd[:80]}")
                    pattern.commands_seen.append(cmd)
                    pattern.tools_used = ["terminal"]
                    pattern.occurrences = 1
                    pattern.sessions.append(self._current_session)
                    pattern.first_seen = datetime.now(timezone.utc).isoformat()
                    pattern.confidence = 0.3
                    pattern.suggested_task_name = f"run-{re.sub(r'[^a-z0-9-]', '-', cmd[:40].lower())}"
                    pattern.suggested_criteria = [
                        {"type": "test_pass", "command": cmd, "weight": 1.0},
                    ]

    def _detect_verify_fail_pattern(self) -> None:
        """Detect: agent declared done → user corrected them."""
        if not self._user_corrections:
            return

        signature = self._make_signature(PatternType.VERIFY_FAIL, "verify_fail")
        pattern = self._get_or_create(signature, PatternType.VERIFY_FAIL,
            description="Agent declared completion but user had to correct")

        pattern.occurrences += len(self._user_corrections)
        pattern.sessions.append(self._current_session)
        pattern.last_seen = datetime.now(timezone.utc).isoformat()
        pattern.confidence = min(1.0, pattern.confidence + 0.18 * len(self._user_corrections))

        # This is the strongest signal — user explicitly corrected the agent
        pattern.suggested_task_name = "verify-before-complete"
        pattern.suggested_criteria = [
            {"type": "test_pass", "command": "echo 'verify all outputs exist'", "weight": 0.5},
            {"type": "content_match", "path": "/tmp/output.md", "pattern": ".+", "weight": 0.5},
        ]

    # -- Suggestion API ------------------------------------------------------

    def suggest_tasks(self, min_occurrences: int = DEFAULT_MIN_OCCURRENCES) -> List[ObservedPattern]:
        """Return patterns that are ready to become task definitions.

        Only returns patterns that:
        - Have been seen at least min_occurrences times
        - Have confidence >= 0.3 (crosses threshold after ~2 occurrences with verification)
        - Have suggested criteria
        """
        suggestions = []
        for pattern in self._patterns.values():
            if pattern.occurrences >= min_occurrences and pattern.confidence >= 0.3:
                if pattern.suggested_criteria:
                    suggestions.append(pattern)

        suggestions.sort(key=lambda p: (p.confidence * p.occurrences), reverse=True)
        return suggestions

    def suggest_task_yaml(self, pattern: ObservedPattern) -> str:
        """Generate a task definition YAML from an observed pattern."""
        criteria_yaml = ""
        for c in pattern.suggested_criteria:
            criteria_yaml += f"  - type: {c['type']}\n"
            for k, v in c.items():
                if k == "type":
                    continue
                if isinstance(v, str):
                    criteria_yaml += f"    {k}: \"{v}\"\n"
                else:
                    criteria_yaml += f"    {k}: {v}\n"

        return f"""# Auto-generated from {pattern.occurrences} observed sessions
# Pattern: {PATTERN_LABELS.get(pattern.pattern_type, pattern.pattern_type)}
# Confidence: {pattern.confidence:.0%}
name: {pattern.suggested_task_name}
description: "{pattern.description}"
domain: general
complexity: {min(8, 2 + pattern.occurrences)}
success_criteria:
{criteria_yaml}timeout_seconds: 120
max_turns: 15
"""

    # -- Persistence ---------------------------------------------------------

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            self._patterns = {
                k: ObservedPattern.from_dict(v)
                for k, v in data.get("patterns", {}).items()
            }
        except Exception as e:
            logger.debug("Failed to load observed patterns: %s", e)

    def _save(self) -> None:
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "patterns": {k: v.to_dict() for k, v in self._patterns.items()},
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Failed to save observed patterns: %s", e)

    def _store_path(self) -> Path:
        return get_hermes_home() / "evolution" / "observed_patterns.json"

    def _prune_old_patterns(self) -> None:
        """Remove patterns with low confidence and few occurrences."""
        if len(self._patterns) <= MAX_STORED_PATTERNS:
            return
        # Sort by confidence * occurrences, keep top
        scored = sorted(
            self._patterns.items(),
            key=lambda x: x[1].confidence * x[1].occurrences,
            reverse=True,
        )
        self._patterns = dict(scored[:MAX_STORED_PATTERNS])

    # -- Helpers -------------------------------------------------------------

    def _get_or_create(self, signature: str, pattern_type: str, description: str) -> ObservedPattern:
        if signature not in self._patterns:
            self._patterns[signature] = ObservedPattern(
                pattern_type=pattern_type,
                description=description,
                first_seen=datetime.now(timezone.utc).isoformat(),
            )
        return self._patterns[signature]

    @staticmethod
    def _make_signature(pattern_type: str, seed: str) -> str:
        return f"{pattern_type}:{seed}"

    @staticmethod
    def _count_roles(messages: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for msg in messages:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            counts[role] += 1
        return dict(counts)

    @staticmethod
    def _extract_tool_names(messages: List[Dict[str, Any]]) -> List[str]:
        tools = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []) or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            tools.append(fn.get("name", ""))
        return [t for t in tools if t]

    @staticmethod
    def _extract_commands(messages: List[Dict[str, Any]]) -> List[str]:
        """Extract shell commands from terminal tool calls."""
        commands = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Look for command patterns
                    for line in content.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("//"):
                            if any(kw in line for kw in ["pytest", "npm ", "go ", "curl ", "docker ", "git ", "python", "make ", "cargo ", "kubectl"]):
                                commands.append(line)
        return commands

    @staticmethod
    def _extract_file_paths(messages: List[Dict[str, Any]]) -> List[str]:
        """Extract file paths mentioned in tool calls."""
        import re
        files = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            # From assistant tool_call args
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []) or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            args_str = fn.get("arguments", "")
                            if isinstance(args_str, str):
                                try:
                                    args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    continue
                                for key in ("path", "file_path", "target_file"):
                                    if key in args:
                                        files.append(str(args[key]))
            # From tool results
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                paths = re.findall(r'(?:/[^\s:]+|[./][^\s:]+\.\w+)', content)
                files.extend(paths[:5])
        return files


# ---------------------------------------------------------------------------
# Global singleton — one observer per process
# ---------------------------------------------------------------------------

_observer: Optional[ConversationObserver] = None


def get_observer() -> ConversationObserver:
    global _observer
    if _observer is None:
        _observer = ConversationObserver()
    return _observer
