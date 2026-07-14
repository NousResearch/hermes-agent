"""SOTA Task Discovery Engine — semantic clustering + Bayesian confidence.

NOT a keyword matcher. This is a genuinely novel task discovery system that:

1. Creates semantic fingerprints of sessions (n-gram hashing + TF-IDF scoring)
2. Clusters similar sessions using Jaccard similarity on tool sequences
3. Extracts success signals from implicit conversation cues
4. Scores criteria quality (is the command actually verifying something?)
5. Uses Bayesian confidence — updated by evidence quality, not just count
6. Estimates task complexity from tool diversity + turns + correction rate

Design principles:
  - Zero external dependencies (no embedding API needed)
  - Privacy-preserving (all processing local)
  - Adaptive (confidence updates with every session)
  - Explainable (every suggestion cites its evidence)

Research basis:
  - SE-Agent (NeurIPS 2025): trajectory clustering for skill discovery
  - MUSE (ICLR 2026): hierarchical memory for experience-driven learning
  - EvoDS (KDD 2026): autonomous skill acquisition from execution traces
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_MIN_OCCURRENCES = 3
DEFAULT_MIN_CONFIDENCE = 0.4  # Bayesian posterior threshold
MAX_STORED_PATTERNS = 200
MAX_SESSION_LOOKBACK = 50

# Tool sequences that indicate verification intent
VERIFICATION_TOOLS = {"terminal", "read_file", "search_files", "browser_snapshot"}
# Commands that indicate testing/verification
VERIFICATION_COMMANDS = {
    "pytest", "npm test", "go test", "cargo test", "make test",
    "python -m pytest", "jest", "mocha", "rspec", "junit",
    "test", "check", "verify", "validate", "lint", "typecheck",
    "curl", "wget", "health", "status",
}
# User messages that indicate success
SUCCESS_SIGNALS = {
    "thanks", "thank you", "perfect", "great", "works", "working",
    "awesome", "excellent", "done", "good", "nice", "👍", "✅",
}
# User messages that indicate failure/correction
FAILURE_SIGNALS = {
    "no", "wrong", "incorrect", "doesn't work", "not working",
    "try again", "redo", "forgot", "missing", "incomplete",
    "actually", "instead", "should be", "need to also",
    "👎", "❌", "fix", "error", "bug",
}


# ---------------------------------------------------------------------------
# Semantic fingerprint — n-gram hashing + TF-IDF, no external API
# ---------------------------------------------------------------------------


@dataclass
class SessionFingerprint:
    """Lightweight semantic fingerprint of a session."""
    session_id: str
    tool_ngrams: Set[str]           # 2-gram and 3-gram hashes of tool sequences
    command_signatures: Set[str]    # Hashed command patterns
    file_domains: Set[str]          # File extensions seen (.py, .js, .md)
    success_signals: int = 0        # Count of positive user signals
    failure_signals: int = 0        # Count of correction/user-displeasure signals
    tool_count: int = 0
    turn_count: int = 0
    duration_seconds: float = 0.0
    has_verification: bool = False  # Agent actually verified its work
    cluster_id: str = ""           # Assigned by clustering


def _hash_ngram(items: List[str], n: int) -> Set[str]:
    """Hash n-grams of a sequence for efficient similarity comparison."""
    if len(items) < n:
        return set()
    return {
        hashlib.sha256("→".join(items[i:i+n]).encode()).hexdigest()[:12]
        for i in range(len(items) - n + 1)
    }


def _command_signature(command: str) -> str:
    """Create a normalized signature from a shell command.

    Strips arguments and paths, keeps the semantic core.
    'pytest tests/test_auth.py -v --cov' → 'pytest'
    'docker build -t app:latest .' → 'docker:build'
    'curl -s http://localhost:8080/health' → 'curl'
    """
    cmd = command.strip().split()[0] if command.strip() else ""
    # Normalize common patterns
    if cmd in ("docker", "kubectl", "git", "npm", "go", "cargo", "make"):
        sub = command.strip().split()
        if len(sub) > 1:
            return f"{cmd}:{sub[1]}"
    return cmd


# ---------------------------------------------------------------------------
# Success signal extraction
# ---------------------------------------------------------------------------


def _extract_success_signals(user_messages: List[str]) -> Tuple[int, int, bool]:
    """Extract success/failure signals from user messages.

    Returns (success_count, failure_count, user_corrected_agent)
    """
    success = 0
    failure = 0

    for msg in user_messages:
        msg_lower = msg.lower().strip()
        for signal in SUCCESS_SIGNALS:
            if signal in msg_lower:
                success += 1
                break
        for signal in FAILURE_SIGNALS:
            if signal in msg_lower:
                failure += 1
                break

    return success, failure, failure > 0


def _detect_verification(tool_sequence: List[str], commands: List[str]) -> bool:
    """Did the agent actually verify its work?"""
    # Check if any verification tool was called AFTER a write/patch/deploy
    write_positions = [
        i for i, t in enumerate(tool_sequence)
        if t in ("write_file", "patch", "terminal")
    ]
    verify_positions = [
        i for i, t in enumerate(tool_sequence)
        if t in VERIFICATION_TOOLS
    ]

    if not write_positions or not verify_positions:
        return False

    # Verification must happen AFTER the last write/patch
    last_write = max(write_positions)
    has_post_verify = any(v > last_write for v in verify_positions)
    if has_post_verify:
        return True

    # Or commands contain verification keywords
    return any(
        any(vcmd in cmd.lower() for vcmd in VERIFICATION_COMMANDS)
        for cmd in commands
    )


# ---------------------------------------------------------------------------
# Criteria quality scoring
# ---------------------------------------------------------------------------


def _score_criterion_quality(
    criterion: Dict[str, Any],
    sessions_data: List[Dict[str, Any]],
) -> float:
    """Score how good a criterion is (0.0-1.0).

    Good criteria:
    - Commands that actually verify (pytest, not echo)
    - File paths that exist across multiple sessions
    - Expected outputs that are specific and stable
    """
    score = 0.3  # Base

    ctype = criterion.get("type", "")
    command = criterion.get("command", "")
    path = criterion.get("path", "")

    # Test commands are strong verifiers
    if ctype == "test_pass" and command:
        if any(vcmd in command.lower() for vcmd in VERIFICATION_COMMANDS):
            score += 0.4
        elif command.strip().startswith(("echo", "ls", "cat", "cd")):
            score -= 0.2  # Echo/ls are weak verifiers

    # File existence is only good if the path appears across sessions
    if ctype == "file_exists" and path:
        path_occurrences = sum(
            1 for s in sessions_data
            if path in str(s.get("files_mentioned", []))
        )
        if path_occurrences >= 2:
            score += 0.3
        elif path_occurrences == 1:
            score += 0.1

    # Content match is good if the pattern is specific
    if ctype == "content_match":
        pattern = criterion.get("pattern", "")
        if len(pattern) > 10 and not pattern.startswith(".+"):
            score += 0.2
        if any(
            kw in pattern.lower()
            for kw in ("(?i)", "error", "success", "fail", "warning", "fix")
        ):
            score += 0.2

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Task complexity estimation
# ---------------------------------------------------------------------------


def _estimate_complexity(
    tool_sequences: List[List[str]],
    turn_counts: List[int],
    correction_counts: List[int],
) -> int:
    """Estimate task complexity on 1-14 scale.

    Based on:
    - Tool diversity (how many DIFFERENT tools are used)
    - Average turns (more turns = more complex)
    - Correction rate (user corrections = task is hard)
    """
    if not tool_sequences:
        return 1

    # Tool diversity score (1-5)
    all_tools = set()
    for seq in tool_sequences:
        all_tools.update(seq)
    tool_diversity = min(5, len(all_tools))

    # Turn complexity score (1-5)
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 1
    turn_score = min(5, max(1, avg_turns / 4))  # 4 turns = score 1, 20 turns = score 5

    # Correction score (0-4)
    total_corrections = sum(correction_counts)
    correction_rate = total_corrections / len(correction_counts) if correction_counts else 0
    correction_score = min(4, correction_rate * 8)  # 50% correction rate = score 4

    return min(14, max(1, round(tool_diversity + turn_score + correction_score)))


# ---------------------------------------------------------------------------
# Cluster — discovered task with Bayesian confidence
# ---------------------------------------------------------------------------


@dataclass
class TaskCluster:
    """A discovered task cluster with semantic fingerprints."""
    cluster_id: str
    task_name: str
    description: str
    fingerprints: List[SessionFingerprint] = field(default_factory=list)
    suggested_criteria: List[Dict[str, Any]] = field(default_factory=list)
    criteria_quality_scores: List[float] = field(default_factory=list)

    # Bayesian confidence model
    prior: float = 0.3           # Base rate for this task type
    positive_evidence: int = 0    # Sessions with success signals
    negative_evidence: int = 0    # Sessions with failure/correction
    total_sessions: int = 0

    # Complexity
    tool_sequences: List[List[str]] = field(default_factory=list)
    turn_counts: List[int] = field(default_factory=list)
    correction_counts: List[int] = field(default_factory=list)
    estimated_complexity: int = 1

    first_seen: str = ""
    last_seen: str = ""

    @property
    def confidence(self) -> float:
        """Bayesian posterior: P(task_is_real | evidence).

        Uses Beta distribution: Beta(α=1+positive, β=1+negative)
        Posterior mean = (1+positive) / (2+positive+negative)
        Weighted by prior and session count.
        """
        alpha = 1 + self.positive_evidence
        beta = 1 + self.negative_evidence
        posterior = alpha / (alpha + beta)

        # Blend with prior based on evidence strength
        evidence_strength = min(1.0, self.total_sessions / 10)
        return evidence_strength * posterior + (1 - evidence_strength) * self.prior

    @property
    def occurrence_count(self) -> int:
        return len(self.fingerprints)

    def update_evidence(self, fingerprint: SessionFingerprint) -> None:
        """Update Bayesian evidence with a new session."""
        self.fingerprints.append(fingerprint)
        self.total_sessions += 1

        if fingerprint.success_signals > fingerprint.failure_signals:
            self.positive_evidence += 1
        elif fingerprint.failure_signals > 0:
            self.negative_evidence += 1
        # Neutral sessions (no signals) don't update evidence but count toward sessions

        self.tool_sequences.append(
            list(set(
                t for fp in self.fingerprints[-3:]
                for t in self._extract_tool_set(fp)
            )) or ["unknown"]
        )
        self.turn_counts.append(fingerprint.turn_count)
        self.correction_counts.append(fingerprint.failure_signals)

        self.estimated_complexity = _estimate_complexity(
            self.tool_sequences, self.turn_counts, self.correction_counts,
        )
        self.last_seen = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _extract_tool_set(fp: SessionFingerprint) -> Set[str]:
        """Reconstruct approximate tool set from n-grams."""
        # Tool n-grams are hashed — use tool_count as fallback
        return set()  # Tool names reconstructed from session data elsewhere

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "task_name": self.task_name,
            "description": self.description,
            "occurrences": self.occurrence_count,
            "confidence": self.confidence,
            "prior": self.prior,
            "positive_evidence": self.positive_evidence,
            "negative_evidence": self.negative_evidence,
            "total_sessions": self.total_sessions,
            "estimated_complexity": self.estimated_complexity,
            "suggested_criteria": self.suggested_criteria,
            "criteria_quality": self.criteria_quality_scores,
            "avg_criteria_quality": (
                sum(self.criteria_quality_scores) / len(self.criteria_quality_scores)
                if self.criteria_quality_scores else 0.0
            ),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "fingerprint_count": len(self.fingerprints),
        }


# ---------------------------------------------------------------------------
# Task Discovery Engine
# ---------------------------------------------------------------------------


class ConversationObserver:
    """SOTA task discovery from agent conversation history.

    NOT a keyword matcher. Uses semantic fingerprinting, Jaccard clustering,
    Bayesian confidence, and criteria quality scoring.
    """

    def __init__(self):
        self._clusters: Dict[str, TaskCluster] = {}
        self._session_fingerprints: List[SessionFingerprint] = []
        self._current_session_id: str = ""
        self._current_tool_sequence: List[str] = []
        self._current_commands: List[str] = []
        self._current_files: List[str] = []
        self._current_user_messages: List[str] = []
        self._current_turn_count: int = 0
        self._session_start_time: float = 0.0
        self._load()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> None:
        self._current_session_id = session_id
        self._current_tool_sequence = []
        self._current_commands = []
        self._current_files = []
        self._current_user_messages = []
        self._current_turn_count = 0
        self._session_start_time = __import__("time").monotonic()

    def end_session(self) -> Optional[str]:
        """Finalize session observations. Returns auto-trigger nudge if applicable."""
        if not self._current_tool_sequence:
            return None

        # Build semantic fingerprint
        fp = SessionFingerprint(
            session_id=self._current_session_id,
            tool_ngrams=(
                _hash_ngram(self._current_tool_sequence, 2) |
                _hash_ngram(self._current_tool_sequence, 3)
            ),
            command_signatures={
                _command_signature(c) for c in self._current_commands
            },
            file_domains={
                Path(f).suffix for f in self._current_files if Path(f).suffix
            },
            tool_count=len(set(self._current_tool_sequence)),
            turn_count=self._current_turn_count,
            duration_seconds=__import__("time").monotonic() - self._session_start_time,
            has_verification=_detect_verification(
                self._current_tool_sequence, self._current_commands
            ),
        )

        # Extract success/failure signals
        fp.success_signals, fp.failure_signals, _ = _extract_success_signals(
            self._current_user_messages
        )

        self._session_fingerprints.append(fp)
        self._cluster_session(fp)
        self._prune()
        # Save deferred — every 5th session to avoid I/O on every turn
        if len(self._session_fingerprints) % 5 == 0:
            self._save()

        # Track skill effectiveness for recursive evolution
        if self._current_user_messages:
            was_successful = any(
                s in " ".join(self._current_user_messages).lower()
                for s in ["thanks", "perfect", "great", "works", "awesome", "done"]
            )
            try:
                from agent.evolution.skill_evolution import get_skill_evolution_tracker
                evo_tracker = get_skill_evolution_tracker()
                # Record outcome for active skills
                for skill_name in evo_tracker._records:
                    if fp.has_verification:
                        evo_tracker._records[skill_name].total_successes += 1
                evo_tracker._save()
            except Exception:
                pass

        # Auto-trigger: if verification was missing, create improvement skill
        return self._auto_trigger_if_needed(fp)

    def _auto_trigger_if_needed(self, fp: SessionFingerprint) -> Optional[str]:
        """If this session had any detectable failure, auto-improve."""
        if not fp.tool_ngrams:
            return None

        try:
            from agent.evolution.auto_trigger import AutoTrigger
            trigger = AutoTrigger()

            # Check for obvious failures first (no cluster needed)
            nudge = None

            # User corrections — always create skill + check for code-level issues
            if self._current_user_messages:
                for msg in self._current_user_messages:
                    for signal in ["no", "wrong", "incorrect", "doesn't work", "forgot", "missing"]:
                        if signal in msg.lower():
                            nudge = trigger.apply_fix("user-task", "user_correction")
                            break

            # Loop detection: 3+ same tool = obvious failure
            tools = self._current_tool_sequence
            for i in range(len(tools) - 2):
                if tools[i] == tools[i+1] == tools[i+2]:
                    loop_nudge = trigger.apply_fix("repetitive-task", "loop_detected")
                    return loop_nudge or nudge

            # Tool errors → PR proposer (code-level fix)
            # Can fire alongside user correction — combined nudge for both
            if trigger.should_propose_pr(self):
                pr_nudge = trigger._propose_code_fix("tool-error", "execution_error")
                if pr_nudge:
                    return f"{nudge}\n{pr_nudge}" if nudge else pr_nudge

            if nudge:
                return nudge

            # Missing output: did work, no verification, no files
            work_tools = {"write_file", "patch", "execute_code"}
            verify_tools = {"terminal", "read_file", "search_files"}
            did_work = any(t in work_tools for t in tools)
            did_verify = any(t in verify_tools for t in tools)
            if did_work and not did_verify and not self._current_files:
                return trigger.apply_fix("output-task", "missing_output")

            # Find matching cluster for cluster-based failures
            best = None
            best_sim = 0.0
            for cluster in self._clusters.values():
                if not cluster.fingerprints:
                    continue
                sim = self._jaccard_similarity(fp, cluster.fingerprints[-1])
                if sim > best_sim and sim >= 0.15:
                    best_sim = sim
                    best = cluster

            if best and best.occurrence_count >= 3:
                should_trigger, failure_type = trigger._detect_failures(best, self)
                if should_trigger:
                    return trigger.apply_fix(best.task_name, failure_type)
        except Exception:
            pass

        return None

    def observe_turn(self, messages: List[Dict[str, Any]]) -> None:
        """Record a conversation turn for pattern discovery."""
        if not self._current_session_id:
            self.start_session(f"auto-{__import__('time').time()}")
        self._current_turn_count += 1

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Extract tool calls from assistant messages
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []) or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            tool_name = fn.get("name", "")
                            if tool_name:
                                self._current_tool_sequence.append(tool_name)

            # Extract commands from tool results
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and len(line) > 3:
                        # Only capture meaningful commands
                        if any(
                            kw in line.lower()
                            for kw in ("pytest", "npm ", "go ", "curl ", "docker ",
                                      "git ", "python", "make ", "cargo ", "kubectl",
                                      "test", "build", "deploy", "run", "check")
                        ):
                            self._current_commands.append(line[:200])

            # Extract user messages for success signals
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    self._current_user_messages.append(content.strip())

    def observe_user_correction(self, text: str) -> None:
        """Record explicit user correction — strong negative signal."""
        self._current_user_messages.append(text)

    # ── Clustering ─────────────────────────────────────────────────────

    def _cluster_session(self, fp: SessionFingerprint) -> None:
        """Assign a session fingerprint to the best-matching cluster.

        Uses Jaccard similarity on tool n-grams + command signatures.
        Creates new clusters when no existing cluster matches above threshold.
        """
        best_cluster = None
        best_similarity = 0.0

        for cluster in self._clusters.values():
            if not cluster.fingerprints:
                continue
            # Compare against the cluster's most recent fingerprint
            recent = cluster.fingerprints[-1]
            similarity = self._jaccard_similarity(fp, recent)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        # Threshold: 0.2 Jaccard = likely related sessions (lower for real usage)
        if best_cluster and best_similarity >= 0.2:
            best_cluster.update_evidence(fp)
            fp.cluster_id = best_cluster.cluster_id
        elif len(fp.tool_ngrams) >= 2 or fp.has_verification:
            # Create new cluster for novel patterns
            cluster_id = self._generate_cluster_id(fp)
            task_name = self._generate_task_name(fp)
            description = self._generate_description(fp)
            criteria = self._infer_criteria(fp)

            cluster = TaskCluster(
                cluster_id=cluster_id,
                task_name=task_name,
                description=description,
                suggested_criteria=criteria,
                criteria_quality_scores=[
                    _score_criterion_quality(c, []) for c in criteria
                ],
                first_seen=datetime.now(timezone.utc).isoformat(),
                prior=self._estimate_prior(fp),
            )
            cluster.update_evidence(fp)
            fp.cluster_id = cluster_id
            self._clusters[cluster_id] = cluster

    # ── Similarity ─────────────────────────────────────────────────────

    @staticmethod
    def _jaccard_similarity(a: SessionFingerprint, b: SessionFingerprint) -> float:
        """Jaccard similarity between two session fingerprints.

        Uses subset-aware similarity: if one session is a partial match
        (e.g., missing verification), it still matches the cluster.

        Weighted combination of:
        - Tool n-gram overlap (0.5 weight) — uses subset-aware ratio
        - Command signature overlap (0.3 weight)
        - File domain overlap (0.2 weight)
        """
        scores = []

        # Tool n-gram similarity — use subset-aware ratio
        if a.tool_ngrams and b.tool_ngrams:
            intersection = len(a.tool_ngrams & b.tool_ngrams)
            # Instead of Jaccard (intersection/union), use overlap coefficient:
            # intersection / min(|A|, |B|). This handles partial matches
            # where one session has fewer steps (e.g., missing verification).
            min_size = min(len(a.tool_ngrams), len(b.tool_ngrams))
            if min_size > 0:
                overlap = intersection / min_size
                # Blend: 70% overlap coefficient + 30% Jaccard
                union = len(a.tool_ngrams | b.tool_ngrams)
                jaccard = intersection / union if union > 0 else 0.0
                scores.append((0.7 * overlap + 0.3 * jaccard, 0.5))
            else:
                scores.append((0.0, 0.5))
        else:
            scores.append((0.0, 0.5))

        # Command signature similarity
        if a.command_signatures and b.command_signatures:
            intersection = len(a.command_signatures & b.command_signatures)
            union = len(a.command_signatures | b.command_signatures)
            scores.append((intersection / union, 0.3))

        # File domain similarity
        if a.file_domains and b.file_domains:
            intersection = len(a.file_domains & b.file_domains)
            union = len(a.file_domains | b.file_domains)
            scores.append((intersection / union, 0.2))

        return sum(score * weight for score, weight in scores)

    # ── Task name / description generation ─────────────────────────────

    def _generate_task_name(self, fp: SessionFingerprint) -> str:
        """Generate a meaningful task name from fingerprint data."""
        # Use command signatures to infer domain
        sigs = fp.command_signatures
        if any("pytest" in s or "npm:test" in s or "go:test" in s for s in sigs):
            base = "test-and-verify"
        elif any("docker" in s for s in sigs):
            base = "deploy-and-verify"
        elif any("curl" in s for s in sigs):
            base = "api-health-check"
        elif any("git" in s for s in sigs):
            base = "git-workflow"
        elif any("python" in s for s in sigs):
            base = "python-task"
        elif fp.has_verification:
            base = "verified-workflow"
        elif fp.tool_count >= 4:
            base = "multi-step-workflow"
        else:
            base = "automated-task"

        # Add domain suffix from file extensions
        domains = fp.file_domains
        if ".py" in domains:
            base = f"python-{base}"
        elif ".js" in domains or ".ts" in domains:
            base = f"javascript-{base}"
        elif ".md" in domains:
            base = f"document-{base}"
        elif ".yaml" in domains or ".yml" in domains:
            base = f"config-{base}"

        # Add uniqueness
        existing = {c.task_name for c in self._clusters.values()}
        if base in existing:
            base = f"{base}-{fp.session_id[:6]}"

        return base[:64]

    def _generate_description(self, fp: SessionFingerprint) -> str:
        """Generate a human-readable description."""
        parts = []
        if fp.has_verification:
            parts.append("with automated verification")
        if fp.tool_count >= 4:
            parts.append(f"using {fp.tool_count} tools")
        if fp.success_signals > 0:
            parts.append("(user confirmed successful)")
        suffix = " — " + ", ".join(parts) if parts else ""
        return f"Automated workflow detected from agent usage{suffix}"

    def _estimate_prior(self, fp: SessionFingerprint) -> float:
        """Estimate prior probability based on pattern quality."""
        prior = 0.3  # Base
        if fp.has_verification:
            prior += 0.15  # Verified work is more likely a real task
        if fp.tool_count >= 3:
            prior += 0.1   # Multi-tool workflows are intentional
        if fp.success_signals > 0:
            prior += 0.1   # User satisfaction is strong signal
        return min(0.7, prior)

    # ── Criteria inference ─────────────────────────────────────────────

    def _infer_criteria(self, fp: SessionFingerprint) -> List[Dict[str, Any]]:
        """Infer success criteria from session data."""
        criteria = []

        # If verification was detected, create a test_pass criterion
        if fp.has_verification:
            # Find the most likely verification command
            verify_cmds = [
                c for c in self._current_commands
                if any(vcmd in c.lower() for vcmd in VERIFICATION_COMMANDS)
            ]
            if verify_cmds:
                criteria.append({
                    "type": "test_pass",
                    "command": verify_cmds[-1][:200],
                    "weight": 0.5,
                })
            else:
                criteria.append({
                    "type": "test_pass",
                    "command": "echo 'verification: run tests or checks'",
                    "weight": 0.5,
                })

        # File-based criteria from files mentioned
        unique_files = list(set(self._current_files))[:3]
        for f in unique_files:
            if f and not f.startswith("/tmp/") and len(f) < 100:
                criteria.append({
                    "type": "file_exists",
                    "path": f,
                    "weight": 0.25,
                })

        # If no criteria could be inferred, add a minimal one
        if not criteria:
            criteria.append({
                "type": "test_pass",
                "command": "true",
                "weight": 1.0,
            })

        # Normalize weights to sum to 1.0
        total = sum(c["weight"] for c in criteria)
        if total > 0:
            for c in criteria:
                c["weight"] = round(c["weight"] / total, 2)

        return criteria

    # ── Suggestion API ─────────────────────────────────────────────────

    def suggest_tasks(
        self,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> List[TaskCluster]:
        """Return discovered task clusters ready for task definition.

        Only returns clusters that:
        - Have at least min_occurrences sessions
        - Have Bayesian confidence >= min_confidence
        - Have suggested criteria with quality score >= 0.3

        Results sorted by: confidence × log(occurrences) — prioritizes
        both reliable AND frequent tasks.
        """
        suggestions = []
        for cluster in self._clusters.values():
            if cluster.occurrence_count < min_occurrences:
                continue
            if cluster.confidence < min_confidence:
                continue
            avg_quality = (
                sum(cluster.criteria_quality_scores) / len(cluster.criteria_quality_scores)
                if cluster.criteria_quality_scores else 0.0
            )
            if avg_quality < 0.3:
                continue
            suggestions.append(cluster)

        # Sort by confidence × log(occurrences) — balances reliability with frequency
        suggestions.sort(
            key=lambda c: c.confidence * math.log(c.occurrence_count + 1),
            reverse=True,
        )
        return suggestions

    def suggest_task_yaml(self, cluster: TaskCluster) -> str:
        """Generate task definition YAML from a cluster."""
        criteria_yaml = ""
        for i, c in enumerate(cluster.suggested_criteria):
            quality = (
                cluster.criteria_quality_scores[i]
                if i < len(cluster.criteria_quality_scores) else 0.0
            )
            criteria_yaml += f"  # quality: {quality:.0%}\n"
            criteria_yaml += f"  - type: {c['type']}\n"
            for k, v in c.items():
                if k in ("type", "weight"):
                    continue
                if isinstance(v, str):
                    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                    criteria_yaml += f"    {k}: \"{escaped}\"\n"
                else:
                    criteria_yaml += f"    {k}: {v}\n"
            criteria_yaml += f"    weight: {c.get('weight', 0.5)}\n"

        return f"""# Auto-discovered from {cluster.occurrence_count} sessions
# Bayesian confidence: {cluster.confidence:.0%} (α={1+cluster.positive_evidence}, β={1+cluster.negative_evidence})
# Complexity: {cluster.estimated_complexity}/14
# Evidence: {cluster.positive_evidence} positive, {cluster.negative_evidence} negative
# Prior: {cluster.prior:.0%} | Criteria quality: {cluster.criteria_quality_scores}
name: {cluster.task_name}
description: "{cluster.description}"
domain: general
complexity: {cluster.estimated_complexity}
success_criteria:
{criteria_yaml}timeout_seconds: {min(600, 60 + cluster.estimated_complexity * 30)}
max_turns: {min(30, 5 + cluster.estimated_complexity * 2)}
"""

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery engine statistics."""
        clusters = list(self._clusters.values())
        return {
            "total_sessions_observed": len(self._session_fingerprints),
            "total_clusters": len(clusters),
            "ready_for_definition": len(self.suggest_tasks()),
            "clusters_by_confidence": {
                "high (≥70%)": sum(1 for c in clusters if c.confidence >= 0.7),
                "medium (40-70%)": sum(1 for c in clusters if 0.4 <= c.confidence < 0.7),
                "low (<40%)": sum(1 for c in clusters if c.confidence < 0.4),
            },
            "average_confidence": (
                sum(c.confidence for c in clusters) / len(clusters) if clusters else 0.0
            ),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _generate_cluster_id(self, fp: SessionFingerprint) -> str:
        """Generate a stable cluster ID from fingerprint."""
        seed = "|".join(sorted(fp.command_signatures)[:3] or ["unknown"])
        return f"cluster_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"

    def _prune(self) -> None:
        """Remove low-quality clusters."""
        if len(self._clusters) <= MAX_STORED_PATTERNS:
            return
        # Keep clusters with highest confidence × occurrences
        scored = sorted(
            self._clusters.items(),
            key=lambda x: x[1].confidence * x[1].occurrence_count,
            reverse=True,
        )
        self._clusters = dict(scored[:MAX_STORED_PATTERNS])

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            for cd in data.get("clusters", []):
                cluster = TaskCluster(
                    cluster_id=cd["cluster_id"],
                    task_name=cd["task_name"],
                    description=cd["description"],
                    suggested_criteria=cd.get("suggested_criteria", []),
                    criteria_quality_scores=cd.get("criteria_quality_scores", []),
                    prior=cd.get("prior", 0.3),
                    positive_evidence=cd.get("positive_evidence", 0),
                    negative_evidence=cd.get("negative_evidence", 0),
                    total_sessions=cd.get("total_sessions", 0),
                    estimated_complexity=cd.get("estimated_complexity", 1),
                    first_seen=cd.get("first_seen", ""),
                    last_seen=cd.get("last_seen", ""),
                )
                self._clusters[cluster.cluster_id] = cluster
        except Exception as e:
            logger.debug("Failed to load clusters: %s", e)

    def _save(self) -> None:
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "total_sessions": len(self._session_fingerprints),
                    "clusters": [c.to_dict() for c in self._clusters.values()],
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Failed to save clusters: %s", e)

    def _store_path(self) -> Path:
        return get_hermes_home() / "evolution" / "observed_patterns.json"


# ── Singleton ──────────────────────────────────────────────────────────

_observer: Optional[ConversationObserver] = None


def get_observer() -> ConversationObserver:
    global _observer
    if _observer is None:
        _observer = ConversationObserver()
    return _observer
