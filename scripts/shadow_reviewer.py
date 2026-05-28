#!/usr/bin/env python3
"""
SHADOW REVIEWER — qwen3-coder:30b local model review layer

Runs asynchronously alongside the primary agent. Reviews every code change
before merge, flags architecture drift, inconsistencies, and potential bugs.
Cost: ZERO (local model). Speed: fast (~2-5s per review).

Architecture:
  Primary Agent → makes code change → fires async shadow review
  Shadow Reviewer → reads diff + context → returns structured findings
  Primary Agent → addresses findings OR overrides with justification
  Ring Quality Gate → final check → output delivered to user

This is Decision 6.9 Level 3 implementation.
"""

import json
import os
import sys
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

# ── Types ──────────────────────────────────────────────────────────

@dataclass
class ReviewFinding:
    severity: str          # "critical" | "warning" | "info" | "style"
    category: str          # "architecture" | "logic" | "consistency" | "safety" | "style"
    message: str           # Human-readable description
    location: str = ""     # File, function, or line reference
    suggested_fix: str = "" # Recommended change
    confidence: float = 0.0 # 0.0 - 1.0 how sure the reviewer is
    auto_fixable: bool = False  # Can this be fixed automatically?

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
            "confidence": round(self.confidence, 2),
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class ReviewResult:
    """Complete result from a shadow review pass."""
    request_id: str = ""
    timestamp: str = ""
    reviewed_diff: str = ""
    findings: List[ReviewFinding] = field(default_factory=list)
    overall_score: float = 1.0  # 1.0 = perfect, 0.0 = reject
    blocked: bool = False       # True if any CRITICAL finding without override
    primary_model_output: str = ""  # What the primary agent produced
    shadow_model_raw_response: str = ""  # What 30B actually said
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def has_blocking_issues(self) -> bool:
        return self.blocked or self.critical_count > 0

    def summary(self) -> str:
        if not self.findings:
            return "✅ Shadow review: no issues found"
        lines = [f"Shadow review: {self.critical_count} critical, {self.warning_count} warnings"]
        for f in self.findings:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵", "style": "⚪"}.get(f.severity, "❓")
            loc = f" ({f.location})" if f.location else ""
            lines.append(f"  {icon} [{f.severity.upper()}] {f.message}{loc}")
            if f.suggested_fix:
                lines.append(f"      Fix: {f.suggested_fix}")
        lines.append(f"\n  Overall score: {self.overall_score:.0%}")
        if self.blocked:
            lines.append("  🚫 BLOCKED — requires resolution before merge")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "reviewed_diff": self.reviewed_diff,
            "findings": [f.to_dict() for f in self.findings],
            "overall_score": self.overall_score,
            "blocked": self.blocked,
            "primary_model_output": self.primary_model_output,
            "shadow_model_raw_response": self.shadow_model_raw_response,
            "metadata": self.metadata,
        }


# ── Shadow Review Prompt ──────────────────────────────────────────

SHADOW_REVIEW_PROMPT = """You are a code review specialist. Review the following code change
against the project context provided. Be thorough but practical — flag
real issues, not style preferences.

RULES:
1. Focus on: logic errors, architecture drift, inconsistent patterns,
   safety/security issues, performance problems, and broken assumptions.
2. Do NOT flag style issues (naming, formatting) unless they cause bugs.
3. If you find a real issue, provide a specific suggested fix.
4. Rate each finding: critical / warning / info / style.
5. Give an honest overall confidence score (0.0-1.0) on the change.

CODE CHANGE (diff):
{changes}

FULL FILE (after change):
{file_after}

PROJECT CONTEXT:
{project_context}

Prior changes in this session:
{session_history}

Respond in this exact JSON format:
{{
  "findings": [
    {{
      "severity": "critical|warning|info|style",
      "category": "architecture|logic|consistency|safety|style|performance",
      "message": "Description of the issue",
      "location": "file.py:line or function name",
      "suggested_fix": "How to fix it",
      "auto_fixable": true/false
    }}
  ],
  "score": 0.0-1.0
}}
"""


# ── Ollama Client (local communication) ───────────────────────────

def call_ollama_30b(prompt: str,
                    model: str = "qwen3-coder:30b-a3b-q4_k_M",
                    timeout: int = 30) -> Optional[str]:
    """
    Call the local 30B model via Ollama API.
    Returns the raw response text, or None on failure.

    This deliberately uses the requests library for simplicity.
    In production, this could use httpx or aiohttp for async.
    """
    try:
        import json
        import urllib.request
        import urllib.error

        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        url = f"{ollama_url}/api/generate"

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temp for code review — deterministic
                "num_ctx": 8192,     # Context window
                "top_p": 0.9,
            },
            "format": "json"  # Ask model to return JSON
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")

    except Exception as e:
        logger = ShadowReviewer._get_logger()
        logger.error(f"Shadow reviewer Ollama call failed: {e}")
        return None


# ── Core Reviewer ─────────────────────────────────────────────────

class ShadowReviewer:
    """
    Async shadow reviewer that checks every code change using local 30B.

    Usage:
        reviewer = ShadowReviewer()
        # Fire and forget — runs in background
        asyncio.create_task(reviewer.review_change(diff, file_after, context))
        # Check results anytime
        result = reviewer.get_latest_result(request_id)
    """

    _instance = None
    _lock = asyncio.Lock()

    def __init__(self, model: str = "qwen3-coder:30b-a3b-q4_k_M"):
        self.model = model
        self._results: Dict[str, ReviewResult] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._total_reviews = 0
        self._total_findings = 0
        self._blocked_count = 0

    @classmethod
    def get_instance(cls) -> "ShadowReviewer":
        """Singleton — one reviewer per process."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _get_logger():
        import logging
        return logging.getLogger("shadow_reviewer")

    # ── Public API ─────────────────────────────────────────────

    def request_review(self,
                       changes: str,
                       file_after: str,
                       project_context: str = "",
                       session_history: str = "",
                       request_id: str = "") -> str:
        """
        Request a shadow review. Returns a request_id for checking results.
        This is NON-BLOCKING — the review happens in the background.
        """
        if not request_id:
            request_id = f"review_{int(time.time()*1000)}"

        result = ReviewResult(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reviewed_diff=changes,
            metadata={"model": self.model, "status": "queued"}
        )
        self._results[request_id] = result

        # Fire review in background
        asyncio.create_task(
            self._run_review(result, changes, file_after, project_context, session_history)
        )

        return request_id

    def get_result(self, request_id: str) -> Optional[ReviewResult]:
        """Get the result of a review. Returns None if not found or still running."""
        return self._results.get(request_id)

    async def wait_for_result(self, request_id: str, timeout: float = 15.0) -> Optional[ReviewResult]:
        """Wait for a review to complete, with timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self._results.get(request_id)
            if result and result.metadata.get("status") in ("complete", "failed"):
                return result
            await asyncio.sleep(0.5)
        return self._results.get(request_id)

    def get_stats(self) -> Dict:
        """Get overall reviewer statistics."""
        return {
            "total_reviews": self._total_reviews,
            "total_findings": self._total_findings,
            "blocked_count": self._blocked_count,
            "avg_findings_per_review": round(self._total_findings / max(1, self._total_reviews), 2),
            "model": self.model,
        }

    # ── Internal ──────────────────────────────────────────────

    async def _run_review(self,
                          result: ReviewResult,
                          changes: str,
                          file_after: str,
                          project_context: str,
                          session_history: str):
        """Execute the actual review against 30B."""
        try:
            result.metadata["status"] = "running"
            self._total_reviews += 1

            prompt = SHADOW_REVIEW_PROMPT.format(
                changes=changes,
                file_after=file_after,
                project_context=project_context[:2000],  # Limit context size
                session_history=session_history[:1000],
            )

            # Call local 30B
            raw_response = await asyncio.to_thread(
                call_ollama_30b, prompt, self.model, 30
            )

            if raw_response is None:
                result.metadata["status"] = "failed"
                result.metadata["error"] = "Ollama call failed"
                self._log_result(result, "FAILED")
                return

            result.shadow_model_raw_response = raw_response

            # Parse the JSON response
            try:
                parsed = json.loads(raw_response.strip())
                findings = parsed.get("findings", [])
                score = parsed.get("score", 1.0)
            except (json.JSONDecodeError, AttributeError):
                # Model didn't return valid JSON — create a generic warning
                findings = [{
                    "severity": "warning",
                    "category": "consistency",
                    "message": f"Shadow model couldn't parse structured review. Raw response: {raw_response[:200]}",
                    "location": "",
                    "suggested_fix": "Retry review or check model availability",
                    "auto_fixable": False
                }]
                score = 0.5

            # Convert to ReviewFinding objects
            for f in findings:
                result.findings.append(ReviewFinding(
                    severity=f.get("severity", "info"),
                    category=f.get("category", "consistency"),
                    message=f.get("message", ""),
                    location=f.get("location", ""),
                    suggested_fix=f.get("suggested_fix", ""),
                    confidence=f.get("confidence", 0.5),
                    auto_fixable=f.get("auto_fixable", False)
                ))

            result.overall_score = score
            finding_count = len(findings)
            self._total_findings += finding_count

            # Determine if blocked
            critical = [f for f in result.findings if f.severity == "critical"]
            if critical:
                result.blocked = True
                self._blocked_count += 1

            result.metadata["status"] = "complete"
            self._log_result(result, "COMPLETE")

        except Exception as e:
            result.metadata["status"] = "failed"
            result.metadata["error"] = str(e)
            self._log_result(result, "ERROR")
            logger = self._get_logger()
            logger.error(f"Shadow review failed: {e}", exc_info=True)

    def _log_result(self, result: ReviewResult, status: str):
        """Log the review result to file."""
        import json
        log_dir = Path(os.path.expanduser("~/.hermes/logs"))
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "shadow_reviewer.jsonl"

        entry = {
            "timestamp": result.timestamp,
            "request_id": result.request_id,
            "status": status,
            "score": result.overall_score,
            "critical": result.critical_count,
            "warnings": result.warning_count,
            "findings_count": len(result.findings),
            "blocked": result.blocked,
            "model": result.metadata.get("model", self.model),
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# ── Integration Hook ─────────────────────────────────────────────

def review_code_change(changes: str,
                       file_after: str,
                       project_context: str = "",
                       session_history: str = "",
                       request_id: str = "") -> str:
    """
    Public API for the gateway to request a shadow review.
    Called from gateway_integration.py during code modification operations.

    Returns: request_id (use to check results with get_result())
    """
    reviewer = ShadowReviewer.get_instance()
    return reviewer.request_review(changes, file_after, project_context, session_history, request_id)


def get_review_result(request_id: str) -> Optional[dict]:
    """Get a review result as a dict (for JSON serialization)."""
    reviewer = ShadowReviewer.get_instance()
    result = reviewer.get_result(request_id)
    if result:
        return result.to_dict() if hasattr(result, 'to_dict') else {
            "request_id": result.request_id,
            "status": result.metadata.get("status", "unknown"),
            "score": result.overall_score,
            "findings": [f.to_dict() for f in result.findings],
            "blocked": result.blocked,
            "summary": result.summary()
        }
    return None


def get_reviewer_stats() -> Dict:
    """Get reviewer statistics."""
    reviewer = ShadowReviewer.get_instance()
    return reviewer.get_stats()


# ── Self-Test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("SHADOW REVIEWER — SELF TEST")
    print("=" * 60)
    print()

    # Test 1: Verbal test (no Ollama required)
    print("Test 1: Module import and class instantiation...")
    reviewer = ShadowReviewer.get_instance()
    print(f"  ✅ Created reviewer instance (model: {reviewer.model})")
    print(f"  ✅ Initial stats: {reviewer.get_stats()}")
    print()

    # Test 2: Simulated review (won't call Ollama without it running)
    print("Test 2: Simulated code change review...")
    fake_diff = """- def authenticate(username, password):
-     if username == "admin" and password == "password":
-         return True
-     return False
+ def authenticate(username, password):
+     query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
+     result = db.execute(query)
+     return len(result) > 0"""

    fake_file = """def authenticate(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return len(result) > 0"""

    request_id = reviewer.request_review(
        changes=fake_diff,
        file_after=fake_file,
        project_context="Authentication module for web application using SQLite database",
        session_history="User asked to add login functionality to the auth module"
    )
    print(f"  ✅ Review requested: {request_id}")
    print()

    # Test 3: Check that it's queued
    result = reviewer.get_result(request_id)
    if result:
        print(f"  ✅ Result status: {result.metadata['status']}")
    print()

    # Test 4: Stats
    print("Test 3: Stats tracking...")
    print(f"  ✅ Stats: {reviewer.get_stats()}")
    print()

    print("=" * 60)
    print("SELF TEST COMPLETE")
    print("=" * 60)
    print()
    print("Note: Full integration test requires Ollama running with")
    print("qwen3-coder:30b-a3b-q4_k_M model loaded.")
    print()
    print("To test with live model:")
    print("  1. ollama run qwen3-coder:30b")
    print("  2. python3 scripts/shadow_reviewer.py")
    print("  3. Check ~/.hermes/logs/shadow_reviewer.jsonl for results")