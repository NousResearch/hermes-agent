"""
Proactive Task Suggestions — Next-step recommendations based on project context.

After the agent completes significant work, this module analyzes what was done
and proactively suggests relevant next actions (inspired by Claude Code's
proactive capabilities).

Design:
- Triggered after N completed tool-call cycles (configurable)
- Analyzes recent file changes, git diff, project structure
- Suggests contextually relevant next steps
- Suggestions are injected as a system hint, not forced
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Project-type detection patterns
_PROJECT_SIGNATURES: dict[str, dict[str, Any]] = {
    "python_package": {
        "files": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
        "dirs": ["src/", "tests/", "docs/"],
        "suggestions": [
            "Run the test suite to verify everything works",
            "Build documentation with docs/",
            "Check for type errors with mypy",
            "Publish to PyPI when ready",
        ],
    },
    "web_app": {
        "files": ["package.json", "webpack.config.js", "vite.config.ts", "next.config.js", "nuxt.config.ts"],
        "dirs": ["src/", "components/", "pages/", "app/"],
        "suggestions": [
            "Run the development server and test in browser",
            "Check for accessibility issues",
            "Run Lighthouse audit",
            "Test responsive design on mobile breakpoints",
        ],
    },
    "react": {
        "files": ["package.json"],
        "keywords": ["react", "react-dom", "jsx", "tsx"],
        "suggestions": [
            "Run TypeScript type check",
            "Test component in Storybook",
            "Check bundle size with source-map-explorer",
            "Verify accessibility with axe-core",
        ],
    },
    "fastapi": {
        "files": ["main.py", "app.py", "api/", "routers/"],
        "keywords": ["fastapi", "uvicorn", "pydantic"],
        "suggestions": [
            "Test API endpoints with curl or Swagger UI",
            "Run pytest with --cov to check coverage",
            "Check for security issues with bandit",
            "Profile performance with locust",
        ],
    },
    "rust": {
        "files": ["Cargo.toml", "Cargo.lock"],
        "suggestions": [
            "Run cargo clippy for linting",
            "Run cargo test with -- --nocapture",
            "Build release binary with cargo build --release",
            "Check for security issues with cargo audit",
        ],
    },
    "node_express": {
        "files": ["package.json", "server.js", "app.js", "routes/"],
        "keywords": ["express", "koa", "fastify"],
        "suggestions": [
            "Run the server and test endpoints",
            "Check for memory leaks with clinic.js",
            "Run load tests",
            "Verify CORS configuration",
        ],
    },
    "llm_project": {
        "files": ["llm_config.json", "prompts/", "agents/", "tools/"],
        "dirs": ["prompts/", "agents/", "skills/"],
        "suggestions": [
            "Test the agent with a complex query",
            "Run evaluation suite",
            "Check prompt injection defenses",
            "Benchmark response quality",
        ],
    },
}


def _detect_project_type(working_dir: str | Path) -> str:
    """Detect the project type based on files and structure."""
    try:
        wd = Path(working_dir)
        if not wd.exists():
            return "unknown"

        # List relevant files
        entries = {p.name for p in wd.iterdir()}
        subdirs = {p.name for p in wd.iterdir() if p.is_dir()}

        # Check each project type
        for ptype, sig in _PROJECT_SIGNATURES.items():
            file_matches = sum(1 for f in sig.get("files", []) if f in entries)
            dir_matches = sum(1 for d in sig.get("dirs", []) if d.rstrip("/") in subdirs)

            if file_matches >= 1 or dir_matches >= 2:
                return ptype

        # Check keywords in pyproject.toml / package.json
        for cfg_file in ["pyproject.toml", "package.json"]:
            cfg_path = wd / cfg_file
            if cfg_path.exists():
                try:
                    content = cfg_path.read_text(encoding="utf-8")
                    for ptype, sig in _PROJECT_SIGNATURES.items():
                        for kw in sig.get("keywords", []):
                            if kw in content.lower():
                                return ptype
                except Exception:
                    pass

    except Exception as e:
        logger.debug("Project type detection failed: %s", e)

    return "unknown"


def _get_git_status(working_dir: str | Path) -> dict[str, Any]:
    """Get git status summary for suggestions."""
    try:
        wd = Path(working_dir)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        changed = [l.strip() for l in result.stdout.splitlines() if l.strip()]

        # Count by type
        modified = [l for l in changed if l.startswith(" M")]
        added = [l for l in changed if l.startswith("??") or l.startswith("A ")]
        deleted = [l for l in changed if l.startswith(" D")]

        return {
            "has_changes": len(changed) > 0,
            "modified": len(modified),
            "added": len(added),
            "deleted": len(deleted),
            "total": len(changed),
        }
    except Exception:
        return {"has_changes": False, "modified": 0, "added": 0, "deleted": 0, "total": 0}


def _get_git_branch(working_dir: str | Path) -> Optional[str]:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class ProactiveTaskSuggester:
    """
    Suggest next steps based on project context and recent activity.

    Usage::

        suggester = ProactiveTaskSuggester(working_dir="/path/to/project")

        # After each tool call cycle:
        if suggester.should_suggest():
            suggestions = suggester.get_suggestions()
            if suggestions:
                return f"\n\n💡 Suggestions: {suggestions}"
    """

    def __init__(
        self,
        working_dir: Optional[str | Path] = None,
        min_cycles_before_first: int = 3,
        min_cycles_between: int = 10,
        max_suggestions: int = 3,
    ):
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self.min_cycles_before_first = min_cycles_before_first
        self.min_cycles_between = min_cycles_between
        self.max_suggestions = max_suggestions

        self._cycle_count = 0
        self._last_suggestion_at = 0.0
        self._suggestion_count = 0
        self._lock = threading.Lock()
        self._project_type: Optional[str] = None
        self._suggestions_cache: list[str] = []

    def record_cycle(self) -> None:
        """Call after each tool-call iteration."""
        with self._lock:
            self._cycle_count += 1

    def should_suggest(self) -> bool:
        """Check if we should generate suggestions now."""
        with self._lock:
            if self._suggestions_cache:
                # Already have cached suggestions
                return True

            if self._cycle_count < self.min_cycles_before_first:
                return False

            elapsed = time.time() - self._last_suggestion_at
            cycles_since_last = self._cycle_count - self._suggestion_count * self.min_cycles_between

            return cycles_since_last >= self.min_cycles_between and elapsed > 30

    def get_suggestions(self) -> list[str]:
        """Get current suggestions, regenerating if stale."""
        with self._lock:
            if self._suggestions_cache:
                return self._suggestions_cache[:self.max_suggestions]

            # Generate fresh suggestions
            suggestions = self._generate_suggestions()
            self._suggestions_cache = suggestions
            self._last_suggestion_at = time.time()
            self._suggestion_count += 1

            return suggestions[:self.max_suggestions]

    def _generate_suggestions(self) -> list[str]:
        """Generate suggestions based on project context."""
        suggestions: list[str] = []

        # Project-type based suggestions
        if self._project_type is None:
            self._project_type = _detect_project_type(self.working_dir)

        if self._project_type != "unknown":
            sig = _PROJECT_SIGNATURES.get(self._project_type, {})
            suggestions.extend(sig.get("suggestions", []))

        # Git-based suggestions
        git_status = _get_git_status(self.working_dir)
        branch = _get_git_branch(self.working_dir)

        if git_status["has_changes"]:
            suggestions.append(
                f"Commit {git_status['total']} changes with a descriptive message"
            )

        if branch:
            if branch != "main" and branch != "master":
                suggestions.append(f"Review changes before merging to main")

        if git_status["modified"] > 5:
            suggestions.append("Run full test suite before committing")

        if git_status["deleted"] > 0:
            suggestions.append("Verify deleted files are intentional")

        # Check for TODO/FIXME comments
        todo_count = self._count_todos()
        if todo_count > 0:
            suggestions.append(f"Address {todo_count} TODO/FIXME comment(s) in the codebase")

        # Deduplicate and shuffle slightly (prefer variety)
        seen = set()
        unique: list[str] = []
        for s in suggestions:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique[: self.max_suggestions * 2]  # Return extra for filtering

    def _count_todos(self) -> int:
        """Count TODO/FIXME/HACK comments in source files."""
        try:
            # Find source files
            exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp"}
            count = 0
            for ext in exts:
                result = subprocess.run(
                    ["find", ".", "-name", f"*{ext}", "-type", "f"],
                    cwd=str(self.working_dir),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
                for fpath in files[:100]:  # Limit to first 100 files
                    try:
                        content = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                        for marker in ["TODO", "FIXME", "HACK", "XXX"]:
                            count += content.upper().count(marker)
                    except Exception:
                        pass
            return count
        except Exception:
            return 0

    def append_suggestions_to_result(self, result: str) -> str:
        """
        If it's time to suggest, append suggestions to a tool result string.
        Call this after significant tool results.
        """
        if not self.should_suggest():
            return result

        suggestions = self.get_suggestions()
        if not suggestions:
            return result

        suggestion_text = "\n".join(f"  💡 {s}" for s in suggestions[:self.max_suggestions])
        return (
            f"{result}\n\n"
            f"\n━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Next steps you might consider:\n"
            f"{suggestion_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
