"""Generate compact task capsules for coding-agent handoffs."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hermes_constants import get_hermes_home


DEFAULT_WORD_BUDGET = 1200
DEFAULT_MAX_FILES = 12

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".turbo",
    "coverage",
    "htmlcov",
}

_SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".sqlite",
    ".db",
    ".lock",
}

_IMPORTANT_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "package.json",
    "pnpm-workspace.yaml",
    "uv.lock",
    "requirements.txt",
    "Makefile",
]

_MEMORY_FILES = ["MEMORY.md", "LEARNINGS.md", "AGENTS.md", "SOUL.md"]


@dataclass(frozen=True)
class RelevantFile:
    path: Path
    reason: str
    score: int
    snippet: str = ""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "task-capsule"


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "task",
        "build",
        "generate",
        "implement",
        "agent",
        "agents",
    }
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        if word in stop or word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out[:20]


def _is_candidate_file(path: Path) -> bool:
    if any(part in _SKIP_DIRS for part in path.parts):
        return False
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return False
    return True


def _iter_candidate_files(repo_path: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            try:
                rel = path.relative_to(repo_path)
            except ValueError:
                continue
            if _is_candidate_file(rel):
                yield path


def _safe_read(path: Path, limit: int = 20000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _line_snippet(text: str, terms: Iterable[str], max_lines: int = 4) -> str:
    term_list = [t.lower() for t in terms]
    lines = text.splitlines()
    matches: list[str] = []
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()
        if any(term in lower for term in term_list):
            cleaned = line.strip()
            if cleaned:
                matches.append(f"L{idx}: {cleaned[:180]}")
        if len(matches) >= max_lines:
            break
    return "\n".join(matches)


def discover_relevant_files(repo_path: Path, title: str, max_files: int) -> list[RelevantFile]:
    terms = _tokenize(title)
    candidates: dict[Path, RelevantFile] = {}

    for name in _IMPORTANT_FILES:
        path = repo_path / name
        if path.exists() and path.is_file():
            candidates[path] = RelevantFile(path, "project guidance/config", 100)

    for path in _iter_candidate_files(repo_path):
        rel = path.relative_to(repo_path)
        rel_text = str(rel).lower()
        score = 0
        reasons: list[str] = []
        for term in terms:
            if term in rel_text:
                score += 20
                reasons.append(f"name matches '{term}'")
        if score <= 0:
            content = _safe_read(path, limit=12000)
            hits = sum(1 for term in terms if term in content.lower())
            if hits:
                score += hits * 8
                reasons.append(f"content matches {hits} task term(s)")
        if score > 0:
            content = _safe_read(path, limit=12000)
            snippet = _line_snippet(content, terms)
            existing = candidates.get(path)
            candidate = RelevantFile(path, "; ".join(reasons), score, snippet)
            if existing is None or candidate.score > existing.score:
                candidates[path] = candidate

    return sorted(candidates.values(), key=lambda f: (-f.score, str(f.path)))[:max_files]


def discover_memory_snippets(title: str, max_snippets: int = 6) -> list[tuple[Path, str]]:
    terms = _tokenize(title)
    hermes_home = get_hermes_home()
    snippets: list[tuple[Path, str]] = []
    for name in _MEMORY_FILES:
        path = hermes_home / name
        if not path.exists():
            continue
        text = _safe_read(path, limit=50000)
        snippet = _line_snippet(text, terms, max_lines=2)
        if snippet:
            snippets.append((path, snippet))
        if len(snippets) >= max_snippets:
            break
    return snippets


def discover_test_commands(repo_path: Path) -> list[str]:
    commands: list[str] = []
    if (repo_path / "scripts" / "run_tests.sh").exists():
        commands.append("scripts/run_tests.sh")
    if (repo_path / "pyproject.toml").exists() or (repo_path / "pytest.ini").exists():
        commands.append("python -m pytest -q")
    if (repo_path / "package.json").exists():
        text = _safe_read(repo_path / "package.json", limit=20000)
        if '"test"' in text:
            commands.append("npm test")
    if (repo_path / "Makefile").exists():
        text = _safe_read(repo_path / "Makefile", limit=20000)
        if re.search(r"^test:", text, re.MULTILINE):
            commands.append("make test")
    return commands or ["<fill in the project-specific test command>"]


def _parse_lines(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\n|;", value)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _trim_to_budget(markdown: str, budget: int) -> str:
    if _word_count(markdown) <= budget:
        return markdown
    remaining = max(0, budget - 6)
    kept: list[str] = []
    for line in markdown.splitlines():
        words = line.split()
        if len(words) <= remaining:
            kept.append(line)
            remaining -= len(words)
            continue
        if remaining > 0:
            kept.append(" ".join(words[:remaining]))
        break
    kept.append("")
    kept.append("[Trimmed to fit word budget.]")
    return "\n".join(kept).rstrip() + "\n"


def build_capsule(
    *,
    title: str,
    repo_path: Path,
    goal: str | None = None,
    constraints: list[str] | None = None,
    acceptance: list[str] | None = None,
    non_goals: list[str] | None = None,
    word_budget: int = DEFAULT_WORD_BUDGET,
    max_files: int = DEFAULT_MAX_FILES,
) -> str:
    repo_path = repo_path.resolve()
    relevant_files = discover_relevant_files(repo_path, title, max_files=max_files)
    memory_snippets = discover_memory_snippets(title)
    test_commands = discover_test_commands(repo_path)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    constraints = constraints or []
    acceptance = acceptance or []
    non_goals = non_goals or []

    if not constraints:
        constraints = [
            "Keep the change focused on the stated goal; avoid unrelated refactors.",
            "Preserve existing project conventions and command surfaces.",
            "Do not commit directly to main; leave changes ready for review unless explicitly told otherwise.",
        ]
    if not acceptance:
        acceptance = [
            "Implementation satisfies the goal and can be handed to a coding agent without extra context.",
            "Relevant tests or smoke checks pass, or any skipped checks are documented.",
        ]
    if not non_goals:
        non_goals = ["Broad redesigns outside the task scope.", "External publishing or credential changes."]

    lines: list[str] = []
    lines.append(f"# Task capsule: {title}")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append(f"Repository: `{repo_path}`")
    lines.append(f"Word budget: {word_budget}")
    lines.append("")
    lines.append("## Goal")
    lines.append(goal.strip() if goal else title.strip())
    lines.append("")
    lines.append("## Non-goals")
    lines.extend(f"- {item}" for item in non_goals)
    lines.append("")
    lines.append("## Constraints")
    lines.extend(f"- {item}" for item in constraints)
    lines.append("")
    lines.append("## Relevant files")
    if relevant_files:
        for item in relevant_files:
            rel = item.path.relative_to(repo_path)
            lines.append(f"- `{rel}` — {item.reason or 'likely relevant'}")
            if item.snippet:
                for snippet_line in item.snippet.splitlines()[:3]:
                    lines.append(f"  - {snippet_line}")
    else:
        lines.append("- No obvious file matches found; start with repository docs and search for task terms.")
    lines.append("")
    lines.append("## Local memory / docs signals")
    if memory_snippets:
        for path, snippet in memory_snippets:
            display = path if path.is_absolute() else path.resolve()
            lines.append(f"- `{display}`")
            for snippet_line in snippet.splitlines():
                lines.append(f"  - {snippet_line}")
    else:
        lines.append("- No matching MEMORY.md / LEARNINGS.md snippets found for the task terms.")
    lines.append("")
    lines.append("## Commands to run")
    lines.extend(f"- `{cmd}`" for cmd in test_commands)
    lines.append("")
    lines.append("## Acceptance criteria")
    lines.extend(f"- {item}" for item in acceptance)
    lines.append("")
    lines.append("## Expected output")
    lines.append("A focused code change plus a short handoff listing changed files, tests run, and any remaining risks.")
    lines.append("")
    lines.append("## Copy/paste prompt")
    lines.append("Use this capsule as the complete handoff. First inspect the listed files, then implement only the Goal while respecting Non-goals and Constraints. Run the listed commands or explain why they were not run.")

    return _trim_to_budget("\n".join(lines).rstrip() + "\n", word_budget)


def default_output_path(title: str) -> Path:
    return get_hermes_home() / "task-capsules" / f"{slugify(title)}.md"


def write_capsule(markdown: str, output_path: Path, overwrite: bool = False) -> Path:
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite or --output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def capsule_command(args: argparse.Namespace) -> None:
    title = args.title.strip()
    repo_path = Path(args.repo).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise SystemExit(f"Repository path does not exist or is not a directory: {repo_path}")

    word_budget = max(250, int(args.word_budget or DEFAULT_WORD_BUDGET))
    markdown = build_capsule(
        title=title,
        repo_path=repo_path,
        goal=args.goal,
        constraints=_parse_lines(args.constraints),
        acceptance=_parse_lines(args.acceptance),
        non_goals=_parse_lines(args.non_goals),
        word_budget=word_budget,
        max_files=max(1, int(args.max_files or DEFAULT_MAX_FILES)),
    )
    output_path = Path(args.output).expanduser() if args.output else default_output_path(title)
    written = write_capsule(markdown, output_path, overwrite=args.overwrite)
    print(f"Wrote task capsule: {written}")
    print(f"Words: {_word_count(markdown)} / {word_budget}")


__all__ = [
    "DEFAULT_WORD_BUDGET",
    "build_capsule",
    "capsule_command",
    "default_output_path",
    "discover_relevant_files",
    "discover_test_commands",
    "slugify",
    "write_capsule",
]
