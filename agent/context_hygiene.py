"""Content-safe context-layer hygiene audit for Hermes.

The audit intentionally returns metadata only. It separates SOUL identity,
project context, skills, memory/profile, and session/trace evidence without
copying raw prompt text, memory rows, skill bodies, paths, or session content.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
CONTENT_POLICY = "metadata_only"

_TASK_PROGRESS_RE = re.compile(
    r"(\bPR\s*#?\d+\b|\bissue\s*#?\d+\b|\bcommit\b|\bHEAD\b|\bSHA\b|"
    r"\bfixed\b|\bimplemented\b|\bcompleted\b|\bphase\b|\btask\b|\btodo\b|"
    r"\b\d+\s+tests?\s+(passed|failed|skipped)\b)",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"(`[^`]+`|\brun\s+[`\w./:-]+|\binstall\b|\bsetup\b|\bworkflow\b|\bsteps?\b)",
    re.IGNORECASE,
)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _file_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"present": False, "bytes": 0, "line_count": 0, "sha256": None}
    text = _read_text(path)
    return {
        "present": True,
        "bytes": path.stat().st_size,
        "line_count": len(text.splitlines()),
        "sha256": _sha256_text(text),
    }


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except OSError:
        return 0


def _issue(layer: str, code: str, *, severity: str = "warning", count: int = 1) -> Dict[str, Any]:
    return {"layer": layer, "code": code, "severity": severity, "count": count}


def _frontmatter_fields(text: str) -> Dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fields: Dict[str, str] = {}
    for raw in match.group(1).splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


def _audit_soul(hermes_home: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = _file_metadata(hermes_home / "SOUL.md")
    if not meta["present"]:
        issues.append(_issue("soul", "soul_missing"))
    return meta


def _file_has_prompt_content(path: Path) -> bool:
    return path.is_file() and bool(_read_text(path).strip())


def _find_git_root(cwd: Path) -> Optional[Path]:
    resolved = cwd.resolve()
    for directory in [resolved, *resolved.parents]:
        if (directory / ".git").exists():
            return directory
    return None


def _discover_project_context_files(cwd: Path) -> List[Path]:
    """Mirror prompt-builder project-context discovery without returning content."""
    resolved = cwd.resolve()

    stop_at = _find_git_root(resolved)
    hermes_search_stopped = False
    for directory in [resolved, *resolved.parents]:
        for name in (".hermes.md", "HERMES.md"):
            candidate = directory / name
            if candidate.is_file():
                if _file_has_prompt_content(candidate):
                    return [candidate]
                hermes_search_stopped = True
                break
        if hermes_search_stopped:
            break
        if stop_at is not None and directory == stop_at:
            break

    for names in (("AGENTS.md", "agents.md"), ("CLAUDE.md", "claude.md")):
        for name in names:
            candidate = resolved / name
            if _file_has_prompt_content(candidate):
                return [candidate]

    cursorrules = resolved / ".cursorrules"
    rules_dir = resolved / ".cursor" / "rules"
    cursor_rules = sorted(rules_dir.glob("*.mdc")) if rules_dir.is_dir() else []
    present = []
    if _file_has_prompt_content(cursorrules):
        present.append(cursorrules)
    present.extend(path for path in cursor_rules if _file_has_prompt_content(path))
    return present


def _audit_project_context(cwd: Optional[Path], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "present": False,
        "cwd_present": cwd is not None,
        "cwd_sha256": _sha256_text(str(cwd)) if cwd is not None else None,
        "file_count": 0,
        "bytes": 0,
        "line_count": 0,
        "sha256": None,
    }
    if cwd is None:
        issues.append(_issue("project_context", "project_context_missing"))
        return result

    present = _discover_project_context_files(cwd)
    if not present:
        issues.append(_issue("project_context", "project_context_missing"))
        return result

    texts = [_read_text(path) for path in present]
    joined = "\n".join(texts)
    result.update({
        "present": True,
        "file_count": len(present),
        "bytes": sum(path.stat().st_size for path in present),
        "line_count": sum(len(text.splitlines()) for text in texts),
        "sha256": _sha256_text(joined),
    })
    return result


def _iter_skill_files(skills_dir: Path) -> Iterable[Path]:
    if not skills_dir.exists():
        return []
    return sorted(path for path in skills_dir.rglob("SKILL.md") if path.is_file())


def _audit_skills(hermes_home: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    skill_files = list(_iter_skill_files(hermes_home / "skills"))
    invalid = 0
    duplicate_names = 0
    seen: set[str] = set()
    named = 0
    total_bytes = 0
    for path in skill_files:
        text = _read_text(path)
        total_bytes += path.stat().st_size
        fields = _frontmatter_fields(text)
        name = fields.get("name")
        description = fields.get("description")
        if name:
            named += 1
            if name in seen:
                duplicate_names += 1
            seen.add(name)
        if not name or not description:
            invalid += 1
    if invalid:
        issues.append(_issue("skills", "skill_frontmatter_incomplete", count=invalid))
    if duplicate_names:
        issues.append(_issue("skills", "skill_duplicate_names", count=duplicate_names))
    return {
        "skill_count": len(skill_files),
        "named_skill_count": named,
        "invalid_frontmatter_count": invalid,
        "duplicate_name_count": duplicate_names,
        "bytes": total_bytes,
    }


def _audit_memory(hermes_home: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    files = [path for path in [hermes_home / "MEMORY.md", hermes_home / "USER.md"] if path.exists() and path.is_file()]
    task_hits = 0
    procedure_hits = 0
    total_lines = 0
    total_bytes = 0
    digest_parts: List[str] = []
    for path in files:
        text = _read_text(path)
        task_hits += len(_TASK_PROGRESS_RE.findall(text))
        procedure_hits += len(_PROCEDURE_RE.findall(text))
        total_lines += len(text.splitlines())
        total_bytes += path.stat().st_size
        digest_parts.append(text)
    if task_hits:
        issues.append(_issue("memory", "memory_contains_task_progress", count=task_hits))
    if procedure_hits:
        issues.append(_issue("memory", "memory_contains_procedure", count=procedure_hits))
    return {
        "memory_files_present": [path.name for path in files],
        "file_count": len(files),
        "bytes": total_bytes,
        "line_count": total_lines,
        "sha256": _sha256_text("\n".join(digest_parts)) if digest_parts else None,
        "task_progress_hits": task_hits,
        "procedure_hits": procedure_hits,
    }


def _audit_sessions_traces(hermes_home: Path) -> Dict[str, Any]:
    sessions_dir = hermes_home / "sessions"
    session_count = 0
    if sessions_dir.exists():
        session_count = sum(1 for path in sessions_dir.glob("session_*.json") if path.is_file())
    harness_dir = hermes_home / "harness"
    return {
        "session_count": session_count,
        "turn_trace_count": _count_jsonl_lines(harness_dir / "turn-traces.jsonl"),
        "harness_event_count": _count_jsonl_lines(harness_dir / "harness-events.jsonl"),
        "replay_case_count": _count_jsonl_lines(harness_dir / "replay-corpus.jsonl"),
    }


def audit_context_hygiene(
    *,
    hermes_home: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a metadata-only audit of context-layer separation.

    The result intentionally excludes raw content and raw paths. Callers can use
    issue codes to decide what to inspect manually without leaking prompt,
    memory, or project text through dashboard/control-plane surfaces.
    """
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    if cwd is not None:
        project_cwd = Path(cwd)
    else:
        project_cwd = Path(os.getenv("TERMINAL_CWD") or os.getcwd())
    issues: List[Dict[str, Any]] = []
    layers = {
        "soul": _audit_soul(home, issues),
        "project_context": _audit_project_context(project_cwd, issues),
        "skills": _audit_skills(home, issues),
        "memory": _audit_memory(home, issues),
        "sessions_traces": _audit_sessions_traces(home),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "content_policy": CONTENT_POLICY,
        "layers": layers,
        "issues": issues,
        "issue_count": len(issues),
    }
