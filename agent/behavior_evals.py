"""Deterministic trajectory checks for repository behavior specifications.

These checks intentionally use only structured tool-call evidence. They do not
infer mutations from arbitrary shell commands and do not call an LLM judge, so
an evaluation can run after a session without changing the runtime prompt or
its cache prefix.
"""

from __future__ import annotations

import json
import ntpath
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional


_CODE_SUFFIXES = {
    ".bash", ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h",
    ".hpp", ".html", ".java", ".js", ".jsx", ".kt",
    ".kts", ".lua", ".m", ".mm", ".php", ".pl", ".ps1", ".py",
    ".pyi", ".rb", ".rs", ".scss", ".sh", ".sql", ".swift",
    ".ts", ".tsx", ".vue", ".zsh",
}
_CODE_FILENAMES = {
    "cargo.toml", "composer.json", "config.yaml", "config.yml", "dockerfile",
    "gemfile", "justfile", "makefile", "package.json", "procfile",
    "pyproject.toml", "requirements.txt", "tsconfig.json", "uv.lock",
}
_CONFIG_SUFFIXES = {".cfg", ".ini", ".json", ".toml", ".xml", ".yaml", ".yml"}
_TEST_COMMAND_RE = re.compile(
    r"(?:^|[;&|()]\s*)"
    r"(?:"
    r"(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?pytest\b|"
    r"(?:python(?:3(?:\.\d+)?)?|py)\s+-m\s+unittest\b|"
    r"uv\s+run\s+pytest\b|"
    r"(?:(?:python(?:3(?:\.\d+)?)?|py)\s+-m\s+|uv\s+run\s+)?ruff\s+check\b|"
    r"(?:(?:python(?:3(?:\.\d+)?)?|py)\s+-m\s+|uv\s+run\s+)?mypy\b|"
    r"(?:(?:python(?:3(?:\.\d+)?)?|py)\s+-m\s+|uv\s+run\s+)?flake8\b|"
    r"(?:(?:python(?:3(?:\.\d+)?)?|py)\s+-m\s+|uv\s+run\s+)?black\s+--check\b|"
    r"(?:npx\s+)?eslint\b|"
    r"scripts[/\\]run_tests\.sh\b|"
    r"npm\s+(?:test\b|run\s+(?:test|lint|build|typecheck)\b)|"
    r"pnpm\s+(?:test\b|run\s+(?:test|lint|build|typecheck)\b)|"
    r"yarn\s+(?:test\b|(?:lint|build|typecheck)\b)|"
    r"cargo\s+(?:test|check|clippy)\b|"
    r"go\s+test\b|"
    r"tox\b|"
    r"nox\b|"
    r"make\s+(?:test|check|lint|build)\b|"
    r"dotnet\s+(?:test|build)\b|"
    r"mvn\s+(?:test|verify)\b|"
    r"gradle\w*\s+(?:test|check|build)\b"
    r")",
    re.IGNORECASE,
)
_MUTATING_ESLINT_RE = re.compile(
    r"(?<![\w-])eslint(?:\.(?:cmd|exe|js))?(?![\w-])[\s\S]*?"
    r"--fix(?![-\w])",
    re.IGNORECASE,
)
_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Persisted tool results may have a valid JSON object followed by
            # injected subdirectory/plugin context. Decode the leading object
            # rather than discarding its structured exit status.
            try:
                parsed, _ = json.JSONDecoder().raw_decode(value.lstrip())
            except (json.JSONDecodeError, TypeError):
                return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (json.JSONDecodeError, TypeError):
            return []
    return calls if isinstance(calls, list) else []


def _normalized_path(path: Any) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    match = re.match(r"^/([A-Za-z])(?:/|$)(.*)", normalized)
    if match:
        normalized = f"{match.group(1).upper()}:/{match.group(2)}"
    return normalized


def _is_code_path(path: str) -> bool:
    normalized = _normalized_path(path)
    if not normalized:
        return False
    leaf = PurePosixPath(normalized).name.lower()
    if leaf in _CODE_FILENAMES:
        return True
    suffix = PurePosixPath(leaf).suffix.lower()
    if suffix in _CODE_SUFFIXES:
        return True
    return suffix in _CONFIG_SUFFIXES and "config" in leaf


def _mutation_paths(tool: str, arguments: Dict[str, Any]) -> List[str]:
    if tool == "write_file":
        path = _normalized_path(arguments.get("path"))
        return [path] if path else []
    if tool != "patch":
        return []
    if arguments.get("mode") == "patch":
        return [_normalized_path(p) for p in _PATCH_PATH_RE.findall(
            str(arguments.get("patch") or "")
        )]
    path = _normalized_path(arguments.get("path"))
    return [path] if path else []


def _mutation_succeeded(content: Any) -> bool:
    payload = _json_object(content)
    if payload:
        if payload.get("success") is False or payload.get("status") == "error":
            return False
        if "verified" in payload:
            return payload.get("verified") is True
        return payload.get("success") is True or payload.get("status") == "success"
    # Behavior scoring is evidence-driven: arbitrary prose is not a
    # structured success signal and must not make the behavior applicable.
    return False


def _terminal_succeeded(content: Any) -> bool:
    payload = _json_object(content)
    if not payload:
        text = str(content or "")
        match = re.search(r"(?:->\s*exit|exit(?:[_ ]code)?[=: ]+)\s*(-?\d+)", text, re.IGNORECASE)
        return bool(match and int(match.group(1)) == 0)
    exit_code = payload.get("exit_code", payload.get("returncode"))
    if exit_code is not None:
        try:
            return int(exit_code) == 0
        except (TypeError, ValueError):
            return False
    return payload.get("status") == "success" and not payload.get("error")


def _message_id(message: Dict[str, Any], fallback: int) -> Any:
    value = message.get("id")
    return value if value is not None else fallback


@lru_cache(maxsize=512)
def _discover_repository_root(path: str) -> Optional[str]:
    """Find a local Git root without invoking Git or mutating the trace."""
    candidate = Path(path)
    current = candidate if candidate.is_dir() else candidate.parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return str(parent)
    return None


def _path_is_in_repository(path: str, repository_root: Optional[str]) -> bool:
    normalized = _normalized_path(path)
    if not normalized:
        return False
    is_absolute = bool(re.match(r"^[A-Za-z]:/", normalized)) or normalized.startswith("/")
    if not is_absolute:
        if not repository_root:
            return ".." not in PurePosixPath(normalized).parts
        normalized = ntpath.join(_normalized_path(repository_root), normalized)
    if not repository_root:
        repository_root = _discover_repository_root(normalized)
    if not repository_root:
        return False
    candidate = ntpath.normcase(ntpath.normpath(normalized))
    root = ntpath.normcase(ntpath.normpath(_normalized_path(repository_root)))
    try:
        return ntpath.commonpath([candidate, root]) == root
    except ValueError:
        return False


def evaluate_code_change_verification(
    messages: Iterable[Dict[str, Any]],
    *,
    repository_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the ``code-change-verification`` trajectory behavior.

    ``n/a`` means no successful structured mutation of a code/config file was
    observed. ``true`` requires a successful recognized verification command
    after the latest successful mutation. ``false`` includes missing or failed
    post-mutation verification.
    """
    ordered = list(messages)
    calls_by_id: Dict[str, Dict[str, Any]] = {}
    pending_by_tool: Dict[str, List[Dict[str, Any]]] = {}

    for index, message in enumerate(ordered):
        if message.get("role") != "assistant":
            continue
        for call in _tool_calls(message):
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if not isinstance(function, dict):
                continue
            tool = str(function.get("name") or "")
            record = {
                "tool": tool,
                "arguments": _json_object(function.get("arguments")),
                "message_id": _message_id(message, index),
            }
            call_id = str(call.get("id") or "")
            if call_id:
                calls_by_id[call_id] = record
            pending_by_tool.setdefault(tool, []).append(record)

    mutation_candidates: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []

    for index, message in enumerate(ordered):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool_name") or "")
        call_id = str(message.get("tool_call_id") or "")
        call = calls_by_id.get(call_id)
        if call is not None:
            candidates = pending_by_tool.get(call["tool"]) or []
            if call in candidates:
                candidates.remove(call)
        if call is None:
            candidates = pending_by_tool.get(tool) or []
            call = candidates.pop(0) if candidates else None
        if call is None:
            continue
        tool = tool or call["tool"]
        result_message_id = _message_id(message, index)

        if tool in {"patch", "write_file"} and _mutation_succeeded(message.get("content")):
            paths = [
                path for path in _mutation_paths(tool, call["arguments"])
                if _is_code_path(path)
            ]
            if paths:
                mutation_candidates.append({
                    "tool": tool,
                    "path": paths[-1],
                    "paths": paths,
                    "message_id": result_message_id,
                    "_sequence": index,
                })
            continue

        if tool == "terminal":
            command = str(call["arguments"].get("command") or "")
            if _TEST_COMMAND_RE.search(command) and not _MUTATING_ESLINT_RE.search(command):
                verifications.append({
                    "command": command,
                    "success": _terminal_succeeded(message.get("content")),
                    "message_id": result_message_id,
                    "_sequence": index,
                })

    last_mutation = None
    for candidate in reversed(mutation_candidates):
        repository_paths = [
            path for path in candidate["paths"]
            if _path_is_in_repository(path, repository_root)
        ]
        if repository_paths:
            last_mutation = {
                **candidate,
                "path": repository_paths[-1],
                "paths": repository_paths,
            }
            break

    if last_mutation is None:
        return {
            "behavior": "code-change-verification",
            "result": "n/a",
            "evidence": {"last_mutation": None, "latest_verification": None},
            "reason": "No successful structured code or configuration mutation was observed.",
        }

    post_mutation = [
        item for item in verifications
        if item["_sequence"] > last_mutation["_sequence"]
    ]
    latest = post_mutation[-1] if post_mutation else None
    passed = bool(latest and latest["success"])
    public_mutation = {k: v for k, v in last_mutation.items() if k != "_sequence"}
    public_latest = (
        {k: v for k, v in latest.items() if k != "_sequence"}
        if latest else None
    )
    reason = (
        "The latest code change was followed by a successful verification command."
        if passed
        else "The latest code change was not followed by a successful verification command."
    )
    return {
        "behavior": "code-change-verification",
        "result": "true" if passed else "false",
        "evidence": {
            "last_mutation": public_mutation,
            "latest_verification": public_latest,
        },
        "reason": reason,
    }
