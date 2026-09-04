"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
from typing import Any, Optional


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})

_AUTOMATIC_PROGRESS_CATALOG: dict[str, frozenset[str]] = {
    "file_changed": frozenset({"file write completed", "patch completed"}),
    "tests_passed": frozenset({"tests passed"}),
    "build_passed": frozenset({"build passed"}),
    "typecheck_passed": frozenset({"typecheck passed"}),
    "lint_passed": frozenset({"lint passed"}),
    "compile_passed": frozenset({"compile passed"}),
    "commit_created": frozenset({"commit created"}),
    "graph_updated": frozenset({"knowledge graph updated"}),
}


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hermes session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False


def _parse_tool_result_dict(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        data = result
    elif isinstance(result, str):
        try:
            data = json.loads(result.strip())
        except Exception:
            return None
    else:
        return None
    return data if isinstance(data, dict) else None


def _clean_terminal_token(token: str) -> str:
    token = token.strip()
    while token.startswith("./"):
        token = token[2:]
    return token


def _strip_shell_wrappers(tokens: list[str]) -> list[str]:
    cleaned = [_clean_terminal_token(t) for t in tokens]
    while cleaned and "=" in cleaned[0] and not cleaned[0].startswith("-"):
        cleaned = cleaned[1:]
    while cleaned and cleaned[0] in {"env", "bash", "sh", "zsh", "dash"}:
        cleaned = cleaned[1:]
    while (
        len(cleaned) >= 2
        and cleaned[0] in {"uv", "poetry", "pipenv"}
        and cleaned[1] == "run"
    ):
        cleaned = cleaned[2:]
    return cleaned


def _classify_terminal_tokens(tokens: list[str]) -> Optional[tuple[str, str]]:
    tokens = _strip_shell_wrappers(tokens)
    if not tokens:
        return None

    if len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "commit":
        return ("commit_created", "commit created")
    if len(tokens) >= 2 and tokens[0] == "graphify" and tokens[1] == "update":
        return ("graph_updated", "knowledge graph updated")

    if any(t.endswith("run_tests.sh") for t in tokens):
        return ("tests_passed", "tests passed")

    base = tokens[0]
    if base in {"pytest", "vitest", "jest", "nosetests"}:
        return ("tests_passed", "tests passed")
    if base == "cargo" and len(tokens) > 1 and tokens[1] == "test":
        return ("tests_passed", "tests passed")
    if base == "go" and len(tokens) > 1 and tokens[1] == "test":
        return ("tests_passed", "tests passed")
    if base in {"npm", "pnpm", "yarn", "bun"}:
        if len(tokens) >= 2 and tokens[1] == "test":
            return ("tests_passed", "tests passed")
        if len(tokens) >= 3 and tokens[1] == "run":
            script = tokens[2]
            if script in {"test", "tests"} or script.startswith("test:"):
                return ("tests_passed", "tests passed")
            if script in {"build", "compile"}:
                return ("build_passed", "build passed")
            if script in {"lint", "eslint"}:
                return ("lint_passed", "lint passed")
            if script in {"typecheck", "check-types", "tc"}:
                return ("typecheck_passed", "typecheck passed")
    if base == "make" and len(tokens) > 1:
        if tokens[1] in {"test", "tests", "check"}:
            return ("tests_passed", "tests passed")
        if tokens[1] in {"build", "all"}:
            return ("build_passed", "build passed")
    if base == "cargo" and len(tokens) > 1 and tokens[1] == "build":
        return ("build_passed", "build passed")
    if base == "go" and len(tokens) > 1 and tokens[1] == "build":
        return ("build_passed", "build passed")
    if base in {"tsc", "mypy", "pyright"}:
        return ("typecheck_passed", "typecheck passed")
    if base in {"eslint", "flake8", "pylint"}:
        return ("lint_passed", "lint passed")
    if base == "ruff" and len(tokens) > 1 and tokens[1] == "check":
        return ("lint_passed", "lint passed")
    if base.startswith("python") and len(tokens) >= 3 and tokens[1] == "-m":
        if tokens[2] in {"pytest", "unittest"}:
            return ("tests_passed", "tests passed")
        if tokens[2] in {"py_compile", "compileall"}:
            return ("compile_passed", "compile passed")
    return None


def _classify_file_mutation_evidence(tool_name: str, data: dict) -> Optional[tuple[str, str]]:
    if tool_name == "write_file":
        if data.get("error"):
            return None
        verified = data.get("verified")
        bytes_written = data.get("bytes_written")
        if verified is True or (
            isinstance(bytes_written, int) and bytes_written > 0
        ):
            return ("file_changed", "file write completed")
        return None
    if tool_name == "patch":
        if data.get("success") is not True or data.get("no_change"):
            return None
        if data.get("files_modified") or data.get("files_created") or data.get("files_deleted"):
            return ("file_changed", "patch completed")
    return None


def classify_automatic_progress_evidence(
    tool_name: str,
    args: dict,
    result: Any,
) -> Optional[tuple[str, str]]:
    """Return sanitized automatic-progress evidence for a completed tool call."""
    if not isinstance(args, dict):
        args = {}

    data = _parse_tool_result_dict(result)
    if data is not None and data.get("status") in {"cancelled", "timeout"}:
        return None

    result_for_failure = result if isinstance(result, str) else (
        json.dumps(data, ensure_ascii=False) if data is not None else ""
    )
    from agent.display import _detect_tool_failure

    is_error, _ = _detect_tool_failure(tool_name, result_for_failure)
    if is_error:
        return None

    if tool_name in FILE_MUTATING_TOOL_NAMES:
        if data is None:
            return None
        return _classify_file_mutation_evidence(tool_name, data)

    if tool_name != "terminal":
        return None

    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    if data is None:
        return None
    exit_code = data.get("exit_code")
    if exit_code is None or int(exit_code) != 0:
        return None

    from agent.verification_evidence import (
        _exit_status_is_attributable,
        _split_shell_segments,
    )

    segments = _split_shell_segments(command)
    if not segments:
        return None
    for idx, segment in enumerate(segments):
        if not _exit_status_is_attributable(segments, idx, exit_code):
            continue
        match = _classify_terminal_tokens(segment.tokens)
        if match is not None:
            return match
    return None


def normalize_automatic_progress_evidence(
    evidence_type: str,
    detail: str,
) -> Optional[tuple[str, str]]:
    """Validate and cap automatic-progress fields against the allowlist."""
    et = str(evidence_type or "").strip()[:32]
    det = str(detail or "").strip()[:64]
    if not et or not det:
        return None
    allowed = _AUTOMATIC_PROGRESS_CATALOG.get(et)
    if not allowed or det not in allowed:
        return None
    if "/" in det or "\\" in det:
        return None
    return et, det
