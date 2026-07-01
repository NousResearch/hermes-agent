"""Self-modification write guardrails for Hermes control paths."""

from __future__ import annotations

import contextvars
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.pr_head_guard import enforce_pr_head_invariant
from agent.runtime_evidence import update_runtime_evidence

_PROTECTED_ROOT_NAMES = frozenset({
    "skills",
    "profiles",
    "docs",
    "hooks",
    "scripts",
    ".github",
})

_PROTECTED_EXACT_FILES = frozenset({
    "config.yaml",
    "config.example.yaml",
})

_WRITE_TOOL_NAMES = frozenset({"write_file", "patch", "skill_manage"})
_TERMINAL_TOOL_NAMES = frozenset({"terminal"})
_EXECUTE_CODE_TOOL_NAMES = frozenset({"execute_code"})

_SELF_IMPROVEMENT_HINTS = (
    r"\bself[- ]?improv(?:e|ement|ing)?\b",
    r"\bruntime\s+guardrail(?:s)?\b",
    r"\bguardrail(?:s)?\s+work\b",
    r"\bhermes\b.*\b(?:config|skill|guardrail|runtime|profile|docs|hooks|scripts|github)\b",
    r"\b(?:config|skill|guardrail|profile|docs|hooks|scripts|github)\s+(?:edit(?:ing)?|update(?:ing)?|modify(?:ing)?|change(?:ing)?|patch(?:ing)?|fix(?:ing)?|rewrite(?:ing)?|refactor(?:ing)?|harden(?:ing)?)\b",
    r"\bedit(?:ing)?\s+(?:the\s+)?(?:hermes\s+)?(?:config|skill|guardrail|profile|docs|hooks|scripts|github)\b",
    r"\bupdate(?:ing)?\s+(?:the\s+)?(?:hermes\s+)?(?:config|skill|guardrail|profile|docs|hooks|scripts|github)\b",
)

_SHELL_WRITE_HINTS = (
    r">>",
    r">",
    r"\btee\b",
    r"\bsed\s+-i\b",
    r"\bperl\s+-pi\b",
    r"\bcp\b",
    r"\bmv\b",
    r"\brm\b",
    r"\btouch\b",
)

_CODE_WRITE_HINTS = (
    r"\bopen\s*\([^)]*'[^']*'[^)]*\b(?:w|a|x|\+)",
    r'\bopen\s*\([^)]*"[^"]*"[^)]*\b(?:w|a|x|\+)',
    r"\.write_text\s*\(",
    r"\.write_bytes\s*\(",
    r"\.unlink\s*\(",
    r"\.rename\s*\(",
    r"\.replace\s*\(",
    r"\bos\.(?:remove|unlink|rename)\s*\(",
    r"\bshutil\.(?:move|copy|copy2)\s*\(",
)

_LIFECYCLE_STATE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "hermes_self_modification_lifecycle",
    default=None,
)


def before_tool_call(function_name: str, function_args: Mapping[str, Any] | None, user_task: str | None) -> str | None:
    """Return a block message when a tool would mutate Hermes control files."""
    args = function_args if isinstance(function_args, Mapping) else {}
    user_task_text = str(user_task or "").strip()

    if function_name in _WRITE_TOOL_NAMES:
        targets = _candidate_targets(function_name, args)
        protected_targets = [target for target in targets if _is_protected_path(target)]
        if not protected_targets:
            return None
        if not _is_explicit_self_mod_request(user_task_text, protected_targets):
            return _blocked_message(function_name, protected_targets)
        repo_root, repo_root_error = _resolve_audit_repo_root(protected_targets)
        if repo_root is None:
            return _blocked_message(function_name, protected_targets, repo_root_error or "pre-edit git status --short could not be captured")
        pr_head_block = enforce_pr_head_invariant(
            repo_root=repo_root,
            action_label="file patch",
        )
        if pr_head_block is not None:
            return _blocked_message(function_name, protected_targets, pr_head_block)
        pre_edit_git_status = _capture_git_status(repo_root)
        if pre_edit_git_status is None:
            return _blocked_message(function_name, protected_targets, "pre-edit git status --short could not be captured")
        _LIFECYCLE_STATE.set({
            "function_name": function_name,
            "protected_targets": list(dict.fromkeys(protected_targets)),
            "repo_root": str(repo_root),
            "pre_edit_git_status": pre_edit_git_status,
            "explicit_user_request": user_task_text,
        })
        update_runtime_evidence(
            latest_user_request=user_task_text or None,
            audit_repo_root=str(repo_root),
            pre_edit_git_status=pre_edit_git_status,
        )
        return None

    if function_name in _TERMINAL_TOOL_NAMES:
        command = str(args.get("command") or args.get("cmd") or "").strip()
        if not command:
            return None
        protected_targets = _extract_protected_mentions(command)
        if not protected_targets or not _contains_write_intent(command):
            return None
        if not _is_explicit_self_mod_request(user_task_text, protected_targets):
            return _blocked_message(function_name, protected_targets)
        repo_root, repo_root_error = _resolve_audit_repo_root(protected_targets)
        if repo_root is None:
            return _blocked_message(function_name, protected_targets, repo_root_error or "pre-edit git status --short could not be captured")
        pr_head_block = enforce_pr_head_invariant(
            repo_root=repo_root,
            action_label="file patch",
        )
        if pr_head_block is not None:
            return _blocked_message(function_name, protected_targets, pr_head_block)
        pre_edit_git_status = _capture_git_status(repo_root)
        if pre_edit_git_status is None:
            return _blocked_message(function_name, protected_targets, "pre-edit git status --short could not be captured")
        _LIFECYCLE_STATE.set({
            "function_name": function_name,
            "protected_targets": list(dict.fromkeys(protected_targets)),
            "repo_root": str(repo_root),
            "pre_edit_git_status": pre_edit_git_status,
            "explicit_user_request": user_task_text,
        })
        update_runtime_evidence(
            latest_user_request=user_task_text or None,
            audit_repo_root=str(repo_root),
            pre_edit_git_status=pre_edit_git_status,
        )
        return None

    if function_name in _EXECUTE_CODE_TOOL_NAMES:
        code = str(args.get("code") or "").strip()
        if not code:
            return None
        protected_targets = _extract_protected_mentions(code)
        if not protected_targets or not _contains_code_write_intent(code):
            return None
        if not _is_explicit_self_mod_request(user_task_text, protected_targets):
            return _blocked_message(function_name, protected_targets)
        repo_root, repo_root_error = _resolve_audit_repo_root(protected_targets)
        if repo_root is None:
            return _blocked_message(function_name, protected_targets, repo_root_error or "pre-edit git status --short could not be captured")
        pr_head_block = enforce_pr_head_invariant(
            repo_root=repo_root,
            action_label="file patch",
        )
        if pr_head_block is not None:
            return _blocked_message(function_name, protected_targets, pr_head_block)
        pre_edit_git_status = _capture_git_status(repo_root)
        if pre_edit_git_status is None:
            return _blocked_message(function_name, protected_targets, "pre-edit git status --short could not be captured")
        _LIFECYCLE_STATE.set({
            "function_name": function_name,
            "protected_targets": list(dict.fromkeys(protected_targets)),
            "repo_root": str(repo_root),
            "pre_edit_git_status": pre_edit_git_status,
            "explicit_user_request": user_task_text,
        })
        update_runtime_evidence(
            latest_user_request=user_task_text or None,
            audit_repo_root=str(repo_root),
            pre_edit_git_status=pre_edit_git_status,
        )
        return None

    return None


def finalize_protected_write_result(function_name: str, result: Any) -> Any:
    """Attach lifecycle evidence to successful protected writes."""
    state = _LIFECYCLE_STATE.get()
    if not isinstance(state, dict) or state.get("function_name") != function_name:
        return result

    try:
        repo_root_text = str(state.get("repo_root") or "").strip()
        repo_root = Path(repo_root_text) if repo_root_text else None
        post_edit_git_status = (_capture_git_status(repo_root) or "").strip() if repo_root else None
        diff_summary = (_capture_git_diff_summary(repo_root) or "").strip() if repo_root else ""
        report = {
            "files_changed": list(state.get("protected_targets") or []),
            "validation_tests_run": [
                "git status --short (pre-edit)",
                "git status --short (post-edit)",
            ],
            "validation_summary": "pre-edit git status --short captured; post-edit git status --short captured",
            "secret_scan_result": _secret_scan_result(function_name, result),
            "final_diff_summary": diff_summary or "no diff available",
            "final_git_status": post_edit_git_status or "unavailable",
            "commit_happened": False,
            "push_happened": False,
            "pre_edit_git_status": state.get("pre_edit_git_status") or "unavailable",
            "explicit_user_request": state.get("explicit_user_request") or "",
        }
        return _merge_report(result, report)
    finally:
        _LIFECYCLE_STATE.set(None)


def clear_lifecycle_state() -> None:
    """Reset pending lifecycle evidence after a tool call finishes."""
    _LIFECYCLE_STATE.set(None)


def _merge_report(result: Any, report: dict[str, Any]) -> Any:
    if isinstance(result, dict):
        merged = dict(result)
        merged["self_modification_report"] = report
        return merged
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            parsed["self_modification_report"] = report
            return json.dumps(parsed, ensure_ascii=False)
    return json.dumps({"result": result, "self_modification_report": report}, ensure_ascii=False)


def _secret_scan_result(function_name: str, result: Any) -> str:
    if isinstance(result, dict):
        for key in ("secret_scan_result", "security_scan", "scan_result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("error", "message"):
            value = result.get(key)
            if isinstance(value, str) and "security scan" in value.lower():
                return value.strip()
    if isinstance(result, str) and "security scan" in result.lower():
        return result.strip()
    return "not reported by tool" if function_name == "skill_manage" else "not applicable"


def _capture_git_status(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.rstrip("\n")


def _resolve_audit_repo_root(protected_targets: Sequence[str]) -> tuple[Path | None, str | None]:
    """Resolve the Git root that owns the protected target path(s).

    If the targets span multiple repositories, block rather than guessing.
    """
    roots: list[Path] = []
    for target in protected_targets:
        root = _nearest_git_root_for_target(target)
        if root is not None and root not in roots:
            roots.append(root)

    if len(roots) > 1:
        root_text = ", ".join(str(root) for root in roots)
        return None, f"protected targets span multiple git roots: {root_text}"

    if len(roots) == 1:
        return roots[0], None

    fallback = _workspace_root()
    if fallback is not None:
        return fallback, None

    return None, "pre-edit git status --short could not be captured"


def _nearest_git_root_for_target(target_text: str) -> Path | None:
    """Return the nearest Git repository root for a protected target path."""
    text = str(target_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path
    for candidate in [resolved, *resolved.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _capture_git_diff_summary(repo_root: Path | None) -> str:
    if repo_root is None:
        return ""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _workspace_root() -> Path | None:
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _blocked_message(function_name: str, protected_targets: Sequence[str], reason: str | None = None) -> str:
    unique_targets = list(dict.fromkeys(protected_targets))
    target_text = ", ".join(unique_targets[:4])
    if len(unique_targets) > 4:
        target_text += f", +{len(unique_targets) - 4} more"
    suffix = f" {reason}." if reason else "."
    return (
        f"Blocked self-modification write via {function_name} for protected path(s): {target_text}{suffix} "
        "This requires explicit user instruction naming the target files; propose the change instead."
    )


def _first_path_arg(args: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_targets(function_name: str, args: Mapping[str, Any]) -> list[str]:
    if function_name == "skill_manage":
        targets: list[str] = []
        file_path = _first_path_arg(args, ("file_path", "path", "filepath"))
        if file_path:
            targets.append(file_path)
        name = str(args.get("name") or "").strip()
        category = str(args.get("category") or "").strip()
        if name:
            if category:
                targets.extend([
                    f"skills/{category}/{name}/SKILL.md",
                    f"skills/{category}/{name}.md",
                ])
            targets.extend([
                f"skills/{name}/SKILL.md",
                f"skills/{name}.md",
            ])
        return targets
    return [
        target
        for target in (
            _first_path_arg(args, ("path", "filepath", "file_path", "target")),
        )
        if target
    ]


def _is_protected_path(path_text: str) -> bool:
    text = str(path_text or "").replace("\\", "/").strip()
    if not text:
        return False
    lower = text.lower()
    if any(exact == lower.rsplit("/", 1)[-1] for exact in _PROTECTED_EXACT_FILES):
        return True
    path = Path(text)
    parts = [part.lower() for part in path.parts]
    if any(part in _PROTECTED_ROOT_NAMES for part in parts):
        return True
    return any(f"/{root}/" in f"/{lower}" for root in _PROTECTED_ROOT_NAMES)


def _extract_protected_mentions(text: str) -> list[str]:
    if not text:
        return []
    normalized = text.replace("\\", "/")
    lower = normalized.lower()
    mentions: list[str] = []
    for exact in _PROTECTED_EXACT_FILES:
        if exact in lower:
            mentions.append(exact)
    for root in _PROTECTED_ROOT_NAMES:
        marker = f"{root}/"
        start = 0
        while True:
            idx = lower.find(marker, start)
            if idx < 0:
                break
            end = idx + len(marker)
            while end < len(normalized) and normalized[end] not in " \t\r\n'\"`|&;<>":
                end += 1
            mentions.append(normalized[idx:end].rstrip(".,);]"))
            start = end
    return list(dict.fromkeys(mentions))


def _is_explicit_self_mod_request(user_task: str, protected_targets: Sequence[str]) -> bool:
    if not user_task or not protected_targets:
        return False
    lower = user_task.lower()
    if not any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in _SELF_IMPROVEMENT_HINTS):
        return False

    target_tokens = set()
    for target in protected_targets:
        normalized = str(target).replace("\\", "/").strip().lower()
        if not normalized:
            continue
        target_tokens.add(normalized)
        try:
            path = Path(normalized)
        except Exception:
            path = None
        if path is not None:
            parts = [part.lower() for part in path.parts if part]
            for part in parts:
                target_tokens.add(part)
            for i in range(len(parts)):
                target_tokens.add("/".join(parts[i:]))
            if len(parts) >= 2:
                target_tokens.add(f"{parts[-2]} {parts[-1]}")
            if parts:
                target_tokens.add(parts[-1])
    return any(token and token in lower for token in target_tokens)


def _contains_write_intent(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _SHELL_WRITE_HINTS + _CODE_WRITE_HINTS)


def _contains_code_write_intent(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _CODE_WRITE_HINTS + _SHELL_WRITE_HINTS)
