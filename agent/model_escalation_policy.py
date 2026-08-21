from __future__ import annotations

from dataclasses import dataclass
import re

VALID_FILE_EXTENSIONS = (
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".sql",
    ".txt",
    ".html",
    ".css",
)


@dataclass(frozen=True)
class SelectionDecision:
    selected_model: str
    reason: str


def is_valid_file_reference(token: str) -> bool:
    token = (token or "").strip().strip("\"'`(),:;")
    if not token:
        return False
    if "/" in token or "\\" in token:
        return True
    lowered = token.lower()
    return any(lowered.endswith(ext) for ext in VALID_FILE_EXTENSIONS)


def count_valid_file_references(task_text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9_./\\:-]+", task_text or "")
    return sum(1 for tok in tokens if is_valid_file_reference(tok))


def _has_fix_request(task_text: str) -> bool:
    text = (task_text or "").lower()
    return any(word in text for word in ("fix", "repair", "resolve", "debug"))


def _has_traceback(task_text: str) -> bool:
    text = (task_text or "").lower()
    return "traceback" in text or "stack trace" in text


def _has_high_risk_intent(task_text: str) -> bool:
    text = (task_text or "").lower()
    return any(word in text for word in ("production", "security", "credential", "secret", "auth", "rollback"))


def _has_architecture_or_multifile_refactor(task_text: str) -> bool:
    text = (task_text or "").lower()
    return (
        "architecture" in text
        or "multi-file refactor" in text
        or "multifile refactor" in text
    )


def _is_hard_complex(task_text: str) -> bool:
    text = (task_text or "").lower()
    file_reference_count = count_valid_file_references(task_text)
    has_traceback = _has_traceback(task_text)
    asks_to_fix_failure = _has_fix_request(task_text)

    return (
        _has_high_risk_intent(task_text)
        or "schema" in text
        or "database migration" in text
        or "destructive operation" in text
        or (has_traceback and asks_to_fix_failure)
        or file_reference_count >= 3
        or _has_architecture_or_multifile_refactor(task_text)
    )


def select_initial_model(task_text: str) -> SelectionDecision:
    if _is_hard_complex(task_text):
        return SelectionDecision(selected_model="gpt-5.4", reason="hard_complex")
    return SelectionDecision(selected_model="gpt-5.4-mini", reason="default_mini")
