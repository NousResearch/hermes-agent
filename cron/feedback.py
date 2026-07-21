"""Validation and compatibility helpers for interactive cron feedback."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

FEEDBACK_CODE_RE = re.compile(r"^[a-z0-9_-]{1,24}$")
FEEDBACK_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_FEEDBACK_CHOICES = 8
MAX_FEEDBACK_LABEL_LENGTH = 64
MAX_FEEDBACK_PROMPT_LENGTH = 200


def normalize_feedback_config(value: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalize a feedback config, returning ``None`` when cleared."""
    if value is None or value == {}:
        return None
    if not isinstance(value, dict):
        raise ValueError("feedback must be an object")

    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("feedback.choices must contain at least one choice")
    if len(choices) > MAX_FEEDBACK_CHOICES:
        raise ValueError(f"feedback.choices supports at most {MAX_FEEDBACK_CHOICES} choices")

    normalized_choices = []
    seen_codes = set()
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("each feedback choice must be an object")
        code = str(choice.get("code") or "").strip().lower()
        label = str(choice.get("label") or "").strip()
        if not FEEDBACK_CODE_RE.fullmatch(code):
            raise ValueError(
                "feedback choice code must use 1-24 lowercase letters, digits, '_' or '-'"
            )
        if code in seen_codes:
            raise ValueError(f"feedback choice code '{code}' is duplicated")
        if not label:
            raise ValueError("feedback choice label cannot be empty")
        if len(label) > MAX_FEEDBACK_LABEL_LENGTH:
            raise ValueError(
                f"feedback choice label cannot exceed {MAX_FEEDBACK_LABEL_LENGTH} characters"
            )
        seen_codes.add(code)
        normalized_choices.append({"code": code, "label": label})

    prompt = str(value.get("prompt") or "").strip()
    if len(prompt) > MAX_FEEDBACK_PROMPT_LENGTH:
        raise ValueError(
            f"feedback prompt cannot exceed {MAX_FEEDBACK_PROMPT_LENGTH} characters"
        )
    return {"prompt": prompt, "choices": normalized_choices}


def feedback_for_job(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return canonical feedback, accepting the pre-PR Telegram-specific key."""
    raw = job.get("feedback")
    if raw is None:
        raw = job.get("telegram_feedback")
    try:
        return normalize_feedback_config(raw)
    except ValueError:
        return None


def feedback_choice_label(job_id: str, action: str) -> Optional[str]:
    """Return the configured label for an action, or ``None`` when invalid."""
    if (
        not FEEDBACK_JOB_ID_RE.fullmatch(job_id or "")
        or not FEEDBACK_CODE_RE.fullmatch(action or "")
    ):
        return None
    from cron.jobs import get_job

    config = feedback_for_job(get_job(job_id) or {})
    if not config:
        return None
    for choice in config["choices"]:
        if choice["code"] == action:
            return choice["label"]
    return None


def feedback_action_allowed(job_id: str, action: str) -> bool:
    """Fail closed unless ``action`` is currently configured for ``job_id``."""
    return feedback_choice_label(job_id, action) is not None
