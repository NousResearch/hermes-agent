"""OpenRouter Decisions API request and response validation.

The Decisions endpoint is deliberately separate from the OpenAI-compatible
chat-completions client: its request and response shapes are not compatible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx


def decisions_endpoint(base_url: str) -> str:
    """Return the Decisions endpoint for an OpenRouter API base URL."""
    parts = urlsplit(base_url)
    if parts.scheme != "https" or parts.netloc.lower() != "openrouter.ai":
        raise ValueError("OpenRouter Decisions requests require an https://openrouter.ai base URL")
    return "https://openrouter.ai/api/alpha/decisions"


def _validated_questions(questions: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if not questions:
        raise ValueError("OpenRouter Decisions requires at least one question")
    normalized: dict[str, dict[str, str]] = {}
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, Mapping):
            raise ValueError("OpenRouter Decisions questions must be a mapping of named questions")
        if question.get("type") != "noul" or not isinstance(question.get("instructions"), str):
            raise ValueError(f"OpenRouter Decisions question {key!r} must be a noul with instructions")
        normalized[key] = {"type": "noul", "instructions": question["instructions"]}
    return normalized


def call_decisions(
    *, api_key: str, base_url: str, model: str, state: Any, questions: Mapping[str, Any],
    timeout: float = 30.0, post: Callable[..., Any] | None = None,
) -> dict[str, float]:
    """Ask OpenRouter a set of noul questions and return calibrated probabilities."""
    if not api_key:
        raise ValueError("OpenRouter Decisions requires an API key")
    if not model:
        raise ValueError("OpenRouter Decisions requires a model")
    normalized_questions = _validated_questions(questions)
    request_post = post or httpx.post
    response = request_post(
        decisions_endpoint(base_url),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "state": state, "questions": normalized_questions},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    answers = payload.get("answers") if isinstance(payload, Mapping) else None
    if not isinstance(answers, Mapping):
        raise ValueError("OpenRouter Decisions response did not contain answers")
    probabilities: dict[str, float] = {}
    for key in normalized_questions:
        answer = answers.get(key)
        value = answer.get("noul") if isinstance(answer, Mapping) and answer.get("type") == "noul" else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"OpenRouter Decisions response has invalid noul answer for {key!r}")
        probabilities[key] = float(value)
    return probabilities
