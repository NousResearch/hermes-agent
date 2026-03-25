"""Shared title-generation helpers backed by the auxiliary LLM."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

_TITLE_RULES = (
    "The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)
_TITLE_MAX_TOKENS = 30
_TITLE_TEMPERATURE = 0.3


def _truncate_snippet(text: str, limit: int = 500) -> str:
    return text[:limit] if text else ""


def _build_title_prompt(subject: str) -> str:
    return f"Generate a short, descriptive title (3-7 words) for {subject}. {_TITLE_RULES}"


def _build_title_messages(
    user_message: str,
    assistant_response: Optional[str] = None,
) -> Optional[list[dict[str, str]]]:
    user_snippet = _truncate_snippet(user_message)
    if not user_snippet:
        return None

    if assistant_response is None:
        return [
            {"role": "system", "content": _build_title_prompt("the following user message")},
            {"role": "user", "content": user_snippet},
        ]

    assistant_snippet = _truncate_snippet(assistant_response)
    return [
        {
            "role": "system",
            "content": _build_title_prompt("a conversation that starts with the following exchange"),
        },
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]


def _clean_title(title: str) -> Optional[str]:
    title = (title or "").strip()
    title = title.strip('"\'')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    if len(title) > 80:
        title = title[:77] + "..."
    return title if title else None


def _extract_title(response: object) -> Optional[str]:
    return _clean_title(response.choices[0].message.content or "")


def _title_request_kwargs(messages: list[dict[str, str]], timeout: float) -> dict:
    return {
        "task": "compression",
        "messages": messages,
        "max_tokens": _TITLE_MAX_TOKENS,
        "temperature": _TITLE_TEMPERATURE,
        "timeout": timeout,
    }


def _generate_title(messages: Optional[list[dict[str, str]]], timeout: float) -> Optional[str]:
    if not messages:
        return None

    try:
        response = call_llm(**_title_request_kwargs(messages, timeout))
        return _extract_title(response)
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return None


def generate_title_from_exchange(
    user_message: str,
    assistant_response: str,
    timeout: float = 15.0,
) -> Optional[str]:
    """Generate a title for a user/assistant exchange."""
    return _generate_title(_build_title_messages(user_message, assistant_response), timeout)


def generate_title_from_message(
    user_message: str,
    timeout: float = 15.0,
) -> Optional[str]:
    """Generate a title for a single user message."""
    return _generate_title(_build_title_messages(user_message), timeout)


async def async_generate_title_from_message(
    user_message: str,
    timeout: float = 15.0,
) -> Optional[str]:
    """Asynchronously generate a title for a single user message off-loop."""
    try:
        return await asyncio.to_thread(generate_title_from_message, user_message, timeout)
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return None
