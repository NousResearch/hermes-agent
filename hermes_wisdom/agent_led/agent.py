"""Agent invocation for agent-led review tasks.

Prompts live as Markdown files under ``prompts/`` at the repository root so
they can be reviewed and edited without touching code. Each task is a
one-shot auxiliary model call with strict JSON output that is validated by
:mod:`hermes_wisdom.agent_led.schemas`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .privacy import model_safe_data

from .schemas import (
    CandidateReviewResult,
    RecipientRecommendation,
    SchemaRejected,
    SharePackage,
    parse_candidate_review,
    parse_recipient_recommendation,
    parse_share_package,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
CANDIDATE_REVIEW_PROMPT = "wisdom_candidate_review.md"
RECIPIENT_RECOMMENDATION_PROMPT = "wisdom_recipient_recommendation.md"
SHARE_PACKAGING_PROMPT = "wisdom_share_packaging.md"

ModelCall = Callable[[list[dict[str, str]], dict[str, Any]], str]


def session_model_call(runtime, *, name: str, max_tokens: int = 6000) -> ModelCall:
    """Bounded tool-free work on the selected conversation's model route."""

    def call(messages, schema):
        from agent.auxiliary_client import call_llm, extract_content_or_reasoning

        if not runtime.get("model") or not runtime.get("provider"):
            raise ValueError("Wisdom work requires an active session model")
        response = call_llm(
            provider=runtime["provider"],
            model=runtime["model"],
            main_runtime=runtime,
            tools=[],
            messages=messages,
            temperature=0,
            timeout=45,
            max_tokens=max_tokens,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": True,
                        "schema": schema,
                    },
                }
            },
        )
        choices = getattr(response, "choices", []) or []
        if choices and getattr(choices[0].message, "tool_calls", None):
            raise ValueError("Wisdom work requested disallowed tools")
        return extract_content_or_reasoning(response)

    return call


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _default_model_call(messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    response = call_llm(
        task="background_review",
        messages=messages,
        temperature=0,
        max_tokens=2500,
        timeout=45,
        tools=[],
        extra_body={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": str(schema.get("title") or "wisdom_agent_led"),
                    "strict": True,
                    "schema": schema,
                },
            }
        },
    )
    return extract_content_or_reasoning(response).strip()


def _run(
    prompt_name: str,
    payload: dict[str, Any],
    model: type[BaseModel],
    parser: Callable[[str], Any],
    *,
    model_call: ModelCall | None = None,
    max_attempts: int = 2,
) -> Any:
    call = model_call or _default_model_call
    messages = [
        {"role": "system", "content": load_prompt(prompt_name)},
        {
            "role": "user",
            "content": json.dumps(
                model_safe_data(payload), sort_keys=True, ensure_ascii=True, default=str
            ),
        },
    ]
    schema = model.model_json_schema()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        raw = call(messages, schema)
        try:
            return parser(raw)
        except SchemaRejected as exc:
            last_error = exc
            logger.warning(
                "Agent-led %s output rejected on attempt %d: %s",
                prompt_name,
                attempt,
                exc,
            )
            messages = [
                *messages[:2],
                {"role": "assistant", "content": model_safe_data(raw)},
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed validation: "
                        + json.dumps(
                            [
                                {"type": error.get("type"), "loc": error.get("loc")}
                                for error in exc.errors
                            ],
                            default=str,
                        )[:2000]
                        + ". Return corrected strict JSON only."
                    ),
                },
            ]
    assert last_error is not None
    raise last_error


def review_candidates(
    payload: dict[str, Any], *, model_call: ModelCall | None = None
) -> CandidateReviewResult:
    return _run(
        CANDIDATE_REVIEW_PROMPT,
        payload,
        CandidateReviewResult,
        parse_candidate_review,
        model_call=model_call,
    )


def recommend_to_recipient(
    payload: dict[str, Any], *, model_call: ModelCall | None = None
) -> RecipientRecommendation:
    return _run(
        RECIPIENT_RECOMMENDATION_PROMPT,
        payload,
        RecipientRecommendation,
        parse_recipient_recommendation,
        model_call=model_call,
    )


def package_for_share(
    payload: dict[str, Any], *, model_call: ModelCall | None = None
) -> SharePackage:
    return _run(
        SHARE_PACKAGING_PROMPT,
        payload,
        SharePackage,
        parse_share_package,
        model_call=model_call,
    )
