"""OpenRouter Decisions API client for ``hermes jev``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"
_QUESTION_TYPES = {"noul", "choice", "score"}

# These thresholds describe advice to the coordinator. This module never approves,
# executes, publishes, deploys, or changes permissions on the coordinator's behalf.
AUTO_APPROVAL_MAX_RISK = 1
AUTO_APPROVAL_MAX_REVIEW_PROBABILITY = 0.1
HUMAN_ESCALATION_MIN_RISK = 3
HUMAN_ESCALATION_MIN_REVIEW_PROBABILITY = 0.9

RISK_RUBRIC = (
    "Trivial — reversible",
    "Low",
    "Moderate",
    "High — needs review",
    "Critical — human must approve",
)

TASK_CLASSIFICATION_QUESTIONS = {
    "risk_level": {
        "type": "score",
        "instructions": (
            "Classify the task's operational risk. Consider reversibility, blast radius, "
            "production impact, security, data loss, permissions, and ambiguity."
        ),
        "criteria": list(RISK_RUBRIC),
    },
    "agent_choice": {
        "type": "choice",
        "instructions": "Choose the best agent arrangement for completing this task.",
        "criteria": {
            "codex": "Code, debugging, implementation, or tests dominate",
            "claude": "Planning, research, specifications, writing, or critique dominate",
            "either": "Either agent is comparably suitable for this routine task",
            "both": "One agent should implement and the other independently review",
        },
    },
    "needs_review": {
        "type": "noul",
        "instructions": (
            "Does this task require independent review because it affects security, "
            "architecture, permissions, production data, or has materially ambiguous requirements?"
        ),
        "criteria": {
            "true": "Any listed concern is present or the cost of an error is substantial",
            "false": "The task is simple, clear, low-risk, and readily reversible",
        },
    },
    "model_class": {
        "type": "choice",
        "instructions": (
            "Choose the least expensive model class likely to complete the task reliably."
        ),
        "criteria": {
            "cheap_fast": "Routine summarization, classification, or simple bounded work",
            "mid": "Normal implementation or analysis with moderate complexity",
            "premium": "High complexity, ambiguity, broad context, or costly failure modes",
            "no_llm": "A deterministic command or program can complete the task without an LLM",
        },
    },
}

MODEL_TIER_CRITERIA = {
    "below_floor": (
        "Does not match the overall capability and reliability floor represented by "
        "Anthropic Claude Sonnet 4.6"
    ),
    "cheap_fast": (
        "Meets the floor and is best for routine, bounded, latency- or cost-sensitive work"
    ),
    "mid": "Meets the floor and is best for normal implementation, analysis, and review",
    "premium": (
        "Meets the floor and is best for complex, ambiguous, broad-context, or high-risk work"
    ),
}
_MODEL_CLASSIFICATION_BATCH_SIZE = 20


class JevInputError(ValueError):
    """Raised when CLI input cannot form a valid Decisions API request."""


def _parse_json_or_file(value: str, *, label: str) -> Any:
    """Parse inline JSON, or JSON from an existing file path."""
    path = Path(value).expanduser()
    try:
        raw = path.read_text(encoding="utf-8") if path.is_file() else value
        return json.loads(raw)
    except OSError as exc:
        raise JevInputError(f"could not read {label} file: {exc}") from exc
    except json.JSONDecodeError as exc:
        source = f"file {path}" if path.is_file() else "inline JSON"
        raise JevInputError(f"invalid {label} JSON in {source}: {exc.msg}") from exc


def parse_state(value: str) -> str | dict[str, Any]:
    """Return an object for JSON object input, otherwise preserve the state string."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(parsed, dict):
        raise JevInputError(
            "state JSON must be an object; use plain text for a string state"
        )
    return parsed


def parse_questions(value: str) -> dict[str, dict[str, Any]]:
    """Load and validate the question mapping accepted by the Decisions API."""
    questions = _parse_json_or_file(value, label="questions")
    if not isinstance(questions, dict) or not questions:
        raise JevInputError(
            "questions must be a non-empty JSON object keyed by question id"
        )

    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise JevInputError("each question id must be a non-empty string")
        if not isinstance(question, dict):
            raise JevInputError(f"question {question_id!r} must be an object")
        question_type = question.get("type")
        if question_type not in _QUESTION_TYPES:
            raise JevInputError(
                f"question {question_id!r} has invalid type {question_type!r}; "
                "expected noul, choice, or score"
            )
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise JevInputError(
                f"question {question_id!r} needs non-empty instructions"
            )

        criteria = question.get("criteria")
        if question_type == "noul":
            valid = criteria is None or (
                isinstance(criteria, dict) and set(criteria) == {"true", "false"}
            )
            expectation = "omitted or an object with exactly true and false keys"
        elif question_type == "choice":
            valid = isinstance(criteria, dict) and len(criteria) >= 2
            expectation = "an object with at least two named options"
        else:
            valid = isinstance(criteria, list) and 2 <= len(criteria) <= 10
            expectation = "an ordered list of 2 to 10 levels"
        if not valid:
            raise JevInputError(
                f"question {question_id!r} criteria must be {expectation} for type {question_type}"
            )
    return questions


def load_openrouter_api_key() -> str:
    """Read the active profile's OpenRouter key without mutating process state."""
    from agent.secret_scope import load_env_file

    env_path = get_hermes_home() / ".env"
    key = (load_env_file(env_path).get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise JevInputError(f"OPENROUTER_API_KEY is not set in {env_path}")
    return key


def request_decision(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST one decision request and return its JSON document."""
    import httpx

    response = httpx.post(
        DECISIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()
    document = response.json()
    if not isinstance(document, dict):
        raise JevInputError("Decisions API returned a non-object JSON response")
    return document


def classify_model_metadata(
    catalog: list[dict[str, Any]], api_key: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Ask Jev to tier provider-neutral, deterministically prefiltered metadata."""
    from hermes_cli.jev_models import metadata_for_jev, provisional_catalog

    candidates, _ = provisional_catalog(catalog)
    assignments: dict[str, dict[str, Any]] = {}
    usage: list[dict[str, Any]] = []
    for offset in range(0, len(candidates), _MODEL_CLASSIFICATION_BATCH_SIZE):
        batch = candidates[offset : offset + _MODEL_CLASSIFICATION_BATCH_SIZE]
        question_ids = [f"model_{offset + index:04d}" for index in range(len(batch))]
        state_candidates = {
            question_id: metadata_for_jev(candidate)
            for question_id, candidate in zip(question_ids, batch, strict=True)
        }
        questions = {
            question_id: {
                "type": "choice",
                "instructions": (
                    f"Classify candidates.{question_id} by observed/declared capability, "
                    "reliability evidence, context, tool support, pricing, latency evidence, "
                    "and appropriate use. Family or version alone must not determine the tier."
                ),
                "criteria": MODEL_TIER_CRITERIA,
            }
            for question_id in question_ids
        }
        result = request_decision(
            {
                "model": JEV_MODEL,
                "state": {
                    "absolute_floor": "anthropic/claude-sonnet-4.6-equivalent",
                    "candidates": state_candidates,
                },
                "questions": questions,
            },
            api_key,
        )
        for question_id, candidate in zip(question_ids, batch, strict=True):
            answer = _answer(result, question_id)
            tier = answer.get("choice")
            if tier not in MODEL_TIER_CRITERIA:
                raise JevInputError(
                    f"Decisions API returned an invalid model tier for {question_id!r}"
                )
            confidence = answer.get("confidence")
            assignments[candidate["canonical_model"]["id"]] = {
                "tier": tier,
                "confidence": (
                    float(confidence)
                    if isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    else None
                ),
            }
        if isinstance(result.get("usage"), dict):
            usage.append(result["usage"])
    return assignments, usage


def _answer(result: dict[str, Any], question_id: str) -> dict[str, Any]:
    """Return one typed answer or reject a malformed Decisions API response."""
    answers = result.get("answers")
    answer = answers.get(question_id) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise JevInputError(f"Decisions API response is missing answer {question_id!r}")
    return answer


def _choice(result: dict[str, Any], question_id: str) -> str:
    answer = _answer(result, question_id)
    choice = answer.get("choice")
    allowed = TASK_CLASSIFICATION_QUESTIONS[question_id]["criteria"]
    if not isinstance(choice, str) or choice not in allowed:
        raise JevInputError(
            f"Decisions API returned an invalid choice for {question_id!r}"
        )
    return choice


def _risk_level(answer: dict[str, Any]) -> tuple[int, float]:
    """Convert Jev's normalized score to the corresponding five-level rubric index."""
    raw_score = answer.get("score")
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        raise JevInputError("Decisions API returned an invalid risk score")
    normalized = min(1.0, max(0.0, float(raw_score)))
    # Half-step boundaries round upward so a borderline risk is never understated.
    level = int(normalized * (len(RISK_RUBRIC) - 1) + 0.5)
    return level, normalized


def _review_probability(answer: dict[str, Any]) -> float:
    probability = answer.get("noul")
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        raise JevInputError(
            "Decisions API returned an invalid needs_review probability"
        )
    return min(1.0, max(0.0, float(probability)))


def _routing_advice(risk: int, review_probability: float) -> str:
    """Apply documented policy thresholds without granting approval or taking action."""
    if (
        risk >= HUMAN_ESCALATION_MIN_RISK
        or review_probability >= HUMAN_ESCALATION_MIN_REVIEW_PROBABILITY
    ):
        return "escalate_to_human"
    if (
        risk <= AUTO_APPROVAL_MAX_RISK
        and review_probability < AUTO_APPROVAL_MAX_REVIEW_PROBABILITY
    ):
        return "eligible_for_coordinator_auto_approval"
    return "recommend_review"


def classify_task(
    task_text: str,
    *,
    api_key: str | None = None,
    catalog: list[dict[str, Any]] | None = None,
    model_classifications: dict[str, dict[str, Any]] | None = None,
    native_catalogs: dict[str, list[str]] | None = None,
    min_context_length: int = 0,
    require_reasoning: bool = False,
    require_structured_outputs: bool = False,
) -> dict[str, Any]:
    """Ask Jev for advisory orchestration classifications and format the recommendation."""
    if not task_text.strip():
        raise JevInputError("task text must not be empty")
    resolved_api_key = api_key or load_openrouter_api_key()
    result = request_decision(
        {
            "model": JEV_MODEL,
            "state": task_text,
            "questions": TASK_CLASSIFICATION_QUESTIONS,
        },
        resolved_api_key,
    )
    risk_answer = _answer(result, "risk_level")
    risk, raw_risk = _risk_level(risk_answer)
    review_answer = _answer(result, "needs_review")
    review_probability = _review_probability(review_answer)
    agent = _choice(result, "agent_choice")
    model_class = _choice(result, "model_class")
    needs_review = review_probability >= 0.5
    disposition = _routing_advice(risk, review_probability)

    from hermes_cli.jev_models import (
        cached_catalog_resolution,
        classify_catalog,
        load_native_catalogs,
        model_requirements,
        resolve_model,
    )

    requirements = model_requirements(
        min_context_length=min_context_length,
        require_tools=True,
        require_reasoning=require_reasoning,
        require_structured_outputs=require_structured_outputs,
    )
    if model_class == "no_llm":
        model_recommendation = {
            "provider": None,
            "provider_model_id": None,
            "canonical_model": None,
            "selected_model_id": None,
            "requested_tier": "no_llm",
            "tier": "no_llm",
            "recommended_use": "deterministic work without an LLM",
            "agent_choice": agent,
            "provider_routes": [],
            "fallback_chain": [],
            "fallback_policy": {
                "route_before_model": True,
                "native_before_openrouter_when_compatible": True,
                "native_catalog_match_required": True,
                "allowed_tiers": [],
                "same_tier_then_higher": True,
                "lower_tier_forbidden": True,
                "runtime_must_not_silently_downgrade": True,
            },
            "requirements": requirements,
            "reasons": [
                "Jev classified the task as deterministic work that does not require an LLM",
                "agent choice was evaluated independently from model choice",
            ],
            "jev_catalog_usage": [],
        }
    else:
        assignments = model_classifications
        catalog_usage: list[dict[str, Any]] = []
        if catalog is None:
            live_catalog, cached_assignments, catalog_usage = cached_catalog_resolution(
                resolved_api_key,
                classify_model_metadata,
                catalog_only=assignments is not None,
            )
            if assignments is None:
                assignments = cached_assignments
        else:
            live_catalog = catalog
        if assignments is None and catalog is not None:
            assignments, catalog_usage = classify_model_metadata(
                live_catalog, resolved_api_key
            )
        if native_catalogs is None:
            native_catalogs = load_native_catalogs() if catalog is None else {}
        candidates, _ = classify_catalog(
            live_catalog, assignments, native_catalogs=native_catalogs
        )
        model_recommendation = resolve_model(
            candidates, model_class, requirements, agent_choice=agent
        )
        model_recommendation["jev_catalog_usage"] = catalog_usage

    return {
        "risk_level": {
            "score": risk,
            "label": RISK_RUBRIC[risk],
            "raw_score": raw_risk,
        },
        "agent_choice": {"choice": agent},
        "needs_review": {
            "noul": review_probability,
            "value": needs_review,
        },
        "model_class": {"choice": model_class},
        "recommendation": {
            "agent": agent,
            "model": model_recommendation,
            "requires_review": needs_review,
            "model_class": model_class,
            "risk": RISK_RUBRIC[risk],
            "disposition": disposition,
            "advisory_only": True,
            "message": (
                f"Recommend agent {agent}; {RISK_RUBRIC[risk]} risk; model tier {model_class}; "
                f"model {model_recommendation['selected_model_id'] or 'none'} selected independently; "
                f"disposition {disposition}. Jev does not approve or execute actions."
            ),
        },
        "usage": result.get("usage", {}),
    }


def cmd_classify_task(args: Any) -> int:
    """Print an advisory orchestration classification for task text."""
    try:
        classification = classify_task(
            args.task_text,
            min_context_length=getattr(args, "min_context_length", 0),
            require_reasoning=getattr(args, "require_reasoning", False),
            require_structured_outputs=getattr(
                args, "require_structured_outputs", False
            ),
        )
    except JevInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            import httpx

            is_http_error = isinstance(exc, httpx.HTTPError)
        except ImportError:
            is_http_error = False
        if not is_http_error:
            raise
        print(f"Error: OpenRouter request failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(classification, indent=2))
    return 0


def cmd_models(args: Any) -> int:
    """Print the current authorized catalog and optional safe fallback order."""
    from hermes_cli.jev_models import (
        cached_catalog_resolution,
        inspect_catalog,
        load_native_catalogs,
        model_requirements,
    )

    try:
        api_key = load_openrouter_api_key()
        requirements = model_requirements(
            min_context_length=getattr(args, "min_context_length", 0),
            require_tools=True,
            require_reasoning=getattr(args, "require_reasoning", False),
            require_structured_outputs=getattr(
                args, "require_structured_outputs", False
            ),
        )
        catalog_only = getattr(args, "catalog_only", False)
        catalog, assignments, usage = cached_catalog_resolution(
            api_key,
            classify_model_metadata,
            catalog_only=catalog_only,
        )
        native_catalogs = load_native_catalogs()
        result = inspect_catalog(
            catalog,
            assignments=assignments,
            requested_tier=getattr(args, "tier", None),
            requirements=requirements,
            native_catalogs=native_catalogs,
        )
        result["jev_usage"] = usage
    except JevInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            import httpx

            is_http_error = isinstance(exc, httpx.HTTPError)
        except ImportError:
            is_http_error = False
        if not is_http_error:
            raise
        print(f"Error: OpenRouter models request failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_jev(args: Any) -> int:
    """Build a Decisions API request, execute it, and print the response JSON."""
    try:
        payload = {
            "model": JEV_MODEL,
            "state": parse_state(args.state),
            "questions": parse_questions(args.questions),
        }
        result = request_decision(payload, load_openrouter_api_key())
    except JevInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        try:
            import httpx

            is_http_error = isinstance(exc, httpx.HTTPError)
        except ImportError:
            is_http_error = False
        if not is_http_error:
            raise
        print(f"Error: Decisions API request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0
