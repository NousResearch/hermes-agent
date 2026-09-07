"""Strict schemas for agent-authored Collective Wisdom output.

Two agent tasks return structured JSON: candidate review (which of the
user's own skills to propose sharing) and recipient recommendation (whether a
teammate's published skill is worth an interruption). Both are validated
here with pydantic; invalid output is rejected, and a conservative repair
pass is attempted before giving up so a single malformed field does not
silently drop a whole review.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION = 1

CANDIDATE_ACTIONS: tuple[str, ...] = ("share", "review", "not_now")
RECIPIENT_ACTIONS: tuple[str, ...] = ("install", "view", "mute")
UPDATE_ACTIONS: tuple[str, ...] = ("update", "view_changes", "mute")

_SECRET_SHAPES = (
    re.compile(r"(?i)\b(sk|ghp|gho|xox[abpr]|AKIA|AIza)[A-Za-z0-9_-]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{16,}"),
)


class SchemaRejected(ValueError):
    """Raised when agent output cannot be validated or repaired."""

    def __init__(self, message: str, *, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def _no_secret_shapes(value: str) -> str:
    for pattern in _SECRET_SHAPES:
        if pattern.search(value):
            raise ValueError("text contains a credential-shaped string")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class UsageEvidence(_Strict):
    """Real invocation evidence; never estimated by the agent."""

    window_days: int = Field(ge=1, le=90)
    invocation_count: int = Field(ge=0)
    days_used: int = Field(ge=0)
    last_used_at: str | None = None
    examples: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("examples")
    @classmethod
    def _examples_clean(cls, value: list[str]) -> list[str]:
        return [_no_secret_shapes(item)[:200] for item in value]


class PortabilityNote(_Strict):
    kind: Literal[
        "env_var",
        "command",
        "account",
        "service",
        "permission",
        "org_specific",
        "path",
        "other",
    ]
    detail: str = Field(min_length=1, max_length=300)
    action: Literal["remove", "generalize", "document", "keep"] = "document"

    @field_validator("detail")
    @classmethod
    def _detail_clean(cls, value: str) -> str:
        return _no_secret_shapes(value)


class CandidateRecommendation(_Strict):
    """One skill the agent proposes the owner share."""

    skill_name: str = Field(min_length=1, max_length=120)
    content_hash: str = Field(min_length=8, max_length=128)
    editorial_name: str = Field(min_length=1, max_length=80)
    one_line_description: str = Field(min_length=1, max_length=200)
    what_it_does: str = Field(min_length=1, max_length=600)
    how_user_relied_on_it: str = Field(min_length=1, max_length=600)
    evidence: UsageEvidence
    why_coworkers_benefit: str = Field(min_length=1, max_length=600)
    audience: str = Field(min_length=1, max_length=120, description="team or organization")
    portability: list[PortabilityNote] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    allowed_actions: list[Literal["share", "review", "not_now"]] = Field(
        default_factory=lambda: list(CANDIDATE_ACTIONS)
    )

    @field_validator(
        "editorial_name",
        "one_line_description",
        "what_it_does",
        "how_user_relied_on_it",
        "why_coworkers_benefit",
        "audience",
    )
    @classmethod
    def _prose_clean(cls, value: str) -> str:
        return _no_secret_shapes(value)

    @field_validator("allowed_actions")
    @classmethod
    def _actions_complete(cls, value: list[str]) -> list[str]:
        # The owner can always defer; the agent may not remove that choice.
        if "not_now" not in value:
            value = [*value, "not_now"]
        return list(dict.fromkeys(value))


class CandidateReviewResult(_Strict):
    """Top-level output of the candidate review prompt."""

    schema_version: Literal[1] = SCHEMA_VERSION
    window_days: int = Field(ge=1, le=90)
    considered: list[str] = Field(default_factory=list)
    recommendations: list[CandidateRecommendation] = Field(default_factory=list, max_length=5)
    nothing_to_recommend_reason: str | None = Field(default=None, max_length=400)
    requires_confirmation: Literal[True] = True

    @field_validator("recommendations")
    @classmethod
    def _unique_skills(cls, value: list[CandidateRecommendation]) -> list[CandidateRecommendation]:
        seen: set[str] = set()
        for item in value:
            if item.skill_name in seen:
                raise ValueError(f"duplicate recommendation for {item.skill_name}")
            seen.add(item.skill_name)
        return value


class RecipientRecommendation(_Strict):
    """Top-level output of the recipient recommendation prompt."""

    schema_version: Literal[1] = SCHEMA_VERSION
    skill_id: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1)
    relevant: bool
    not_relevant_reason: str | None = Field(default=None, max_length=400)
    editorial_name: str | None = Field(default=None, max_length=80)
    outcome_one_liner: str | None = Field(default=None, max_length=200)
    publisher_name: str | None = Field(default=None, max_length=120)
    publisher_uses_it_for: str | None = Field(default=None, max_length=300)
    evidence: UsageEvidence | None = None
    why_it_helps_recipient: str | None = Field(default=None, max_length=400)
    safety_summary: str | None = Field(default=None, max_length=200)
    installation_count: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    allowed_actions: list[Literal["install", "view", "mute"]] = Field(
        default_factory=lambda: list(RECIPIENT_ACTIONS)
    )
    installed: Literal[False] = False

    @field_validator(
        "editorial_name",
        "outcome_one_liner",
        "publisher_uses_it_for",
        "why_it_helps_recipient",
        "safety_summary",
    )
    @classmethod
    def _prose_clean(cls, value: str | None) -> str | None:
        return _no_secret_shapes(value) if value else value

    @field_validator("allowed_actions")
    @classmethod
    def _mute_always_available(cls, value: list[str]) -> list[str]:
        if "mute" not in value:
            value = [*value, "mute"]
        return list(dict.fromkeys(value))

    def require_complete_when_relevant(self) -> "RecipientRecommendation":
        if not self.relevant:
            return self
        missing = [
            name
            for name in (
                "editorial_name",
                "outcome_one_liner",
                "publisher_name",
                "publisher_uses_it_for",
                "why_it_helps_recipient",
            )
            if not getattr(self, name)
        ]
        if missing:
            raise SchemaRejected(
                "relevant recommendation is missing required fields: " + ", ".join(missing)
            )
        return self


class PackagingRequirement(_Strict):
    kind: Literal["command", "account", "service", "permission", "env_var", "skill", "script"]
    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=300)
    handoff: str | None = Field(
        default=None,
        max_length=300,
        description="How the recipient obtains it; never the value itself.",
    )

    @field_validator("purpose", "handoff")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return _no_secret_shapes(value) if value else value


class PackagedFile(_Strict):
    path: str = Field(min_length=1, max_length=255)
    content: str
    generalized_from_original: bool = False

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        from ..package import _validate_package_path

        _validate_package_path(value, source="proposed package")
        return value

    @field_validator("content")
    @classmethod
    def _content_clean(cls, value: str) -> str:
        return _no_secret_shapes(value)


class SharePackage(_Strict):
    """Output of the structured packaging task in the Share flow."""

    schema_version: Literal[1] = SCHEMA_VERSION
    skill_name: str = Field(min_length=1, max_length=120)
    source_content_hash: str = Field(min_length=8, max_length=128)
    editorial_name: str = Field(min_length=1, max_length=80)
    plain_description: str = Field(min_length=1, max_length=400)
    files: list[PackagedFile] = Field(min_length=1, max_length=64)
    requirements: list[PackagingRequirement] = Field(default_factory=list, max_length=40)
    setup_instructions: list[str] = Field(default_factory=list, max_length=20)
    credential_handoff: list[str] = Field(default_factory=list, max_length=20)
    compatibility_limits: list[str] = Field(default_factory=list, max_length=20)
    verification_step: str = Field(min_length=1, max_length=400)
    removed_or_generalized: list[str] = Field(default_factory=list, max_length=40)
    related_skills: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("skill_name")
    @classmethod
    def _skill_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}", value):
            raise ValueError("skill_name must be a single safe identifier")
        return value

    @field_validator("files")
    @classmethod
    def _has_skill_md(cls, value: list[PackagedFile]) -> list[PackagedFile]:
        from ..package import _validate_package_path

        keys = [_validate_package_path(item.path, source="proposed package")[1] for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("package contains colliding paths")
        if not any(item.path == "SKILL.md" for item in value):
            raise ValueError("package must include SKILL.md")
        return value

    @field_validator(
        "setup_instructions", "credential_handoff", "compatibility_limits", "removed_or_generalized"
    )
    @classmethod
    def _lines_clean(cls, value: list[str]) -> list[str]:
        return [_no_secret_shapes(line)[:400] for line in value]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _loads_lenient(raw: str) -> Any:
    text = raw.strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _repair(model: type[BaseModel], data: Any, errors: list[dict[str, Any]]) -> Any:
    """Conservative repair: drop offending list items, never invent content."""
    if not isinstance(data, dict):
        raise SchemaRejected("agent output is not a JSON object", errors=errors)
    repaired = dict(data)
    dropped = False
    for error in errors:
        loc = error.get("loc") or ()
        if len(loc) >= 2 and isinstance(loc[1], int) and isinstance(repaired.get(loc[0]), list):
            items = list(repaired[loc[0]])
            if 0 <= loc[1] < len(items):
                items.pop(loc[1])
                repaired[loc[0]] = items
                dropped = True
        elif len(loc) == 1 and error.get("type") == "extra_forbidden":
            repaired.pop(loc[0], None)
            dropped = True
    if not dropped:
        raise SchemaRejected("agent output failed schema validation", errors=errors)
    return repaired


def parse_model(model: type[BaseModel], raw: str | dict[str, Any], *, repair: bool = True) -> Any:
    """Validate ``raw`` against ``model``; attempt one repair pass; else reject."""
    try:
        data = raw if isinstance(raw, dict) else _loads_lenient(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SchemaRejected(f"agent output is not valid JSON: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        if not repair:
            raise SchemaRejected("agent output failed schema validation", errors=errors) from exc
    repaired = _repair(model, data, errors)
    try:
        return model.model_validate(repaired)
    except ValidationError as exc:
        raise SchemaRejected(
            "agent output failed schema validation after repair", errors=exc.errors()
        ) from exc


def parse_candidate_review(raw: str | dict[str, Any]) -> CandidateReviewResult:
    return parse_model(CandidateReviewResult, raw)


def parse_recipient_recommendation(raw: str | dict[str, Any]) -> RecipientRecommendation:
    parsed: RecipientRecommendation = parse_model(RecipientRecommendation, raw)
    return parsed.require_complete_when_relevant()


def parse_share_package(raw: str | dict[str, Any]) -> SharePackage:
    return parse_model(SharePackage, raw, repair=False)


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()
