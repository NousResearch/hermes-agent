"""Strict private contract types for the Becky Actions bridge."""

from __future__ import annotations

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from gateway.action_journal import MutationEvent, MutationPage, MutationStatus

ACTION_METHODS = ["execute_one_shot", "list_mutations", "start_loop"]


class _ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionCapabilities(_ActionModel):
    schema_version: Literal["1"]
    policy_version: Literal["1"]
    methods: list[Literal["execute_one_shot", "list_mutations", "start_loop"]] = Field(
        min_length=3, max_length=3
    )

    @field_validator("methods")
    @classmethod
    def _exact_methods(cls, value: list[str]) -> list[str]:
        if set(value) != set(ACTION_METHODS):
            raise ValueError("incompatible action methods")
        return value


class MutationListRequest(_ActionModel):
    after_cursor: Annotated[StrictStr, Field(min_length=1, max_length=512)] | None
    limit: Annotated[StrictInt, Field(ge=1, le=100)]


class OneShotRequest(_ActionModel):
    title: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    text: Annotated[StrictStr, Field(min_length=1, max_length=4_000)]
    idempotency_key: UUID
    note_default: Literal["obsidian", "apple_notes", "google_docs"]
    policy_version: Literal["1"]


class OneShotResult(_ActionModel):
    schema_version: Literal["1"]
    disposition: Literal["succeeded", "failed", "needs_loop", "ignored"]
    event: MutationEvent | None

    @model_validator(mode="after")
    def _event_matches_disposition(self) -> Self:
        attempted = self.disposition in {"succeeded", "failed"}
        if attempted != (self.event is not None):
            raise ValueError("one-shot event does not match disposition")
        if self.event is not None and self.event.status.value != self.disposition:
            raise ValueError("one-shot event status does not match disposition")
        return self


class StartLoopRequest(_ActionModel):
    title: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    context: Annotated[StrictStr, Field(min_length=1, max_length=2_000)]
    prior_status: Literal["succeeded", "failed"]
    idempotency_key: UUID


class StartLoopResult(_ActionModel):
    schema_version: Literal["1"]
    state: Literal["completed", "pending"]
    title: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    telegram_url: Annotated[
        StrictStr,
        Field(
            pattern=(
                r"^https://t\.me/c/[1-9][0-9]*/[1-9][0-9]*"
                r"(?:/[1-9][0-9]*\?single)?$"
            )
        ),
    ]


def parse_one_shot_params(params: object) -> OneShotRequest | None:
    if not isinstance(params, dict):
        return None
    try:
        return OneShotRequest.model_validate(params)
    except Exception:
        return None


def parse_start_loop_params(params: object) -> StartLoopRequest | None:
    if not isinstance(params, dict):
        return None
    try:
        return StartLoopRequest.model_validate(params)
    except Exception:
        return None


def action_capabilities() -> dict[str, object]:
    return ActionCapabilities(
        schema_version="1",
        policy_version="1",
        methods=list(ACTION_METHODS),
    ).model_dump(mode="json")


def parse_mutation_list_params(params: object) -> MutationListRequest | None:
    if not isinstance(params, dict):
        return None
    try:
        return MutationListRequest.model_validate(params)
    except Exception:
        return None


def dump_mutation_page(page: MutationPage) -> dict[str, object]:
    return page.model_dump(mode="json")


__all__ = [
    "ACTION_METHODS",
    "ActionCapabilities",
    "OneShotRequest",
    "OneShotResult",
    "StartLoopRequest",
    "StartLoopResult",
    "MutationListRequest",
    "MutationEvent",
    "MutationStatus",
    "action_capabilities",
    "dump_mutation_page",
    "parse_one_shot_params",
    "parse_mutation_list_params",
    "parse_start_loop_params",
]
