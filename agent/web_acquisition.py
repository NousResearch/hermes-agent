"""Validated requests, outcomes and provenance for native web acquisition."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.web_acquisition_errors import WebFailureData, failure_from_data


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class WebSearchRequest(_Contract):
    """Search through native routing and caching without model-tool presentation."""

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def _nonblank_query(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("A search query must contain non-whitespace characters")
        return query


class WebSearchHit(_Contract):
    """Search metadata without upstream diagnostics or arbitrary vendor fields."""

    url: str = Field(min_length=1)
    title: str = ""
    description: str = ""
    position: int | None = Field(default=None, ge=0)


WebAttemptRoute = Literal["selected", "managed-fallback", "keyless", "keyless-rescue"]


class WebAttemptSuccess(_Contract):
    """A completed acquisition attempt, with the actual provider when known."""

    status: Literal["succeeded"] = "succeeded"
    provider: str | None = Field(default=None, min_length=1)
    route: WebAttemptRoute


class WebAttemptFailure(_Contract):
    """An acquisition attempt which failed before a later alternative ran."""

    status: Literal["failed"] = "failed"
    provider: str | None = Field(default=None, min_length=1)
    route: WebAttemptRoute
    failure: WebFailureData

    @field_validator("failure", mode="before")
    @classmethod
    def _safe_failure(cls, failure: object) -> WebFailureData:
        return failure_from_data(failure).to_failure()


WebAttempt = Annotated[
    WebAttemptSuccess | WebAttemptFailure, Field(discriminator="status")
]


class WebSearchSuccess(_Contract):
    """Search hits and provenance after native cache, fallback and rescue handling."""

    status: Literal["ok"] = "ok"
    selected_provider: str = Field(min_length=1)
    served_provider: str | None = Field(default=None, min_length=1)
    cache: Literal["hit", "miss", "disabled"]
    hits: tuple[WebSearchHit, ...]
    attempts: tuple[WebAttempt, ...] = Field(default=(), max_length=8)


class WebSearchFailure(_Contract):
    """The primary failure and any subsequent unsuccessful acquisition attempts."""

    status: Literal["failed"] = "failed"
    selected_provider: str | None = Field(default=None, min_length=1)
    failure: WebFailureData
    attempts: tuple[WebAttempt, ...] = Field(default=(), max_length=8)

    @field_validator("failure", mode="before")
    @classmethod
    def _safe_failure(cls, failure: object) -> WebFailureData:
        return failure_from_data(failure).to_failure()


WebSearchResult = Annotated[
    WebSearchSuccess | WebSearchFailure, Field(discriminator="status")
]


WebExtractFormat = Literal["markdown", "html"]


class WebExtractCapabilities(_Contract):
    """Options which a provider guarantees to apply during extraction."""

    formats: tuple[WebExtractFormat, ...] = ()
    max_characters: bool = False


class WebExtractRequest(_Contract):
    """Acquire pages with explicit content requirements, before presentation limits."""

    urls: tuple[str, ...] = Field(min_length=1)
    format: WebExtractFormat | None = None
    required_coverage: Literal["any", "full"] = "any"
    max_characters: int | None = Field(default=None, ge=1)

    @field_validator("urls")
    @classmethod
    def _nonblank_urls(cls, urls: tuple[str, ...]) -> tuple[str, ...]:
        if any(not url.strip() for url in urls):
            raise ValueError(
                "Each extraction URL must contain non-whitespace characters"
            )
        return urls


class WebExtractPageSuccess(_Contract):
    """An attributed page, with coverage and provenance preserved across cache hits."""

    status: Literal["ok"] = "ok"
    requested_url: str = Field(min_length=1)
    resolved_url: str = Field(min_length=1)
    title: str = ""
    content: str
    coverage: Literal["full", "partial", "unknown"]
    served_provider: str | None = Field(default=None, min_length=1)
    cache: Literal["hit", "miss", "disabled"]
    attempts: tuple[WebAttempt, ...] = Field(default=(), max_length=8)


class WebExtractPageFailure(_Contract):
    """A page failure without upstream diagnostics or arbitrary vendor metadata."""

    status: Literal["failed"] = "failed"
    requested_url: str = Field(min_length=1)
    failure: WebFailureData
    attempts: tuple[WebAttempt, ...] = Field(default=(), max_length=8)

    @field_validator("failure", mode="before")
    @classmethod
    def _safe_failure(cls, failure: object) -> WebFailureData:
        return failure_from_data(failure).to_failure()


WebExtractPage = Annotated[
    WebExtractPageSuccess | WebExtractPageFailure, Field(discriminator="status")
]


class WebExtractSuccess(_Contract):
    """One result for each requested URL, including duplicates and per-page failures."""

    status: Literal["ok"] = "ok"
    selected_provider: str = Field(min_length=1)
    results: tuple[WebExtractPage, ...]


class WebExtractFailure(_Contract):
    """A whole-call refusal or routing failure, without echoing secret URLs."""

    status: Literal["failed"] = "failed"
    selected_provider: str | None = Field(default=None, min_length=1)
    failure: WebFailureData

    @field_validator("failure", mode="before")
    @classmethod
    def _safe_failure(cls, failure: object) -> WebFailureData:
        return failure_from_data(failure).to_failure()


WebExtractResult = Annotated[
    WebExtractSuccess | WebExtractFailure, Field(discriminator="status")
]
