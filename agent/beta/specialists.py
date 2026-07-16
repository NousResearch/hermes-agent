"""Declarative specialist catalog for Beta orchestration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SpecialistManifestError(ValueError):
    """Raised when the specialist catalog is invalid."""


class Specialist(BaseModel):
    """Validated specialist manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    name: str
    description: str
    capabilities: tuple[str, ...] = Field(min_length=1)
    keywords: tuple[str, ...] = Field(min_length=1)
    allowed_toolsets: tuple[str, ...] = ()
    blocked_tools: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    max_risk: Literal["low", "medium", "high"] = "low"
    memory_access: Literal["none", "read", "read_write"] = "none"
    memory_scope: str | None = None
    max_concurrency: int = Field(default=1, ge=1)
    enabled: bool = True

    @field_validator("name", "description")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("capabilities", "keywords", "allowed_toolsets", "blocked_tools")
    @classmethod
    def normalized_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values if value.strip()))
        if not normalized and values:
            raise ValueError("must contain non-empty strings")
        return normalized


class SpecialistRegistry:
    """Lookup-only registry built entirely from specialist manifests."""

    def __init__(self, specialists: Iterable[Specialist]):
        self._by_id: dict[str, Specialist] = {}
        for specialist in specialists:
            if specialist.id in self._by_id:
                raise SpecialistManifestError(f"duplicate specialist id: {specialist.id}")
            self._by_id[specialist.id] = specialist

    def get(self, specialist_id: str) -> Specialist | None:
        return self._by_id.get(specialist_id)

    def enabled(self) -> tuple[Specialist, ...]:
        return tuple(specialist for specialist in self._by_id.values() if specialist.enabled)

    def __iter__(self):
        return iter(self._by_id.values())

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SpecialistRegistry":
        source = Path(path) if path else Path(__file__).with_name("specialists.yaml")
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SpecialistManifestError(f"cannot load specialist catalog {source}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("specialists"), list):
            raise SpecialistManifestError("specialist catalog must contain a 'specialists' list")

        specialists = []
        for index, manifest in enumerate(data["specialists"]):
            try:
                specialists.append(Specialist.model_validate(manifest))
            except ValidationError as exc:
                raise SpecialistManifestError(f"invalid specialist at index {index}: {exc}") from exc
        return cls(specialists)


@lru_cache(maxsize=1)
def default_specialist_registry() -> SpecialistRegistry:
    """Return the packaged specialist registry."""
    return SpecialistRegistry.load()

