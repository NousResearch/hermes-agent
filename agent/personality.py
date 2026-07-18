"""Shared parsing and rendering for configured personality definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


CLEAR_PERSONALITY_ALIASES = frozenset({"none", "default", "neutral"})
_PERSONALITY_FIELDS = ("description", "system_prompt", "tone", "style")


def is_personality_clear_request(requested_name: Any) -> bool:
    """Return whether a requested name means to clear the personality overlay."""
    normalized_name = str(requested_name or "").strip().lower()
    return not normalized_name or normalized_name in CLEAR_PERSONALITY_ALIASES


def _value_type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


class PersonalityConfigError(ValueError):
    """Raised when a selected personality has an invalid definition."""

    def __init__(self, personality_name: str, detail: str):
        self.personality_name = personality_name
        self.detail = detail
        super().__init__(f"Invalid personality '{personality_name}': {detail}")


class PersonalityNotFoundError(LookupError):
    """Raised when a requested personality is absent from the configured catalog."""

    def __init__(
        self,
        requested_name: str,
        normalized_name: str,
        available_names: tuple[str, ...],
    ):
        self.requested_name = requested_name
        self.normalized_name = normalized_name
        self.available_names = available_names
        super().__init__(f"Unknown personality: {requested_name}")


@dataclass(frozen=True)
class PersonalityDefinition:
    """Validated, immutable representation of one configured personality."""

    name: str
    description: str = ""
    system_prompt: str = ""
    tone: str = ""
    style: str = ""

    @classmethod
    def parse(cls, name: str, value: Any) -> "PersonalityDefinition":
        """Parse the legacy string or mapping representation."""
        if isinstance(value, str):
            return cls(name=name, system_prompt=value)
        if not isinstance(value, Mapping):
            raise PersonalityConfigError(
                name,
                "expected a string or mapping, "
                f"got {_value_type_name(value)}",
            )

        parsed: dict[str, str] = {}
        for field_name in _PERSONALITY_FIELDS:
            field_value = value.get(field_name, "")
            if not isinstance(field_value, str):
                raise PersonalityConfigError(
                    name,
                    f"field '{field_name}' must be a string, "
                    f"got {_value_type_name(field_value)}",
                )
            parsed[field_name] = field_value

        return cls(name=name, **parsed)

    def render(self) -> str:
        """Render exactly the text historically appended to the system prompt."""
        parts = [self.system_prompt]
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.style:
            parts.append(f"Style: {self.style}")
        return "\n".join(part for part in parts if part)


def resolve_personality(
    requested_name: str,
    personalities: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve clear aliases or render one selected catalog entry.

    Returns ``("", "")`` for the existing clear aliases and otherwise returns
    the normalized configured name plus its rendered prompt.
    """
    raw_name = str(requested_name or "").strip()
    normalized_name = raw_name.lower()
    if is_personality_clear_request(normalized_name):
        return "", ""

    if not isinstance(personalities, Mapping):
        raise PersonalityConfigError(
            normalized_name,
            "personality catalog must be a mapping, "
            f"got {_value_type_name(personalities)}",
        )
    if normalized_name not in personalities:
        available_names = tuple(str(name) for name in personalities)
        raise PersonalityNotFoundError(
            requested_name=raw_name,
            normalized_name=normalized_name,
            available_names=available_names,
        )

    definition = PersonalityDefinition.parse(
        normalized_name,
        personalities[normalized_name],
    )
    return normalized_name, definition.render()
