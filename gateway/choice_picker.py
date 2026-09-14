"""Optional page results for the existing native finite-choice picker.

The callback stays ``async (chat_id, value)``. A string finishes the picker;
an opted-in callback can return ChoicePage to replace its current page.
Values are private callback data, never native action IDs or display labels.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Awaitable, Callable, Mapping, Sequence, TypeAlias

MAX_PAGE_CHOICES = 12
PAGE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ChoicePage:
    title: str
    choices: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.title, str)
            or not self.title.strip()
            or len(self.title) > 2048
        ):
            raise ValueError("Choice page title must contain 1-2048 characters")
        if not 1 <= len(self.choices) <= MAX_PAGE_CHOICES:
            raise ValueError("Choice pages require 1-12 choices, including navigation")
        frozen = []
        for choice in self.choices:
            value = choice.get("value")
            label = choice.get("label") or value
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError("Choice value must contain 1-1024 characters")
            if not isinstance(label, str) or not label or len(label) > 4096:
                raise ValueError("Choice label must contain 1-4096 characters")
            frozen.append(
                MappingProxyType({
                    "value": value,
                    "label": label,
                    "is_current": choice.get("is_current") is True,
                    "full_width": choice.get("full_width") is True,
                })
            )
        object.__setattr__(self, "choices", tuple(frozen))


@dataclass(frozen=True)
class ChoiceProgress:
    """Render feedback before slow work, keeping this selection claimed.

    complete is invoked once only after feedback is displayed. It returns a
    final string or another page; delivery retries/idempotency stay consumer-owned.
    """

    text: str
    complete: Callable[[], Awaitable[str | ChoicePage]]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > 2048
        ):
            raise ValueError("Choice progress must contain 1-2048 characters")
        if not callable(self.complete):
            raise ValueError("Choice progress requires a completion callback")


ChoiceResult: TypeAlias = str | ChoicePage | ChoiceProgress
ChoiceCallback: TypeAlias = Callable[[str, str], Awaitable[ChoiceResult]]


def choice_action(token: str, revision: int, index: int) -> str:
    return f"cp:{token}:{revision}:{index}"


def choice_index(action: str, token: str, revision: int, count: int) -> int | None:
    prefix = f"cp:{token}:{revision}:"
    if not isinstance(action, str) or not action.startswith(prefix):
        return None
    suffix = action[len(prefix) :]
    if not suffix.isascii() or not suffix.isdecimal() or len(suffix) > 2:
        return None
    index = int(suffix)
    return index if 0 <= index < count else None


def choice_label(choice: Mapping[str, object], limit: int = 100) -> str:
    label = " ".join(str(choice.get("label") or choice.get("value") or "").split())
    return label if len(label) <= limit else label[: limit - 1] + "\u2026"
