"""Confirmation gate for local-secretary write / publish / destructive actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ActionCategory(str, Enum):
    READ = "read"
    LOCAL_GENERATION = "local_generation"
    WRITE = "write"
    EXTERNAL_PUBLISH = "external_publish"
    DESTRUCTIVE = "destructive"
    SHELL = "shell"


class WriteActionError(PermissionError):
    """Raised when a gated action runs without explicit user confirmation."""


_CONFIRMATION_REQUIRED = {
    ActionCategory.WRITE,
    ActionCategory.EXTERNAL_PUBLISH,
    ActionCategory.DESTRUCTIVE,
    ActionCategory.SHELL,
}

_ACTION_ALIASES: dict[str, ActionCategory] = {
    "gmail_search": ActionCategory.READ,
    "gmail_read": ActionCategory.READ,
    "calendar_list": ActionCategory.READ,
    "calendar_read": ActionCategory.READ,
    "web_search": ActionCategory.READ,
    "news_collect": ActionCategory.READ,
    "emergency_news_collect": ActionCategory.READ,
    "tts_generate": ActionCategory.LOCAL_GENERATION,
    "irodori_tts": ActionCategory.LOCAL_GENERATION,
    "x_draft_post": ActionCategory.LOCAL_GENERATION,
    "gmail_send": ActionCategory.WRITE,
    "gmail_draft_send": ActionCategory.WRITE,
    "calendar_create": ActionCategory.WRITE,
    "calendar_update": ActionCategory.WRITE,
    "calendar_delete": ActionCategory.DESTRUCTIVE,
    "x_publish_post": ActionCategory.EXTERNAL_PUBLISH,
    "shell_exec": ActionCategory.SHELL,
}


def classify_action(action_type: str | ActionCategory) -> ActionCategory:
    if isinstance(action_type, ActionCategory):
        return action_type
    key = (action_type or "").strip().lower()
    if key in _ACTION_ALIASES:
        return _ACTION_ALIASES[key]
    if key in {c.value for c in ActionCategory}:
        return ActionCategory(key)
    return ActionCategory.WRITE


def confirmation_required(category: ActionCategory) -> bool:
    return category in _CONFIRMATION_REQUIRED


@dataclass(frozen=True)
class GateResult:
    ok: bool
    action: str
    category: ActionCategory
    confirmation_required: bool
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.ok,
            "action": self.action,
            "category": self.category.value,
            "confirmation_required": self.confirmation_required,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def require_write_confirmation(
    action_type: str | ActionCategory,
    *,
    confirmed: bool = False,
    detail: str | None = None,
) -> None:
    category = classify_action(action_type)
    if not confirmation_required(category):
        return
    if confirmed:
        return
    msg = f"User confirmation required for {category.value} action"
    if detail:
        msg = f"{msg}: {detail}"
    raise WriteActionError(msg)


def check_write_action(
    action_type: str | ActionCategory,
    *,
    confirmed: bool = False,
    detail: str | None = None,
) -> GateResult:
    action_name = action_type.value if isinstance(action_type, ActionCategory) else str(action_type)
    category = classify_action(action_type)
    needs_confirmation = confirmation_required(category)
    if needs_confirmation and not confirmed:
        error = f"User confirmation required for {category.value} action"
        if detail:
            error = f"{error}: {detail}"
        return GateResult(
            ok=False,
            action=action_name,
            category=category,
            confirmation_required=True,
            error=error,
        )
    return GateResult(
        ok=True,
        action=action_name,
        category=category,
        confirmation_required=needs_confirmation,
    )


def write_action_gate(action_type: str | ActionCategory) -> Callable[[F], F]:
    """Decorator: block write/publish/destructive/shell unless confirmed=True."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            confirmed = bool(kwargs.pop("confirmed", False))
            require_write_confirmation(action_type, confirmed=confirmed)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
