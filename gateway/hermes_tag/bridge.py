"""Compatibility bridge from existing gateway session identities."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from .errors import IncompleteScope
from .model import ExternalIdentity, SurfaceRef


def _value(source: object, *names: str) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            value = source[name]
        else:
            value = getattr(source, name, None)
        if value is not None and value != "":
            if isinstance(value, Enum):
                return value.value
            return value
    return None


def _text(source: object, names: tuple[str, ...], label: str) -> str:
    value = _value(source, *names)
    if value is None:
        raise IncompleteScope(f"gateway source is missing {label}")
    result = str(value).strip()
    if not result:
        raise IncompleteScope(f"gateway source is missing {label}")
    return result


def surface_from_session_source(
    source: object,
    *,
    profile: str | None = None,
    platform: str | None = None,
) -> SurfaceRef:
    """Normalize a current SessionSource-like object into a stable surface.

    Tenant/workspace identity is mandatory. It is never inferred from a chat id
    because identical channel ids on another transport/account are not the same
    authority boundary.
    """
    resolved_platform = platform or _text(
        source, ("platform", "source", "kind"), "platform"
    )
    resolved_profile = profile or _text(source, ("profile",), "profile")
    scope_id = _text(
        source,
        (
            "scope_id",
            "workspace_id",
            "team_id",
            "guild_id",
            "server_id",
            "corp_id",
            "tenant_id",
            "account_id",
        ),
        "tenant/workspace scope",
    )
    chat_id = _text(
        source,
        ("chat_id", "channel_id", "conversation_id", "room_id", "target_id"),
        "chat/channel id",
    )
    thread = _value(
        source,
        "thread_id",
        "thread_ts",
        "topic_id",
        "message_thread_id",
    )
    return SurfaceRef(
        platform=str(resolved_platform),
        profile=str(resolved_profile),
        scope_id=scope_id,
        chat_id=chat_id,
        thread_id=str(thread) if thread is not None else None,
    )


def identity_from_session_source(
    source: object,
    *,
    surface: SurfaceRef | None = None,
    external_id: str | None = None,
    display_name: str | None = None,
) -> ExternalIdentity:
    """Normalize the authenticated transport actor, never a model-supplied id."""
    resolved_surface = surface or surface_from_session_source(source)
    actor_id = external_id or _text(
        source,
        ("user_id", "actor_id", "sender_id", "author_id", "from_id"),
        "authenticated actor id",
    )
    actor_name = display_name
    if actor_name is None:
        value = _value(source, "display_name", "user_name", "sender_name", "author_name")
        actor_name = str(value) if value is not None else None
    return ExternalIdentity(
        platform=resolved_surface.platform,
        profile=resolved_surface.profile,
        scope_id=resolved_surface.scope_id,
        external_id=actor_id,
        display_name=actor_name,
    )
