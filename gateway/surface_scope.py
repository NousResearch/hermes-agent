"""Immutable, default-deny scope for plugin-owned Telegram task surfaces."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_CHAT_RE = re.compile(r"-?[1-9][0-9]{0,19}")
_THREAD_RE = re.compile(r"[1-9][0-9]{0,19}")
_BOARD_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_TASK_RE = re.compile(r"t_[a-f0-9]{8,64}")


@dataclass(frozen=True)
class TelegramRoute:
    profile: str
    platform: str
    chat_id: str
    thread_id: str | None


@dataclass(frozen=True)
class TaskResource:
    board: str
    task_id: str


@dataclass(frozen=True)
class TelegramSurfaceScope:
    """Exact routes and task resources copied out of plugin configuration."""

    routes: frozenset[TelegramRoute]
    task_resources: frozenset[TaskResource]

    def allows_route(self, profile: Any, platform: Any, chat_id: Any, thread_id: Any) -> bool:
        platform_value = getattr(platform, "value", platform)
        route = TelegramRoute(
            profile=str(profile or ""),
            platform=str(platform_value or "").lower(),
            chat_id=str(chat_id or ""),
            thread_id=None if thread_id is None or thread_id == "" else str(thread_id),
        )
        return route in self.routes

    def allows_task(self, profile: Any, board: Any, task_id: Any) -> bool:
        profile_value = str(profile or "")
        return (any(route.profile == profile_value for route in self.routes)
                and TaskResource(str(board or ""), str(task_id or "")) in self.task_resources)

    def allows_card(
        self, profile: Any, platform: Any, chat_id: Any, thread_id: Any,
        board: Any, task_id: Any,
    ) -> bool:
        return (self.allows_route(profile, platform, chat_id, thread_id)
                and self.allows_task(profile, board, task_id))


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)!r}")
    return value


def _bounded_list(value: Any, label: str) -> list:
    if type(value) is not list or not value or len(value) > 4096:
        raise ValueError(f"{label} must be a non-empty list of at most 4096 entries")
    return value


def _parse_route(value: Any) -> TelegramRoute:
    route = _exact_mapping(value, {"profile", "platform", "chat_id", "thread_id"}, "scope route")
    profile = route["profile"]
    platform = route["platform"]
    chat_id = route["chat_id"]
    thread_id = route["thread_id"]
    if type(profile) is not str or _PROFILE_RE.fullmatch(profile) is None:
        raise ValueError("scope route profile must be an exact profile id")
    if platform != "telegram":
        raise ValueError("scope route platform must be 'telegram'")
    if type(chat_id) is not str or _CHAT_RE.fullmatch(chat_id) is None:
        raise ValueError("scope route chat_id must be an exact Telegram numeric id string")
    if thread_id is not None and (
        type(thread_id) is not str or _THREAD_RE.fullmatch(thread_id) is None
    ):
        raise ValueError("scope route thread_id must be null or an exact positive numeric id string")
    return TelegramRoute(profile, platform, chat_id, thread_id)


def _parse_resource(value: Any) -> TaskResource:
    resource = _exact_mapping(value, {"board", "task_id"}, "scope task resource")
    board, task_id = resource["board"], resource["task_id"]
    if type(board) is not str or _BOARD_RE.fullmatch(board) is None:
        raise ValueError("scope task resource board must be an exact board slug")
    if type(task_id) is not str or _TASK_RE.fullmatch(task_id) is None:
        raise ValueError("scope task resource task_id must be an exact canonical task id")
    return TaskResource(board, task_id)


def parse_surface_scope(value: Any, *, require_tasks: bool) -> TelegramSurfaceScope:
    """Validate the complete scope before a plugin surface is registered.

    The same immutable value may be shared by independent todo, card, decision and
    detail grants. Possessing a scope never grants one of those capabilities.
    """
    if type(value) is not dict or set(value) not in ({"routes"}, {"routes", "task_resources"}):
        raise ValueError("surface scope must contain routes and optional task_resources")
    scope = value
    routes = tuple(_parse_route(item) for item in _bounded_list(scope["routes"], "scope routes"))
    raw_resources = scope.get("task_resources", [])
    if type(raw_resources) is not list or len(raw_resources) > 4096:
        raise ValueError("scope task_resources must be a list of at most 4096 entries")
    resources = tuple(_parse_resource(item) for item in raw_resources)
    if require_tasks and not resources:
        raise ValueError("this surface requires at least one exact task resource")
    if len(set(routes)) != len(routes) or len(set(resources)) != len(resources):
        raise ValueError("surface scope entries must be unique")
    return TelegramSurfaceScope(frozenset(routes), frozenset(resources))
