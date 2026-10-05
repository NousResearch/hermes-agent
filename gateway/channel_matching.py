"""ID-first channel configuration matching, shared by overrides and profile routes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

logger = logging.getLogger(__name__)
_PATTERN_METACHARS = frozenset("^$+?[](){}|\\")


@dataclass(frozen=True)
class ChannelNames:
    chat: tuple[str, ...] = ()
    thread: tuple[str, ...] = ()
    parent: tuple[str, ...] = ()
    guild: tuple[str, ...] = ()

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.chat + self.thread + self.parent))


NameResolver = Callable[[], ChannelNames]


@lru_cache(maxsize=1024)
def compile_name_pattern(value: str) -> re.Pattern | None:
    # Deliberately follows #109676: '.' and '*' alone do not opt a literal into regex.
    if value.lstrip("+-").isdigit() or not any(char in _PATTERN_METACHARS for char in value):
        return None
    try:
        return re.compile(value)
    except re.error as exc:
        logger.warning("Channel name pattern %r is not a valid regex; regex matching disabled: %s", value, exc)
        return None


def prepare_name_patterns(values: Iterable[str]) -> None:
    for value in values:
        if value:
            compile_name_pattern(value)


def name_matches(value: str, names: Iterable[str], level: int) -> bool:
    names = tuple(names)
    if level >= 1 and value in names:
        return True
    pattern = compile_name_pattern(value) if level >= 2 and names else None
    return pattern is not None and any(pattern.match(name) for name in names)


def lazy_names(resolver: NameResolver | None) -> NameResolver:
    @lru_cache(maxsize=1)
    def resolve() -> ChannelNames:
        return resolver() if resolver is not None else ChannelNames()
    return resolve


def channel_override_lookup_keys(chat_id, *, thread_id=None, parent_id=None) -> list[str]:
    return list(dict.fromkeys(str(key) for key in (chat_id, thread_id, parent_id) if key))


def get_channel_override(config, platform, chat_id: str, *, thread_id=None, parent_id=None,
                         name_resolver: NameResolver | None = None):
    platforms = getattr(config, "platforms", None) or {}
    platform_config = platforms.get(platform)
    overrides = platform_config.channel_overrides if platform_config else {}
    if not overrides:
        return None
    for key in channel_override_lookup_keys(chat_id, thread_id=thread_id, parent_id=parent_id):
        if (override := overrides.get(key)) is not None:
            return override
    names = name_resolver().channel_names if name_resolver else ()
    for name in names:
        if (override := overrides.get(name)) is not None:
            return override
    for key, override in overrides.items():
        if name_matches(key, names, 2):
            return override
    return None
