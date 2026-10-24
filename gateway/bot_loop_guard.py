"""Sliding-window budget for bot-authored inbound messages.

``{PLATFORM}_ALLOW_BOTS`` only decides admission, so two Hermes profiles replying to each other never stop.
The guard counts admitted bot messages per conversation and drops further ones for ``cooldown_seconds``
once ``max_events`` land inside ``window_seconds``. Settings: config.yaml ``gateway.bot_loop_guard``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Hashable, Mapping, Tuple

__all__ = ["BotLoopGuard", "BotLoopGuardSettings", "load_settings", "settings_from_config"]

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class BotLoopGuardSettings:
    enabled: bool = True
    max_events: int = 20
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    # Platform -> user ids that must be COUNTED AS BOTS even though the platform does not mark them
    # (``source.is_bot`` stays False). See :func:`_normalize_bot_senders`.
    bot_senders: Mapping[str, frozenset] = field(default_factory=dict)


def _as_bool(raw, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower() if raw is not None else ""
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _as_positive(raw, default: float) -> float:
    if isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _as_positive_int(raw, default: int) -> int:
    value = _as_positive(raw, 0.0)
    return int(value) if value >= 1 and value == int(value) else default


def _normalize_bot_senders(raw) -> Dict[str, frozenset]:
    """Parse ``gateway.bot_loop_guard.bot_senders`` into ``{platform: frozenset(user ids)}``.

    Both shapes are accepted and equivalent (the mapping reads better in YAML)::

        bot_senders:
          matrix: ["@heimdall:nanopi.example.org"]
        bot_senders: ["matrix:@heimdall:nanopi.example.org"]

    WHY this exists: ``{PLATFORM}_ALLOW_BOTS`` gates admission, and the guard's ``is_bot`` test only
    works where the platform marks bot authors. Matrix has no such flag, so a peer bot reaches the
    gateway as an ordinary AUTHORIZED user -- ``source.is_bot`` is False, the guard never counts it,
    and two Hermes instances in one room answer each other without a bound (observed live: room
    traffic ping-ponging on model-fallback and interrupt notices). Naming the peer here makes the
    guard treat its messages as bot-authored: same window budget, same cooldown.

    Keys and ids are matched case-insensitively, and an id is split from its platform on the FIRST
    colon only (Matrix ids contain colons: ``@user:server``).
    """
    grouped: Dict[str, set] = {}

    def _add(platform, user_id) -> None:
        platform = str(platform or "").strip().lower()
        user_id = str(user_id or "").strip().lower()
        if platform and user_id:
            grouped.setdefault(platform, set()).add(user_id)

    if isinstance(raw, Mapping):
        for platform, ids in raw.items():
            if isinstance(ids, str):
                # ``hermes config set`` writes a list value as its literal string, so the same
                # normalization the other allowlists use is applied here.
                from gateway.platforms._shared import decode_json_list_literal as _decode

                decoded = _decode(ids)
                ids = decoded if isinstance(decoded, list) else ids.split(",")
            if not isinstance(ids, (list, tuple, set, frozenset)):
                continue
            for user_id in ids:
                _add(platform, user_id)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        for entry in raw:
            # Strings only: str() on a dict/list entry yields a bogus "platform:id" pair rather
            # than being ignored as unusable.
            if not isinstance(entry, str):
                continue
            platform, sep, user_id = entry.partition(":")
            if sep:
                _add(platform, user_id)
    return {platform: frozenset(ids) for platform, ids in grouped.items()}


def settings_from_config(cfg) -> BotLoopGuardSettings:
    """Read ``gateway.bot_loop_guard`` from a loaded config dict. Unusable values keep the default."""
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    if not isinstance(block, dict):
        return BotLoopGuardSettings()
    defaults = BotLoopGuardSettings()
    return BotLoopGuardSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        max_events=_as_positive_int(block.get("max_events"), defaults.max_events),
        window_seconds=_as_positive(block.get("window_seconds"), defaults.window_seconds),
        cooldown_seconds=_as_positive(block.get("cooldown_seconds"), defaults.cooldown_seconds),
        bot_senders=_normalize_bot_senders(block.get("bot_senders")),
    )


def load_settings() -> BotLoopGuardSettings:
    """Settings from the live config.yaml. Defaults when the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly())
    except Exception:
        return BotLoopGuardSettings()


class BotLoopGuard:
    """Per-conversation sliding window with a cooldown once the budget trips. Thread-safe.
    Settings are re-read on every call so a config.yaml edit takes effect without a restart."""

    def __init__(
        self,
        settings: Callable[[], BotLoopGuardSettings] = load_settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._events: dict[Hashable, deque[float]] = {}
        self._cooldown_until: dict[Hashable, float] = {}
        self._last_sweep = 0.0

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    def is_declared_bot(self, platform: str, user_id: str) -> bool:
        """True when ``user_id`` is named as a bot author for ``platform`` in the live settings.

        Read through the same settings callable as the budget, so a config.yaml edit takes effect on
        the next message without a restart.
        """
        declared = getattr(self._settings(), "bot_senders", None) or {}
        ids = declared.get(str(platform or "").strip().lower())
        if not ids:
            return False
        return str(user_id or "").strip().lower() in ids

    def blocked(self, conversation: Hashable) -> bool:
        """True while ``conversation`` is cooling down. Reads only, so callers may ask as often as they like."""
        if not self._settings().enabled:
            return False
        with self._lock:
            return self._cooldown_until.get(conversation, 0.0) > self._clock()

    def admit(self, conversation: Hashable) -> tuple[bool, str]:
        """Count one admitted bot-authored message for ``conversation``.
        Returns ``(allowed, state)``; state is ``disabled``, ``ok``, ``tripped`` (this message started the cooldown) or ``cooldown``."""
        settings = self._settings()
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        with self._lock:
            self._sweep(now, settings)
            if self._cooldown_until.get(conversation, 0.0) > now:
                return False, "cooldown"
            events = self._events.setdefault(conversation, deque())
            cutoff = now - settings.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= settings.max_events:
                self._cooldown_until[conversation] = now + settings.cooldown_seconds
                events.clear()
                return False, "tripped"
            events.append(now)
            return True, "ok"

    def _sweep(self, now: float, settings: BotLoopGuardSettings) -> None:
        """Drop idle conversations at most once per window so memory stays bounded."""
        if now - self._last_sweep < settings.window_seconds:
            return
        self._last_sweep = now
        idle_cutoff = now - settings.window_seconds
        for key in [k for k, dq in self._events.items() if not dq or dq[-1] <= idle_cutoff]:
            del self._events[key]
        for key in [k for k, until in self._cooldown_until.items() if until <= now]:
            del self._cooldown_until[key]
