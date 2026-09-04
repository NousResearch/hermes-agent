"""Sliding-window budget for bot-authored inbound messages (#91481).

``{PLATFORM}_ALLOW_BOTS`` only decides admission. When two Hermes profiles reply to each
other, every reply satisfies the ``mentions`` test again, so nothing ends the exchange.
The guard counts admitted bot-authored messages per conversation and, once a conversation
exceeds ``max_events`` inside ``window_seconds``, drops further bot messages there for
``cooldown_seconds``. Human traffic never enters the guard.

Settings live in config.yaml under ``gateway.bot_loop_guard``:
``enabled`` (true), ``max_events`` (20), ``window_seconds`` (300), ``cooldown_seconds`` (600).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Hashable, Tuple

__all__ = ["BotLoopGuard", "BotLoopGuardSettings", "load_settings", "settings_from_config"]

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class BotLoopGuardSettings:
    enabled: bool = True
    max_events: int = 20
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0


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


def settings_from_config(cfg) -> BotLoopGuardSettings:
    """Read ``gateway.bot_loop_guard`` from a loaded config dict; unusable values keep the default."""
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    if not isinstance(block, dict):
        return BotLoopGuardSettings()
    defaults = BotLoopGuardSettings()
    return BotLoopGuardSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        max_events=int(_as_positive(block.get("max_events"), defaults.max_events)),
        window_seconds=_as_positive(block.get("window_seconds"), defaults.window_seconds),
        cooldown_seconds=_as_positive(block.get("cooldown_seconds"), defaults.cooldown_seconds),
    )


def load_settings() -> BotLoopGuardSettings:
    """Settings from the live config.yaml; defaults when the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly())
    except Exception:
        return BotLoopGuardSettings()


class BotLoopGuard:
    """Per-conversation sliding window with a cooldown once the budget trips. Thread-safe.

    Settings are re-read on every call so a config.yaml edit takes effect without a restart;
    ``load_config_readonly`` caches on the file signature, so the read is cheap.
    """

    def __init__(
        self,
        settings: Callable[[], BotLoopGuardSettings] = load_settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._events: Dict[Hashable, Deque[float]] = {}
        self._cooldown_until: Dict[Hashable, float] = {}
        self._last_sweep = 0.0

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    def admit(self, conversation: Hashable) -> Tuple[bool, str]:
        """Count one admitted bot-authored message for ``conversation``.

        Returns ``(allowed, state)`` with state one of ``disabled``, ``ok``, ``tripped``
        (this message exceeded the budget and started the cooldown) or ``cooldown``.
        """
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
