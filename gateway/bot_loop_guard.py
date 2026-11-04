"""Sliding-window budget for bot-authored inbound messages.

``{PLATFORM}_ALLOW_BOTS`` only decides admission, so two Hermes profiles replying to each other never stop.
The guard counts admitted bot messages per conversation and drops further ones for ``cooldown_seconds``
once ``max_events`` land inside ``window_seconds``. Settings: config.yaml ``gateway.bot_loop_guard``.

The chain rules (``chain_admit``) end an exchange long before that budget: a human message starts a
new chain; each bot message after it is one hop and ``max_hops`` caps them; a bot repeating the same
text inside one chain is one hop, not a new ask; and with ``bot_replies_expect_reply`` false (the
default) a bot answering the reply this bot just gave it is not answered again (#137027).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Hashable, Tuple

__all__ = ["BotLoopGuard", "BotLoopGuardSettings", "configured_bot_user", "load_settings", "settings_from_config"]

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class BotLoopGuardSettings:
    enabled: bool = True
    max_events: int = 20
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    max_hops: int = 4
    bot_replies_expect_reply: bool = False


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
        max_hops=_as_positive_int(block.get("max_hops"), defaults.max_hops),
        bot_replies_expect_reply=_as_bool(block.get("bot_replies_expect_reply"), defaults.bot_replies_expect_reply),
    )


def configured_bot_user(platform_config, user_id) -> bool:
    """Whether ``user_id`` is in the adapter's ``bot_users`` list (config.yaml platform ``extra``): the
    opt-in that marks a peer bot on a platform whose events carry no bot flag (Mattermost, Matrix,
    Signal, IRC, ...), so this guard and the bot-peer delivery rules engage for it (#137027)."""
    if user_id is None or user_id == "":
        return False
    configured = (getattr(platform_config, "extra", None) or {}).get("bot_users")
    if isinstance(configured, str):
        configured = configured.split(",")
    return isinstance(configured, (list, tuple, set)) and str(user_id) in {str(v).strip() for v in configured}


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
        self._chains: dict[Hashable, _Chain] = {}

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def tracking_chains(self) -> bool:
        """Whether any bot exchange is open, so a human message has a chain to reset."""
        return bool(self._chains)

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

    def chain_admit(self, conversation: Hashable, sender: str, text: str, *, is_bot: bool) -> tuple[bool, str]:
        """Apply the chain rules to one admitted inbound message. Returns ``(allowed, state)``; state is
        ``disabled``, ``human`` (chain reset), ``ok``, ``max_hops``, ``duplicate`` or ``reply_to_reply``."""
        settings = self._settings()
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        with self._lock:
            self._sweep(now, settings)
            if not is_bot:
                self._chains.pop(conversation, None)
                return True, "human"
            chain = self._chains.get(conversation)
            if chain is None or now - chain.updated > settings.cooldown_seconds:
                chain = self._chains[conversation] = _Chain()
            chain.updated = now
            ask = (sender, hashlib.sha256(" ".join(str(text or "").split()).encode()).hexdigest())
            if ask in chain.asks:
                return False, "duplicate"
            if not settings.bot_replies_expect_reply and chain.last_trigger == sender:
                return False, "reply_to_reply"
            if chain.hops >= settings.max_hops:
                return False, "max_hops"
            chain.hops += 1
            chain.asks.add(ask)
            chain.last_trigger = sender
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
        for key in [k for k, chain in self._chains.items() if now - chain.updated > settings.cooldown_seconds]:
            del self._chains[key]


class _Chain:
    """One bot-to-bot exchange since the last human message in a conversation."""
    __slots__ = ("asks", "hops", "last_trigger", "updated")

    def __init__(self) -> None:
        self.hops = 0
        self.asks: set[tuple[str, str]] = set()
        self.last_trigger: str = ""
        self.updated = 0.0
