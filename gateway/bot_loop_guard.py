"""Sliding-window budget for bot-authored inbound messages.

``{PLATFORM}_ALLOW_BOTS`` only decides admission, so two Hermes profiles replying to each other never stop.
The guard counts admitted bot messages per conversation and drops further ones for ``cooldown_seconds``
once ``max_events`` land inside ``window_seconds``. Settings: config.yaml ``gateway.bot_loop_guard``.

The chain rules (``chain_admit``) end an exchange long before that budget: a human message starts a
new chain; each bot message after it is one hop and ``max_hops`` caps them; a bot repeating the same
text inside one chain is one hop, not a new ask; and with ``bot_replies_expect_reply`` false (the
default) a bot answering the reply this bot delivered to it is not answered again (#137027).
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


def _callable_takes_arg(fn: Callable) -> bool:
    """Whether ``fn`` accepts a positional ``tier`` argument, so ``BotLoopGuard`` can call a tier-aware
    settings callable as ``fn(tier)`` while still calling a legacy 0-arg one (e.g. a test's plain
    ``lambda: settings`` or ``functools.partial``) as ``fn()`` -- no caller needs to change."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return True
    return False


def _merge_tier_block(base: BotLoopGuardSettings, tier_block) -> BotLoopGuardSettings:
    """Override only the fields ``tier_block`` sets; everything else keeps ``base`` (the flat/legacy
    value). ``tier_block`` missing/empty returns ``base`` unchanged -- this is what keeps a config.yaml
    with no ``claude_mode``/``local_mode`` sub-block behaving exactly as before (§3 of the design doc)."""
    if not isinstance(tier_block, dict) or not tier_block:
        return base
    return BotLoopGuardSettings(
        enabled=_as_bool(tier_block.get("enabled"), base.enabled),
        max_events=_as_positive_int(tier_block.get("max_events"), base.max_events),
        window_seconds=_as_positive(tier_block.get("window_seconds"), base.window_seconds),
        cooldown_seconds=_as_positive(tier_block.get("cooldown_seconds"), base.cooldown_seconds),
        max_hops=_as_positive_int(tier_block.get("max_hops"), base.max_hops),
        bot_replies_expect_reply=_as_bool(tier_block.get("bot_replies_expect_reply"), base.bot_replies_expect_reply),
    )


def settings_from_config(cfg, tier: str = "default") -> BotLoopGuardSettings:
    """Read ``gateway.bot_loop_guard`` from a loaded config dict. Unusable values keep the default.

    ``tier`` is ``"default"`` (ignore ``claude_mode``/``local_mode`` -- legacy/flat behaviour,
    unconditionally 100% backward compatible), ``"claude"`` or ``"local"``: for the latter two, the
    matching ``claude_mode``/``local_mode`` sub-block overrides the flat keys above, field by field
    (bot_loop_guard_split_design.md §3/§4.1). A sub-block that is absent or empty changes nothing, so a
    config.yaml with no sub-blocks behaves identically for every tier.
    """
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    if not isinstance(block, dict):
        return BotLoopGuardSettings()
    defaults = BotLoopGuardSettings()
    base = BotLoopGuardSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        max_events=_as_positive_int(block.get("max_events"), defaults.max_events),
        window_seconds=_as_positive(block.get("window_seconds"), defaults.window_seconds),
        cooldown_seconds=_as_positive(block.get("cooldown_seconds"), defaults.cooldown_seconds),
        max_hops=_as_positive_int(block.get("max_hops"), defaults.max_hops),
        bot_replies_expect_reply=_as_bool(block.get("bot_replies_expect_reply"), defaults.bot_replies_expect_reply),
    )
    if tier not in ("claude", "local"):
        return base
    return _merge_tier_block(base, block.get(f"{tier}_mode"))


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


def load_settings(tier: str = "default") -> BotLoopGuardSettings:
    """Settings from the live config.yaml for ``tier`` (see ``settings_from_config``). Defaults when
    the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly(), tier)
    except Exception:
        return BotLoopGuardSettings()


class BotLoopGuard:
    """Per-conversation sliding window with a cooldown once the budget trips. Thread-safe.
    Settings are re-read on every call so a config.yaml edit takes effect without a restart."""

    def __init__(
        self,
        settings: Callable[..., BotLoopGuardSettings] = load_settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._settings_takes_tier = _callable_takes_arg(settings)
        self._clock = clock
        self._lock = threading.Lock()
        self._events: dict[Hashable, deque[float]] = {}
        self._cooldown_until: dict[Hashable, float] = {}
        self._last_sweep = 0.0
        self._chains: dict[Hashable, _Chain] = {}

    def _resolved_settings(self, tier: str = "default") -> BotLoopGuardSettings:
        """``self._settings(tier)`` for a tier-aware settings callable; ``self._settings()`` for a
        legacy 0-arg one (checked once, at construction) -- so a caller passing a plain ``lambda:
        settings`` (every pre-tier test and caller) keeps working unchanged."""
        return self._settings(tier) if self._settings_takes_tier else self._settings()

    @staticmethod
    def _bucket_key(conversation: Hashable, tier: str) -> Hashable:
        """``tier="default"`` keeps the legacy bucket key (just ``conversation``) so the sliding-window
        state is unchanged for every caller that does not pass a tier. An explicit ``"claude"``/``"local"``
        tier gets its own bucket, so a claude-tier burst cannot spend a local-tier peer's budget or vice
        versa (bot_loop_guard_split_design.md §4.1)."""
        return conversation if tier == "default" else (conversation, tier)

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def tracking_chains(self) -> bool:
        """Whether any bot exchange is open, so a human message has a chain to reset."""
        return bool(self._chains)

    def blocked(self, conversation: Hashable, tier: str = "default") -> bool:
        """True while ``conversation`` (at ``tier``) is cooling down. Reads only, so callers may ask as
        often as they like."""
        if not self._resolved_settings(tier).enabled:
            return False
        with self._lock:
            return self._cooldown_until.get(self._bucket_key(conversation, tier), 0.0) > self._clock()

    def admit(self, conversation: Hashable, tier: str = "default") -> tuple[bool, str]:
        """Count one admitted bot-authored message for ``conversation`` at ``tier`` (``"default"``,
        ``"claude"`` or ``"local"`` -- each tier has its own budget once explicitly tiered, see
        ``_bucket_key``). Returns ``(allowed, state)``; state is ``disabled``, ``ok``, ``tripped`` (this
        message started the cooldown) or ``cooldown``."""
        settings = self._resolved_settings(tier)
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        key = self._bucket_key(conversation, tier)
        with self._lock:
            self._sweep(now, settings)
            if self._cooldown_until.get(key, 0.0) > now:
                return False, "cooldown"
            events = self._events.setdefault(key, deque())
            cutoff = now - settings.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= settings.max_events:
                self._cooldown_until[key] = now + settings.cooldown_seconds
                events.clear()
                return False, "tripped"
            events.append(now)
            return True, "ok"

    def chain_admit(
        self, conversation: Hashable, sender: str, text: str, *, is_bot: bool, tier: str = "default"
    ) -> tuple[bool, str]:
        """Apply the chain rules to one admitted inbound message. ``tier`` is this message's sender's
        tier; sticky-strict (bot_loop_guard_split_design.md §4.2/§5): once any message in this chain is
        ``"claude"``, ``_Chain.ever_claude`` latches True and every later message in the same chain is
        judged under claude-tier settings even if its own sender is ``"local"`` -- a local-tier bot
        cannot relax the limits a claude-tier one in the same exchange is held to. ``tier="default"``
        (the legacy default) never sets ``ever_claude``, so a caller that never passes a tier sees the
        chain behave exactly as before. Returns ``(allowed, state)``; state is ``disabled``, ``human``
        (chain reset), ``ok``, ``max_hops``, ``duplicate`` or ``reply_to_reply``."""
        settings = self._resolved_settings(tier)
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
            if tier == "claude":
                chain.ever_claude = True
            effective_tier = "claude" if chain.ever_claude else tier
            settings = self._resolved_settings(effective_tier)
            chain.updated = now
            ask = (sender, hashlib.sha256(" ".join(str(text or "").split()).encode()).hexdigest())
            if ask in chain.asks:
                return False, "duplicate"
            if not settings.bot_replies_expect_reply and chain.replied_to == sender:
                return False, "reply_to_reply"
            if chain.hops >= settings.max_hops:
                return False, "max_hops"
            chain.hops += 1
            chain.asks.add(ask)
            return True, "ok"

    def note_reply(self, conversation: Hashable, recipient: str) -> None:
        """Record that this bot's reply to bot ``recipient`` went out. Only a delivered reply makes the
        recipient's next message an answer to an answer: a message still waiting for its turn, or one
        the turn chose to leave unanswered, does not."""
        if not self._resolved_settings().enabled:
            return
        with self._lock:
            chain = self._chains.get(conversation)
            if chain is not None:
                chain.replied_to = recipient

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
    __slots__ = ("asks", "ever_claude", "hops", "replied_to", "updated")

    def __init__(self) -> None:
        self.hops = 0
        self.asks: set[tuple[str, str]] = set()
        self.replied_to: str = ""
        self.updated = 0.0
        self.ever_claude: bool = False
