"""Platform-agnostic dynamic tool reactions mixin.

Provides the full lifecycle state machine for emoji reactions during message
processing:

    on_processing_start  → add persona emoji
    on_tool_call_start   → swap to tool-specific emoji (with cooldown)
    on_processing_complete → swap to final emoji (persona / ❌)

Platforms opt in by:
1. Inheriting ``DynamicReactionMixin`` (before ``BasePlatformAdapter`` in MRO)
2. Implementing the three primitives:
   - ``_reaction_add(msg_ref, emoji) -> bool``
   - ``_reaction_remove(msg_ref, emoji) -> bool``
   - ``_reaction_msg_key(event) -> Optional[Hashable]``

For replace-all platforms (Telegram), override ``_reaction_replace_mode = True``
and implement ``_reaction_set(msg_ref, emoji) -> bool`` instead of add/remove.

The mixin resolves ``dynamic_reactions``, ``persona_emoji``, and
``reaction_cooldown`` from config once at init.  Platforms that don't call
``_init_reaction_mixin()`` get zero behavior — all hooks short-circuit.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)


class DynamicReactionMixin:
    """Shared dynamic tool-reaction logic for any messaging platform.

    Call ``_init_reaction_mixin()`` at the end of your adapter's ``__init__``
    to activate.  Without that call every hook is a no-op.
    """

    # Subclass overrides ──────────────────────────────────────────────────

    # Set True for platforms where setting a reaction replaces all existing
    # reactions (e.g. Telegram).  When True, the mixin calls
    # ``_reaction_set`` instead of ``_reaction_add`` + ``_reaction_remove``.
    _reaction_replace_mode: bool = False

    # ── Primitives (subclass MUST implement) ─────────────────────────────

    async def _reaction_add(self, msg_ref: Any, emoji: str) -> bool:
        """Add *emoji* to the message identified by *msg_ref*.

        *msg_ref* is whatever ``_reaction_resolve_message`` returns — a raw
        Discord message object, a ``(chat_id, message_id)`` tuple, etc.
        """
        return False

    async def _reaction_remove(self, msg_ref: Any, emoji: str) -> bool:
        """Remove *emoji* from the message identified by *msg_ref*."""
        return False

    async def _reaction_set(self, msg_ref: Any, emoji: str) -> bool:
        """Replace all reactions on the message with *emoji*.

        Only used when ``_reaction_replace_mode`` is True.
        """
        return False

    def _reaction_resolve_message(self, event: Any) -> Any:
        """Extract a platform-native message reference from *event*.

        Return ``None`` if the event doesn't carry enough info to react.
        The returned object is passed verbatim to ``_reaction_add`` /
        ``_reaction_remove`` / ``_reaction_set``.
        """
        return None

    def _reaction_msg_key(self, event: Any) -> Optional[Hashable]:
        """Return a hashable key that uniquely identifies the message.

        Used for tracking active reactions and cooldown timestamps.
        Return ``None`` to skip reaction handling for this event.
        """
        return None

    def _reaction_translate_emoji(self, emoji: str) -> Optional[str]:
        """Translate a Unicode emoji to the platform's native format.

        Return ``None`` if the emoji is not supported on this platform
        (the mixin will fall back to the default tool emoji ``⚙️``).

        Default implementation returns the emoji unchanged (Unicode passthrough).
        """
        return emoji

    # ── Init ─────────────────────────────────────────────────────────────

    def _init_reaction_mixin(self) -> None:
        """Initialize mixin state.  Call from adapter ``__init__``."""
        # Per-message tracking: msg_key → currently displayed emoji
        self._rxn_active: Dict[Hashable, str] = {}
        # Per-message tracking: msg_key → resolved message reference
        self._rxn_msg_refs: Dict[Hashable, Any] = {}
        # Cooldown: msg_key → monotonic timestamp of last swap
        self._rxn_last_swap: Dict[Hashable, float] = {}

        # Resolve config once
        self._rxn_persona_emoji: str = self._rxn_resolve_persona_emoji()
        self._rxn_dynamic: bool = self._rxn_resolve_dynamic_reactions()
        self._rxn_cooldown: float = self._rxn_resolve_cooldown()
        self._rxn_initialized: bool = True

    # ── Config resolution ────────────────────────────────────────────────

    def _rxn_resolve_persona_emoji(self) -> str:
        """Resolve persona emoji from platform config → global config → default."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if emoji := extra.get("persona_emoji"):
            return emoji
        try:
            from hermes_cli.config import load_config
            return load_config().get("persona_emoji") or "👀"
        except Exception:
            return "👀"

    def _rxn_resolve_dynamic_reactions(self) -> bool:
        """Resolve dynamic_reactions flag from platform → global → True."""
        if not self._rxn_reactions_enabled():
            return False
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if "dynamic_reactions" in extra:
            return bool(extra["dynamic_reactions"])
        try:
            from hermes_cli.config import load_config
            return bool(load_config().get("dynamic_reactions", True))
        except Exception:
            return True

    def _rxn_resolve_cooldown(self) -> float:
        """Resolve reaction_cooldown from platform config → default 1.0s."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        return float(extra.get("reaction_cooldown", 1.0))

    def _rxn_reactions_enabled(self) -> bool:
        """Check if reactions are enabled at all.

        Delegates to the adapter's own ``_reactions_enabled()`` if it exists
        as a callable, or reads it as a bool attribute.  Otherwise returns True.
        """
        attr = getattr(self, "_reactions_enabled", None)
        if callable(attr):
            return attr()
        if attr is not None:
            return bool(attr)
        return True

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    async def _rxn_on_processing_start(self, event: Any) -> None:
        """Add persona emoji when processing begins."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_reactions_enabled():
            return

        msg_ref = self._reaction_resolve_message(event)
        if msg_ref is None:
            return
        key = self._reaction_msg_key(event)
        if key is None:
            return

        emoji = self._rxn_persona_emoji
        translated = self._reaction_translate_emoji(emoji)
        if translated is None:
            translated = "👀"

        if self._reaction_replace_mode:
            await self._reaction_set(msg_ref, translated)
        else:
            await self._reaction_add(msg_ref, translated)

        self._rxn_active[key] = translated
        self._rxn_msg_refs[key] = msg_ref

    async def _rxn_on_tool_call_start(self, event: Any, tool_name: str) -> None:
        """Swap reaction to tool-specific emoji (with cooldown)."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_dynamic:
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return
        msg_ref = self._rxn_msg_refs.get(key)
        if msg_ref is None:
            # Try resolving from event directly (fallback)
            msg_ref = self._reaction_resolve_message(event)
            if msg_ref is None:
                return
            self._rxn_msg_refs[key] = msg_ref

        # Cooldown check
        now = time.monotonic()
        last = self._rxn_last_swap.get(key, 0.0)
        if now - last < self._rxn_cooldown:
            return

        from agent.display import get_tool_emoji
        raw_emoji = get_tool_emoji(tool_name, default="⚙️")
        tool_emoji = self._reaction_translate_emoji(raw_emoji)
        if tool_emoji is None:
            tool_emoji = self._reaction_translate_emoji("⚙️") or "⚙️"

        current = self._rxn_active.get(key)
        if current == tool_emoji:
            return  # Already showing this emoji

        if self._reaction_replace_mode:
            await self._reaction_set(msg_ref, tool_emoji)
        else:
            await self._reaction_add(msg_ref, tool_emoji)
            if current and current != tool_emoji:
                await self._reaction_remove(msg_ref, current)

        self._rxn_active[key] = tool_emoji
        self._rxn_last_swap[key] = now

    async def _rxn_on_processing_complete(self, event: Any, outcome: Any) -> None:
        """Replace active reaction with final emoji."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_reactions_enabled():
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return
        msg_ref = self._rxn_msg_refs.pop(key, None)
        if msg_ref is None:
            msg_ref = self._reaction_resolve_message(event)
        current = self._rxn_active.pop(key, None)
        self._rxn_last_swap.pop(key, None)

        if msg_ref is None:
            return

        # Import here to avoid circular imports at module level
        from gateway.platforms.base import ProcessingOutcome

        if outcome == ProcessingOutcome.CANCELLED:
            # Just clean up, don't change the reaction
            if current:
                if not self._reaction_replace_mode:
                    await self._reaction_remove(msg_ref, current)
            return

        if outcome == ProcessingOutcome.SUCCESS:
            final = self._rxn_persona_emoji
        else:
            final = "❌"

        translated = self._reaction_translate_emoji(final)
        if translated is None:
            translated = self._reaction_translate_emoji("❌") or "❌"

        if self._reaction_replace_mode:
            await self._reaction_set(msg_ref, translated)
        else:
            if translated != current:
                await self._reaction_add(msg_ref, translated)
            if current and current != translated:
                await self._reaction_remove(msg_ref, current)
