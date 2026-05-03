"""
Discord interactive UI component builders.

Provides View, Button, Select, and Modal builders that allow agents to send
rich interactive messages through the send_message tool.  Component interactions
are routed back to the agent as MessageEvent instances.

Uses discord.py's native View system (v1 components).  Views are stored in a
ComponentStore keyed by message_id for interaction routing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discord import guard (mirrors discord.py adapter pattern)
# ---------------------------------------------------------------------------
try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUSTOM_ID_PREFIX = "hermes:"
"""Prefix baked into every custom_id so we can distinguish agent-created
components from other bot interactions or native Discord UI."""

DISCORD_CUSTOM_ID_MAX = 100
"""Discord enforces a 100-character limit on custom_id strings."""

DISCORD_BUTTONS_PER_ROW = 5
DISCORD_SELECTS_PER_ROW = 1
DISCORD_ACTION_ROWS_MAX = 5

VIEW_DEFAULT_TIMEOUT = 300  # 5 minutes, matches _ExecApprovalView

# Button style mapping: agent-friendly names → discord.ButtonStyle
_BUTTON_STYLE_MAP: Dict[str, "discord.ButtonStyle"] = {
    "primary": discord.ButtonStyle.blurple if discord else None,
    "secondary": discord.ButtonStyle.grey if discord else None,
    "success": discord.ButtonStyle.green if discord else None,
    "danger": discord.ButtonStyle.red if discord else None,
    "link": discord.ButtonStyle.link if discord else None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_custom_id(custom_id: str) -> str:
    """Ensure *custom_id* does not exceed Discord's 100-char limit.

    If truncation is needed, a short content-hash suffix is appended so IDs
    remain unique even after truncation.
    """
    if len(custom_id) <= DISCORD_CUSTOM_ID_MAX:
        return custom_id
    hsh = hashlib.sha256(custom_id.encode()).hexdigest()[:8]
    return f"{custom_id[:DISCORD_CUSTOM_ID_MAX - 9]}_{hsh}"


def _format_interaction_text(
    kind: str,
    *,
    label: str = "",
    value: str = "",
    custom_id: str = "",
    values: Optional[List[str]] = None,
) -> str:
    """Build a human-readable structured text for an interaction event."""
    parts = [f"[{kind}]"]
    if label:
        parts.append(f"label='{label}'")
    if value:
        parts.append(f"value='{value}'")
    if custom_id:
        parts.append(f"custom_id='{custom_id}'")
    if values:
        parts.append(f"values={values}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Component spec types (plain dicts, no external dependencies)
# ---------------------------------------------------------------------------

# Expected component spec shapes (JSON from the agent):
#
#   {
#       "action_rows": [
#           {
#               "buttons": [
#                   {"label": "...", "style": "primary", "custom_id": "...",
#                    "emoji": "✅", "disabled": false},
#                   ...
#               ],
#               "select": {          # optional, mutually exclusive with buttons
#                   "placeholder": "Pick one…",
#                   "custom_id": "...",
#                   "options": [
#                       {"label": "A", "value": "a", "description": "Option A"},
#                       ...
#                   ],
#                   "min_values": 1,
#                   "max_values": 1,
#               }
#           },
#           ...
#       ],
#       "timeout": 300,
#       "single_use": true,
#   }


# ---------------------------------------------------------------------------
# ComponentStore — tracks live views for interaction routing
# ---------------------------------------------------------------------------

@dataclass
class _TrackedView:
    """Internal bookkeeping for a single tracked component view."""

    view: Any  # discord.ui.View (None for REST-sent components)
    session_key: str
    interaction_callback: Callable
    source: Any  # SessionSource
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    rest_sent: bool = False
    """True if the components were sent via REST API (aiohttp) rather than
    through discord.py's websocket adapter.  REST-sent components have no
    discord.py View attached and need the adapter's on_raw_interaction
    handler to acknowledge and route them."""
    chat_id: str = ""
    """Discord channel ID (populated for REST-sent components so the
    adapter can build a proper MessageEvent without a View reference)."""


class ComponentStore:
    """Singleton-style store mapping message_id → tracked view metadata.

    The Discord adapter should register views here after sending a message
    with components.  When a Discord interaction arrives whose custom_id
    starts with ``CUSTOM_ID_PREFIX``, the adapter can look up the originating
    view via the interaction's message reference.

    Thread-safety note: all access is expected to happen on the asyncio event
    loop thread (same as discord.py callbacks), so no explicit locking.
    """

    def __init__(self) -> None:
        self._store: Dict[str, _TrackedView] = {}

    # -- public API --------------------------------------------------------

    def register(
        self,
        message_id: str,
        view: Any,
        session_key: str,
        interaction_callback: Callable,
        source: Any,
        rest_sent: bool = False,
        chat_id: str = "",
    ) -> None:
        """Register a view for the given *message_id*.

        Parameters
        ----------
        rest_sent:
            If True, the components were sent via REST (aiohttp) rather than
            through the discord.py websocket adapter.  The adapter's
            ``on_raw_interaction`` handler will be responsible for
            acknowledging and routing these interactions.
        chat_id:
            Discord channel ID (required when *rest_sent* is True).
        """
        self._store[message_id] = _TrackedView(
            view=view,
            session_key=session_key,
            interaction_callback=interaction_callback,
            source=source,
            rest_sent=rest_sent,
            chat_id=chat_id,
        )
        logger.debug(
            "Registered component view for message %s (session=%s, rest_sent=%s)",
            message_id,
            session_key,
            rest_sent,
        )

    def get(self, message_id: str) -> Optional[_TrackedView]:
        """Return tracked metadata for *message_id*, or ``None``."""
        return self._store.get(message_id)

    def remove(self, message_id: str) -> None:
        """Remove a tracked view entry."""
        self._store.pop(message_id, None)

    def cleanup(self, max_age: float = 600.0) -> int:
        """Remove entries older than *max_age* seconds.

        Returns the number of entries removed.
        """
        now = time.time()
        expired = [
            mid for mid, tv in self._store.items()
            if now - tv.created_at > max_age or tv.resolved
        ]
        for mid in expired:
            del self._store[mid]
        if expired:
            logger.debug("Cleaned up %d expired component entries", len(expired))
        return len(expired)

    @property
    def size(self) -> int:
        """Number of currently tracked views."""
        return len(self._store)


# Module-level singleton
component_store = ComponentStore()


# ---------------------------------------------------------------------------
# AgentComponentView — generic discord.ui.View built from a JSON spec
# ---------------------------------------------------------------------------

class AgentComponentView(discord.ui.View if discord else object):  # type: ignore[misc]
    """A :class:`discord.ui.View` dynamically constructed from an agent's
    component specification dict.

    Parameters
    ----------
    spec:
        The full component spec dict (see module docstring for shape).
    message_id:
        The Discord snowflake ID of the message this view is attached to.
    session_key:
        Hermes session key for routing the interaction back.
    source:
        :class:`SessionSource` that identifies the originating session.
    interaction_callback:
        Async callable ``(MessageEvent) -> None`` invoked when a user
        interacts with any component in this view.
    timeout:
        View timeout in seconds (default 300).
    single_use:
        If ``True`` (default), components are disabled after the first
        interaction.
    """

    def __init__(
        self,
        spec: Dict[str, Any],
        message_id: str,
        session_key: str,
        source: Any,
        interaction_callback: Callable,
        timeout: float = VIEW_DEFAULT_TIMEOUT,
        single_use: bool = True,
    ) -> None:
        if not DISCORD_AVAILABLE:
            raise RuntimeError("discord.py is not installed")
        super().__init__(timeout=timeout)
        self._message_id = message_id
        self._session_key = session_key
        self._source = source
        self._interaction_callback = interaction_callback
        self._single_use = single_use
        self._resolved = False
        self._spec = spec
        self._build_from_spec(spec)

    # -- spec parsing -------------------------------------------------------

    def _build_from_spec(self, spec: Dict[str, Any]) -> None:
        """Populate this View with buttons/selects from *spec*."""
        action_rows: List[Dict[str, Any]] = spec.get("action_rows", [])
        if not action_rows:
            logger.warning("Component spec has no action_rows — view will be empty")
            return

        for row_idx, row_spec in enumerate(action_rows):
            if row_idx >= DISCORD_ACTION_ROWS_MAX:
                logger.warning(
                    "Component spec has more than %d action rows; truncating",
                    DISCORD_ACTION_ROWS_MAX,
                )
                break

            buttons = row_spec.get("buttons", [])
            select_spec = row_spec.get("select")

            if buttons and select_spec:
                logger.warning(
                    "Action row %d has both buttons and a select menu; "
                    "Discord only allows one type per row. Select will be ignored.",
                    row_idx,
                )
                select_spec = None

            if buttons:
                self._add_buttons(buttons)
            elif select_spec:
                self._add_select(select_spec)

    def _add_buttons(self, buttons: List[Dict[str, Any]]) -> None:
        """Add up to 5 buttons for the current action row."""
        for btn_spec in buttons[:DISCORD_BUTTONS_PER_ROW]:
            try:
                btn = self._make_button(btn_spec)
                if btn is not None:
                    self.add_item(btn)
            except Exception:
                logger.warning(
                    "Failed to create button from spec %s",
                    btn_spec,
                    exc_info=True,
                )

    def _add_select(self, select_spec: Dict[str, Any]) -> None:
        """Add a string select menu for the current action row."""
        try:
            select = self._make_select(select_spec)
            if select is not None:
                self.add_item(select)
        except Exception:
            logger.warning(
                "Failed to create select from spec %s",
                select_spec,
                exc_info=True,
            )

    def _make_button(self, spec: Dict[str, Any]) -> Optional[Any]:
        """Build a :class:`discord.ui.Button` from a button spec dict.

        Returns ``None`` for link-style buttons with invalid URLs.
        """
        style_name = spec.get("style", "secondary")
        style = _BUTTON_STYLE_MAP.get(style_name)
        if style is None:
            logger.warning("Unknown button style '%s'; falling back to secondary", style_name)
            style = _BUTTON_STYLE_MAP["secondary"]

        label = str(spec.get("label", ""))[:80]  # Discord label limit
        emoji = spec.get("emoji")
        disabled = bool(spec.get("disabled", False))
        custom_id = spec.get("custom_id", "")

        # Link buttons don't use custom_id; they use url instead
        url = spec.get("url")
        if style_name == "link":
            if not url:
                logger.warning("Link button has no URL; skipping")
                return None
            btn = discord.ui.Button(
                style=style,
                label=label or None,
                emoji=emoji,
                url=url,
                disabled=disabled,
            )
            # Link buttons are not interactive — no callback needed
            return btn

        if not custom_id:
            logger.warning("Non-link button has no custom_id; skipping")
            return None

        full_custom_id = _truncate_custom_id(f"{CUSTOM_ID_PREFIX}{custom_id}")

        btn = discord.ui.Button(
            style=style,
            label=label or None,
            emoji=emoji,
            custom_id=full_custom_id,
            disabled=disabled,
        )
        btn.callback = self._make_button_callback(label, custom_id, full_custom_id)
        return btn

    def _make_select(self, spec: Dict[str, Any]) -> Optional[Any]:
        """Build a :class:`discord.ui.Select` from a select spec dict."""
        custom_id = spec.get("custom_id", "")
        if not custom_id:
            logger.warning("Select menu has no custom_id; skipping")
            return None

        full_custom_id = _truncate_custom_id(f"{CUSTOM_ID_PREFIX}{custom_id}")
        placeholder = str(spec.get("placeholder", "Select an option…"))[:100]
        min_values = max(1, int(spec.get("min_values", 1)))
        max_values = min(25, int(spec.get("max_values", 1)))
        if max_values < min_values:
            max_values = min_values
        disabled = bool(spec.get("disabled", False))

        options: List[discord.SelectOption] = []
        for opt in spec.get("options", [])[:25]:
            opt_label = str(opt.get("label", ""))[:100]
            opt_value = str(opt.get("value", opt_label))[:100]
            opt_desc = opt.get("description")
            if opt_desc:
                opt_desc = str(opt_desc)[:100]
            opt_emoji = opt.get("emoji")
            opt_default = bool(opt.get("default", False))
            options.append(
                discord.SelectOption(
                    label=opt_label,
                    value=opt_value,
                    description=opt_desc,
                    emoji=opt_emoji,
                    default=opt_default,
                )
            )

        if not options:
            logger.warning("Select menu has no options; skipping")
            return None

        select = discord.ui.Select(
            custom_id=full_custom_id,
            placeholder=placeholder,
            min_values=min_values,
            max_values=max_values,
            options=options,
            disabled=disabled,
        )
        select.callback = self._make_select_callback(custom_id, full_custom_id)
        return select

    # -- interaction callbacks ----------------------------------------------

    def _make_button_callback(
        self, label: str, original_custom_id: str, full_custom_id: str
    ) -> Callable:
        """Return an async callback for a button interaction."""

        async def _callback(interaction: discord.Interaction) -> None:
            await self._handle_interaction(
                interaction,
                kind="Button clicked",
                label=label,
                original_custom_id=original_custom_id,
            )

        return _callback

    def _make_select_callback(
        self, original_custom_id: str, full_custom_id: str
    ) -> Callable:
        """Return an async callback for a select-menu interaction."""

        async def _callback(interaction: discord.Interaction) -> None:
            values = list(interaction.data.get("values", []))
            # Use the first selected value as label for readability
            label = values[0] if values else ""
            await self._handle_interaction(
                interaction,
                kind="Select chosen",
                label=label,
                original_custom_id=original_custom_id,
                values=values,
            )

        return _callback

    async def _handle_interaction(
        self,
        interaction: discord.Interaction,
        *,
        kind: str,
        label: str,
        original_custom_id: str,
        values: Optional[List[str]] = None,
    ) -> None:
        """Common handler for all component interactions.

        1. Acknowledge the interaction (edit the message).
        2. Format structured text describing what was clicked.
        3. Build a MessageEvent and forward it via the callback.
        4. Optionally disable the component (single-use mode).
        """
        if self._resolved:
            # Already handled (race guard)
            try:
                await interaction.response.defer()
            except Exception:
                pass
            return

        # --- format the interaction as structured text ---------------------
        text = _format_interaction_text(
            kind,
            label=label,
            custom_id=original_custom_id,
            values=values,
        )

        # --- update the message to acknowledge the click -------------------
        try:
            # Disable clicked component(s) after interaction
            if self._single_use:
                for child in self.children:
                    if isinstance(child, (discord.ui.Button, discord.ui.Select)):
                        child.disabled = True
                self._resolved = True
                # Also mark the tracked view as resolved
                tracked = component_store.get(self._message_id)
                if tracked:
                    tracked.resolved = True

            await interaction.response.edit_message(view=self)
        except Exception:
            logger.debug(
                "Failed to edit message on interaction for %s; "
                "attempting deferred response",
                self._message_id,
                exc_info=True,
            )
            try:
                await interaction.response.defer()
            except Exception:
                pass

        # --- build a MessageEvent and route it to the gateway ---------------
        from gateway.platforms.base import MessageEvent, MessageType

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=self._source,
            raw_message=interaction,
            message_id=str(interaction.message.id) if interaction.message else None,
            reply_to_message_id=self._message_id,
        )

        try:
            result = self._interaction_callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.error(
                "Error routing component interaction for message %s",
                self._message_id,
                exc_info=True,
            )

    # -- lifecycle ----------------------------------------------------------

    async def on_timeout(self) -> None:
        """Disable all children when the view expires."""
        self._resolved = True
        tracked = component_store.get(self._message_id)
        if tracked:
            tracked.resolved = True
        for child in self.children:
            child.disabled = True
        logger.debug(
            "AgentComponentView timed out for message %s", self._message_id
        )


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_view_from_spec(
    spec: Dict[str, Any],
    message_id: str,
    session_key: str,
    source: Any,
    interaction_callback: Callable,
) -> Optional[AgentComponentView]:
    """Factory: create an :class:`AgentComponentView` from a JSON spec.

    Parameters
    ----------
    spec:
        Component specification dict (see module docstring).
    message_id:
        Discord message ID the view will be attached to.
    session_key:
        Hermes session key for routing.
    source:
        :class:`~gateway.session.SessionSource` identifying the session.
    interaction_callback:
        Async callable ``(MessageEvent) -> None`` to dispatch interactions.

    Returns
    -------
    AgentComponentView or ``None`` if the spec is invalid or discord.py is
    not available.
    """
    if not DISCORD_AVAILABLE:
        logger.warning("discord.py not available; cannot build component view")
        return None

    if not isinstance(spec, dict):
        logger.warning("Component spec is not a dict: %r", spec)
        return None

    action_rows = spec.get("action_rows")
    if not action_rows:
        logger.warning("Component spec has empty or missing action_rows")
        return None

    if not isinstance(action_rows, list):
        logger.warning("Component spec action_rows is not a list: %r", action_rows)
        return None

    timeout = float(spec.get("timeout", VIEW_DEFAULT_TIMEOUT))
    single_use = bool(spec.get("single_use", True))

    try:
        view = AgentComponentView(
            spec=spec,
            message_id=message_id,
            session_key=session_key,
            source=source,
            interaction_callback=interaction_callback,
            timeout=timeout,
            single_use=single_use,
        )
    except Exception:
        logger.error(
            "Failed to build AgentComponentView from spec for message %s",
            message_id,
            exc_info=True,
        )
        return None

    # Register in the component store
    component_store.register(
        message_id=message_id,
        view=view,
        session_key=session_key,
        interaction_callback=interaction_callback,
        source=source,
    )

    return view


def is_agent_component(custom_id: str) -> bool:
    """Return ``True`` if *custom_id* belongs to an agent-created component."""
    return bool(custom_id) and custom_id.startswith(CUSTOM_ID_PREFIX)
