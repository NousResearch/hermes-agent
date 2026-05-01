"""Discord-side wiring for the unified skill trigger framework.

Companion file to ``gateway/platforms/discord.py``. Hosts:

- :class:`DiscordInteractionsHandler` — composition handler injected into
  :class:`gateway.platforms.discord.DiscordAdapter` to receive button
  interactions and inbound reactions.
- :class:`SkillButtonView` — convenience ``discord.ui.View`` subclass that
  delegates button callbacks to the handler. Skills emit buttons via this
  helper; raw ``discord.ui.View`` subclasses bypass routing intentionally
  (discord.py 2.7+ routes View callbacks BEFORE the global ``on_interaction``
  event, so a custom View opts out of the resolver).

Architecture notes (see plan §2b for full context):

- Composition over inheritance: ``DiscordInteractionsHandler`` does NOT
  subclass ``DiscordAdapter``. The adapter instantiates the handler in
  ``__init__`` and calls into it from ``connect()``-time event handlers.
- Skill enumeration via callable injection: ``skill_provider`` is a
  ``Callable[[], List[SkillEntry]]`` — typically wraps
  ``agent.skill_utils.iter_all_skill_frontmatters``. There is no
  ``_available_skills`` attribute on adapters; the resolver receives the
  callable's result on each invocation.
- ``custom_id`` shape: ``skill_<skill_name>_<action>`` — the resolver
  matches against ``custom_id_pattern`` via ``fnmatch``, so skills can
  declare e.g. ``custom_id_pattern: "skill_approve_*"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

try:
    import discord  # type: ignore
    DISCORD_AVAILABLE = True
except ImportError:
    discord = None  # type: ignore
    DISCORD_AVAILABLE = False

from gateway.skill_resolver import SkillEntry, has_explicit_triggers, resolve_event_skills

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from gateway.platforms.discord import DiscordAdapter

logger = logging.getLogger(__name__)


# ── Public custom_id helpers ───────────────────────────────────────────────


SKILL_CUSTOM_ID_PREFIX = "skill_"


# Phase 0.5 token-leak instrumentation: per-decision counters for inbound
# Discord events. Single-event-loop access (discord.py) means a plain dict
# is safe — no asyncio.Lock needed. Mention counters are pre-declared here
# but wiring is deferred to PR-B (the mention hook does not yet exist).
_event_counters: dict[str, int] = {
    "discord.reactions.skipped_non_bot_author": 0,
    "discord.reactions.skipped_no_match": 0,
    "discord.reactions.invoked": 0,
    "discord.mentions.skipped_no_match": 0,
    "discord.mentions.invoked": 0,
}


def get_event_counters() -> dict[str, int]:
    """Return an immutable snapshot of the inbound-event decision counters."""
    return dict(_event_counters)


def make_skill_custom_id(skill_name: str, action: str) -> str:
    """Build a canonical custom_id for skill-routed button interactions.

    Skills declare ``custom_id_pattern`` like ``skill_<name>_*`` to match
    button clicks routed via :class:`SkillButtonView`.
    """
    safe_name = skill_name.replace(" ", "_")
    safe_action = action.replace(" ", "_")
    return f"{SKILL_CUSTOM_ID_PREFIX}{safe_name}_{safe_action}"


def is_skill_custom_id(custom_id: Optional[str]) -> bool:
    """Return True if a custom_id should route to the skill resolver."""
    return isinstance(custom_id, str) and custom_id.startswith(SKILL_CUSTOM_ID_PREFIX)


# ── DiscordInteractionsHandler ────────────────────────────────────────────


class DiscordInteractionsHandler:
    """Composition handler for inbound Discord interactions and reactions.

    Lifecycle:
        1. Adapter ``__init__`` instantiates the handler with a callable that
           returns the current skill list.
        2. ``connect()`` registers ``on_interaction`` and (when
           ``inbound_routing`` is enabled) raw reaction handlers that delegate
           to the methods on this class.
        3. Each delegate call resolves matching skills and dispatches a
           synthetic ``MessageEvent`` through ``adapter.handle_message``.

    The handler holds NO event-loop state of its own; restart-safety follows
    from the adapter's existing reconnect logic.
    """

    def __init__(
        self,
        adapter: "DiscordAdapter",
        skill_provider: Callable[[], List[SkillEntry]],
    ) -> None:
        self._adapter = adapter
        self._skill_provider = skill_provider
        self._explicit_triggers_cache: Optional[bool] = None

    # ── Cached corpus signal ─────────────────────────────────────────────

    def explicit_triggers_present(self) -> bool:
        """Cached check: does the deployed skill corpus use the new schema?

        Used by Feishu BC fallback (and any other adapter that needs to know
        whether to fall through to broadcast routing). Cached on first call;
        ``invalidate_cache()`` clears it after a skill reload.
        """
        if self._explicit_triggers_cache is None:
            try:
                self._explicit_triggers_cache = has_explicit_triggers(self._skill_provider())
            except Exception:
                logger.exception("explicit_triggers_present: skill_provider raised — assuming False")
                self._explicit_triggers_cache = False
        return self._explicit_triggers_cache

    def invalidate_cache(self) -> None:
        self._explicit_triggers_cache = None

    # ── Button interactions ─────────────────────────────────────────────

    async def handle_skill_button_interaction(self, interaction: "discord.Interaction") -> None:
        """Route a button click to matching skills.

        Called from a ``SkillButtonView`` callback OR from the global
        ``on_interaction`` handler when the interaction's ``custom_id`` starts
        with :data:`SKILL_CUSTOM_ID_PREFIX`. Builds a payload, asks the
        resolver, and dispatches a synthetic MessageEvent with
        ``auto_skill=<matched names>``.
        """
        custom_id = getattr(interaction.data, "get", lambda _k, _d=None: None)(
            "custom_id"
        ) if interaction.data else None
        if not custom_id:
            # Defensive — non-button interaction or missing payload.
            return

        try:
            skills = self._skill_provider()
        except Exception:
            logger.exception("handle_skill_button_interaction: skill_provider raised")
            return

        payload = self._build_button_payload(interaction, custom_id)
        matched = resolve_event_skills("button", payload, skills)
        if not matched:
            logger.debug(
                "handle_skill_button_interaction: no skill matched custom_id=%r",
                custom_id,
            )
            return

        await self._dispatch_synthetic(
            interaction=interaction,
            text=f"[button] {custom_id}",
            matched_skills=matched,
        )

    # ── Inbound mentions ────────────────────────────────────────────────

    async def handle_inbound_mention(
        self,
        message: "discord.Message",
        normalized_text: str,
    ) -> Optional[List[str]]:
        """Resolve skill matches for an @mention event.

        Companion to :meth:`handle_inbound_reaction` — routes the mention
        through the skill resolver. The caller (``DiscordAdapter._handle_message``)
        decides what to do with the result:

        - On match: dispatches the existing message pipeline with
          ``auto_skill=<matched names>`` set, saving the LLM round-trip for
          general-purpose reasoning.
        - On no match: caller may early-return (saving an LLM invoke) when
          explicit triggers are configured, or fall through to the legacy
          path when not.

        Args:
            message: The inbound ``discord.Message`` (used for channel
                metadata; we only read attributes — no fetch I/O).
            normalized_text: The mention-stripped text already computed by
                ``_handle_message``. Avoids re-stripping the bot's own
                ``<@id>`` token.

        Returns:
            List of matching skill names on a match, or ``None`` on no match
            / fail-safe. Returning a list (vs ``[]``) lets the caller cleanly
            distinguish "matched" from "did not match".
        """
        try:
            skills = self._skill_provider()
        except Exception:
            logger.exception("handle_inbound_mention: skill_provider raised")
            return None

        channel = getattr(message, "channel", None)
        channel_name = getattr(channel, "name", "") or "" if channel is not None else ""
        channel_id = str(getattr(channel, "id", "") or "") if channel is not None else ""
        payload: Dict[str, Any] = {
            "text": normalized_text,
            "channel": channel_name,
            "channel_id": channel_id,
        }
        matched = resolve_event_skills("mention", payload, skills)
        if not matched:
            _event_counters["discord.mentions.skipped_no_match"] += 1
            return None

        _event_counters["discord.mentions.invoked"] += 1
        return matched

    # ── Inbound reactions ───────────────────────────────────────────────

    async def handle_inbound_reaction(
        self,
        payload: "discord.RawReactionActionEvent",
        action: str,
    ) -> None:
        """Route a reaction add/remove event to matching skills.

        Only fires when ``config.extra.reactions.inbound_routing`` is enabled
        AND the user is not the bot itself. Builds a structured payload from
        the raw event and asks the resolver.
        """
        if not DISCORD_AVAILABLE:
            return
        client = self._adapter._client  # type: ignore[attr-defined]
        if client is not None and client.user is not None and payload.user_id == client.user.id:
            # Bot's own reactions are skipped (Vector 2 author filter).
            _event_counters["discord.reactions.skipped_non_bot_author"] += 1
            return  # ignore the bot's own reactions

        # Reacted-message author filter: only route reactions on the bot's
        # own messages. Try the in-memory message cache first (zero API
        # cost) before falling back to channel.fetch_message.
        # discord.py state._messages is a Deque[Message] (or None when
        # max_messages=None); iterate, don't .get(). Wrapped in try/except
        # because _connection._messages is a discord.py internal: a minor-
        # version rename or restructure should fall through to fetch_message
        # silently rather than break reaction routing.
        if client is None or client.user is None:
            return
        reacted_msg = None
        try:
            connection = getattr(client, "_connection", None)
            cache = getattr(connection, "_messages", None) if connection is not None else None
            if cache:
                reacted_msg = next((m for m in cache if m.id == payload.message_id), None)
        except (AttributeError, KeyError, TypeError):
            logger.debug(
                "handle_inbound_reaction: message cache lookup failed; falling through to fetch_message"
            )
            reacted_msg = None

        if reacted_msg is None:
            try:
                channel = client.get_channel(payload.channel_id) or await client.fetch_channel(payload.channel_id)
                reacted_msg = await channel.fetch_message(payload.message_id)
            except Exception:
                logger.debug(
                    "handle_inbound_reaction: failed to fetch message %s in channel %s; aborting",
                    payload.message_id,
                    payload.channel_id,
                )
                return  # fail-safe

        if reacted_msg.author.id != client.user.id:
            _event_counters["discord.reactions.skipped_non_bot_author"] += 1
            return  # not bot's message

        try:
            skills = self._skill_provider()
        except Exception:
            logger.exception("handle_inbound_reaction: skill_provider raised")
            return

        emoji_str = str(payload.emoji)
        resolver_payload = self._build_reaction_payload(payload, emoji_str, action)
        matched = resolve_event_skills("reaction", resolver_payload, skills)
        if not matched:
            _event_counters["discord.reactions.skipped_no_match"] += 1
            return

        # Reactions don't carry interaction objects; build a synthetic event
        # via a fabricated channel/message context.
        _event_counters["discord.reactions.invoked"] += 1
        synthetic_text = f"[reaction:{action}] {emoji_str} on message {payload.message_id}"
        await self._dispatch_reaction_synthetic(
            payload=payload,
            text=synthetic_text,
            matched_skills=matched,
        )

    # ── Internal: payload builders ──────────────────────────────────────

    def _build_button_payload(
        self,
        interaction: "discord.Interaction",
        custom_id: str,
    ) -> Dict[str, Any]:
        channel_name = ""
        channel = getattr(interaction, "channel", None)
        if channel is not None:
            channel_name = getattr(channel, "name", "") or ""
        return {
            "custom_id": custom_id,
            "user_id": str(interaction.user.id) if interaction.user else "",
            "channel": channel_name,
            "channel_id": str(interaction.channel_id) if interaction.channel_id else "",
        }

    def _build_reaction_payload(
        self,
        payload: "discord.RawReactionActionEvent",
        emoji_str: str,
        action: str,
    ) -> Dict[str, Any]:
        return {
            "emoji": emoji_str,
            "action": action,  # "add" or "remove"
            "user_id": str(payload.user_id),
            "channel_id": str(payload.channel_id),
            "message_id": str(payload.message_id),
            # Channel name is not on the raw event; the adapter resolves it
            # lazily during dispatch. Channel filtering against names is best
            # effort — a skill needing strict filtering should match by id
            # via channel_filter (which currently checks `channel`).
            "channel": str(payload.channel_id),
        }

    # ── Internal: dispatch ──────────────────────────────────────────────

    async def _dispatch_synthetic(
        self,
        interaction: "discord.Interaction",
        text: str,
        matched_skills: List[str],
    ) -> None:
        """Build a MessageEvent for a button interaction and run the adapter's
        message handling pipeline with auto_skill set."""
        # Defer the interaction so Discord doesn't time it out while the skill
        # prompts the LLM. The adapter's outbound message handling will follow
        # up via interaction.followup.* paths if a skill response is generated.
        try:
            if not interaction.response.is_done():  # type: ignore[union-attr]
                await interaction.response.defer(ephemeral=False, thinking=True)  # type: ignore[union-attr]
        except Exception:
            logger.debug("interaction.response.defer failed — continuing", exc_info=True)

        try:
            event = self._adapter._build_slash_event(interaction, text)  # type: ignore[attr-defined]
            event.auto_skill = matched_skills
        except Exception:
            logger.exception("_build_slash_event failed for button interaction")
            return

        await self._adapter.handle_message(event)  # type: ignore[attr-defined]

    async def _dispatch_reaction_synthetic(
        self,
        payload: "discord.RawReactionActionEvent",
        text: str,
        matched_skills: List[str],
    ) -> None:
        """Build a MessageEvent for an inbound reaction and run the adapter's
        message handling pipeline with auto_skill set.

        Reactions don't carry an Interaction object, so we use the adapter's
        existing source-building pipeline anchored on the channel + user.
        """
        from gateway.platforms.base import MessageEvent, MessageType  # noqa: WPS433

        # Build a lightweight source from raw reaction fields. The adapter
        # builds richer sources via its message pipeline; here we keep it
        # minimal and rely on the adapter's downstream resolution for any
        # missing context.
        try:
            source = self._adapter.build_source(  # type: ignore[attr-defined]
                chat_id=str(payload.channel_id),
                chat_name="",
                chat_type="dm" if payload.guild_id is None else "group",
                user_id=str(payload.user_id),
                user_name="",
                thread_id=None,
                chat_topic=None,
            )
        except Exception:
            logger.exception("build_source failed for reaction event")
            return

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            auto_skill=matched_skills,
        )
        await self._adapter.handle_message(event)  # type: ignore[attr-defined]


# ── SkillButtonView helper ────────────────────────────────────────────────


if DISCORD_AVAILABLE:

    class SkillButtonView(discord.ui.View):
        """A ``discord.ui.View`` that routes button clicks to the skill resolver.

        Skills SHOULD use this helper to emit buttons. Skills that subclass
        ``discord.ui.View`` directly bypass the resolver — discord.py routes
        View callbacks before the global ``on_interaction`` event, so the
        custom subclass owns dispatch.

        Args:
            handler: The :class:`DiscordInteractionsHandler` registered on
                the adapter (typically ``adapter._interactions``).
            skill_name: Skill that owns this view. Used to build canonical
                ``custom_id`` strings via :func:`make_skill_custom_id`.
            actions: Mapping of button label → action key. The action key is
                appended to the canonical custom_id.
            button_styles: Optional mapping of button label → ``discord.ButtonStyle``
                value. Labels not present here default to ``ButtonStyle.primary``.
                Passed by :func:`tools.discord_button_tool` when the LLM specifies
                per-button styles. Not needed for direct Python callers that accept
                the primary default.
            timeout: Discord view timeout in seconds (default 180).
        """

        def __init__(
            self,
            handler: "DiscordInteractionsHandler",
            skill_name: str,
            actions: Dict[str, str],
            *,
            button_styles: Optional[Dict[str, Any]] = None,
            timeout: float = 180.0,
        ) -> None:
            super().__init__(timeout=timeout)
            self._handler = handler
            self._skill_name = skill_name
            for label, action in actions.items():
                custom_id = make_skill_custom_id(skill_name, action)
                style = (
                    (button_styles or {}).get(label)
                    or discord.ButtonStyle.primary  # type: ignore[attr-defined]
                )
                button = discord.ui.Button(  # type: ignore[attr-defined]
                    label=label,
                    custom_id=custom_id,
                    style=style,
                )
                button.callback = self._make_callback()
                self.add_item(button)

        def _make_callback(self):
            handler = self._handler

            async def _callback(interaction: discord.Interaction) -> None:
                await handler.handle_skill_button_interaction(interaction)

            return _callback

else:  # discord.py not installed — keep import-time symbol, raise on use

    class SkillButtonView:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("discord.py not installed; SkillButtonView unavailable")
