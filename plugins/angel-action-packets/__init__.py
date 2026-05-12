"""Angel Action Packet native approval-card bridge.

This plugin is opt-in (``plugins.enabled``) and intentionally dependency-light.
When enabled inside a gateway process, it watches the Angel resource governor
for pending Discord-addressed Action Packets and posts native Discord component
cards through the live Discord adapter/client. It does not approve packets,
execute runner work, install anything, or mutate gateway/runtime state outside
of governor card audit rows created while posting cards.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_GOVERNOR_PATH = "/opt/angel-resource-governor"
DEFAULT_POLL_SECONDS = 5.0
_TASK: asyncio.Task | None = None


@dataclass(frozen=True)
class DiscordDestination:
    channel_id: str
    thread_id: str | None = None

    @property
    def target_id(self) -> str:
        """Discord target to send into: thread ID when supplied, else channel."""
        return self.thread_id or self.channel_id


def parse_discord_destination(value: str) -> DiscordDestination:
    """Parse ``discord:<channel_id>[:<thread_id>]`` Action Packet destinations.

    The bridge intentionally accepts only explicit numeric Discord destinations.
    Non-Discord routes, empty components, non-numeric IDs, and extra components
    are rejected so the poster never silently sends cards to the wrong target.
    """
    if not isinstance(value, str):
        raise ValueError("destination must be a string")
    parts = value.strip().split(":")
    if len(parts) not in {2, 3} or parts[0] != "discord":
        raise ValueError("destination must be discord:<channel_id>[:<thread_id>]")
    channel_id = parts[1]
    thread_id = parts[2] if len(parts) == 3 else None
    if not channel_id or not channel_id.isdigit():
        raise ValueError("discord channel_id must be numeric")
    if thread_id is not None and (not thread_id or not thread_id.isdigit()):
        raise ValueError("discord thread_id must be numeric")
    return DiscordDestination(channel_id=channel_id, thread_id=thread_id)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _poll_interval() -> float:
    raw = os.environ.get("ANGEL_ACTION_PACKET_CARD_POLL_SECONDS", "")
    try:
        value = float(raw) if raw else DEFAULT_POLL_SECONDS
    except ValueError:
        value = DEFAULT_POLL_SECONDS
    return max(1.0, min(value, 60.0))


def _governor() -> Any:
    path = os.environ.get("ANGEL_RESOURCE_GOVERNOR_PATH", DEFAULT_GOVERNOR_PATH)
    if path and path not in sys.path:
        sys.path.insert(0, path)
    from angel_resource_governor.governor import QueueGovernor

    return QueueGovernor.from_env()


def _discord_adapter(adapters: dict[Any, Any]) -> Any | None:
    for key, adapter in (adapters or {}).items():
        key_value = getattr(key, "value", key)
        if str(key_value).lower() == "discord" or str(getattr(adapter, "name", "")).lower() == "discord":
            return adapter
    return None


def _is_unexpired(packet: dict[str, Any]) -> bool:
    expires_at = str(packet.get("expires_at") or "")
    if not expires_at:
        return True
    try:
        return datetime.fromisoformat(expires_at) > datetime.now(timezone.utc)
    except ValueError:
        # Malformed governor rows should not crash the gateway; let the
        # governor/card creation path reject if necessary.
        return True


def _packet_has_posted_card(governor: Any, action_id: str) -> bool:
    """Return true when a non-empty Discord message_id is already recorded."""
    try:
        with governor.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM action_packet_cards
                WHERE action_id = ?
                  AND COALESCE(message_id, '') != ''
                  AND status IN ('pending', 'resolving', 'resolved')
                LIMIT 1
                """,
                (action_id,),
            ).fetchone()
            return row is not None
    except Exception:
        logger.debug("could not inspect Action Packet card state", exc_info=True)
        return False


def _audit(governor: Any, action_id: str, event: str, detail: str) -> None:
    try:
        with governor.connect() as conn:
            governor.audit(conn, "system", event, "action_packets", action_id, detail[:1000])
    except Exception:
        logger.debug("could not audit %s for %s", event, action_id, exc_info=True)


async def send_action_packet_card_to_destination(
    adapter: Any,
    governor: Any,
    packet: dict[str, Any],
    destination: DiscordDestination,
) -> bool:
    """Post one native Action Packet card into a channel or thread target."""
    client = getattr(adapter, "_client", None)
    if client is None:
        return False
    try:
        import discord
        from angel_resource_governor.ux import action_packet_discord_card
        from gateway.platforms.discord import _component_check_auth
    except Exception:
        logger.debug("Action Packet card dependencies unavailable", exc_info=True)
        return False

    target_id = destination.target_id
    card_row = governor.create_action_packet_card(packet["id"], channel_id=target_id)
    card = action_packet_discord_card(packet)
    embed = discord.Embed(
        title=card["title"],
        description=card["description"],
        color=_discord_color(discord, card["color"]),
    )
    for name, value, inline in card["fields"]:
        embed.add_field(name=name, value=value, inline=inline)
    embed.set_footer(text=card["footer"])

    allowed_user_ids = getattr(adapter, "_allowed_user_ids", set())
    allowed_role_ids = getattr(adapter, "_allowed_role_ids", set())
    view = _build_action_packet_view(
        discord,
        governor,
        card_row,
        allowed_user_ids,
        allowed_role_ids,
        _component_check_auth,
    )

    channel = client.get_channel(int(target_id))
    if channel is None:
        channel = await client.fetch_channel(int(target_id))
    message = await channel.send(embed=embed, view=view)
    governor.attach_action_packet_card_message(card_row["id"], str(message.id))
    _audit(
        governor,
        packet["id"],
        "action_packet_card_auto_posted",
        f"discord channel={destination.channel_id} thread={destination.thread_id or ''} message={message.id}",
    )
    return True


def _build_action_packet_view(
    discord: Any,
    governor: Any,
    card: dict[str, Any],
    allowed_user_ids: set,
    allowed_role_ids: set,
    component_check_auth: Any,
) -> Any:
    class ActionPacketView(discord.ui.View):
        def __init__(self) -> None:
            super().__init__(timeout=900)
            self.resolved = False

        async def _resolve(self, interaction: Any, custom_id: str) -> None:
            if self.resolved:
                await interaction.response.send_message("Already handled.", ephemeral=True)
                return
            actor_id = str(getattr(interaction.user, "id", "") or "")
            actor_is_approver = bool(allowed_user_ids or allowed_role_ids) and bool(
                component_check_auth(interaction, allowed_user_ids, allowed_role_ids)
            )
            result = governor.resolve_action_packet_card_button(
                custom_id,
                actor_user_id=actor_id,
                discord_interaction_id=str(getattr(interaction, "id", "") or ""),
                actor_is_approver=actor_is_approver,
            )
            if not result.get("ok"):
                await interaction.response.send_message(_button_error(result), ephemeral=True)
                return
            self.resolved = True
            for child in self.children:
                child.disabled = True
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                if result.get("decision") == "approved":
                    embed.color = discord.Color.green()
                    embed.set_footer(text=f"Approved by {interaction.user.display_name}; runner queued")
                    _spawn_approved_execution_runner(result.get("action_id"))
                else:
                    embed.color = discord.Color.red()
                    embed.set_footer(text=f"Rejected by {interaction.user.display_name}")
            await interaction.response.edit_message(embed=embed, view=self)

    approve = discord.ui.Button(
        label="Approve",
        style=discord.ButtonStyle.success,
        custom_id=card["approve_custom_id"],
    )
    reject = discord.ui.Button(
        label="Reject",
        style=discord.ButtonStyle.danger,
        custom_id=card["reject_custom_id"],
    )

    async def approve_callback(interaction: Any) -> None:
        await view._resolve(interaction, card["approve_custom_id"])

    async def reject_callback(interaction: Any) -> None:
        await view._resolve(interaction, card["reject_custom_id"])

    view = ActionPacketView()
    approve.callback = approve_callback
    reject.callback = reject_callback
    view.add_item(approve)
    view.add_item(reject)
    return view


def _button_error(result: dict[str, Any]) -> str:
    return {
        "not_authorized": "You are not authorized to use this button.",
        "expired": "This Action Packet expired.",
        "card_already_resolved": "This Action Packet card was already handled.",
        "action_packet_already_resolved": "This Action Packet was already handled.",
        "unknown_button": "This Action Packet button is no longer valid.",
    }.get(result.get("reason"), "Action Packet decision was rejected.")


def _spawn_approved_execution_runner(action_id: Any) -> None:
    # Import lazily so parser/idempotency tests do not need the local Angel
    # runtime. The runner path re-checks governor approval before executing.
    try:
        from angel_action_packets.bridge import _spawn_approved_execution_runner as spawn
    except Exception:
        logger.debug("approved execution runner spawn hook unavailable", exc_info=True)
        return
    spawn(action_id)


def _discord_color(discord: Any, name: str) -> Any:
    return {
        "blue": discord.Color.blue(),
        "gold": discord.Color.gold(),
        "orange": discord.Color.orange(),
        "red": discord.Color.red(),
        "light_grey": discord.Color.light_grey(),
    }.get(name, discord.Color.orange())


async def poll_once(adapter: Any, governor: Any) -> int:
    """Post cards for eligible pending Discord Action Packets once."""
    posted = 0
    try:
        packets = governor.list_action_packets(status="pending")
    except Exception:
        logger.debug("could not list pending Action Packets", exc_info=True)
        return posted

    for packet_ref in packets:
        action_id = str(packet_ref.get("id") or "")
        if not action_id:
            continue
        try:
            packet = governor.show_action_packet(action_id)
            if packet.get("status") != "pending" or not _is_unexpired(packet):
                continue
            destination = parse_discord_destination(str(packet.get("approval_destination") or ""))
            if _packet_has_posted_card(governor, action_id):
                continue
            if await send_action_packet_card_to_destination(adapter, governor, packet, destination):
                posted += 1
        except ValueError as exc:
            _audit(governor, action_id, "action_packet_card_auto_post_skipped", str(exc))
        except Exception as exc:
            _audit(governor, action_id, "action_packet_card_auto_post_failed", str(exc))
            logger.warning("Action Packet auto-card post failed for %s: %s", action_id, exc)
    return posted


async def _poll_loop(adapter: Any, governor: Any) -> None:
    interval = _poll_interval()
    logger.info("Angel Action Packet Discord card bridge started; poll interval %.1fs", interval)
    while True:
        await poll_once(adapter, governor)
        await asyncio.sleep(interval)


def _gateway_startup(**kwargs: Any) -> None:
    global _TASK
    if _truthy(os.environ.get("ANGEL_ACTION_PACKET_AUTO_CARD_ENABLED"), default=True) is False:
        logger.info("Angel Action Packet auto-card bridge disabled by environment")
        return
    if _TASK is not None and not _TASK.done():
        return
    adapter = _discord_adapter(kwargs.get("adapters") or {})
    if adapter is None:
        logger.debug("Angel Action Packet auto-card bridge not started: Discord adapter unavailable")
        return
    try:
        governor = _governor()
    except Exception:
        logger.debug("Angel Action Packet auto-card bridge not started: governor unavailable", exc_info=True)
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("Angel Action Packet auto-card bridge requires a running gateway event loop")
        return
    _TASK = loop.create_task(_poll_loop(adapter, governor), name="angel-action-packet-auto-card")


def register(ctx: Any) -> None:
    ctx.register_hook("gateway_startup", _gateway_startup)
