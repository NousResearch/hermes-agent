"""Discord slash command wiring and interaction helpers."""

from __future__ import annotations

from typing import Any

from gateway.platforms.discord_impl import components as discord_components
from gateway.platforms.discord_impl import native_commands

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    discord = None
    DISCORD_AVAILABLE = False


def register_slash_commands(tree: Any, adapter: Any) -> None:
    """Register Discord slash commands on the command tree."""
    native_commands.register_slash_commands(tree, adapter)


def build_slash_event(adapter: Any, interaction: discord.Interaction, text: str):
    """Build a MessageEvent from a Discord slash command interaction."""
    return native_commands.build_slash_event(adapter, interaction, text)


def create_component_runtime() -> discord_components.DiscordComponentRuntime:
    """Create a generic Discord component runtime."""
    return discord_components.DiscordComponentRuntime()


if DISCORD_AVAILABLE:

    def create_exec_approval_view(
        adapter: Any,
        approval_id: str,
        allowed_user_ids: set,
        runtime: discord_components.DiscordComponentRuntime | None = None,
    ):
        """Build a generic component-runtime approval view."""
        runtime = runtime or create_component_runtime()
        view = discord_components.ManagedComponentView(runtime, timeout=300)
        allowed = tuple(str(user_id) for user_id in allowed_user_ids)

        async def _resolve(invocation: discord_components.DiscordComponentInvocation, decision: str, color: Any) -> bool:
            resolver = getattr(adapter, "_resolve_exec_approval", None)
            if not callable(resolver):
                await invocation.deny("Approval resolver is unavailable~")
                return False

            result = resolver(decision=decision, approval_id=approval_id)
            if hasattr(result, "__await__"):
                result = await result

            embed = invocation.interaction.message.embeds[0] if invocation.interaction.message.embeds else None
            if embed:
                embed.color = color
                embed.set_footer(text=f"{decision} by {invocation.interaction.user.display_name}")

            invocation.disable_all()
            await invocation.interaction.response.edit_message(embed=embed, view=view)

            followup = getattr(invocation.interaction, "followup", None)
            if followup is not None and hasattr(followup, "send"):
                await followup.send(str(result), ephemeral=True)
            return True

        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Allow Once",
                style="success",
                allowed_user_ids=allowed,
                handler=lambda invocation: _resolve(
                    invocation,
                    "allow-once",
                    discord.Color.green(),
                ),
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Always Allow",
                style="primary",
                allowed_user_ids=allowed,
                handler=lambda invocation: _resolve(
                    invocation,
                    "allow-always",
                    discord.Color.blue(),
                ),
            )
        )
        view.add_button(
            discord_components.DiscordButtonSpec(
                label="Deny",
                style="danger",
                allowed_user_ids=allowed,
                handler=lambda invocation: _resolve(
                    invocation,
                    "deny",
                    discord.Color.red(),
                ),
            )
        )
        return view

else:  # pragma: no cover - import guard
    create_exec_approval_view = None
