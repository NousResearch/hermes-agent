"""Discord slash command and approval UI helpers."""

from __future__ import annotations

from typing import Any

from gateway.platforms.base import MessageEvent, MessageType

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    discord = None
    DISCORD_AVAILABLE = False


def register_slash_commands(tree: Any, adapter: Any) -> None:
    """Register Discord slash commands on the command tree."""
    if tree is None:
        return

    @tree.command(name="new", description="Start a new conversation")
    async def slash_new(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/reset", "New conversation started~")

    @tree.command(name="reset", description="Reset your Hermes session")
    async def slash_reset(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/reset", "Session reset~")

    @tree.command(name="model", description="Show or change the model")
    @discord.app_commands.describe(name="Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.")
    async def slash_model(interaction: discord.Interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/model {name}".strip())

    @tree.command(name="reasoning", description="Show or change reasoning effort")
    @discord.app_commands.describe(effort="Reasoning effort: xhigh, high, medium, low, minimal, or none.")
    async def slash_reasoning(interaction: discord.Interaction, effort: str = ""):
        await interaction.response.defer(ephemeral=True)
        event = adapter._build_slash_event(interaction, f"/reasoning {effort}".strip())
        await adapter.handle_message(event)

    @tree.command(name="personality", description="Set a personality")
    @discord.app_commands.describe(name="Personality name. Leave empty to list available.")
    async def slash_personality(interaction: discord.Interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/personality {name}".strip())

    @tree.command(name="retry", description="Retry your last message")
    async def slash_retry(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/retry", "Retrying~")

    @tree.command(name="undo", description="Remove the last exchange")
    async def slash_undo(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/undo")

    @tree.command(name="status", description="Show Hermes session status")
    async def slash_status(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/status", "Status sent~")

    @tree.command(name="sethome", description="Set this chat as the home channel")
    async def slash_sethome(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/sethome")

    @tree.command(name="stop", description="Stop the running Hermes agent")
    async def slash_stop(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/stop", "Stop requested~")

    @tree.command(name="compress", description="Compress conversation context")
    async def slash_compress(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/compress")

    @tree.command(name="title", description="Set or show the session title")
    @discord.app_commands.describe(name="Session title. Leave empty to show current.")
    async def slash_title(interaction: discord.Interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/title {name}".strip())

    @tree.command(name="resume", description="Resume a previously-named session")
    @discord.app_commands.describe(name="Session name to resume. Leave empty to list sessions.")
    async def slash_resume(interaction: discord.Interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/resume {name}".strip())

    @tree.command(name="usage", description="Show token usage for this session")
    async def slash_usage(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/usage")

    @tree.command(name="provider", description="Show available providers")
    async def slash_provider(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/provider")

    @tree.command(name="help", description="Show available commands")
    async def slash_help(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/help")

    @tree.command(name="insights", description="Show usage insights and analytics")
    @discord.app_commands.describe(days="Number of days to analyze (default: 7)")
    async def slash_insights(interaction: discord.Interaction, days: int = 7):
        await adapter._run_simple_slash(interaction, f"/insights {days}")

    @tree.command(name="reload-mcp", description="Reload MCP servers from config")
    async def slash_reload_mcp(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/reload-mcp")

    @tree.command(name="voice", description="Toggle voice reply mode")
    @discord.app_commands.describe(mode="Voice mode: on, off, tts, channel, leave, or status")
    @discord.app_commands.choices(mode=[
        discord.app_commands.Choice(name="channel — join your voice channel", value="channel"),
        discord.app_commands.Choice(name="leave — leave voice channel", value="leave"),
        discord.app_commands.Choice(name="on — voice reply to voice messages", value="on"),
        discord.app_commands.Choice(name="tts — voice reply to all messages", value="tts"),
        discord.app_commands.Choice(name="off — text only", value="off"),
        discord.app_commands.Choice(name="status — show current mode", value="status"),
    ])
    async def slash_voice(interaction: discord.Interaction, mode: str = ""):
        await interaction.response.defer(ephemeral=True)
        event = adapter._build_slash_event(interaction, f"/voice {mode}".strip())
        await adapter.handle_message(event)

    @tree.command(name="update", description="Update Hermes Agent to the latest version")
    async def slash_update(interaction: discord.Interaction):
        await adapter._run_simple_slash(interaction, "/update", "Update initiated~")

    @tree.command(name="thread", description="Create a new thread and start a Hermes session in it")
    @discord.app_commands.describe(
        name="Thread name",
        message="Optional first message to send to Hermes in the thread",
        auto_archive_duration="Auto-archive in minutes (60, 1440, 4320, 10080)",
    )
    async def slash_thread(
        interaction: discord.Interaction,
        name: str,
        message: str = "",
        auto_archive_duration: int = 1440,
    ):
        await interaction.response.defer(ephemeral=True)
        await adapter._handle_thread_create_slash(interaction, name, message, auto_archive_duration)


def build_slash_event(adapter: Any, interaction: discord.Interaction, text: str) -> MessageEvent:
    """Build a MessageEvent from a Discord slash command interaction."""
    dm_channel_cls = getattr(discord, "DMChannel", None) if discord else None
    is_dm = isinstance(interaction.channel, dm_channel_cls) if dm_channel_cls else False
    chat_type = "dm" if is_dm else "group"
    chat_name = ""
    if not is_dm and hasattr(interaction.channel, "name"):
        chat_name = interaction.channel.name
        if hasattr(interaction.channel, "guild") and interaction.channel.guild:
            chat_name = f"{interaction.channel.guild.name} / #{chat_name}"

    chat_topic = getattr(interaction.channel, "topic", None)
    source = adapter.build_source(
        chat_id=str(interaction.channel_id),
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=str(interaction.user.id),
        user_name=interaction.user.display_name,
        chat_topic=chat_topic,
    )

    msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=source,
        raw_message=interaction,
    )


if DISCORD_AVAILABLE:

    class ExecApprovalView(discord.ui.View):
        """Interactive button view for exec approval of dangerous commands."""

        def __init__(self, approval_id: str, allowed_user_ids: set):
            super().__init__(timeout=300)
            self.approval_id = approval_id
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Verify the user clicking is authorized."""
            if not self.allowed_user_ids:
                return True
            return str(interaction.user.id) in self.allowed_user_ids

        async def _resolve(
            self, interaction: discord.Interaction, action: str, color: discord.Color
        ):
            """Resolve the approval and update the message."""
            if self.resolved:
                await interaction.response.send_message(
                    "This approval has already been resolved~", ephemeral=True
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized to approve commands~", ephemeral=True
                )
                return

            self.resolved = True

            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                embed.color = color
                embed.set_footer(text=f"{action} by {interaction.user.display_name}")

            for child in self.children:
                child.disabled = True

            await interaction.response.edit_message(embed=embed, view=self)

            try:
                from tools.approval import approve_permanent

                if action == "allow_always":
                    approve_permanent(self.approval_id)
            except ImportError:
                pass

        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_once", discord.Color.green())

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_always", discord.Color.blue())

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "deny", discord.Color.red())

        async def on_timeout(self):
            """Handle view timeout -- disable buttons and mark as expired."""
            self.resolved = True
            for child in self.children:
                child.disabled = True

else:  # pragma: no cover - import guard
    ExecApprovalView = None
