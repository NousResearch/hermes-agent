"""Structured Discord native command registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.discord_impl import command_sessions

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    discord = None
    DISCORD_AVAILABLE = False


@dataclass(frozen=True)
class DiscordChoiceSpec:
    name: str
    value: str


@dataclass(frozen=True)
class DiscordArgSpec:
    name: str
    description: str
    kind: str = "str"  # "str" or "int"
    default: Any = ""
    choices: tuple[DiscordChoiceSpec, ...] = ()


@dataclass(frozen=True)
class DiscordNativeCommandSpec:
    name: str
    description: str
    route: str  # "simple", "dispatch", "thread"
    command_factory: Callable[..., str] | None = None
    args: tuple[DiscordArgSpec, ...] = ()
    followup_msg: Optional[str] = None
    defer_ephemeral: bool = True


def _join_command(base: str, *parts: Any) -> str:
    suffix = " ".join(str(part).strip() for part in parts if str(part).strip())
    return f"{base} {suffix}".strip()


def get_command_specs() -> tuple[DiscordNativeCommandSpec, ...]:
    """Return the structured Discord-native command spec set."""
    voice_choices = (
        DiscordChoiceSpec("channel — join your voice channel", "channel"),
        DiscordChoiceSpec("leave — leave voice channel", "leave"),
        DiscordChoiceSpec("on — voice reply to voice messages", "on"),
        DiscordChoiceSpec("tts — voice reply to all messages", "tts"),
        DiscordChoiceSpec("off — text only", "off"),
        DiscordChoiceSpec("status — show current mode", "status"),
    )

    return (
        DiscordNativeCommandSpec(
            "new",
            "Start a new conversation",
            "simple",
            lambda: "/reset",
            followup_msg="New conversation started~",
        ),
        DiscordNativeCommandSpec(
            "reset",
            "Reset your Hermes session",
            "simple",
            lambda: "/reset",
            followup_msg="Session reset~",
        ),
        DiscordNativeCommandSpec("help", "Show available commands", "simple", lambda: "/help"),
        DiscordNativeCommandSpec(
            "commands",
            "Show the full command catalog",
            "simple",
            lambda: "/commands",
        ),
        DiscordNativeCommandSpec(
            "whoami",
            "Show the Discord identity Hermes sees",
            "simple",
            lambda: "/whoami",
        ),
        DiscordNativeCommandSpec(
            "status",
            "Show Hermes session status",
            "simple",
            lambda: "/status",
            followup_msg="Status sent~",
        ),
        DiscordNativeCommandSpec(
            "model",
            "Show or change the model",
            "simple",
            lambda name="": _join_command("/model", name),
            args=(
                DiscordArgSpec(
                    "name",
                    "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "models",
            "Open the interactive model picker",
            "simple",
            lambda: "/models",
        ),
        DiscordNativeCommandSpec(
            "reasoning",
            "Show or change reasoning effort",
            "dispatch",
            lambda effort="": _join_command("/reasoning", effort),
            args=(
                DiscordArgSpec(
                    "effort",
                    "Reasoning effort: xhigh, high, medium, low, minimal, or none.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "personality",
            "Set a personality",
            "simple",
            lambda name="": _join_command("/personality", name),
            args=(
                DiscordArgSpec(
                    "name",
                    "Personality name. Leave empty to list available.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "retry",
            "Retry your last message",
            "simple",
            lambda: "/retry",
            followup_msg="Retrying~",
        ),
        DiscordNativeCommandSpec("undo", "Remove the last exchange", "simple", lambda: "/undo"),
        DiscordNativeCommandSpec(
            "sethome",
            "Set this chat as the home channel",
            "simple",
            lambda: "/sethome",
        ),
        DiscordNativeCommandSpec(
            "stop",
            "Stop the running Hermes agent",
            "simple",
            lambda: "/stop",
            followup_msg="Stop requested~",
        ),
        DiscordNativeCommandSpec(
            "compress",
            "Compress conversation context",
            "simple",
            lambda: "/compress",
        ),
        DiscordNativeCommandSpec(
            "title",
            "Set or show the session title",
            "simple",
            lambda name="": _join_command("/title", name),
            args=(
                DiscordArgSpec(
                    "name",
                    "Session title. Leave empty to show current.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "resume",
            "Resume a previously-named session",
            "simple",
            lambda name="": _join_command("/resume", name),
            args=(
                DiscordArgSpec(
                    "name",
                    "Session name to resume. Leave empty to list sessions.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "usage",
            "Show token usage for this session",
            "simple",
            lambda: "/usage",
        ),
        DiscordNativeCommandSpec(
            "provider",
            "Show available providers",
            "simple",
            lambda: "/provider",
        ),
        DiscordNativeCommandSpec(
            "insights",
            "Show usage insights and analytics",
            "simple",
            lambda days=7: _join_command("/insights", days),
            args=(
                DiscordArgSpec(
                    "days",
                    "Number of days to analyze (default: 7)",
                    kind="int",
                    default=7,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "reload-mcp",
            "Reload MCP servers from config",
            "simple",
            lambda: "/reload-mcp",
        ),
        DiscordNativeCommandSpec(
            "voice",
            "Toggle voice reply mode",
            "dispatch",
            lambda mode="": _join_command("/voice", mode),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Voice mode: on, off, tts, channel, leave, or status",
                    choices=voice_choices,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "update",
            "Update Hermes Agent to the latest version",
            "simple",
            lambda: "/update",
            followup_msg="Update initiated~",
        ),
        DiscordNativeCommandSpec(
            "thread",
            "Create a new thread and start a Hermes session in it",
            "thread",
            args=(
                DiscordArgSpec("name", "Thread name"),
                DiscordArgSpec(
                    "message",
                    "Optional first message to send to Hermes in the thread",
                    default="",
                ),
                DiscordArgSpec(
                    "auto_archive_duration",
                    "Auto-archive in minutes (60, 1440, 4320, 10080)",
                    kind="int",
                    default=1440,
                ),
            ),
        ),
    )


def extract_inline_shortcut(text: str) -> tuple[str | None, str]:
    """Return the first supported inline shortcut and remaining text."""
    return command_sessions.extract_inline_shortcut(text)


def _register_zero_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    @tree.command(name=spec.name, description=spec.description)
    async def callback(interaction: Any, _spec: DiscordNativeCommandSpec = spec):
        await _dispatch(adapter, interaction, _spec)


def _register_single_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    arg = spec.args[0]

    if arg.choices:
        choices_decorator = discord.app_commands.choices(
            **{
                arg.name: [
                    discord.app_commands.Choice(name=choice.name, value=choice.value)
                    for choice in arg.choices
                ]
            }
        )
    else:
        choices_decorator = lambda fn: fn

    if arg.name == "name":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(name=arg.description)
        @choices_decorator
        async def callback(
            interaction: Any,
            name=arg.default,
            _spec: DiscordNativeCommandSpec = spec,
        ):
            await _dispatch(adapter, interaction, _spec, name=name)
        return

    if arg.name == "effort":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(effort=arg.description)
        @choices_decorator
        async def callback(
            interaction: Any,
            effort=arg.default,
            _spec: DiscordNativeCommandSpec = spec,
        ):
            await _dispatch(adapter, interaction, _spec, effort=effort)
        return

    if arg.name == "days":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(days=arg.description)
        @choices_decorator
        async def callback(
            interaction: Any,
            days: int = arg.default,
            _spec: DiscordNativeCommandSpec = spec,
        ):
            await _dispatch(adapter, interaction, _spec, days=days)
        return

    if arg.name == "mode":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(mode=arg.description)
        @choices_decorator
        async def callback(
            interaction: Any,
            mode=arg.default,
            _spec: DiscordNativeCommandSpec = spec,
        ):
            await _dispatch(adapter, interaction, _spec, mode=mode)
        return

    raise ValueError(f"Unsupported Discord arg registration for {spec.name}:{arg.name}")


def _register_thread_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    @tree.command(name=spec.name, description=spec.description)
    @discord.app_commands.describe(
        name=spec.args[0].description,
        message=spec.args[1].description,
        auto_archive_duration=spec.args[2].description,
    )
    async def callback(
        interaction: Any,
        name: str,
        message: str = "",
        auto_archive_duration: int = 1440,
        _spec: DiscordNativeCommandSpec = spec,
    ):
        await _dispatch(
            adapter,
            interaction,
            _spec,
            name=name,
            message=message,
            auto_archive_duration=auto_archive_duration,
        )


def register_slash_commands(tree: Any, adapter: Any) -> None:
    """Register Discord slash commands from structured specs."""
    if tree is None or not DISCORD_AVAILABLE:
        return

    for spec in get_command_specs():
        if spec.route == "thread":
            _register_thread_command(tree, adapter, spec)
            continue
        if len(spec.args) == 0:
            _register_zero_arg_command(tree, adapter, spec)
            continue
        if len(spec.args) == 1:
            _register_single_arg_command(tree, adapter, spec)
            continue
        raise ValueError(f"Unsupported Discord command spec shape for {spec.name}")


def build_slash_event(adapter: Any, interaction: Any, text: str):
    """Build a slash event via the command-session helper."""
    return command_sessions.build_slash_event(adapter, interaction, text)


async def _dispatch(adapter: Any, interaction: Any, spec: DiscordNativeCommandSpec, **kwargs: Any) -> None:
    if spec.route == "simple":
        command_text = spec.command_factory(**kwargs) if spec.command_factory else f"/{spec.name}"
        await adapter._run_simple_slash(interaction, command_text, spec.followup_msg)
        return

    if spec.route == "dispatch":
        if spec.defer_ephemeral:
            await interaction.response.defer(ephemeral=True)
        command_text = spec.command_factory(**kwargs) if spec.command_factory else f"/{spec.name}"
        event = adapter._build_slash_event(interaction, command_text)
        await adapter.handle_message(event)
        return

    if spec.route == "thread":
        if spec.defer_ephemeral:
            await interaction.response.defer(ephemeral=True)
        await adapter._handle_thread_create_slash(
            interaction,
            kwargs["name"],
            kwargs.get("message", ""),
            kwargs.get("auto_archive_duration", 1440),
        )
        return

    raise ValueError(f"Unsupported Discord command route: {spec.route}")
