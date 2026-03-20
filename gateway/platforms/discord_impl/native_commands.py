"""Structured Discord native command registration and command UX helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from gateway.platforms.discord_impl import components as discord_components
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
    description: Optional[str] = None


@dataclass(frozen=True)
class DiscordArgSpec:
    name: str
    description: str
    kind: str = "str"  # "str" or "int"
    default: Any = ""
    required: bool = False
    choices: tuple[DiscordChoiceSpec, ...] = ()
    prefer_autocomplete: bool = False
    allow_fallback_menu: bool = False


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
    vc_choices = (
        DiscordChoiceSpec("join — join your current voice channel", "join"),
        DiscordChoiceSpec("leave — leave the active voice channel", "leave"),
        DiscordChoiceSpec("status — show current voice channel status", "status"),
    )
    think_choices = (
        DiscordChoiceSpec("off — disable reasoning effort", "off"),
        DiscordChoiceSpec("minimal — minimum reasoning", "minimal"),
        DiscordChoiceSpec("low — lighter reasoning", "low"),
        DiscordChoiceSpec("medium — balanced reasoning", "medium"),
        DiscordChoiceSpec("high — deeper reasoning", "high"),
        DiscordChoiceSpec("xhigh — maximum reasoning", "xhigh"),
    )
    reasoning_choices = (
        DiscordChoiceSpec("off — hide reasoning blocks", "off"),
        DiscordChoiceSpec("on — show reasoning blocks", "on"),
        DiscordChoiceSpec("hide — hide reasoning blocks", "hide"),
        DiscordChoiceSpec("show — show reasoning blocks", "show"),
        DiscordChoiceSpec("none — disable reasoning effort", "none"),
        DiscordChoiceSpec("minimal — minimum reasoning", "minimal"),
        DiscordChoiceSpec("low — lighter reasoning", "low"),
        DiscordChoiceSpec("medium — balanced reasoning", "medium"),
        DiscordChoiceSpec("high — deeper reasoning", "high"),
        DiscordChoiceSpec("xhigh — maximum reasoning", "xhigh"),
    )
    approval_choices = (
        DiscordChoiceSpec("allow-once — run this command once", "allow-once"),
        DiscordChoiceSpec("allow-always — permanently allow this pattern", "allow-always"),
        DiscordChoiceSpec("deny — reject this command", "deny"),
    )
    send_choices = (
        DiscordChoiceSpec("on — allow send_message for this session", "on"),
        DiscordChoiceSpec("off — block send_message for this session", "off"),
        DiscordChoiceSpec("inherit — use the default behavior", "inherit"),
    )
    activation_choices = (
        DiscordChoiceSpec("mention — require an explicit mention", "mention"),
        DiscordChoiceSpec("always — respond without a mention", "always"),
    )
    session_choices = (
        DiscordChoiceSpec("idle — inactivity auto-unfocus window", "idle"),
        DiscordChoiceSpec("max-age — hard focus lifetime", "max-age"),
        DiscordChoiceSpec("status — show current focus binding", "status"),
    )
    context_choices = (
        DiscordChoiceSpec("list — concise context summary", "list"),
        DiscordChoiceSpec("detail — full context breakdown", "detail"),
        DiscordChoiceSpec("json — machine-readable snapshot", "json"),
    )
    allowlist_choices = (
        DiscordChoiceSpec("list — show approved command patterns", "list"),
        DiscordChoiceSpec("add — permanently approve a pattern", "add"),
        DiscordChoiceSpec("remove — remove a stored pattern", "remove"),
    )
    config_choices = (
        DiscordChoiceSpec("show — show the current on-disk config", "show"),
        DiscordChoiceSpec("get — read one config or env key", "get"),
        DiscordChoiceSpec("set — write one config or env key", "set"),
        DiscordChoiceSpec("unset — remove one config or env key", "unset"),
    )
    debug_choices = (
        DiscordChoiceSpec("show — show runtime override status", "show"),
        DiscordChoiceSpec("set — set a runtime override", "set"),
        DiscordChoiceSpec("unset — remove a runtime override", "unset"),
        DiscordChoiceSpec("reset — clear all runtime overrides", "reset"),
    )
    subagent_choices = (
        DiscordChoiceSpec("list — list sub-agent runs", "list"),
        DiscordChoiceSpec("kill — kill a sub-agent run", "kill"),
        DiscordChoiceSpec("log — show sub-agent logs", "log"),
        DiscordChoiceSpec("info — inspect a sub-agent run", "info"),
        DiscordChoiceSpec("send — send input to a sub-agent", "send"),
        DiscordChoiceSpec("steer — steer a sub-agent", "steer"),
        DiscordChoiceSpec("spawn — spawn a sub-agent", "spawn"),
    )
    acp_choices = (
        DiscordChoiceSpec("spawn — create an ACP session", "spawn"),
        DiscordChoiceSpec("cancel — cancel ACP work", "cancel"),
        DiscordChoiceSpec("steer — steer ACP work", "steer"),
        DiscordChoiceSpec("close — close an ACP session", "close"),
        DiscordChoiceSpec("status — show ACP status", "status"),
        DiscordChoiceSpec("set-mode — update ACP mode", "set-mode"),
        DiscordChoiceSpec("set — set ACP runtime options", "set"),
        DiscordChoiceSpec("cwd — change ACP cwd", "cwd"),
        DiscordChoiceSpec("permissions — change ACP permissions", "permissions"),
        DiscordChoiceSpec("timeout — change ACP timeout", "timeout"),
        DiscordChoiceSpec("model — change ACP model", "model"),
        DiscordChoiceSpec("reset-options — clear ACP options", "reset-options"),
        DiscordChoiceSpec("doctor — inspect ACP health", "doctor"),
        DiscordChoiceSpec("install — install ACP tooling", "install"),
        DiscordChoiceSpec("sessions — list ACP sessions", "sessions"),
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
            "context",
            "Show context summary, detail, or JSON",
            "simple",
            lambda mode="": _join_command("/context", mode),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Context mode: list, detail, or json",
                    choices=context_choices,
                    default="list",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "export-session",
            "Export the current session snapshot",
            "simple",
            lambda path="": _join_command("/export-session", path),
            args=(
                DiscordArgSpec(
                    "path",
                    "Optional export path (.html or .json). Leave empty for the default HTML export.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "export",
            "Alias for /export-session",
            "simple",
            lambda path="": _join_command("/export", path),
            args=(
                DiscordArgSpec(
                    "path",
                    "Optional export path (.html or .json). Leave empty for the default HTML export.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "whoami",
            "Show the Discord identity Hermes sees",
            "simple",
            lambda: "/whoami",
        ),
        DiscordNativeCommandSpec(
            "focus",
            "Bind the current Discord thread to this Hermes session",
            "dispatch",
            lambda name="": _join_command("/focus", name),
            args=(
                DiscordArgSpec(
                    "name",
                    "Optional label for the current thread binding",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "unfocus",
            "Remove the current Discord thread binding",
            "simple",
            lambda: "/unfocus",
        ),
        DiscordNativeCommandSpec(
            "agents",
            "Show Discord thread bindings for this session",
            "simple",
            lambda: "/agents",
        ),
        DiscordNativeCommandSpec(
            "session",
            "Manage thread binding idle and max-age controls",
            "dispatch",
            lambda mode="", value="": _join_command("/session", mode, value),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Thread binding control: idle, max-age, or status",
                    choices=session_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "value",
                    "Duration like 30m, 2h, 1d, or off",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "id",
            "Alias for /whoami",
            "simple",
            lambda: "/id",
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
                    choices=reasoning_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "think",
            "Set reasoning effort quickly",
            "dispatch",
            lambda effort="": _join_command("/think", effort),
            args=(
                DiscordArgSpec(
                    "effort",
                    "Thinking effort: off, minimal, low, medium, high, or xhigh.",
                    choices=think_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
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
            "compact",
            "Compress conversation context with optional instructions",
            "simple",
            lambda instructions="": _join_command("/compact", instructions),
            args=(
                DiscordArgSpec(
                    "instructions",
                    "Optional compression guidance for what to preserve.",
                    default="",
                ),
            ),
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
            "skill",
            "Run a skill by name",
            "dispatch",
            lambda name="", input="": _join_command("/skill", name, input),
            args=(
                DiscordArgSpec(
                    "name",
                    "Skill name to invoke",
                    default="",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "input",
                    "Optional user input for the skill",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "subagents",
            "Inspect or control sub-agent runs",
            "simple",
            lambda mode="", payload="": _join_command("/subagents", mode, payload),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Sub-agent action: list, kill, log, info, send, steer, or spawn",
                    choices=subagent_choices,
                    default="list",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "payload",
                    "Optional sub-agent command arguments",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "kill",
            "Abort a running sub-agent",
            "simple",
            lambda target="": _join_command("/kill", target),
            args=(
                DiscordArgSpec(
                    "target",
                    "Target sub-agent id, ordinal, or all",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "steer",
            "Steer a running sub-agent",
            "simple",
            lambda target="", message="": _join_command("/steer", target, message),
            args=(
                DiscordArgSpec(
                    "target",
                    "Target sub-agent id or ordinal",
                    default="",
                ),
                DiscordArgSpec(
                    "message",
                    "Steer message",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "tell",
            "Alias for /steer",
            "simple",
            lambda target="", message="": _join_command("/tell", target, message),
            args=(
                DiscordArgSpec(
                    "target",
                    "Target sub-agent id or ordinal",
                    default="",
                ),
                DiscordArgSpec(
                    "message",
                    "Steer message",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "acp",
            "Inspect or control ACP runtime sessions",
            "simple",
            lambda mode="", payload="": _join_command("/acp", mode, payload),
            args=(
                DiscordArgSpec(
                    "mode",
                    "ACP action to run",
                    choices=acp_choices,
                    default="status",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "payload",
                    "Optional ACP command arguments",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "bash",
            "Run a host shell command through gateway control",
            "simple",
            lambda command="": _join_command("/bash", command),
            args=(
                DiscordArgSpec(
                    "command",
                    "Host shell command to run",
                    default="",
                ),
            ),
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
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "vc",
            "Join, leave, or inspect Discord voice channel state",
            "dispatch",
            lambda mode="": _join_command("/vc", mode),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Voice channel command: join, leave, or status",
                    choices=vc_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "approve",
            "Resolve a pending command approval",
            "dispatch",
            lambda decision="", approval_id="": _join_command("/approve", approval_id, decision),
            args=(
                DiscordArgSpec(
                    "decision",
                    "Approval decision: allow-once, allow-always, or deny",
                    choices=approval_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "approval_id",
                    "Approval ID shown in the approval prompt footer. Optional for the current chat.",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "allowlist",
            "Inspect or edit the command allowlist",
            "simple",
            lambda mode="", entry="": _join_command("/allowlist", mode, entry),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Allowlist action: list, add, or remove",
                    choices=allowlist_choices,
                    default="list",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "entry",
                    "Pattern key to add or remove. Leave empty for list.",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "config",
            "Show or update Hermes configuration",
            "simple",
            lambda mode="", key="", value="": _join_command("/config", mode, key, value),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Config action: show, get, set, or unset",
                    choices=config_choices,
                    default="show",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "key",
                    "Dotted config path or env var name",
                    default="",
                ),
                DiscordArgSpec(
                    "value",
                    "Value to set. YAML scalars and JSON-style lists work.",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "debug",
            "Inspect runtime override parity status",
            "simple",
            lambda mode="", key="", value="": _join_command("/debug", mode, key, value),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Debug action: show, set, unset, or reset",
                    choices=debug_choices,
                    default="show",
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
                DiscordArgSpec(
                    "key",
                    "Runtime override key",
                    default="",
                ),
                DiscordArgSpec(
                    "value",
                    "Runtime override value",
                    default="",
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "send",
            "Control whether this session may use send_message",
            "dispatch",
            lambda mode="": _join_command("/send", mode),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Send policy: on, off, or inherit",
                    choices=send_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
                ),
            ),
        ),
        DiscordNativeCommandSpec(
            "activation",
            "Control mention-vs-always activation for this Discord chat",
            "dispatch",
            lambda mode="": _join_command("/activation", mode),
            args=(
                DiscordArgSpec(
                    "mode",
                    "Activation mode: mention or always",
                    choices=activation_choices,
                    prefer_autocomplete=True,
                    allow_fallback_menu=True,
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
            "restart",
            "Restart the Hermes gateway",
            "simple",
            lambda: "/restart",
            followup_msg="Restart scheduled~",
        ),
        DiscordNativeCommandSpec(
            "dock-telegram",
            "Dock replies for this session to the Telegram home channel",
            "simple",
            lambda: "/dock-telegram",
        ),
        DiscordNativeCommandSpec(
            "dock-discord",
            "Dock replies for this session to the Discord home channel",
            "simple",
            lambda: "/dock-discord",
        ),
        DiscordNativeCommandSpec(
            "dock-slack",
            "Dock replies for this session to the Slack home channel",
            "simple",
            lambda: "/dock-slack",
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


def _resolve_arg_choices(
    adapter: Any,
    spec: DiscordNativeCommandSpec,
    arg: DiscordArgSpec,
    *,
    interaction: Any | None = None,
    current_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[DiscordChoiceSpec, ...]:
    del adapter, interaction, current_kwargs
    if spec.name == "skill" and arg.name == "name":
        try:
            from agent.skill_commands import get_skill_commands

            choices: list[DiscordChoiceSpec] = []
            for command_name, payload in sorted(get_skill_commands().items()):
                value = command_name.lstrip("/")
                description = str(payload.get("description") or "Skill command")
                choices.append(DiscordChoiceSpec(value, value, description[:100] or None))
            return tuple(choices)
        except Exception:
            return ()
    return arg.choices


def _format_choice_label(label: str, *, limit: int = 80) -> str:
    if len(label) <= limit:
        return label
    return label[: limit - 3] + "..."


async def _send_or_edit_interaction_view(interaction: Any, content: str, view: Any) -> None:
    response = getattr(interaction, "response", None)
    if response is not None and hasattr(response, "edit_message") and getattr(interaction, "message", None) is not None:
        try:
            await response.edit_message(content=content, view=view)
            return
        except Exception:
            pass
    if response is not None and hasattr(response, "send_message"):
        is_done = getattr(response, "is_done", None)
        if not callable(is_done) or not is_done():
            await response.send_message(content, ephemeral=True, view=view)
            return
    followup = getattr(interaction, "followup", None)
    if followup is not None and hasattr(followup, "send"):
        await followup.send(content, ephemeral=True, view=view)


async def _open_arg_fallback(
    adapter: Any,
    interaction: Any,
    spec: DiscordNativeCommandSpec,
    arg: DiscordArgSpec,
    current_kwargs: dict[str, Any],
) -> bool:
    choices = _resolve_arg_choices(adapter, spec, arg, interaction=interaction, current_kwargs=current_kwargs)
    if not choices or not arg.allow_fallback_menu:
        return False

    runtime = getattr(adapter, "_component_runtime", None)
    if runtime is None or getattr(discord_components, "ManagedComponentView", None) is None:
        return False

    view = discord_components.ManagedComponentView(runtime, timeout=300)
    allowed_user_id = str(getattr(getattr(interaction, "user", None), "id", "") or "")
    prompt = f"Choose `{arg.name}` for `/{spec.name}`."

    async def _choose(invocation: discord_components.DiscordComponentInvocation, value: str) -> bool:
        next_kwargs = dict(current_kwargs)
        next_kwargs[arg.name] = value
        await _dispatch(adapter, invocation.interaction, spec, **next_kwargs)
        return True

    if len(choices) <= 5:
        for choice in choices:
            view.add_button(
                discord_components.DiscordButtonSpec(
                    label=_format_choice_label(choice.name, limit=40),
                    style="primary",
                    allowed_user_ids=(allowed_user_id,),
                    handler=lambda invocation, value=choice.value: _choose(invocation, value),
                )
            )
    else:
        view.add_select(
            discord_components.DiscordSelectSpec(
                select_type="string",
                placeholder=arg.description,
                options=tuple(
                    discord_components.DiscordSelectOptionSpec(
                        label=_format_choice_label(choice.name, limit=100),
                        value=choice.value,
                        description=choice.description,
                    )
                    for choice in choices[:25]
                ),
                allowed_user_ids=(allowed_user_id,),
                handler=lambda invocation: _choose(invocation, invocation.values[0]),
            )
        )

    await _send_or_edit_interaction_view(interaction, prompt, view)
    return True


async def _autocomplete_choices(
    adapter: Any,
    spec: DiscordNativeCommandSpec,
    arg: DiscordArgSpec,
    interaction: Any,
    current: str,
) -> list[Any]:
    choices = _resolve_arg_choices(adapter, spec, arg, interaction=interaction)
    query = str(current or "").strip().lower()
    filtered = [
        choice
        for choice in choices
        if not query
        or query in choice.name.lower()
        or query in choice.value.lower()
    ]
    return [
        discord.app_commands.Choice(name=choice.name, value=choice.value)
        for choice in filtered[:25]
    ]


def _build_choices_decorator(
    adapter: Any,
    spec: DiscordNativeCommandSpec,
    arg: DiscordArgSpec,
):
    choices = _resolve_arg_choices(adapter, spec, arg)
    use_autocomplete = bool(arg.prefer_autocomplete and choices)
    if use_autocomplete:
        autocomplete_factory = getattr(discord.app_commands, "autocomplete", None)
        if autocomplete_factory is None:
            return (lambda fn: fn), None

        async def autocomplete_callback(interaction, current):
            return await _autocomplete_choices(
                adapter,
                spec,
                arg,
                interaction,
                current,
            )

        return None, discord.app_commands.autocomplete(
            **{arg.name: autocomplete_callback}
        )

    if choices:
        return discord.app_commands.choices(
            **{
                arg.name: [
                    discord.app_commands.Choice(name=choice.name, value=choice.value)
                    for choice in choices[:25]
                ]
            }
        ), None

    identity = lambda fn: fn
    return identity, None


def _register_zero_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    @tree.command(name=spec.name, description=spec.description)
    async def callback(interaction: Any):
        await _dispatch(adapter, interaction, spec)


def _register_single_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    arg = spec.args[0]
    choices_decorator, autocomplete_decorator = _build_choices_decorator(adapter, spec, arg)
    if choices_decorator is None:
        choices_decorator = lambda fn: fn
    if autocomplete_decorator is None:
        autocomplete_decorator = lambda fn: fn

    if arg.name == "name":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(name=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            name: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, name=name)
        return

    if arg.name == "effort":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(effort=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            effort: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, effort=effort)
        return

    if arg.name == "days":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(days=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            days: int = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, days=days)
        return

    if arg.name == "mode":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(mode=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            mode: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, mode=mode)
        return

    if arg.name == "path":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(path=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            path: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, path=path)
        return

    if arg.name == "instructions":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(instructions=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            instructions: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, instructions=instructions)
        return

    if arg.name == "target":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(target=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            target: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, target=target)
        return

    if arg.name == "command":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(command=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            command: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, command=command)
        return

    if arg.name == "decision":
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(decision=arg.description)
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            decision: str = arg.default,
        ):
            await _dispatch(adapter, interaction, spec, decision=decision)
        return

    raise ValueError(f"Unsupported Discord arg registration for {spec.name}:{arg.name}")


def _register_double_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    first, second = spec.args
    choices_decorator, autocomplete_decorator = _build_choices_decorator(adapter, spec, first)
    if choices_decorator is None:
        choices_decorator = lambda fn: fn
    if autocomplete_decorator is None:
        autocomplete_decorator = lambda fn: fn

    if (first.name, second.name) == ("decision", "approval_id"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            decision=first.description,
            approval_id=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            decision: str = first.default,
            approval_id: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                decision=decision,
                approval_id=approval_id,
            )
        return

    if (first.name, second.name) == ("mode", "value"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            mode=first.description,
            value=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            mode: str = first.default,
            value: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                mode=mode,
                value=value,
            )
        return

    if (first.name, second.name) == ("mode", "entry"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            mode=first.description,
            entry=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            mode: str = first.default,
            entry: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                mode=mode,
                entry=entry,
            )
        return

    if (first.name, second.name) == ("name", "input"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            name=first.description,
            input=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            name: str = first.default,
            input: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                name=name,
                input=input,
            )
        return

    if (first.name, second.name) == ("mode", "payload"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            mode=first.description,
            payload=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            mode: str = first.default,
            payload: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                mode=mode,
                payload=payload,
            )
        return

    if (first.name, second.name) == ("target", "message"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            target=first.description,
            message=second.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            target: str = first.default,
            message: str = second.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                target=target,
                message=message,
            )
        return

    raise ValueError(
        f"Unsupported Discord command spec shape for {spec.name}:{first.name},{second.name}"
    )


def _register_triple_arg_command(tree: Any, adapter: Any, spec: DiscordNativeCommandSpec) -> None:
    first, second, third = spec.args
    choices_decorator, autocomplete_decorator = _build_choices_decorator(adapter, spec, first)
    if choices_decorator is None:
        choices_decorator = lambda fn: fn
    if autocomplete_decorator is None:
        autocomplete_decorator = lambda fn: fn

    if (first.name, second.name, third.name) == ("mode", "key", "value"):
        @tree.command(name=spec.name, description=spec.description)
        @discord.app_commands.describe(
            mode=first.description,
            key=second.description,
            value=third.description,
        )
        @autocomplete_decorator
        @choices_decorator
        async def callback(
            interaction: Any,
            mode: str = first.default,
            key: str = second.default,
            value: str = third.default,
        ):
            await _dispatch(
                adapter,
                interaction,
                spec,
                mode=mode,
                key=key,
                value=value,
            )
        return

    raise ValueError(
        f"Unsupported Discord command spec shape for {spec.name}:{first.name},{second.name},{third.name}"
    )


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
    ):
        await _dispatch(
            adapter,
            interaction,
            spec,
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
        if len(spec.args) == 2:
            _register_double_arg_command(tree, adapter, spec)
            continue
        if len(spec.args) == 3:
            _register_triple_arg_command(tree, adapter, spec)
            continue
        raise ValueError(f"Unsupported Discord command spec shape for {spec.name}")


def build_slash_event(adapter: Any, interaction: Any, text: str):
    """Build a slash event via the command-session helper."""
    return command_sessions.build_slash_event(adapter, interaction, text)


async def _dispatch(adapter: Any, interaction: Any, spec: DiscordNativeCommandSpec, **kwargs: Any) -> None:
    missing_choice_arg = next(
        (
            arg
            for arg in spec.args
            if arg.allow_fallback_menu
            and not str(kwargs.get(arg.name, "") or "").strip()
            and _resolve_arg_choices(adapter, spec, arg, interaction=interaction, current_kwargs=kwargs)
        ),
        None,
    )
    if missing_choice_arg is not None:
        opened = await _open_arg_fallback(
            adapter,
            interaction,
            spec,
            missing_choice_arg,
            kwargs,
        )
        if opened:
            return

    if spec.route == "simple":
        command_text = spec.command_factory(**kwargs) if spec.command_factory else f"/{spec.name}"
        await adapter._run_simple_slash(interaction, command_text, spec.followup_msg)
        return

    if spec.route == "dispatch":
        if spec.defer_ephemeral:
            await interaction.response.defer(ephemeral=True)
        command_text = spec.command_factory(**kwargs) if spec.command_factory else f"/{spec.name}"
        response = await adapter._invoke_native_slash_command(interaction, command_text)
        if response:
            await adapter._send_native_slash_content(interaction, response)
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
