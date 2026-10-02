"""Canonical slash command definitions and import-light metadata queries.

Consumers supply configuration gates; discovery is never authorization.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from hermes_constants import INDICATOR_STYLES

@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command."""
    name: str                          # canonical name without slash: "background"
    description: str                   # human-readable description
    category: str                      # "Session", "Configuration", etc.
    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]"
    subcommands: tuple[str, ...] = ()  # tab-completable subcommands
    cli_only: bool = False             # only available in CLI
    gateway_only: bool = False         # only available in gateway/messaging
    gateway_config_gate: str | None = None  # config dotpath; truthy overrides cli_only for gateway
    # Mid-run (agent busy) gateway behavior (gateway/run.py Guard-2 dispatcher): "dispatch" = run
    # while busy (normal handler or the ``busy_handler`` variant); "reject" = refuse mid-run
    # (generic "Agent is running" unless ``busy_handler`` names a reject message);
    # "interrupt_then_dispatch" = interrupt first (/stop, /new, /reset; Guard 1, platforms/base.py).
    busy_policy: str = "reject"
    busy_handler: str | None = None  # key of a special mid-run handler in Guard-2 table
    # Shared executor key (a string, not a callable: keeps this module
    # import-light for the gateway).
    execute: str | None = None
    argument_mode: str | None = None  # desktop composer: options|text|mixed; None inferred
    # Desktop availability: None = offered; "hidden" = runs but out of the popover; else a reason.
    desktop: str | None = None


VALID_BUSY_POLICIES: frozenset[str] = frozenset({"dispatch", "reject", "interrupt_then_dispatch"})


COMMAND_REGISTRY: list[CommandDef] = [
    # Session
    CommandDef("start", "Acknowledge platform start pings without a reply", "Session",
               gateway_only=True, busy_policy="dispatch", busy_handler="start"),
    CommandDef("new", "Start a new session (fresh session ID + history)", "Session",
               aliases=("reset",), args_hint="[name]",
               busy_policy="interrupt_then_dispatch", busy_handler="new"),
    CommandDef("topic", "Enable or inspect Telegram DM topic sessions", "Session",
               gateway_only=True, args_hint="[off|help|session-id]"),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("redraw", "Force a full UI repaint (recovers from terminal drift)", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("history", "Show conversation history", "Session",
               cli_only=True, desktop="terminal"),
    CommandDef("save", "Export the current conversation (bare /save shows usage)", "Session",
               args_hint="<json|md|html> [filename] [redact]"),
    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    CommandDef("prompt", "Compose your next prompt in $EDITOR (markdown), then send it", "Session",
               cli_only=True, args_hint="[initial text]", aliases=("compose",)),
    CommandDef("undo", "Back up N user turns and re-prompt (default 1)", "Session",
               args_hint="[N]"),
    CommandDef("title", "Set a title for the current session", "Session", args_hint="[name]"),
    CommandDef("handoff", "Hand off this session to a messaging platform (Telegram, Discord, etc.)", "Session",
               args_hint="<platform>", cli_only=True, argument_mode="options"),
    CommandDef("branch", "Branch the current session (new thread on Discord/Telegram/Slack/Matrix; --here stays here)",
               "Session", aliases=("fork",), args_hint="[--here] [name]"),
    CommandDef("worktree", "Show, list, create, or prune isolated git worktrees", "Session",
               cli_only=True, args_hint="[new [name]|list|prune [--dry-run]]",
               subcommands=("new", "list", "prune")),
    CommandDef("compress", "Compress conversation context (add 'here [N]' to keep recent N turns; --preview shows what would happen)", "Session",
               aliases=("compact",), args_hint="[here [N] | focus topic | --preview|--dry-run]"),
    CommandDef("rollback", "List or restore filesystem checkpoints (restores keep your hand-edits; --all overrides)", "Session",
               args_hint="[number] [--all]"),
    CommandDef("snapshot", "Create or restore state snapshots of Hermes config/state", "Session",
               cli_only=True, aliases=("snap",), args_hint="[create|restore <id>|prune]",
               desktop="terminal"),
    CommandDef("export", "Export a profile (config, skills, theme) to a shareable archive", "Configuration",
               cli_only=True, args_hint="[profile] [-o output.tar.gz]"),
    CommandDef("import", "Import a shared profile archive as a new profile", "Configuration",
               cli_only=True, args_hint="<archive.tar.gz> [--name <name>]"),
    CommandDef("stop", "Kill all running background processes", "Session",
               busy_policy="interrupt_then_dispatch", busy_handler="stop"),
    CommandDef("pause", "Pause new work globally (emergency stop); '/pause off' resumes", "Session",
               gateway_only=True, args_hint="[reason | off]", busy_policy="dispatch"),
    CommandDef("approve", "Approve a pending dangerous command", "Session",
               gateway_only=True, args_hint="[session|always]", busy_policy="dispatch",
               desktop="messaging"),
    CommandDef("deny", "Deny a pending dangerous command (optionally with a reason)", "Session",
               gateway_only=True, args_hint="[all] [reason]", busy_policy="dispatch",
               desktop="messaging"),
    CommandDef("bg", "Run a prompt in a separate background session", "Session",
               args_hint="<prompt>", busy_policy="dispatch"),
    CommandDef("btw", "Ask a side question about the current conversation without interrupting it", "Session",
               args_hint="<question>", busy_policy="dispatch"),
    CommandDef("agents", "Show active agents and running tasks", "Session",
               aliases=("tasks",), busy_policy="dispatch"),
    CommandDef("journey", "Open the learning journey timeline",
               "Session", aliases=("learning", "memory-graph"), cli_only=True,
               args_hint="[list|delete <id>|edit <id>]", subcommands=("list", "delete", "edit")),
    CommandDef("queue", "Queue a prompt for the next turn, or list/edit/rm/move/clear queued prompts", "Session",
               aliases=("q",), args_hint="[<prompt>|list|edit N <prompt>|rm N|move A B|clear]",
               subcommands=("list", "edit", "rm", "move", "clear", "add"),
               busy_policy="dispatch", busy_handler="queue"),
    CommandDef("steer", "Inject a message after the next tool call without interrupting", "Session",
               aliases=("s",), args_hint="<prompt>", busy_policy="dispatch", busy_handler="steer"),
    CommandDef("goal", "Set a standing goal Hermes works on across turns until achieved", "Session",
               args_hint="[text | draft <text> | show | gate add <cmd> | pause | resume | clear | status | wait <pid> | unwait]",
               argument_mode="mixed", busy_policy="dispatch", busy_handler="goal"),
    CommandDef("heartbeat", "Set a recurring prompt that re-enters this session when idle", "Session",
               aliases=("hb",), args_hint="[every <interval> <prompt> | status | pause | resume | clear]",
               subcommands=("status", "pause", "resume", "clear"),
               busy_policy="dispatch"),
    CommandDef("refine", "Review this conversation now and save lessons to memory/skills", "Session",
               args_hint="[focus instructions]"),
    CommandDef("review", "Spawn an independent subagent to review the work just discussed (PR, code, docs)", "Session",
               args_hint="[review instructions]"),
    CommandDef("loop", "Re-run a prompt on a recurring interval in this session", "Session",
               aliases=("proactive",),
               args_hint="[interval] <prompt> [--times N] [--until <condition>] | status | pause | resume | stop",
               argument_mode="mixed", busy_policy="dispatch", busy_handler="loop"),
    CommandDef("plan", "Write a markdown implementation plan to .hermes/plans/ without executing anything", "Session",
               args_hint="[task]"),
    CommandDef("moa", "Run one prompt through the default Mixture of Agents preset, then restore your model", "Session",
               args_hint="<prompt>", busy_policy="reject", busy_handler="moa"),
    CommandDef("subgoal", "Add or manage extra criteria on the active goal", "Session",
               args_hint="[text | remove N | clear]", busy_policy="dispatch"),
    CommandDef("status", "Show session, model, token, and context info", "Session",
               busy_policy="dispatch"),
    CommandDef("egress", "Show Docker egress proxy status", "Session",
               args_hint="[status]", subcommands=("status",), busy_policy="dispatch",
               busy_handler="egress", execute="egress"),
    CommandDef("context", "Show detailed context window view with usage gauge, category breakdown, compression stats, and throughput", "Session",
               aliases=("ctx",), args_hint="[all]", subcommands=("all",), busy_policy="dispatch"),
    CommandDef("whoami", "Show your slash command access (admin / user)", "Info"),
    CommandDef("profile", "Show active profile name and home directory", "Info",
               busy_policy="dispatch", execute="profile"),
    CommandDef("sethome", "Set this chat as the home channel", "Session",
               gateway_only=True, aliases=("set-home",), desktop="terminal"),
    CommandDef("resume", "Resume a previously-named session", "Session",
               args_hint="[name]", argument_mode="mixed"),
    CommandDef("sessions", "Browse and resume previous sessions", "Session"),

    # Configuration
    CommandDef("config", "Show current configuration", "Configuration",
               cli_only=True, desktop="terminal"),
    CommandDef("model", "Switch model (session-scoped; --global to persist)", "Configuration",
               args_hint="[model] [--provider name] [--reasoning level] [--global|--session] [--refresh]",
               busy_policy="reject", busy_handler="model", desktop="hidden"),
    CommandDef("codex-runtime", "Toggle codex app-server runtime for OpenAI/Codex models",
               "Configuration", aliases=("codex_runtime",), args_hint="[auto|codex_app_server]",
               busy_policy="reject", busy_handler="codex-runtime"),
    CommandDef("personality", "Set a predefined personality", "Configuration",
               args_hint="[name]", argument_mode="options"),
    CommandDef("statusbar", "Toggle the context/model status bar", "Configuration",
               cli_only=True, aliases=("sb",), desktop="terminal"),
    CommandDef("battery", "Toggle a color-coded battery indicator in the status bar",
               "Configuration", cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("timestamps", "Toggle [HH:MM] timestamps on messages and /history", "Configuration",
               cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status"), aliases=("ts",)),
    CommandDef("diff", "Show git changes in the working directory", "Info",
               args_hint="[staged|all|session] [--stat] [path...]",
               subcommands=("staged", "all", "session")),
    CommandDef("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
               "Configuration", cli_only=True, gateway_config_gate="display.tool_progress_command",
               busy_policy="dispatch", desktop="terminal"),
    CommandDef("focus", "Toggle focus view — show only your prompt and the final response",
               "Configuration", cli_only=True, args_hint="[on|off|status]",
               subcommands=("on", "off", "status")),
    CommandDef("footer", "Toggle gateway runtime-metadata footer on final replies",
               "Configuration", args_hint="[on|off|status]", subcommands=("on", "off", "status"),
               busy_policy="dispatch", desktop="terminal"),
    CommandDef("yolo", "Toggle YOLO mode (skip all dangerous command approvals)",
               "Configuration", busy_policy="dispatch"),
    CommandDef("approvals", "Show or set the persistent dangerous-command approval mode",
               "Configuration", args_hint="[manual|smart|off]",
               subcommands=("manual", "smart", "off")),
    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration",
               args_hint="[level|show|hide|full|clamp] [--global]",
               subcommands=("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "show", "hide", "on", "off", "full", "clamp", "--global")),
    CommandDef("fast", "Fast mode — OpenAI Priority Processing / Anthropic Fast Mode (normal/fast/auto/cold)", "Configuration",
               args_hint="[normal|fast|auto|cold|status] [--global]",
               subcommands=("normal", "fast", "auto", "cold", "status", "on", "off", "--global"),
               desktop="advanced"),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration",
               cli_only=True, args_hint="[name]", argument_mode="options"),
    CommandDef("indicator", "Pick the TUI busy-indicator style", "Configuration",
               cli_only=True, args_hint=f"[{'|'.join(INDICATOR_STYLES)}]",
               subcommands=INDICATOR_STYLES, desktop="terminal"),
    CommandDef("voice", "Toggle voice mode", "Configuration",
               args_hint="[on|off|tts|status]", subcommands=("on", "off", "tts", "status"),
               desktop="composer-voice"),
    CommandDef("wake", "Toggle the 'Hey Hermes' wake word listener", "Configuration",
               cli_only=True, args_hint="[on|off|status]", subcommands=("on", "off", "status")),
    CommandDef("busy", "Control how messages behave while Hermes is working", "Configuration",
               args_hint="[queue|steer|interrupt|status]",
               subcommands=("queue", "steer", "interrupt", "status"),
               busy_policy="dispatch", desktop="terminal"),

    # Tools & Skills
    CommandDef("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
               args_hint="[list|disable|enable] [name...]", cli_only=True, argument_mode="options"),
    CommandDef("toolsets", "List available toolsets", "Tools & Skills",
               cli_only=True, desktop="terminal"),
    CommandDef("skills", "Search, install, inspect, or manage skills",
               "Tools & Skills", cli_only=True,
               gateway_config_gate="skills.write_approval",
               subcommands=("search", "browse", "inspect", "install", "audit",
                            "pending", "approve", "reject", "diff", "approval"),
               desktop="settings"),
    CommandDef("memory", "Review pending memory writes / toggle the approval gate",
               "Tools & Skills", args_hint="[pending|approve|reject|approval] [id|on|off]",
               subcommands=("pending", "approve", "reject", "approval")),
    CommandDef("bundles", "List skill bundles (aliases /<name> for multiple skills)",
               "Tools & Skills", execute="bundles"),
    CommandDef("pet", "Toggle or adopt a petdex mascot (/pet, /pet list, /pet <slug>)", "Tools & Skills",
               cli_only=True, args_hint="[toggle|list|scale <n>|<slug>]", subcommands=("toggle", "list", "scale", "off")),
    CommandDef("hatch", "Generate a new petdex pet from a description",
               "Tools & Skills", cli_only=True, aliases=("generate-pet",), args_hint="[description]"),
    CommandDef("learn", "Learn a reusable skill from anything you describe (dirs, URLs, this chat, notes)",
               "Tools & Skills", args_hint="<what to learn from>"),
    CommandDef("init", "Generate or update AGENTS.md project instructions from a repo scan",
               "Tools & Skills", args_hint="[notes]"),
    CommandDef("cron", "Manage scheduled tasks", "Tools & Skills",
               cli_only=True, args_hint="[subcommand]",
               subcommands=("list", "add", "create", "edit", "pause", "resume", "run", "remove"),
               desktop="terminal"),
    CommandDef("suggestions", "Review suggested automations (accept/dismiss)",
               "Tools & Skills", aliases=("suggest",), args_hint="[accept|dismiss N | catalog]",
               subcommands=("accept", "dismiss", "catalog", "clear")),
    CommandDef("blueprint", "Set up an automation from a blueprint template",
               "Tools & Skills", aliases=("bp",), args_hint="[name] [slot=value ...]"),
    CommandDef("curator", "Background skill maintenance (status, run, pin, archive, list-archived)",
               "Tools & Skills", args_hint="[subcommand]",
               subcommands=("status", "run", "pause", "resume", "pin", "unpin", "restore", "list-archived"),
               desktop="advanced"),
    CommandDef("kanban", "Multi-profile collaboration board (tasks, links, comments)",
               "Tools & Skills", args_hint="[subcommand]",
               subcommands=("init", "boards", "create", "list", "ls", "show", "assign",
                            "reclaim", "reassign", "diagnostics", "diag", "link", "unlink",
                            "claim", "comment", "complete", "edit", "block", "unblock",
                            "archive", "tail", "dispatch", "stats", "notify-subscribe",
                            "notify-list", "notify-unsubscribe", "log", "runs",
                            "heartbeat", "assignees", "context", "specify", "gc"),
               busy_policy="dispatch", desktop="advanced"),
    CommandDef("reload", "Reload .env variables into the running session", "Tools & Skills",
               cli_only=True, desktop="terminal"),
    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
               aliases=("reload_mcp",), desktop="advanced"),
    CommandDef("reload-skills", "Re-scan ~/.hermes/skills/ for newly installed or removed skills",
               "Tools & Skills", aliases=("reload_skills",), desktop="advanced"),
    CommandDef("browser", "Connect browser tools to your live Chromium-family browser via CDP, or switch to Browser Use mode", "Tools & Skills",
               cli_only=True, args_hint="[connect|disconnect|status|use]",
               subcommands=("connect", "disconnect", "status", "use")),
    CommandDef("plugins", "List installed plugins and their status",
               "Tools & Skills", cli_only=True, desktop="terminal"),

    # Info
    CommandDef("commands", "Browse all commands and skills (paginated)", "Info",
               gateway_only=True, args_hint="[page]", busy_policy="dispatch",
               execute="gateway_commands"),
    CommandDef("help", "Show available commands (/help skills lists skill commands, /help <text> filters)", "Info", busy_policy="dispatch",
               execute="gateway_help", args_hint="[skills|<filter>]"),
    CommandDef("palette", "Open the fuzzy command palette (also Ctrl+P)", "Info",
               cli_only=True, busy_policy="dispatch"),
    CommandDef("restart", "Gracefully restart the gateway after draining active runs", "Session",
               gateway_only=True, busy_policy="dispatch", desktop="terminal"),
    CommandDef("usage", "Show token usage and rate limits; `reset` redeems a banked Codex limit reset", "Info",
               args_hint="[reset [--force]]"),
    CommandDef("subscription", "View your Nous plan and change it in the browser", "Info",
               cli_only=True, aliases=("upgrade",)),
    CommandDef("login", "Sign in with a Nous account (keeps your connectors)", "Info",
               busy_policy="dispatch", desktop="settings"),
    CommandDef("topup", "Show your Nous balance and manage billing on the portal", "Info"),
    CommandDef("insights", "Show usage insights and analytics", "Info",
               args_hint="[days]", desktop="advanced"),
    CommandDef("platforms", "Show gateway/messaging platform status", "Info",
               cli_only=True, aliases=("gateway",), desktop="terminal"),
    CommandDef("platform", "Pause, resume, or list a failing gateway platform", "Info",
               gateway_only=True, args_hint="<pause|resume|list> [name]"),
    CommandDef("copy", "Copy the last assistant response to clipboard", "Info",
               cli_only=True, args_hint="[number]", desktop="terminal"),
    CommandDef("paste", "Attach clipboard image from your clipboard", "Info",
               cli_only=True, desktop="terminal"),
    CommandDef("image", "Attach a local image file for your next prompt", "Info",
               cli_only=True, args_hint="<path>", desktop="terminal"),
    CommandDef("update", "Update Hermes Agent to the latest version", "Info",
               busy_policy="dispatch", desktop="terminal"),
    CommandDef("version", "Show Hermes Agent version", "Info", aliases=("v",),
               busy_policy="dispatch", execute="version"),
    CommandDef("debug", "Upload debug report (system info + logs) and get shareable links", "Info",
               args_hint="[nous|local]"),

    # Exit
    CommandDef("quit", "Exit the CLI (use --delete to also remove session history)", "Exit",
               cli_only=True, aliases=("exit",), args_hint="[--delete]", desktop="terminal")]


# Distinguishes ``mixed`` (subcommands plus free-text) from ``options``; no subcommands => ``text``.
_PROSE_HINTS = ("<prompt>", "[text", "instructions", "[interval]", "<what")


def infer_argument_mode(cmd: CommandDef) -> str | None:
    """Composer mode: explicit on the CommandDef, else inferred from its args."""
    if cmd.argument_mode in {"options", "text", "mixed"}:
        return cmd.argument_mode
    hint = (cmd.args_hint or "").strip()
    if cmd.subcommands:
        prose = hint and any(token in hint.lower() for token in _PROSE_HINTS)
        return "mixed" if prose else "options"
    return "text" if hint else None


def command_desktop_meta(cmd: CommandDef) -> dict[str, str | None]:
    """Wire shape for ``commands.catalog`` — reads the CommandDef, nothing else."""
    return {"argument_mode": infer_argument_mode(cmd), "desktop": cmd.desktop}


def desktop_surface_registry() -> dict[str, str | None]:
    """``/name`` (and every alias) -> ``desktop`` disposition for EVERY registry command.

    Offered built-ins (no ``desktop=`` value) map to ``None``: the desktop needs their names
    too, or offline it cannot tell ``/context`` from a skill command and files it under
    Skills / routes it down the extension path until the live catalog warms up.

    The desktop app reads this live from ``commands.catalog``; the copy committed at
    ``apps/desktop/src/lib/desktop-slash-registry.json`` (``scripts/dump_desktop_slash_registry.py``)
    is its offline fallback before the catalog answers, so the registry stays the ONLY place a
    command's desktop disposition is authored. A test on each side fails when the two drift.
    """
    return {
        f"/{key}": cmd.desktop or None
        for cmd in COMMAND_REGISTRY
        for key in (cmd.name, *cmd.aliases)
    }


# Every name and alias -> its CommandDef.
_COMMAND_LOOKUP: dict[str, CommandDef] = {
    key: cmd for cmd in COMMAND_REGISTRY for key in (cmd.name, *cmd.aliases)}


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias (leading slash optional) to its CommandDef."""
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))


from plugin_runtime.host_bindings import bind_plugin_host

bind_plugin_host(command_resolver=resolve_command)


# Subcommands lookup: "/cmd" -> ["sub1", ...]; explicit ``subcommands`` first (in
# registry order), then pipe patterns in args_hint ("[on|off|status]") as fallback.
SUBCOMMANDS: dict[str, list[str]] = {
    f"/{_cmd.name}": list(_cmd.subcommands) for _cmd in COMMAND_REGISTRY if _cmd.subcommands}
_PIPE_SUBS_RE = re.compile(r"[a-z]+(?:\|[a-z]+)+")
for _cmd in COMMAND_REGISTRY:
    _m = _PIPE_SUBS_RE.search(_cmd.args_hint) if _cmd.args_hint else None
    if _m and f"/{_cmd.name}" not in SUBCOMMANDS:
        SUBCOMMANDS[f"/{_cmd.name}"] = _m.group(0).split("|")


# All names + aliases the gateway dispatches. Config-gated commands are
# included; their handler checks the gate at runtime.
GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
    name for cmd in COMMAND_REGISTRY if not cmd.cli_only or cmd.gateway_config_gate
    for name in (cmd.name, *cmd.aliases))


def is_gateway_known_command(name: str | None) -> bool:
    """True if ``name`` is a built-in or plugin gateway slash command (plugins looked
    up lazily); decides whether the gateway emits ``command:<name>`` hooks."""
    if not name:
        return False
    return name in GATEWAY_KNOWN_COMMANDS or any(
        plugin_name == name for plugin_name, _d, _h in plugin_command_entries())


# Commands with explicit mid-run handling (busy_policy != "reject"). Kept
# under its historical name for introspection/tests; the real bypass set is
# every resolvable command (see should_bypass_active_session).
ACTIVE_SESSION_BYPASS_COMMANDS: frozenset[str] = frozenset(
    cmd.name for cmd in COMMAND_REGISTRY if cmd.busy_policy != "reject")


def is_interrupt_then_dispatch(command_name: str | None) -> bool:
    """Guard 1 (gateway/platforms/base.py) routes these through the cancel-handoff path."""
    cmd = resolve_command(command_name) if command_name else None
    return cmd is not None and cmd.busy_policy == "interrupt_then_dispatch"


def should_bypass_active_session(command_name: str | None) -> bool:
    """True for any resolvable slash command: every recognized command is dispatched mid-run
    (Guard-2 handler or the "busy" catch-all), never queued — gateway.run's safety net discards
    command text reaching the pending queue, so a queued mid-run /model (or /reasoning, /voice,
    /insights, /title, /resume, /retry, /undo, /compress, /usage, /reload-mcp, /sethome, /reset)
    would silently interrupt the agent AND get discarded — a zero-char response. See issue
    #5057 / PRs #6252, #10370, #4665. ACTIVE_SESSION_BYPASS_COMMANDS remains the subset with
    explicit Level-2 handlers; the rest fall through to the catch-all.

    See #10370, #4665, #5057, #6252.
    """
    return resolve_command(command_name) is not None if command_name else False


def is_gateway_available(cmd: CommandDef, enabled_config_gates: Iterable[str] = ()) -> bool:
    """Surface availability from supplied canonical gate names, never authorization."""
    if not cmd.cli_only:
        return True
    return bool(cmd.gateway_config_gate and cmd.name in enabled_config_gates)


def available_commands(surface: str, *, enabled_config_gates: Iterable[str] = ()) -> tuple[CommandDef, ...]:
    """Registry-order built-ins available on CLI, gateway, or Desktop surfaces.

    Desktop dispositions remain metadata for the caller's presentation/dispatch.
    Gate values must be resolved by the application in its current profile scope.
    """
    gates = frozenset(enabled_config_gates)
    if surface == "gateway":
        return tuple(cmd for cmd in COMMAND_REGISTRY if is_gateway_available(cmd, gates))
    if surface in {"cli", "desktop"}:
        return tuple(cmd for cmd in COMMAND_REGISTRY if not cmd.gateway_only)
    raise ValueError(f"Unknown command surface: {surface}")


def plugin_command_entries() -> list[tuple[str, str, str]]:
    """(name, description, args_hint) for ``PluginContext.register_command`` slash commands.
    Lazy so importing this module never forces plugin discovery."""
    try:
        from plugin_runtime.api import get_plugin_commands
        commands = get_plugin_commands() or {}
    except Exception:
        return []
    return [(name, str(meta.get("description") or f"Run /{name}"),
             str(meta.get("args_hint") or "").strip())
            for name, meta in commands.items() if isinstance(name, str) and isinstance(meta, dict)]


__all__ = [
    "CommandDef", "COMMAND_REGISTRY", "VALID_BUSY_POLICIES", "SUBCOMMANDS",
    "GATEWAY_KNOWN_COMMANDS", "ACTIVE_SESSION_BYPASS_COMMANDS", "resolve_command",
    "available_commands", "is_gateway_available", "command_desktop_meta",
    "desktop_surface_registry", "infer_argument_mode", "is_gateway_known_command",
    "is_interrupt_then_dispatch", "should_bypass_active_session", "plugin_command_entries",
]
