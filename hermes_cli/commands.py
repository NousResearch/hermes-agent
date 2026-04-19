     1|"""Slash command definitions and autocomplete for the Hermes CLI.
     2|
     3|Central registry for all slash commands. Every consumer -- CLI help, gateway
     4|dispatch, Telegram BotCommands, Slack subcommand mapping, autocomplete --
     5|derives its data from ``COMMAND_REGISTRY``.
     6|
     7|To add a command: add a ``CommandDef`` entry to ``COMMAND_REGISTRY``.
     8|To add an alias: set ``aliases=("short",)`` on the existing ``CommandDef``.
     9|"""
    10|
    11|from __future__ import annotations
    12|
    13|import os
    14|import re
    15|import shutil
    16|import subprocess
    17|import time
    18|from collections.abc import Callable, Mapping
    19|from dataclasses import dataclass
    20|from typing import Any
    21|
    22|# prompt_toolkit is an optional CLI dependency — only needed for
    23|# SlashCommandCompleter and SlashCommandAutoSuggest.  Gateway and test
    24|# environments that lack it must still be able to import this module
    25|# for resolve_command, gateway_help_lines, and COMMAND_REGISTRY.
    26|try:
    27|    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    28|    from prompt_toolkit.completion import Completer, Completion
    29|except ImportError:  # pragma: no cover
    30|    AutoSuggest = object  # type: ignore[assignment,misc]
    31|    Completer = object    # type: ignore[assignment,misc]
    32|    Suggestion = None     # type: ignore[assignment]
    33|    Completion = None     # type: ignore[assignment]
    34|
    35|
    36|# ---------------------------------------------------------------------------
    37|# CommandDef dataclass
    38|# ---------------------------------------------------------------------------
    39|
    40|@dataclass(frozen=True)
    41|class CommandDef:
    42|    """Definition of a single slash command."""
    43|
    44|    name: str                          # canonical name without slash: "background"
    45|    description: str                   # human-readable description
    46|    category: str                      # "Session", "Configuration", etc.
    47|    aliases: tuple[str, ...] = ()      # alternative names: ("bg",)
    48|    args_hint: str = ""                # argument placeholder: "<prompt>", "[name]"
    49|    subcommands: tuple[str, ...] = ()  # tab-completable subcommands
    50|    cli_only: bool = False             # only available in CLI
    51|    gateway_only: bool = False         # only available in gateway/messaging
    52|    gateway_config_gate: str | None = None  # config dotpath; when truthy, overrides cli_only for gateway
    53|
    54|
    55|# ---------------------------------------------------------------------------
    56|# Central registry -- single source of truth
    57|# ---------------------------------------------------------------------------
    58|
    59|COMMAND_REGISTRY: list[CommandDef] = [
    60|    # Session
    61|    CommandDef("new", "Start a new session (fresh session ID + history)", "Session",
    62|               aliases=("reset",)),
    63|    CommandDef("clear", "Clear screen and start a new session", "Session",
    64|               cli_only=True),
    65|    CommandDef("history", "Show conversation history", "Session",
    66|               cli_only=True),
    67|    CommandDef("save", "Save the current conversation", "Session",
    68|               cli_only=True),
    69|    CommandDef("retry", "Retry the last message (resend to agent)", "Session"),
    70|    CommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    71|    CommandDef("title", "Set a title for the current session", "Session",
    72|               args_hint="[name]"),
    73|    CommandDef("branch", "Branch the current session (explore a different path)", "Session",
    74|               aliases=("fork",), args_hint="[name]"),
    75|    CommandDef("compress", "Manually compress conversation context", "Session",
    76|               args_hint="[focus topic]"),
    77|    CommandDef("rollback", "List or restore filesystem checkpoints", "Session",
    78|               args_hint="[number]"),
    79|    CommandDef("snapshot", "Create or restore state snapshots of Hermes config/state", "Session",
    80|               aliases=("snap",), args_hint="[create|restore <id>|prune]"),
    81|    CommandDef("stop", "Kill all running background processes", "Session"),
    82|    CommandDef("approve", "Approve a pending dangerous command", "Session",
    83|               gateway_only=True, args_hint="[session|always]"),
    84|    CommandDef("deny", "Deny a pending dangerous command", "Session",
    85|               gateway_only=True),
    86|    CommandDef("background", "Run a prompt in the background", "Session",
    87|               aliases=("bg",), args_hint="<prompt>"),
    88|    CommandDef("btw", "Ephemeral side question using session context (no tools, not persisted)", "Session",
    89|               args_hint="<question>"),
    90|    CommandDef("agents", "Show active agents and running tasks", "Session",
    91|               aliases=("tasks",)),
    92|    CommandDef("queue", "Queue a prompt for the next turn (doesn't interrupt)", "Session",
    93|               aliases=("q",), args_hint="<prompt>"),
    94|    CommandDef("steer", "Inject a message after the next tool call without interrupting", "Session",
    95|               args_hint="<prompt>"),
    96|    CommandDef("status", "Show session info", "Session"),
    97|    CommandDef("profile", "Show active profile name and home directory", "Info"),
    98|    CommandDef("sethome", "Set this chat as the home channel", "Session",
    99|               gateway_only=True, aliases=("set-home",)),
   100|    CommandDef("resume", "Resume a previously-named session", "Session",
   101|               args_hint="[name]"),
   102|
   103|    # Configuration
   104|    CommandDef("config", "Show current configuration", "Configuration",
   105|               cli_only=True),
   106|    CommandDef("model", "Switch model for this session", "Configuration", args_hint="[model] [--provider name] [--global]"),
   107|    CommandDef("provider", "Show available providers and current provider",
   108|               "Configuration"),
   109|    CommandDef("gquota", "Show Google Gemini Code Assist quota usage", "Info"),
   110|
   111|    CommandDef("personality", "Set a predefined personality", "Configuration",
   112|               args_hint="[name]"),
   113|    CommandDef("statusbar", "Toggle the context/model status bar", "Configuration",
   114|               cli_only=True, aliases=("sb",)),
   115|    CommandDef("verbose", "Cycle tool progress display: off -> new -> all -> verbose",
   116|               "Configuration", cli_only=True,
   117|               gateway_config_gate="display.tool_progress_command"),
   118|    CommandDef("yolo", "Toggle YOLO mode (skip all dangerous command approvals)",
   119|               "Configuration"),
   120|    CommandDef("reasoning", "Manage reasoning effort and display", "Configuration",
   121|               args_hint="[level|show|hide]",
   122|               subcommands=("none", "minimal", "low", "medium", "high", "xhigh", "show", "hide", "on", "off")),
   123|    CommandDef("fast", "Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode (Normal/Fast)", "Configuration",
   124|               args_hint="[normal|fast|status]",
   125|               subcommands=("normal", "fast", "status", "on", "off")),
    CommandDef("skin", "Show or change the display skin/theme", "Configuration", cli_only=True, args_hint="[name]"),
    CommandDef("markdown", "Toggle markdown rendering for responses", "Configuration", cli_only=True, aliases=("md",), args_hint="[on|off]", subcommands=("on", "off")),
   135|    CommandDef("voice", "Toggle voice mode", "Configuration",
   136|               args_hint="[on|off|tts|status]", subcommands=("on", "off", "tts", "status")),
   137|
   138|    # Tools & Skills
   139|    CommandDef("tools", "Manage tools: /tools [list|disable|enable] [name...]", "Tools & Skills",
   140|               args_hint="[list|disable|enable] [name...]", cli_only=True),
   141|    CommandDef("toolsets", "List available toolsets", "Tools & Skills",
   142|               cli_only=True),
   143|    CommandDef("skills", "Search, install, inspect, or manage skills",
   144|               "Tools & Skills", cli_only=True,
   145|               subcommands=("search", "browse", "inspect", "install")),
   146|    CommandDef("cron", "Manage scheduled tasks", "Tools & Skills",
   147|               cli_only=True, args_hint="[subcommand]",
   148|               subcommands=("list", "add", "create", "edit", "pause", "resume", "run", "remove")),
   149|    CommandDef("reload", "Reload .env variables into the running session", "Tools & Skills"),
   150|    CommandDef("reload-mcp", "Reload MCP servers from config", "Tools & Skills",
   151|               aliases=("reload_mcp",)),
   152|    CommandDef("browser", "Connect browser tools to your live Chrome via CDP", "Tools & Skills",
   153|               cli_only=True, args_hint="[connect|disconnect|status]",
   154|               subcommands=("connect", "disconnect", "status")),
   155|    CommandDef("plugins", "List installed plugins and their status",
   156|               "Tools & Skills", cli_only=True),
   157|
   158|    # Info
   159|    CommandDef("commands", "Browse all commands and skills (paginated)", "Info",
   160|               gateway_only=True, args_hint="[page]"),
   161|    CommandDef("help", "Show available commands", "Info"),
   162|    CommandDef("restart", "Gracefully restart the gateway after draining active runs", "Session",
   163|               gateway_only=True),
   164|    CommandDef("usage", "Show token usage and rate limits for the current session", "Info"),
   165|    CommandDef("insights", "Show usage insights and analytics", "Info",
   166|               args_hint="[days]"),
   167|    CommandDef("platforms", "Show gateway/messaging platform status", "Info",
   168|               cli_only=True, aliases=("gateway",)),
   169|    CommandDef("copy", "Copy the last assistant response to clipboard", "Info",
   170|               cli_only=True, args_hint="[number]"),
   171|    CommandDef("paste", "Attach clipboard image from your clipboard", "Info",
   172|               cli_only=True),
   173|    CommandDef("image", "Attach a local image file for your next prompt", "Info",
   174|               cli_only=True, args_hint="<path>"),
   175|    CommandDef("update", "Update Hermes Agent to the latest version", "Info",
   176|               gateway_only=True),
   177|    CommandDef("debug", "Upload debug report (system info + logs) and get shareable links", "Info"),
   178|
   179|    # Exit
   180|    CommandDef("quit", "Exit the CLI", "Exit",
   181|               cli_only=True, aliases=("exit",)),
   182|]
   183|
   184|
   185|# ---------------------------------------------------------------------------
   186|# Derived lookups -- rebuilt once at import time, refreshed by rebuild_lookups()
   187|# ---------------------------------------------------------------------------
   188|
   189|def _build_command_lookup() -> dict[str, CommandDef]:
   190|    """Map every name and alias to its CommandDef."""
   191|    lookup: dict[str, CommandDef] = {}
   192|    for cmd in COMMAND_REGISTRY:
   193|        lookup[cmd.name] = cmd
   194|        for alias in cmd.aliases:
   195|            lookup[alias] = cmd
   196|    return lookup
   197|
   198|
   199|_COMMAND_LOOKUP: dict[str, CommandDef] = _build_command_lookup()
   200|
   201|
   202|def resolve_command(name: str) -> CommandDef | None:
   203|    """Resolve a command name or alias to its CommandDef.
   204|
   205|    Accepts names with or without the leading slash.
   206|    """
   207|    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))
   208|
   209|
   210|def _build_description(cmd: CommandDef) -> str:
   211|    """Build a CLI-facing description string including usage hint."""
   212|    if cmd.args_hint:
   213|        return f"{cmd.description} (usage: /{cmd.name} {cmd.args_hint})"
   214|    return cmd.description
   215|
   216|
   217|# Backwards-compatible flat dict: "/command" -> description
   218|COMMANDS: dict[str, str] = {}
   219|for _cmd in COMMAND_REGISTRY:
   220|    if not _cmd.gateway_only:
   221|        COMMANDS[f"/{_cmd.name}"] = _build_description(_cmd)
   222|        for _alias in _cmd.aliases:
   223|            COMMANDS[f"/{_alias}"] = f"{_cmd.description} (alias for /{_cmd.name})"
   224|
   225|# Backwards-compatible categorized dict
   226|COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
   227|for _cmd in COMMAND_REGISTRY:
   228|    if not _cmd.gateway_only:
   229|        _cat = COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {})
   230|        _cat[f"/{_cmd.name}"] = COMMANDS[f"/{_cmd.name}"]
   231|        for _alias in _cmd.aliases:
   232|            _cat[f"/{_alias}"] = COMMANDS[f"/{_alias}"]
   233|
   234|
   235|# Subcommands lookup: "/cmd" -> ["sub1", "sub2", ...]
   236|SUBCOMMANDS: dict[str, list[str]] = {}
   237|for _cmd in COMMAND_REGISTRY:
   238|    if _cmd.subcommands:
   239|        SUBCOMMANDS[f"/{_cmd.name}"] = list(_cmd.subcommands)
   240|
   241|# Also extract subcommands hinted in args_hint via pipe-separated patterns
   242|# e.g. args_hint="[on|off|tts|status]" for commands that don't have explicit subcommands.
   243|# NOTE: If a command already has explicit subcommands, this fallback is skipped.
   244|# Use the `subcommands` field on CommandDef for intentional tab-completable args.
   245|_PIPE_SUBS_RE = re.compile(r"[a-z]+(?:\|[a-z]+)+")
   246|for _cmd in COMMAND_REGISTRY:
   247|    key = f"/{_cmd.name}"
   248|    if key in SUBCOMMANDS or not _cmd.args_hint:
   249|        continue
   250|    m = _PIPE_SUBS_RE.search(_cmd.args_hint)
   251|    if m:
   252|        SUBCOMMANDS[key] = m.group(0).split("|")
   253|
   254|
   255|# ---------------------------------------------------------------------------
   256|# Gateway helpers
   257|# ---------------------------------------------------------------------------
   258|
   259|# Set of all command names + aliases recognized by the gateway.
   260|# Includes config-gated commands so the gateway can dispatch them
   261|# (the handler checks the config gate at runtime).
   262|GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
   263|    name
   264|    for cmd in COMMAND_REGISTRY
   265|    if not cmd.cli_only or cmd.gateway_config_gate
   266|    for name in (cmd.name, *cmd.aliases)
   267|)
   268|
   269|
   270|# Commands with explicit Level-2 running-agent handlers in gateway/run.py.
   271|# Listed here for introspection / tests; semantically a subset of
   272|# "all resolvable commands" — which is the real bypass set (see
   273|# should_bypass_active_session below).
   274|ACTIVE_SESSION_BYPASS_COMMANDS: frozenset[str] = frozenset(
   275|    {
   276|        "agents",
   277|        "approve",
   278|        "background",
   279|        "commands",
   280|        "deny",
   281|        "help",
   282|        "new",
   283|        "profile",
   284|        "queue",
   285|        "restart",
   286|        "status",
   287|        "steer",
   288|        "stop",
   289|        "update",
   290|    }
   291|)
   292|
   293|
   294|def should_bypass_active_session(command_name: str | None) -> bool:
   295|    """Return True for any resolvable slash command.
   296|
   297|    Rationale: every gateway-registered slash command either has a
   298|    specific Level-2 handler in gateway/run.py (/stop, /new, /model,
   299|    /approve, etc.) or reaches the running-agent catch-all that returns
   300|    a "busy — wait or /stop first" response. In both paths the command
   301|    is dispatched, not queued.
   302|
   303|    Queueing is always wrong for a recognized slash command because the
   304|    safety net in gateway.run discards any command text that reaches
   305|    the pending queue — which meant a mid-run /model (or /reasoning,
   306|    /voice, /insights, /title, /resume, /retry, /undo, /compress,
   307|    /usage, /provider, /reload-mcp, /sethome, /reset) would silently
   308|    interrupt the agent AND get discarded, producing a zero-char
   309|    response. See issue #5057 / PRs #6252, #10370, #4665.
   310|
   311|    ACTIVE_SESSION_BYPASS_COMMANDS remains the subset of commands with
   312|    explicit Level-2 handlers; the rest fall through to the catch-all.
   313|    """
   314|    return resolve_command(command_name) is not None if command_name else False
   315|
   316|
   317|def _resolve_config_gates() -> set[str]:
   318|    """Return canonical names of commands whose ``gateway_config_gate`` is truthy.
   319|
   320|    Reads ``config.yaml`` and walks the dot-separated key path for each
   321|    config-gated command.  Returns an empty set on any error so callers
   322|    degrade gracefully.
   323|    """
   324|    gated = [c for c in COMMAND_REGISTRY if c.gateway_config_gate]
   325|    if not gated:
   326|        return set()
   327|    try:
   328|        from hermes_cli.config import read_raw_config
   329|        cfg = read_raw_config()
   330|    except Exception:
   331|        return set()
   332|    result: set[str] = set()
   333|    for cmd in gated:
   334|        val: Any = cfg
   335|        for key in cmd.gateway_config_gate.split("."):
   336|            if isinstance(val, dict):
   337|                val = val.get(key)
   338|            else:
   339|                val = None
   340|                break
   341|        if val:
   342|            result.add(cmd.name)
   343|    return result
   344|
   345|
   346|def _is_gateway_available(cmd: CommandDef, config_overrides: set[str] | None = None) -> bool:
   347|    """Check if *cmd* should appear in gateway surfaces (help, menus, mappings).
   348|
   349|    Unconditionally available when ``cli_only`` is False.  When ``cli_only``
   350|    is True but ``gateway_config_gate`` is set, the command is available only
   351|    when the config value is truthy.  Pass *config_overrides* (from
   352|    ``_resolve_config_gates()``) to avoid re-reading config for every command.
   353|    """
   354|    if not cmd.cli_only:
   355|        return True
   356|    if cmd.gateway_config_gate:
   357|        overrides = config_overrides if config_overrides is not None else _resolve_config_gates()
   358|        return cmd.name in overrides
   359|    return False
   360|
   361|
   362|def gateway_help_lines() -> list[str]:
   363|    """Generate gateway help text lines from the registry."""
   364|    overrides = _resolve_config_gates()
   365|    lines: list[str] = []
   366|    for cmd in COMMAND_REGISTRY:
   367|        if not _is_gateway_available(cmd, overrides):
   368|            continue
   369|        args = f" {cmd.args_hint}" if cmd.args_hint else ""
   370|        alias_parts: list[str] = []
   371|        for a in cmd.aliases:
   372|            # Skip internal aliases like reload_mcp (underscore variant)
   373|            if a.replace("-", "_") == cmd.name.replace("-", "_") and a != cmd.name:
   374|                continue
   375|            alias_parts.append(f"`/{a}`")
   376|        alias_note = f" (alias: {', '.join(alias_parts)})" if alias_parts else ""
   377|        lines.append(f"`/{cmd.name}{args}` -- {cmd.description}{alias_note}")
   378|    return lines
   379|
   380|
   381|def telegram_bot_commands() -> list[tuple[str, str]]:
   382|    """Return (command_name, description) pairs for Telegram setMyCommands.
   383|
   384|    Telegram command names cannot contain hyphens, so they are replaced with
   385|    underscores.  Aliases are skipped -- Telegram shows one menu entry per
   386|    canonical command.
   387|    """
   388|    overrides = _resolve_config_gates()
   389|    result: list[tuple[str, str]] = []
   390|    for cmd in COMMAND_REGISTRY:
   391|        if not _is_gateway_available(cmd, overrides):
   392|            continue
   393|        tg_name = _sanitize_telegram_name(cmd.name)
   394|        if tg_name:
   395|            result.append((tg_name, cmd.description))
   396|    return result
   397|
   398|
   399|_CMD_NAME_LIMIT = 32
   400|"""Max command name length shared by Telegram and Discord."""
   401|
   402|# Backward-compat alias — tests and external code may reference the old name.
   403|_TG_NAME_LIMIT = _CMD_NAME_LIMIT
   404|
   405|# Telegram Bot API allows only lowercase a-z, 0-9, and underscores in
   406|# command names.  This regex strips everything else after initial conversion.
   407|_TG_INVALID_CHARS = re.compile(r"[^a-z0-9_]")
   408|_TG_MULTI_UNDERSCORE = re.compile(r"_{2,}")
   409|
   410|
   411|def _sanitize_telegram_name(raw: str) -> str:
   412|    """Convert a command/skill/plugin name to a valid Telegram command name.
   413|
   414|    Telegram requires: 1-32 chars, lowercase a-z, digits 0-9, underscores only.
   415|    Steps: lowercase → replace hyphens with underscores → strip all other
   416|    invalid characters → collapse consecutive underscores → strip leading/
   417|    trailing underscores.
   418|    """
   419|    name = raw.lower().replace("-", "_")
   420|    name = _TG_INVALID_CHARS.sub("", name)
   421|    name = _TG_MULTI_UNDERSCORE.sub("_", name)
   422|    return name.strip("_")
   423|
   424|
   425|def _clamp_command_names(
   426|    entries: list[tuple[str, str]],
   427|    reserved: set[str],
   428|) -> list[tuple[str, str]]:
   429|    """Enforce 32-char command name limit with collision avoidance.
   430|
   431|    Both Telegram and Discord cap slash command names at 32 characters.
   432|    Names exceeding the limit are truncated.  If truncation creates a duplicate
   433|    (against *reserved* names or earlier entries in the same batch), the name is
   434|    shortened to 31 chars and a digit ``0``-``9`` is appended to differentiate.
   435|    If all 10 digit slots are taken the entry is silently dropped.
   436|    """
   437|    used: set[str] = set(reserved)
   438|    result: list[tuple[str, str]] = []
   439|    for name, desc in entries:
   440|        if len(name) > _CMD_NAME_LIMIT:
   441|            candidate = name[:_CMD_NAME_LIMIT]
   442|            if candidate in used:
   443|                prefix = name[:_CMD_NAME_LIMIT - 1]
   444|                for digit in range(10):
   445|                    candidate = f"{prefix}{digit}"
   446|                    if candidate not in used:
   447|                        break
   448|                else:
   449|                    # All 10 digit slots exhausted — skip entry
   450|                    continue
   451|            name = candidate
   452|        if name in used:
   453|            continue
   454|        used.add(name)
   455|        result.append((name, desc))
   456|    return result
   457|
   458|
   459|# Backward-compat alias.
   460|_clamp_telegram_names = _clamp_command_names
   461|
   462|
   463|# ---------------------------------------------------------------------------
   464|# Shared skill/plugin collection for gateway platforms
   465|# ---------------------------------------------------------------------------
   466|
   467|def _collect_gateway_skill_entries(
   468|    platform: str,
   469|    max_slots: int,
   470|    reserved_names: set[str],
   471|    desc_limit: int = 100,
   472|    sanitize_name: "Callable[[str], str] | None" = None,
   473|) -> tuple[list[tuple[str, str, str]], int]:
   474|    """Collect plugin + skill entries for a gateway platform.
   475|
   476|    Priority order:
   477|      1. Plugin slash commands (take precedence over skills)
   478|      2. Built-in skill commands (fill remaining slots, alphabetical)
   479|
   480|    Only skills are trimmed when the cap is reached.
   481|    Hub-installed skills are excluded.  Per-platform disabled skills are
   482|    excluded.
   483|
   484|    Args:
   485|        platform: Platform identifier for per-platform skill filtering
   486|            (``"telegram"``, ``"discord"``, etc.).
   487|        max_slots: Maximum number of entries to return (remaining slots after
   488|            built-in/core commands).
   489|        reserved_names: Names already taken by built-in commands.  Mutated
   490|            in-place as new names are added.
   491|        desc_limit: Max description length (40 for Telegram, 100 for Discord).
   492|        sanitize_name: Optional name transform applied before clamping, e.g.
   493|            :func:`_sanitize_telegram_name` for Telegram.  May return an
   494|            empty string to signal "skip this entry".
   495|
   496|    Returns:
   497|        ``(entries, hidden_count)`` where *entries* is a list of
   498|        ``(name, description, cmd_key)`` triples and *hidden_count* is the
   499|        number of skill entries dropped due to the cap.  ``cmd_key`` is the
   500|        original ``/skill-name`` key from :func:`get_skill_commands`.
   501|