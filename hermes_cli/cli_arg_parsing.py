"""Slash-command argument parsing for the interactive CLI (``CLICommandsMixin``).

The flag/word tables plus the small token- and text-helpers the handlers parse their
arguments with. Split out of ``hermes_cli/cli_commands_mixin`` verbatim (byte-for-byte);
the origin module re-exports every name, so ``cli_commands_mixin.<name>`` stays green.
"""

from __future__ import annotations

import json
import shlex


def _command_arg(cmd: str, *, lower: bool = False) -> str:
    """Everything after the slash-command word, stripped (optionally lowercased)."""
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _shlex_args(cmd: str) -> list:
    """Tokens after the command word; falls back to whitespace split on unbalanced quotes."""
    try:
        return shlex.split(cmd)[1:] if cmd else []
    except ValueError:
        return (cmd or "").split()[1:]


def _take_flag(parts: list, flag: str):
    """Pop ``flag VALUE`` out of ``parts``: ``(rest, value, ok)``; ok=False when the value is
    missing (caller prints usage)."""
    if flag not in parts:
        return parts, None, True
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return parts, None, False
    return parts[:idx] + parts[idx + 2:], parts[idx + 1], True


def _summarize_paths(paths, limit: int = 5) -> str:
    """``a, b, c (+N more)`` for a list of paths."""
    more = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return ", ".join(paths[:limit]) + more


def _ellipsize(text: str, limit: int) -> str:
    """``text[:limit]`` plus ``...`` when truncated."""
    return f"{text[:limit]}{'...' if len(text) > limit else ''}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


# Small data tables.

# /cron flag tables: flag -> opts key. Order-sensitive in _parse_cron_flags: bool flags never
# consume a value; --repeat is int-validated separately.
_CRON_BOOL_FLAGS = {"--clear-skills": "clear_skills", "--all": "all"}
_CRON_LIST_FLAGS = {"--skill": "skills", "--add-skill": "add_skills", "--remove-skill": "remove_skills"}
_CRON_VALUE_FLAGS = {"--name": "name", "--deliver": "deliver", "--prompt": "prompt", "--schedule": "schedule"}
# /cron subcommand -> CLICommandsMixin method name.
_CRON_SUBCOMMANDS = {
    "list": "_cron_list", "add": "_cron_add", "create": "_cron_add", "edit": "_cron_edit",
    **{k: "_cron_job_action" for k in ("pause", "resume", "run", "remove", "rm", "delete")}}

_ON_WORDS = {"on", "enable", "true", "1"}
_OFF_WORDS = {"off", "disable", "false", "0"}

# /busy mode -> what Enter does while Hermes is working (status line / post-set explanation).
_BUSY_MODE_SHORT = {
    "queue": "queues for next turn", "steer": "steers into current run (after next tool call)",
    "interrupt": "redirects current run immediately"}
_BUSY_MODE_LONG = {
    "queue": "Enter will queue follow-up input while Hermes is busy.",
    "steer": "Enter will steer your message into the current run (after the next tool call).",
    "interrupt": "Enter will redirect the current run while Hermes is busy; /stop still cancels it.",
}

# /fast argument -> (service_tier value, persisted config value)
_FAST_TIERS = {
    "fast": ("priority", "fast"), "on": ("priority", "fast"), "normal": (None, "normal"),
    "off": (None, "normal"), "auto": ("auto", "auto"), "cold": ("cold", "cold")}

# /reasoning display toggles: arg -> (attr, value, headline, follow-up note)
_REASONING_TOGGLES = {
    **dict.fromkeys(("show", "on"), ("show_reasoning", True, "ON",
                                     "Model thinking will be shown during and after each response.")),
    **dict.fromkeys(("hide", "off"), ("show_reasoning", False, "OFF", "")),
    **dict.fromkeys(("full", "all"), ("reasoning_full", True, "FULL",
                                      "The post-response recap box will print complete thinking.")),
    **dict.fromkeys(("clamp", "collapse", "short"), ("reasoning_full", False, "CLAMPED to 10 lines", "")),
}

# /bg AIAgent provider-routing kwargs -> HermesCLI attribute carrying the value.
_BG_PROVIDER_KWARGS = {
    "providers_allowed": "_providers_only", "providers_ignored": "_providers_ignore",
    "providers_order": "_providers_order", "provider_sort": "_provider_sort",
    "provider_require_parameters": "_provider_require_params",
    "provider_data_collection": "_provider_data_collection",
    "openrouter_min_coding_score": "_openrouter_min_coding_score", "fallback_model": "_fallback_model"}

# /worktree subcommand -> CLICommandsMixin method name (all need a repo root).
_WORKTREE_SUBCOMMANDS = {
    **dict.fromkeys(("prune", "gc", "clean"), "_worktree_prune"),
    **dict.fromkeys(("list", "ls"), "_worktree_list"),
    **dict.fromkeys(("new", "add", "create"), "_worktree_new")}

# Message fields copied verbatim onto a /branch row (plus role / tool_name / api_content).
_BRANCH_COPY_KEYS = ("content", "tool_calls", "tool_call_id", "reasoning", "reasoning_details",
                     "codex_reasoning_items", "codex_message_items", "timestamp")

_HATCH_PROGRESS = {"compose": "  ┊ composing spritesheet…", "save": "  ┊ saving…"}

# /diff argument -> mode (anything else is a path; --stat/stat is the stat flag).
_DIFF_MODES = {
    "staged": "staged", "--staged": "staged", "cached": "staged", "--cached": "staged",
    "all": "all", "--all": "all", "head": "all", "session": "session"}
_DIFF_LABELS = {"working": "Unstaged", "staged": "Staged", "all": "All (vs HEAD)"}


def _split_scope_flags(raw: str):
    """``(arg, explicit_global)`` for /reasoning + /fast: session scope by default, ``--global``
    persists to config.yaml, ``--session`` is an explicit no-op (parity with /model)."""
    tokens = raw.strip().lower().split()
    return " ".join(t for t in tokens if t not in ("--global", "--session")), "--global" in tokens


def _scope_outcome(explicit_global: bool, saved: bool) -> str:
    """Parenthetical tail for a scoped setting change."""
    if saved:
        return "(saved to config)"
    if explicit_global:
        return "(session only; config save failed)"
    return "(this session — use --global to persist)"


def _toggle_target(arg: str, current: bool):
    """Resolve a ``/x [on|off|status]`` argument: "status" for a status query, a bool for the
    new state (bare arg toggles), or None when the argument is unrecognized."""
    if arg in {"status", "?"}:
        return "status"
    if arg in _ON_WORDS:
        return True
    if arg in _OFF_WORDS:
        return False
    if arg == "":
        return not current
    return None


def _cron_api(**kwargs) -> dict:
    """Call the cronjob model tool and decode its JSON reply."""
    from tools.cronjob_tools import cronjob as cronjob_tool
    return json.loads(cronjob_tool(**kwargs))


def _normalize_skills(values) -> list:
    """Strip, drop empties, and dedupe (order-preserving)."""
    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _parse_cron_flags(tokens):
    """Parse /cron flags into an opts dict (None after printing an error for a bad --repeat)."""
    opts = {
        "name": None, "deliver": None, "repeat": None, "prompt": None, "schedule": None,
        "skills": [], "add_skills": [], "remove_skills": [],
        "clear_skills": False, "all": False, "positionals": []}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        has_value = i + 1 < len(tokens)
        if token in _CRON_BOOL_FLAGS:
            opts[_CRON_BOOL_FLAGS[token]] = True
            i += 1
        elif token in _CRON_LIST_FLAGS and has_value:
            opts[_CRON_LIST_FLAGS[token]].append(tokens[i + 1])
            i += 2
        elif token == "--repeat" and has_value:
            try:
                opts["repeat"] = int(tokens[i + 1])
            except ValueError:
                return print("(._.) --repeat must be an integer")
            i += 2
        elif token in _CRON_VALUE_FLAGS and has_value:
            opts[_CRON_VALUE_FLAGS[token]] = tokens[i + 1]
            i += 2
        else:
            opts["positionals"].append(token)
            i += 1
    return opts
