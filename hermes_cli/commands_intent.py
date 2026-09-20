"""Intent-to-slash-command matching and keyword search for Hermes CLI.

Maps common natural language action prompts (in English, Hinglish, Hindi,
Spanish, etc.) to canonical Hermes slash commands, allowing users to discover
and execute commands without having to memorize exact slash syntax.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class IntentMatch(NamedTuple):
    command: str          # Canonical slash command without leading slash, e.g. "new", "model"
    description: str      # Clean action description
    matched_phrase: str   # The trigger phrase that matched


# Multi-lingual & natural language prompt patterns mapped to canonical slash commands
INTENT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Session reset & fresh start -> /new (alias /reset)
    (
        re.compile(
            r"\b(new\s+(session|chat|conversation)|fresh\s+(session|chat|start)|"
            r"reset\s+(session|chat|conversation|all|history)|clear\s+(session|chat|conversation|memory|history)|"
            r"restart\s+(session|chat)|wipe\s+(chat|history|session)|"
            r"nayi\s+chat|naya\s+session|chat\s+reset|reset\s+kardo|shuru\s+se|"
            r"nueva\s+sesion|reiniciar\s+chat|recommencer)\b",
            re.IGNORECASE,
        ),
        "new",
        "Start a new session (fresh session ID + history)",
    ),
    # Terminal clear screen -> /clear
    (
        re.compile(
            r"\b(clear\s+screen|cls|clear\s+terminal|clean\s+screen|screen\s+clear)\b",
            re.IGNORECASE,
        ),
        "clear",
        "Clear screen and start a new session",
    ),
    # Model switching -> /model
    (
        re.compile(
            r"\b(switch\s+model|change\s+model|choose\s+model|select\s+model|pick\s+model|"
            r"model\s+selector|model\s+list|change\s+llm|switch\s+llm|"
            r"model\s+badlo|dusra\s+model|model\s+change|"
            r"cambiar\s+modelo|changer\s+de\s+modele)\b",
            re.IGNORECASE,
        ),
        "model",
        "Select or inspect the active LLM model",
    ),
    # Session cost & spend -> /cost
    (
        re.compile(
            r"\b(session\s+cost|token\s+cost|how\s+much\s+(cost|spent|spend)|check\s+cost|show\s+cost|"
            r"total\s+spend|api\s+cost|pricing|"
            r"kitna\s+kharcha|cost\s+kitni|kitne\s+paise|kharcha\s+kitna|"
            r"cuanto\s+cuesta|combien\s+coute)\b",
            re.IGNORECASE,
        ),
        "cost",
        "Show current session cost breakdown",
    ),
    # Token usage & rate limits -> /usage
    (
        re.compile(
            r"\b(token\s+usage|tokens\s+used|token\s+count|how\s+many\s+tokens|rate\s+limit|"
            r"context\s+limit|tokens\s+remaining|"
            r"tokens\s+kitne|token\s+kitna|usage\s+dikhao)\b",
            re.IGNORECASE,
        ),
        "usage",
        "Show token usage and rate limits",
    ),
    # Conversation history -> /history
    (
        re.compile(
            r"\b(show\s+history|chat\s+history|conversation\s+history|previous\s+messages|"
            r"history\s+dikhao|purani\s+baatein|historial)\b",
            re.IGNORECASE,
        ),
        "history",
        "Show conversation history",
    ),
    # Save & export -> /save
    (
        re.compile(
            r"\b(save\s+(chat|conversation|session|transcript)|"
            r"export\s+(chat|conversation|session|transcript)|"
            r"download\s+(chat|history)|backup\s+chat|chat\s+save)\b",
            re.IGNORECASE,
        ),
        "save",
        "Export the current conversation",
    ),
    # Undo turn -> /undo
    (
        re.compile(
            r"\b(undo\s+(turn|message|last)|go\s+back|revert\s+(turn|last)|back\s+up\s+turn|"
            r"piche\s+jao|undo\s+karo|deshacer)\b",
            re.IGNORECASE,
        ),
        "undo",
        "Back up N user turns and re-prompt",
    ),
    # Retry -> /retry
    (
        re.compile(
            r"\b(retry\s+(last|message|prompt)|try\s+again|resend\s+(last|message)|"
            r"dobaara\s+bhejo|phir\s+se\s+bhejo|reintentar)\b",
            re.IGNORECASE,
        ),
        "retry",
        "Retry the last message (resend to agent)",
    ),
    # Context compression -> /compress (alias /compact)
    (
        re.compile(
            r"\b(compress\s+(context|history|session|chat)|"
            r"compact\s+(context|history|session|chat|memory)|"
            r"shrink\s+context|reduce\s+tokens|compactar)\b",
            re.IGNORECASE,
        ),
        "compress",
        "Compress conversation context",
    ),
    # Tools management -> /tools
    (
        re.compile(
            r"\b(manage\s+tools|list\s+tools|enable\s+tools?|disable\s+tools?|"
            r"mcp\s+servers?|toolset|tools\s+list|tools\s+dikhao)\b",
            re.IGNORECASE,
        ),
        "tools",
        "Inspect, enable, or disable toolsets and MCP servers",
    ),
    # Status & health -> /status
    (
        re.compile(
            r"\b(agent\s+status|system\s+status|session\s+status|health\s+check|status\s+check|"
            r"status\s+dikhao)\b",
            re.IGNORECASE,
        ),
        "status",
        "Show current agent and session status",
    ),
    # Help & commands browse -> /help
    (
        re.compile(
            r"\b(help\s+me|show\s+commands|list\s+commands|what\s+commands|available\s+commands|"
            r"commands\s+list|help\s+menu|commands\s+dikhao|kya\s+commands)\b",
            re.IGNORECASE,
        ),
        "help",
        "Show available commands",
    ),
    # Palette search -> /palette
    (
        re.compile(
            r"\b(command\s+palette|search\s+commands|find\s+command|fuzzy\s+commands)\b",
            re.IGNORECASE,
        ),
        "palette",
        "Open the fuzzy command palette",
    ),
]

# Keyword tags for slash command fuzzy search (e.g. typing /token, /pricing, /persona, /clean, /wipe)
KEYWORD_TAGS: dict[str, list[str]] = {
    "cost": ["pricing", "spend", "spending", "price", "bill", "billing", "usd", "expense", "kharcha"],
    "usage": ["token", "tokens", "limit", "rate", "consumption", "quota"],
    "model": ["switch", "llm", "provider", "select", "badlo", "choose"],
    "new": ["clean", "fresh", "reset", "wipe", "start", "restart", "nayi", "naya"],
    "clear": ["cls", "screen", "terminal", "clean"],
    "compress": ["compact", "shrink", "prune", "reduce", "summary"],
    "personality": ["persona", "character", "tone", "system", "prompt"],
    "skin": ["theme", "style", "ui", "colors", "appearance"],
    "tools": ["mcp", "toolset", "server", "capabilities"],
    "save": ["export", "download", "backup", "log"],
    "branch": ["fork", "split", "alternate"],
    "undo": ["revert", "back", "piche"],
    "retry": ["again", "resend", "dobaara"],
}


def match_prompt_intent(text: str) -> list[IntentMatch]:
    """Match natural language prompt text against command intent patterns.

    Returns matching IntentMatch entries if text corresponds to a meta-action.
    Ignores inputs that are excessively long to avoid false positives on large prompts.
    """
    cleaned = text.strip()
    if not cleaned or len(cleaned) > 100:
        return []

    matches: list[IntentMatch] = []
    seen: set[str] = set()

    for pattern, cmd, desc in INTENT_PATTERNS:
        m = pattern.search(cleaned)
        if m and cmd not in seen:
            seen.add(cmd)
            matches.append(IntentMatch(command=cmd, description=desc, matched_phrase=m.group(0)))

    return matches


def match_slash_keywords(word: str) -> list[tuple[str, str, str]]:
    """Match a typed slash word (e.g. 'token', 'pricing') against keyword tags and descriptions.

    Returns list of (command_name, description, matched_keyword).
    """
    if not word or len(word) < 2:
        return []

    lowered = word.lower()
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for cmd, tags in KEYWORD_TAGS.items():
        for tag in tags:
            if tag.startswith(lowered) or (len(lowered) >= 4 and lowered in tag):
                if cmd not in seen:
                    seen.add(cmd)
                    results.append((cmd, f"Matched keyword '{tag}'", tag))
                break

    return results
