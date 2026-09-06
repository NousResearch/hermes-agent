"""Single owner for personality overlays.

* ``display.personality`` holds the selected NAME (empty = no overlay).
* ``agent.system_prompt`` is the user-owned manual overlay; personality code never writes it.
* ``agent.personalities`` holds user-defined/overridden personalities, overlaying built-ins by name.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

#: Names that mean "no personality overlay".
NEUTRAL_PERSONALITY_NAMES = frozenset({"", "none", "default", "neutral"})

#: Built-in personalities, available on every surface without any config.
BUILTIN_PERSONALITIES: Dict[str, str] = {
    "helpful": "You are a helpful, friendly AI assistant.",
    "concise": "You are a concise assistant. Keep responses brief and to the point.",
    "technical": "You are a technical expert. Provide detailed, accurate technical information.",
    "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
    "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
    "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
    "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
    "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
    "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
    "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga!",
    "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
    "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
    "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
    "hype": "YOOO LET'S GOOOO!!! I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS!",
}


def _get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of None/non-dict intermediate nodes."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def prompt_text(value: Any) -> str:
    """Normalize config prompt values from YAML (str | list | None) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def render_personality_prompt(value: Any) -> str:
    """Render a string or structured personality definition to prompt text."""
    if isinstance(value, dict):
        parts = [value.get("system_prompt", "")]
        if value.get("tone"):
            parts.append(f'Tone: {value["tone"]}')
        if value.get("style"):
            parts.append(f'Style: {value["style"]}')
        return "\n".join(str(part).strip() for part in parts if str(part).strip())
    return prompt_text(value)


def describe_personality(value: Any, width: int = 50) -> str:
    """Short preview line for list UIs (CLI table, gateway /personality list)."""
    if isinstance(value, dict):
        preview = value.get("description") or str(value.get("system_prompt", ""))
    else:
        preview = str(value)
    preview = preview.strip().replace("\n", " ")
    return preview[:width] + ("..." if len(preview) > width else "")


def normalize_personality_name(value: Any) -> str:
    """Canonical form of a personality name ('' for any neutral spelling)."""
    name = str(value or "").strip().lower()
    return "" if name in NEUTRAL_PERSONALITY_NAMES else name


def available_personalities(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Built-ins overlaid by the user's ``agent.personalities`` (user wins)."""
    merged: Dict[str, Any] = dict(BUILTIN_PERSONALITIES)
    user = _get(cfg, "agent", "personalities", default={})
    if isinstance(user, dict):
        for name, definition in user.items():
            key = str(name).strip().lower()
            if key and key not in NEUTRAL_PERSONALITY_NAMES:
                merged[key] = definition
    return merged


def resolve_personality(value: Any, cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Resolve a requested personality to ``(canonical_name, prompt_text)``."""
    name = normalize_personality_name(value)
    if not name:
        return "", ""
    personalities = available_personalities(cfg)
    if name not in personalities:
        names = ", ".join(f"`{n}`" for n in sorted(personalities))
        raise ValueError(f"Unknown personality: `{str(value).strip()}`.\n\nAvailable: `none`, {names}")
    return name, render_personality_prompt(personalities[name])


def active_personality_name(cfg: Optional[Dict[str, Any]]) -> str:
    """The currently selected personality name ('' when none is active)."""
    name = normalize_personality_name(_get(cfg, "display", "personality", default=""))
    return name if name and name in available_personalities(cfg) else ""


def resolve_ephemeral_system_prompt(cfg: Optional[Dict[str, Any]]) -> str:
    """Session overlay: ``display.personality`` when it names a known personality, else the
    user-owned ``agent.system_prompt``. Callers still prefer ``HERMES_EPHEMERAL_SYSTEM_PROMPT``."""
    name = active_personality_name(cfg)
    if name:
        return render_personality_prompt(available_personalities(cfg)[name])
    return prompt_text(_get(cfg, "agent", "system_prompt", default=""))


def persist_personality(value: Any) -> bool:
    """Persist the selection — the ONLY sanctioned write path. Writes the canonical name (or '')
    to ``display.personality`` atomically (comments/ordering preserved); never touches
    ``agent.system_prompt``. Returns True on success."""
    name = normalize_personality_name(value)
    try:
        from hermes_constants import get_hermes_home
        from utils import atomic_roundtrip_yaml_update

        config_path = get_hermes_home() / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_roundtrip_yaml_update(config_path, "display.personality", name)
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        return True
    except Exception:
        return False
