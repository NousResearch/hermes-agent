"""Hermes CLI skin helpers.

Keeps the lore state, themed banner fragments, response framing, and
lightweight easter-egg helpers separate from the main CLI loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sqlite3
import struct
import time
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_SKIN = "hermes"
VALID_SKINS = {"hermes", "ares", "posideon"}
MOD_SKINS = {"ares", "posideon"}
MOD_DIRS = {
    "ares": Path(__file__).resolve().parent.parent / "ares agent mod",
    "posideon": Path(__file__).resolve().parent.parent / "posideon agent mod",
}

_DEFAULT_ARES_CRIMSON = "#9F1C1C"
_DEFAULT_ARES_BLOOD = "#6B1717"
_DEFAULT_ARES_EMBER = "#DD4A3A"
_DEFAULT_ARES_BRONZE = "#C7A96B"
_DEFAULT_ARES_SAND = "#F1E6CF"
_DEFAULT_ARES_ASH = "#6E584B"
_DEFAULT_ARES_STEEL = "#51433B"
_DEFAULT_ARES_OBSIDIAN = "#1A1513"
_DEFAULT_ARES_INK = "#241B18"
_DEFAULT_ARES_PATINA = "#8E6A42"

_DEFAULT_PIXEL_FONT = {
    "A": (
        " 111 ",
        "1   1",
        "1   1",
        "11111",
        "1   1",
        "1   1",
        "1   1",
    ),
    "R": (
        "1111 ",
        "1   1",
        "1   1",
        "1111 ",
        "1 1  ",
        "1  1 ",
        "1   1",
    ),
    "E": (
        "11111",
        "1    ",
        "1    ",
        "1111 ",
        "1    ",
        "1    ",
        "11111",
    ),
    "D": (
        "1111 ",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        "1111 ",
    ),
    "I": (
        "11111",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "11111",
    ),
    "O": (
        " 111 ",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        "1   1",
        " 111 ",
    ),
    "S": (
        " 1111",
        "1    ",
        "1    ",
        " 111 ",
        "    1",
        "    1",
        "1111 ",
    ),
    "-": (
        "     ",
        "     ",
        "     ",
        "11111",
        "     ",
        "     ",
        "     ",
    ),
    "G": (
        " 1111",
        "1    ",
        "1    ",
        "1 111",
        "1   1",
        "1   1",
        " 111 ",
    ),
    "N": (
        "1   1",
        "11  1",
        "1 1 1",
        "1  11",
        "1   1",
        "1   1",
        "1   1",
    ),
    "T": (
        "11111",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
        "  1  ",
    ),
    " ": (
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
    ),
}


def _normalize_skin_token(name: str | None) -> str:
    if not name:
        return DEFAULT_SKIN
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_SKIN,
        "classic": "hermes",
        "classic_gold": "hermes",
        "winged": "ares",
        "holographic": "ares",
        "poseidon": "posideon",
        "ocean": "posideon",
        "trident": "posideon",
    }
    return aliases.get(normalized, normalized)


def _active_mod_skin(name: str | None = None) -> str | None:
    normalized = _normalize_skin_token(name if name is not None else os.getenv("HERMES_CLI_SKIN"))
    return normalized if normalized in MOD_SKINS else None


@lru_cache(maxsize=3)
def _load_mod(skin_name: str | None = None):
    resolved_skin = _active_mod_skin(skin_name)
    if resolved_skin is None:
        return None
    path = MOD_DIRS[resolved_skin] / "mod.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"{resolved_skin}_agent_mod_payload", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mod_attr(name: str, default, skin_name: str | None = None):
    mod = _load_mod(_active_mod_skin(skin_name))
    if mod is None:
        return default
    return getattr(mod, name, default)


def _call_mod(name: str, *args, skin_name: str | None = None, **kwargs):
    mod = _load_mod(_active_mod_skin(skin_name))
    if mod is None:
        return None
    func = getattr(mod, name, None)
    if not callable(func):
        return None
    return func(*args, **kwargs)


ARES_CRIMSON = _mod_attr("ARES_CRIMSON", _DEFAULT_ARES_CRIMSON)
ARES_BLOOD = _mod_attr("ARES_BLOOD", _DEFAULT_ARES_BLOOD)
ARES_EMBER = _mod_attr("ARES_EMBER", _DEFAULT_ARES_EMBER)
ARES_BRONZE = _mod_attr("ARES_BRONZE", _DEFAULT_ARES_BRONZE)
ARES_SAND = _mod_attr("ARES_SAND", _DEFAULT_ARES_SAND)
ARES_ASH = _mod_attr("ARES_ASH", _DEFAULT_ARES_ASH)
ARES_STEEL = _mod_attr("ARES_STEEL", _DEFAULT_ARES_STEEL)
ARES_OBSIDIAN = _mod_attr("ARES_OBSIDIAN", _DEFAULT_ARES_OBSIDIAN)
ARES_INK = _mod_attr("ARES_INK", _DEFAULT_ARES_INK)
ARES_PATINA = _mod_attr("ARES_PATINA", _DEFAULT_ARES_PATINA)
PIXEL_FONT = _mod_attr("PIXEL_FONT", _DEFAULT_PIXEL_FONT)


CADUCEUS_FRAMES = (
    """[#7FFFD4]         ╭──╮             ╭──╮[/]
[#7FFFD4]    ╭────╯  ╰────╮   ╭────╯  ╰────╮[/]
[#5EF0A5]   ╱   ╭──╮  ╭──╮ ╲ ╱ ╭──╮  ╭──╮   ╲[/]
[#5EF0A5]  ╱   ╱   ╰──╯   ╲ ╳ ╱   ╰──╯   ╲   ╲[/]
[#FFD700]             ╭────╫────╮[/]
[#FFD700]          ╭──╯   ╱╳╲   ╰──╮[/]
[#FFBF00]          ╰──╮  ╱╱ ╲╲  ╭──╯[/]
[#FFBF00]             │  ╲╲ ╱╱  │[/]
[#CD7F32]          ╭──╯  ╱╱ ╲╲  ╰──╮[/]
[#CD7F32]          ╰──╮  ╲╲ ╱╱  ╭──╯[/]
[#B8860B]             ╰────╥────╯[/]
[#5EF0A5]          winged relay aligned[/]""",
    """[#7FFFD4]       ╭───╮             ╭───╮[/]
[#7FFFD4]   ╭─────╯   ╰─────╮   ╭─────╯   ╰─────╮[/]
[#5EF0A5]  ╱   ╭───╮  ╭───╮  ╲ ╱  ╭───╮  ╭───╮   ╲[/]
[#5EF0A5] ╱   ╱    ╰──╯    ╲  ╳  ╱    ╰──╯    ╲   ╲[/]
[#FFD700]             ╭────╫────╮[/]
[#FFD700]           ╭─╯   ╱╳╲   ╰─╮[/]
[#FFBF00]           ╰─╮  ╱╱ ╲╲  ╭─╯[/]
[#FFBF00]             │  ╲╲ ╱╱  │[/]
[#CD7F32]           ╭─╯  ╱╱ ╲╲  ╰─╮[/]
[#CD7F32]           ╰─╮  ╲╲ ╱╱  ╭─╯[/]
[#B8860B]             ╰────╨────╯[/]
[#5EF0A5]         sandals cross the grid[/]""",
    """[#7FFFD4]      ╭────╮           ╭────╮[/]
[#7FFFD4]  ╭─────╯    ╰─────╮ ╭─────╯    ╰─────╮[/]
[#5EF0A5] ╱   ╭────╮  ╭────╮ ╳ ╭────╮  ╭────╮   ╲[/]
[#5EF0A5]╱   ╱     ╰──╯     ╲╳╱╱     ╰──╯     ╲   ╲[/]
[#FFD700]            ╭─────╬─────╮[/]
[#FFD700]          ╭─╯    ╱╳╲    ╰─╮[/]
[#FFBF00]          ╰─╮   ╱╱ ╲╲   ╭─╯[/]
[#FFBF00]            │   ╲╲ ╱╱   │[/]
[#CD7F32]          ╭─╯   ╱╱ ╲╲   ╰─╮[/]
[#CD7F32]          ╰─╮   ╲╲ ╱╱   ╭─╯[/]
[#B8860B]            ╰────╫────╯[/]
[#5EF0A5]         messenger surge primed[/]""",
    """[#7FFFD4]        ╭──╮             ╭──╮[/]
[#7FFFD4]   ╭─────╯  ╰─────╮   ╭─────╯  ╰─────╮[/]
[#5EF0A5]  ╱   ╭──╮  ╭──╮   ╲ ╱   ╭──╮  ╭──╮   ╲[/]
[#5EF0A5] ╱   ╱   ╰──╯   ╲   ╳   ╱   ╰──╯   ╲   ╲[/]
[#FFD700]             ╭────╫────╮[/]
[#FFD700]          ╭──╯   ╱╳╲   ╰──╮[/]
[#FFBF00]          ╰──╮  ╱╱ ╲╲  ╭──╯[/]
[#FFBF00]             │  ╲╲ ╱╱  │[/]
[#CD7F32]          ╭──╯  ╱╱ ╲╲  ╰──╮[/]
[#CD7F32]          ╰──╮  ╲╲ ╱╱  ╭──╯[/]
[#B8860B]             ╰────╫────╯[/]
[#5EF0A5]         courier lattice stable[/]""",
)

CADUCEUS_FRAMES_ASCENDED = (
    """[#7FFFD4]      ✦  ╭──╮           ╭──╮  ✦[/]
[#7FFFD4]   ✦ ╭────╯  ╰────╮   ╭────╯  ╰────╮ ✦[/]
[#5EF0A5]   ╱  ╭───╮  ╭───╮ ╲ ╱ ╭───╮  ╭───╮  ╲[/]
[#5EF0A5]  ╱  ╱  ✦ ╰──╯ ✦  ╲ ╳ ╱  ✦ ╰──╯ ✦  ╲  ╲[/]
[#FFD700]             ╭────◉────╮[/]
[#FFD700]          ╭──╯   ╱╳╲   ╰──╮[/]
[#FFBF00]          ╰──╮  ╱╱◉╲╲  ╭──╯[/]
[#FFBF00]             │  ╲╲◉╱╱  │[/]
[#CD7F32]          ╭──╯  ╱╱◉╲╲  ╰──╮[/]
[#CD7F32]          ╰──╮  ╲╲◉╱╱  ╭──╯[/]
[#B8860B]             ╰────◉────╯[/]
[#5EF0A5]        ascended relay unlocked[/]""",
    """[#7FFFD4]    ✦  ╭────╮         ╭────╮  ✦[/]
[#7FFFD4] ✦ ╭─────╯    ╰─────╮ ╭─────╯    ╰─────╮ ✦[/]
[#5EF0A5] ╱  ╭────╮  ╭────╮  ╳ ╳  ╭────╮  ╭────╮  ╲[/]
[#5EF0A5]╱  ╱   ✦ ╰──╯ ✦   ╲╳╱╳╲╱   ✦ ╰──╯ ✦   ╲  ╲[/]
[#FFD700]            ╭─────◉─────╮[/]
[#FFD700]          ╭─╯    ╱╳╲    ╰─╮[/]
[#FFBF00]          ╰─╮   ╱╱◉╲╲   ╭─╯[/]
[#FFBF00]            │   ╲╲◉╱╱   │[/]
[#CD7F32]          ╭─╯   ╱╱◉╲╲   ╰─╮[/]
[#CD7F32]          ╰─╮   ╲╲◉╱╱   ╭─╯[/]
[#B8860B]            ╰────◉────╯[/]
[#5EF0A5]         golden telemetry online[/]""",
)

MESSENGER_TITLES = tuple(
    _mod_attr("MESSENGER_TITLES", ("Ares Dispatch", "War Scroll", "Iron Decree"))
)

TRICKSTER_CORRECTIONS = dict(
    _mod_attr(
        "TRICKSTER_CORRECTIONS",
        {
            "teh": "the",
            "adn": "and",
            "heremes": "Ares",
            "definately": "definitely",
            "wierd": "weird",
        },
    )
)

COIN_SPIN_FRAMES = tuple(_mod_attr("COIN_SPIN_FRAMES", ("◐", "◓", "◑", "◒", "◐", "◎")))
DI20_GLYPHS = tuple(_mod_attr("DI20_GLYPHS", ("◢", "◣", "◤", "◥", "⬢", "⬡")))


@dataclass
class HermesLoreState:
    """Persistent progression state for the Hermes skin."""

    sessions: int = 0
    user_messages: int = 0
    clever_replies: int = 0
    published_skills: list[str] = field(default_factory=list)

    @property
    def wing_level(self) -> int:
        if self.sessions >= 50:
            return 2
        if self.sessions >= 15:
            return 1
        return 0

    @property
    def glow_enabled(self) -> bool:
        return self.clever_replies >= 100

    @property
    def orbiting_skills(self) -> list[str]:
        return self.published_skills[:4]

    def summary(self) -> str:
        wing_state = "iron phalanx" if self.wing_level >= 2 else "spartan command"
        glow_state = "ember glow" if self.glow_enabled else "ashen glow"
        return f"{wing_state} · {glow_state}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _mix_rgb(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    ratio = max(0.0, min(ratio, 1.0))
    return tuple(
        int(round(low[index] + (high[index] - low[index]) * ratio))
        for index in range(3)
    )


def _is_near_white(red: int, green: int, blue: int) -> bool:
    """Return True for bright neutral pixels that should read as transparency."""
    luminance = (red * 299 + green * 587 + blue * 114) / 1000.0
    spread = max(red, green, blue) - min(red, green, blue)
    return luminance >= 220 and spread <= 40


def build_mod_masthead() -> str:
    """Render a readable pixel-font masthead for the active mod banner."""
    payload = _call_mod("build_masthead")
    if payload:
        return payload
    payload = _call_mod("build_ares_masthead")
    if payload:
        return payload
    red_rows = (
        "#C42E2E",
        "#B52727",
        ARES_CRIMSON,
        ARES_CRIMSON,
        "#861C1C",
        ARES_BLOOD,
        ARES_BLOOD,
    )
    bronze_rows = (
        "#E1C98E",
        "#D5B978",
        ARES_BRONZE,
        ARES_BRONZE,
        "#B48E58",
        ARES_PATINA,
        "#775735",
    )
    text = get_mod_brand_name().upper().replace(" ", "-")
    lines: list[str] = []

    for row_index in range(7):
        line: list[str] = []
        active_color: str | None = None
        for char_index, char in enumerate(text):
            glyph = PIXEL_FONT.get(char, PIXEL_FONT[" "])[row_index]
            if char == " ":
                if active_color is not None:
                    line.append("[/]")
                    active_color = None
                line.append("   ")
            else:
                palette = red_rows if char_index < 4 else bronze_rows
                color = palette[row_index]
                for pixel in glyph:
                    if pixel == " ":
                        if active_color is not None:
                            line.append("[/]")
                            active_color = None
                        line.append(" ")
                        continue
                    if color != active_color:
                        if active_color is not None:
                            line.append("[/]")
                        line.append(f"[{color}]")
                        active_color = color
                    line.append("█")
            if active_color is not None:
                line.append("[/]")
                active_color = None
            line.append(" ")
        lines.append("".join(line).rstrip())

    return "\n".join(lines)


def build_ares_masthead() -> str:
    """Backward-compatible alias for the active mod masthead."""
    return build_mod_masthead()


def get_mod_brand_name() -> str:
    return str(_mod_attr("BRAND_NAME", "Ares Agent"))


def get_mod_version_title(version: str) -> str:
    payload = _call_mod("get_version_title", version)
    if payload:
        return payload
    return f"{get_mod_brand_name()} {version}"


def get_mod_asset_dir() -> Path:
    payload = _call_mod("get_asset_dir")
    if payload:
        return Path(payload)
    active_mod = _active_mod_skin()
    if active_mod:
        return MOD_DIRS[active_mod]
    return MOD_DIRS["ares"]


def get_mod_help_footer(tool_count: int, skill_count: int) -> str:
    payload = _call_mod("get_help_footer", tool_count, skill_count)
    if payload:
        return payload
    return f"{tool_count} tools · {skill_count} skills · /help - for Sparta"


def get_mod_unit_designation() -> str:
    return str(
        _mod_attr(
            "UNIT_DESIGNATION",
            "UNIT DESIGNATION: MILITARY INTELLIGENCE // WAR DEPARTMENT // Ares-001",
        )
    )


def get_mod_welcome_message() -> str:
    payload = _call_mod("get_welcome_message")
    if payload:
        return payload
    return "Welcome to Ares Agent! Type your message or /help for commands."


def get_mod_placeholder_text() -> str:
    payload = _call_mod("get_placeholder_text")
    if payload:
        return payload
    return "ask Ares, or try /flip, /roll d20, /skin hermes"


def get_mod_hint_bar(agent_running: bool, glyph: str, orbit_count: int) -> str:
    payload = _call_mod("get_hint_bar", agent_running, glyph, orbit_count)
    if payload:
        return payload
    if agent_running:
        return f"  {glyph} warpath in flight · type to interrupt · Ctrl+C to break"
    return f"  {glyph} shield line ready · rituals /flip /roll d20 · orbit {orbit_count}"


def get_mod_omens_title() -> str:
    return str(_mod_attr("OMENS_TITLE", "Ares Omens"))


def get_mod_rituals() -> tuple[tuple[str, str], ...]:
    return tuple(
        _mod_attr(
            "RITUALS",
            (
                ("/flip", "shield omen"),
                ("/roll d20", "weighted Ares dice"),
                ("flip coin", "local shortcut"),
                ("roll dice", "local shortcut"),
            ),
        )
    )


def get_mod_skin_status_label() -> str:
    return str(_mod_attr("SKIN_STATUS_LABEL", "Skin"))


def get_mod_prompt_frames(active: bool) -> tuple[str, ...]:
    key = "ACTIVE_PROMPT_FRAMES" if active else "IDLE_PROMPT_FRAMES"
    default = ("⟪⚔⟫ ", "⟪▲⟫ ", "⟪⛨⟫ ", "⟪⚔⟫ ") if active else ("⚔ ", "⛨ ", "▲ ", "⚔ ")
    return tuple(_mod_attr(key, default))


def get_mod_spinner_wings(frame_idx: int) -> tuple[str, str]:
    wings = tuple(_mod_attr("SPINNER_WINGS", (("⟪⚔", "⚔⟫"), ("⟪▲", "▲⟫"), ("⟪╸", "╺⟫"), ("⟪⛨", "⛨⟫"))))
    return wings[frame_idx % len(wings)]


def get_mod_waiting_faces() -> tuple[str, ...]:
    return tuple(_mod_attr("WAITING_FACES", ("(⚔)", "(⛨)", "(▲)", "(<> )", "(/)")))


def get_mod_thinking_faces() -> tuple[str, ...]:
    return tuple(_mod_attr("THINKING_FACES", ("(⚔)", "(⛨)", "(▲)", "(⌁)", "(<> )")))


def get_mod_thinking_verbs() -> tuple[str, ...]:
    return tuple(
        _mod_attr(
            "THINKING_VERBS",
            (
                "forging",
                "marching",
                "sizing the field",
                "holding the line",
                "hammering plans",
                "tempering steel",
                "plotting impact",
                "raising the shield",
            ),
        )
    )


def get_mod_assistant_name() -> str:
    return str(_mod_attr("ASSISTANT_NAME", get_mod_brand_name().split()[0]))


def get_mod_agent_glyph() -> str:
    return str(_mod_attr("AGENT_GLYPH", "⚔"))


def get_mod_compact_tagline() -> str:
    return str(_mod_attr("COMPACT_TAGLINE", "Spartan CLI Skin"))


def get_mod_compact_description() -> str:
    return str(_mod_attr("COMPACT_DESCRIPTION", "War-forged terminal interface"))


def get_mod_progress_labels() -> tuple[str, str, str]:
    return tuple(_mod_attr("PROGRESS_LABELS", ("Shield rise", "Ember glow", "Scroll orbit")))


def get_mod_next_labels() -> tuple[str, str]:
    return tuple(_mod_attr("NEXT_PROGRESS_LABELS", ("Next shield rise", "Next glow")))


def set_active_skin_globals(skin_name: str | None = None) -> str:
    """Refresh cached palette and animation globals after a runtime skin switch."""
    resolved_skin = _active_mod_skin(skin_name)

    global ARES_CRIMSON, ARES_BLOOD, ARES_EMBER, ARES_BRONZE, ARES_SAND
    global ARES_ASH, ARES_STEEL, ARES_OBSIDIAN, ARES_INK, ARES_PATINA
    global PIXEL_FONT, MESSENGER_TITLES, TRICKSTER_CORRECTIONS, COIN_SPIN_FRAMES
    global DI20_GLYPHS

    ARES_CRIMSON = _mod_attr("ARES_CRIMSON", _DEFAULT_ARES_CRIMSON, skin_name=resolved_skin)
    ARES_BLOOD = _mod_attr("ARES_BLOOD", _DEFAULT_ARES_BLOOD, skin_name=resolved_skin)
    ARES_EMBER = _mod_attr("ARES_EMBER", _DEFAULT_ARES_EMBER, skin_name=resolved_skin)
    ARES_BRONZE = _mod_attr("ARES_BRONZE", _DEFAULT_ARES_BRONZE, skin_name=resolved_skin)
    ARES_SAND = _mod_attr("ARES_SAND", _DEFAULT_ARES_SAND, skin_name=resolved_skin)
    ARES_ASH = _mod_attr("ARES_ASH", _DEFAULT_ARES_ASH, skin_name=resolved_skin)
    ARES_STEEL = _mod_attr("ARES_STEEL", _DEFAULT_ARES_STEEL, skin_name=resolved_skin)
    ARES_OBSIDIAN = _mod_attr("ARES_OBSIDIAN", _DEFAULT_ARES_OBSIDIAN, skin_name=resolved_skin)
    ARES_INK = _mod_attr("ARES_INK", _DEFAULT_ARES_INK, skin_name=resolved_skin)
    ARES_PATINA = _mod_attr("ARES_PATINA", _DEFAULT_ARES_PATINA, skin_name=resolved_skin)
    PIXEL_FONT = _mod_attr("PIXEL_FONT", _DEFAULT_PIXEL_FONT, skin_name=resolved_skin)
    MESSENGER_TITLES = tuple(
        _mod_attr("MESSENGER_TITLES", ("Ares Dispatch", "War Scroll", "Iron Decree"), skin_name=resolved_skin)
    )
    TRICKSTER_CORRECTIONS = dict(
        _mod_attr(
            "TRICKSTER_CORRECTIONS",
            {
                "teh": "the",
                "adn": "and",
                "heremes": "Ares",
                "definately": "definitely",
                "wierd": "weird",
            },
            skin_name=resolved_skin,
        )
    )
    COIN_SPIN_FRAMES = tuple(_mod_attr("COIN_SPIN_FRAMES", ("◐", "◓", "◑", "◒", "◐", "◎"), skin_name=resolved_skin))
    DI20_GLYPHS = tuple(_mod_attr("DI20_GLYPHS", ("◢", "◣", "◤", "◥", "⬢", "⬡"), skin_name=resolved_skin))
    return normalize_skin_name(skin_name)


def normalize_skin_name(name: str | None) -> str:
    """Normalize user-facing skin names."""
    normalized = resolve_skin_request(name)
    return normalized if normalized is not None else DEFAULT_SKIN


def resolve_skin_request(name: str | None) -> str | None:
    """Resolve a requested skin name or alias without falling back silently."""
    if name is None:
        return None
    normalized = _normalize_skin_token(name)
    return normalized if normalized in VALID_SKINS else None


def is_hermes_skin(name: str | None) -> bool:
    """Return True when the Hermes visual skin is active."""
    return normalize_skin_name(name) == "hermes"


def is_ares_skin(name: str | None) -> bool:
    """Return True when the Ares visual skin is active."""
    return normalize_skin_name(name) == "ares"


def is_posideon_skin(name: str | None) -> bool:
    """Return True when the Posideon visual skin is active."""
    return normalize_skin_name(name) == "posideon"


def is_mod_skin(name: str | None) -> bool:
    """Return True when a custom mod skin is active."""
    return normalize_skin_name(name) in MOD_SKINS


def get_hermes_home() -> Path:
    """Resolve the Hermes home directory."""
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def get_lore_path() -> Path:
    """Path to the Hermes skin lore file."""
    return get_hermes_home() / "hermes_lore.json"


def _load_lore_blob() -> dict:
    path = get_lore_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_lore_blob(blob: dict) -> None:
    path = get_lore_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def record_published_skill(skill_name: str, target: str = "github") -> None:
    """Persist a successfully published skill for lore progression."""
    if not skill_name:
        return
    blob = _load_lore_blob()
    published = blob.setdefault("published_skills", [])
    published.append(
        {
            "name": skill_name,
            "target": target,
            "published_at": int(time.time()),
        }
    )
    blob["published_skills"] = published[-12:]
    _save_lore_blob(blob)


def _is_clever_reply(content: str) -> bool:
    """Heuristic for counting substantial user inputs."""
    if not content:
        return False
    text = content.strip()
    if len(text) >= 90:
        return True
    if text.count("\n") >= 2 or "```" in text:
        return True
    words = [word for word in text.split() if word]
    if len(words) >= 14:
        return True
    trigger_phrases = (
        "why ",
        "how ",
        "compare ",
        "design ",
        "debug ",
        "implement ",
        "refactor ",
        "architecture",
    )
    lowered = text.lower()
    return any(phrase in lowered for phrase in trigger_phrases)


def load_lore_state(session_db=None) -> HermesLoreState:
    """Load session-derived lore progression for the Hermes skin."""
    lore = HermesLoreState()
    conn = getattr(session_db, "_conn", None)
    owns_connection = False

    if conn is None:
        db_path = get_hermes_home() / "state.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            owns_connection = True

    try:
        if conn is not None:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE source = ?", ("cli",))
            row = cursor.fetchone()
            lore.sessions = int(row[0]) if row else 0

            cursor = conn.execute(
                """SELECT m.content
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.source = ? AND m.role = ?""",
                ("cli", "user"),
            )
            user_messages = [row[0] or "" for row in cursor.fetchall()]
            lore.user_messages = len(user_messages)
            lore.clever_replies = sum(1 for message in user_messages if _is_clever_reply(message))
    finally:
        if owns_connection and conn is not None:
            conn.close()

    blob = _load_lore_blob()
    published = blob.get("published_skills", [])
    lore.published_skills = _iter_unique(
        entry.get("name", "")
        for entry in reversed(published)
        if isinstance(entry, dict)
    )
    return lore


def build_holographic_grid(width: int, phase: int = 0) -> str:
    """Return a single holographic grid line for the banner."""
    width = max(width, 24)
    motifs = ("╳━", "━╳", "╸╺", "╺╸")
    motif = motifs[phase % len(motifs)]
    repeated = (motif * ((width // len(motif)) + 3))[:width]
    return f"[{ARES_BLOOD}]{repeated}[/]"


def build_progress_meter(label: str, current: int, total: int, width: int = 10) -> str:
    """Render a compact progress meter with Rich markup."""
    total = max(total, 1)
    current = max(0, min(current, total))
    width = max(width, 4)
    filled = round((current / total) * width)
    if current > 0:
        filled = max(1, filled)
    bar = ("■" * filled) + ("·" * (width - filled))
    return f"[bold {ARES_BRONZE}]{label:<12}[/] [{ARES_CRIMSON}]{bar}[/] [{ARES_SAND}]{current}/{total}[/]"


def build_orbit_line(lore: HermesLoreState, phase: int = 0, width: int = 32) -> str:
    """Render a compact orbit line for published skill scrolls."""
    width = max(width, 24)
    skills = lore.orbiting_skills[:4]
    slots = ["◌"] * max(4, len(skills) or 4)
    for index, name in enumerate(skills):
        slots[index] = name[:8]
    rotation = phase % len(slots)
    rotated = slots[rotation:] + slots[:rotation]
    orbit = "  ·  ".join(rotated)
    if len(orbit) > width:
        orbit = orbit[: width - 1] + "…"
    return f"◜ {orbit} ◞"


def build_relay_telemetry(
    lore: HermesLoreState,
    phase: int = 0,
    width: int = 42,
    *,
    active: bool = False,
) -> str:
    """Build a plain-text telemetry ribbon for banner and prompt UI."""
    payload = _call_mod("build_relay_telemetry", lore, phase, width, active=active)
    if payload:
        return payload
    width = max(width, 28)
    courier = "▲" if active else "△"
    beacons = ["•", "◦", "•"]
    track_width = max(width - 18, 10)
    track = ["·"] * track_width
    marker = phase % track_width
    track[marker] = courier
    if track_width > 6:
        track[(marker + 4) % track_width] = "╾"
        track[(marker - 4) % track_width] = "╼"
        track[(track_width // 3)] = beacons[(phase // 2) % len(beacons)]
        track[(2 * track_width // 3)] = beacons[(phase // 3 + 1) % len(beacons)]
    status = _mod_attr("ACTIVE_STATUS", "warpath active") if active else _mod_attr("IDLE_STATUS", "warpath primed")
    orbit = len(lore.orbiting_skills)
    return f"{''.join(track)}  {status}  orbit {orbit}"


@lru_cache(maxsize=8)
def _load_png_rgba(asset_name: str) -> tuple[int, int, int, tuple[bytes, ...]] | None:
    """Load a local PNG asset without requiring Pillow."""
    candidate_paths = (get_mod_asset_dir() / asset_name,)
    data = None
    for path in candidate_paths:
        try:
            data = path.read_bytes()
            break
        except OSError:
            continue
    if data is None:
        return None

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    width = height = bit_depth = color_type = None
    idat_chunks: list[bytes] = []
    cursor = 8
    channels_map = {0: 1, 2: 3, 4: 2, 6: 4}

    while cursor + 8 <= len(data):
        length = struct.unpack(">I", data[cursor : cursor + 4])[0]
        cursor += 4
        chunk_type = data[cursor : cursor + 4]
        cursor += 4
        chunk = data[cursor : cursor + length]
        cursor += length
        cursor += 4  # CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth != 8 or color_type not in channels_map:
        return None

    try:
        raw = zlib.decompress(b"".join(idat_chunks))
    except zlib.error:
        return None

    channels = channels_map[color_type]
    stride = width * channels
    rows: list[bytes] = []
    offset = 0
    previous = bytearray(stride)

    def paeth(a: int, b: int, c: int) -> int:
        predictor = a + b - c
        pa = abs(predictor - a)
        pb = abs(predictor - b)
        pc = abs(predictor - c)
        if pa <= pb and pa <= pc:
            return a
        if pb <= pc:
            return b
        return c

    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        scanline = bytearray(raw[offset : offset + stride])
        offset += stride

        if filter_type == 1:
            for index in range(stride):
                left = scanline[index - channels] if index >= channels else 0
                scanline[index] = (scanline[index] + left) & 0xFF
        elif filter_type == 2:
            for index in range(stride):
                scanline[index] = (scanline[index] + previous[index]) & 0xFF
        elif filter_type == 3:
            for index in range(stride):
                left = scanline[index - channels] if index >= channels else 0
                scanline[index] = (scanline[index] + ((left + previous[index]) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(stride):
                left = scanline[index - channels] if index >= channels else 0
                up = previous[index]
                up_left = previous[index - channels] if index >= channels else 0
                scanline[index] = (scanline[index] + paeth(left, up, up_left)) & 0xFF

        rows.append(bytes(scanline))
        previous = scanline

    return width, height, channels, tuple(rows)


def _rgba_at(row: bytes, channels: int, x: int) -> tuple[int, int, int, int]:
    start = x * channels
    pixel = row[start : start + channels]
    if channels == 1:
        gray = pixel[0]
        return gray, gray, gray, 255
    if channels == 2:
        gray, alpha = pixel
        return gray, gray, gray, alpha
    if channels == 3:
        red, green, blue = pixel
        return red, green, blue, 255
    red, green, blue, alpha = pixel
    return red, green, blue, alpha


def _find_visible_bbox(
    width: int,
    height: int,
    channels: int,
    rows: tuple[bytes, ...],
    *,
    ignore_white: bool,
    crop: bool,
) -> tuple[int, int, int, int]:
    if not crop:
        return 0, 0, width - 1, height - 1

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for y, row in enumerate(rows):
        for x in range(width):
            red, green, blue, alpha = _rgba_at(row, channels, x)
            if alpha < 8:
                continue
            if ignore_white and _is_near_white(red, green, blue):
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return 0, 0, width - 1, height - 1
    return min_x, min_y, max_x, max_y


def _fit_image_box(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    scale = max(source_width / max(target_width, 1), source_height / max(target_height, 1), 1.0)
    render_width = max(1, min(target_width, int(round(source_width / scale))))
    render_height = max(1, min(target_height, int(round(source_height / scale))))
    return render_width, render_height


def _average_region_color(
    rows: tuple[bytes, ...],
    channels: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    ignore_white: bool,
) -> tuple[int, int, int] | None:
    total_red = 0.0
    total_green = 0.0
    total_blue = 0.0
    total_weight = 0.0

    for y in range(y0, y1):
        row = rows[y]
        for x in range(x0, x1):
            red, green, blue, alpha = _rgba_at(row, channels, x)
            if alpha < 8:
                continue
            if ignore_white and _is_near_white(red, green, blue):
                continue
            weight = alpha / 255.0
            total_red += red * weight
            total_green += green * weight
            total_blue += blue * weight
            total_weight += weight

    if total_weight <= 0.0:
        return None

    return (
        int(round(total_red / total_weight)),
        int(round(total_green / total_weight)),
        int(round(total_blue / total_weight)),
    )


def _mode_region_color(
    rows: tuple[bytes, ...],
    channels: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    ignore_white: bool,
) -> tuple[int, int, int] | None:
    counts: dict[tuple[int, int, int], float] = {}

    for y in range(y0, y1):
        row = rows[y]
        for x in range(x0, x1):
            red, green, blue, alpha = _rgba_at(row, channels, x)
            if alpha < 8:
                continue
            if ignore_white and _is_near_white(red, green, blue):
                continue
            key = (red, green, blue)
            counts[key] = counts.get(key, 0.0) + (alpha / 255.0)

    if not counts:
        return None

    return max(counts.items(), key=lambda item: item[1])[0]


def _average_region_darkness(
    rows: tuple[bytes, ...],
    channels: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> float:
    total_darkness = 0.0
    total_weight = 0.0

    for y in range(y0, y1):
        row = rows[y]
        for x in range(x0, x1):
            red, green, blue, alpha = _rgba_at(row, channels, x)
            if alpha < 8:
                continue
            luminance = (red * 299 + green * 587 + blue * 114) / 1000.0
            weight = alpha / 255.0
            total_darkness += ((255.0 - luminance) / 255.0) * weight
            total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return total_darkness / total_weight


def _render_half_block_grid(grid: list[list[tuple[int, int, int] | None]]) -> tuple[str, ...]:
    lines: list[str] = []
    for index in range(0, len(grid), 2):
        top_row = grid[index]
        bottom_row = grid[index + 1] if index + 1 < len(grid) else [None] * len(top_row)
        line: list[str] = []
        for top_color, bottom_color in zip(top_row, bottom_row):
            if top_color and bottom_color:
                line.append(
                    f"[{_rgb_to_hex(top_color)} on {_rgb_to_hex(bottom_color)}]▀[/]"
                )
            elif top_color:
                line.append(f"[{_rgb_to_hex(top_color)}]▀[/]")
            elif bottom_color:
                line.append(f"[{_rgb_to_hex(bottom_color)}]▄[/]")
            else:
                line.append(" ")
        lines.append("".join(line).rstrip())
    return tuple(lines)


def _color_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum((left[index] - right[index]) ** 2 for index in range(3))


def _pick_quadrant_palette(
    colors: list[tuple[int, int, int] | None],
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, int]:
    """Choose background, foreground, and bitmask for a 2x2 color cell."""
    indexed_colors = [color for color in colors if color is not None]
    if not indexed_colors:
        return None, None, 0

    unique = list(dict.fromkeys(indexed_colors))
    if len(unique) == 1:
        foreground = unique[0]
        mask = 0
        for bit_index, color in enumerate(colors):
            if color is not None:
                mask |= 1 << bit_index
        return None, foreground, mask

    best_background: tuple[int, int, int] | None = None
    best_foreground: tuple[int, int, int] | None = None
    best_mask = 0
    best_score: int | None = None

    for background in unique:
        for foreground in unique:
            if foreground == background:
                continue
            score = 0
            mask = 0
            for bit_index, color in enumerate(colors):
                if color is None:
                    continue
                distance_to_background = _color_distance(color, background)
                distance_to_foreground = _color_distance(color, foreground)
                if distance_to_foreground < distance_to_background:
                    score += distance_to_foreground
                    mask |= 1 << bit_index
                else:
                    score += distance_to_background
            if best_score is None or score < best_score:
                best_score = score
                best_background = background
                best_foreground = foreground
                best_mask = mask

    return best_background, best_foreground, best_mask


def _render_quadrant_grid(grid: list[list[tuple[int, int, int] | None]]) -> tuple[str, ...]:
    glyphs = {
        0: " ",
        1: "▘",
        2: "▝",
        3: "▀",
        4: "▖",
        5: "▌",
        6: "▞",
        7: "▛",
        8: "▗",
        9: "▚",
        10: "▐",
        11: "▜",
        12: "▄",
        13: "▙",
        14: "▟",
        15: "█",
    }
    lines: list[str] = []

    for y in range(0, len(grid), 2):
        top_row = grid[y]
        bottom_row = grid[y + 1] if y + 1 < len(grid) else [None] * len(top_row)
        line: list[str] = []
        for x in range(0, len(top_row), 2):
            quadrant_colors = [
                top_row[x],
                top_row[x + 1] if x + 1 < len(top_row) else None,
                bottom_row[x],
                bottom_row[x + 1] if x + 1 < len(bottom_row) else None,
            ]
            background, foreground, mask = _pick_quadrant_palette(quadrant_colors)
            glyph = glyphs[mask]
            if foreground is None:
                line.append(" ")
            elif background is None:
                if mask == 0:
                    line.append(" ")
                else:
                    line.append(f"[{_rgb_to_hex(foreground)}]{glyph}[/]")
            elif mask == 0:
                line.append(f"[on {_rgb_to_hex(background)}] [/]")
            elif mask == 15:
                line.append(f"[{_rgb_to_hex(foreground)}]{glyph}[/]")
            else:
                line.append(
                    f"[{_rgb_to_hex(foreground)} on {_rgb_to_hex(background)}]{glyph}[/]"
                )
        lines.append("".join(line).rstrip())

    return tuple(lines)


def _render_braille_grid(grid: list[list[tuple[int, int, int] | None]]) -> tuple[str, ...]:
    """Render a dense 2x4-per-cell braille approximation for hero art."""
    dot_bits = {
        (0, 0): 0x01,
        (0, 1): 0x02,
        (0, 2): 0x04,
        (1, 0): 0x08,
        (1, 1): 0x10,
        (1, 2): 0x20,
        (0, 3): 0x40,
        (1, 3): 0x80,
    }
    lines: list[str] = []

    for y in range(0, len(grid), 4):
        line: list[str] = []
        for x in range(0, len(grid[0]), 2):
            mask = 0
            active_colors: list[tuple[int, int, int]] = []
            for dy in range(4):
                row_index = y + dy
                if row_index >= len(grid):
                    continue
                row = grid[row_index]
                for dx in range(2):
                    col_index = x + dx
                    if col_index >= len(row):
                        continue
                    color = row[col_index]
                    if color is None:
                        continue
                    mask |= dot_bits[(dx, dy)]
                    active_colors.append(color)

            if mask == 0:
                line.append(" ")
                continue

            count = len(active_colors)
            foreground = (
                sum(color[0] for color in active_colors) // count,
                sum(color[1] for color in active_colors) // count,
                sum(color[2] for color in active_colors) // count,
            )
            glyph = chr(0x2800 + mask)
            line.append(f"[{_rgb_to_hex(foreground)}]{glyph}[/]")
        lines.append("".join(line).rstrip())

    return tuple(lines)


@lru_cache(maxsize=16)
def _render_sprite_asset(asset_name: str, width: int, height: int) -> tuple[str, ...] | None:
    loaded = _load_png_rgba(asset_name)
    if loaded is None:
        return None

    image_width, image_height, channels, rows = loaded
    left, top, right, bottom = _find_visible_bbox(
        image_width,
        image_height,
        channels,
        rows,
        ignore_white=True,
        crop=True,
    )
    source_width = right - left + 1
    source_height = bottom - top + 1
    target_pixel_width = max(width * 2, 2)
    target_pixel_height = max(height * 4, 4)
    render_width = target_pixel_width
    render_height = target_pixel_height

    grid: list[list[tuple[int, int, int] | None]] = [
        [None for _ in range(target_pixel_width)] for _ in range(target_pixel_height)
    ]
    for sample_y in range(render_height):
        y0 = top + int(sample_y * source_height / render_height)
        y1 = top + int((sample_y + 1) * source_height / render_height)
        if y1 <= y0:
            y1 = min(bottom + 1, y0 + 1)
        for sample_x in range(render_width):
            x0 = left + int(sample_x * source_width / render_width)
            x1 = left + int((sample_x + 1) * source_width / render_width)
            if x1 <= x0:
                x1 = min(right + 1, x0 + 1)
            grid[sample_y][sample_x] = _mode_region_color(
                rows,
                channels,
                x0,
                x1,
                y0,
                y1,
                ignore_white=True,
            )

    return _render_braille_grid(grid)


@lru_cache(maxsize=16)
def _render_mural_asset(asset_name: str, width: int, height: int) -> tuple[str, ...] | None:
    loaded = _load_png_rgba(asset_name)
    if loaded is None:
        return None

    image_width, image_height, channels, rows = loaded
    left, top, right, bottom = _find_visible_bbox(
        image_width,
        image_height,
        channels,
        rows,
        ignore_white=False,
        crop=False,
    )
    source_width = right - left + 1
    source_height = bottom - top + 1
    target_pixel_height = max(height * 2, 2)
    render_width, render_height = _fit_image_box(
        source_width,
        source_height,
        width,
        target_pixel_height,
    )
    pad_x = max((width - render_width) // 2, 0)
    pad_y = max((target_pixel_height - render_height) // 2, 0)

    low = _hex_to_rgb(ARES_OBSIDIAN)
    high = _hex_to_rgb(ARES_PATINA)
    grid: list[list[tuple[int, int, int] | None]] = [
        [None for _ in range(width)] for _ in range(target_pixel_height)
    ]
    for sample_y in range(render_height):
        y0 = top + int(sample_y * source_height / render_height)
        y1 = top + int((sample_y + 1) * source_height / render_height)
        if y1 <= y0:
            y1 = min(bottom + 1, y0 + 1)
        for sample_x in range(render_width):
            x0 = left + int(sample_x * source_width / render_width)
            x1 = left + int((sample_x + 1) * source_width / render_width)
            if x1 <= x0:
                x1 = min(right + 1, x0 + 1)
            darkness = _average_region_darkness(rows, channels, x0, x1, y0, y1)
            if darkness < 0.07:
                continue
            grid[pad_y + sample_y][pad_x + sample_x] = _mix_rgb(
                low,
                high,
                min(1.0, darkness * 1.35),
            )

    return _render_half_block_grid(grid)


@lru_cache(maxsize=None)
def _load_spartan_rows(x_step: int, y_step: int) -> tuple[str, ...] | None:
    """Load and downsample the Spartan emblem pixel art for terminal display."""
    path = get_mod_asset_dir() / "spartan_emblem_pixel_art_transparent.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    pixels = payload.get("pixels")
    transparent = payload.get("transparent", ".")
    if not isinstance(pixels, list) or not pixels:
        return None

    rows: list[str] = []

    for y in range(0, len(pixels), y_step):
        downsampled: list[str] = []
        for x in range(0, len(pixels[0]), x_step):
            block = [row[x : x + x_step] for row in pixels[y : y + y_step]]
            values = [
                char
                for row in block
                for char in row
                if char and char != transparent
            ]
            if not values:
                downsampled.append(transparent)
                continue
            counts: dict[str, int] = {}
            for char in values:
                counts[char] = counts.get(char, 0) + 1
            downsampled.append(max(counts.items(), key=lambda item: item[1])[0])
        rows.append("".join(downsampled))

    while rows and not rows[0].strip(transparent):
        rows.pop(0)
    while rows and not rows[-1].strip(transparent):
        rows.pop()
    if not rows:
        return None

    left = min(
        idx
        for row in rows
        for idx, char in enumerate(row)
        if char != transparent
    )
    right = max(
        idx
        for row in rows
        for idx, char in enumerate(row)
        if char != transparent
    )
    return tuple(row[left : right + 1] for row in rows)


def _render_emblem_row(row: str) -> str:
    palette = {
        "0": ARES_SAND,
        "1": "#D8C08A",
        "2": ARES_BRONZE,
        "3": "#B98D48",
        "4": "#8D6E4A",
        "5": ARES_ASH,
        "6": ARES_STEEL,
        "7": ARES_BLOOD,
        "8": "#821B1B",
        "9": ARES_CRIMSON,
        "a": "#B22727",
        "b": "#C73030",
        "c": ARES_EMBER,
        "d": "#7D2626",
        "e": ARES_OBSIDIAN,
    }
    out: list[str] = []
    active_color: str | None = None
    for char in row:
        if char == ".":
            if active_color is not None:
                out.append("[/]")
                active_color = None
            out.append(" ")
            continue
        color = palette.get(char, ARES_ASH)
        if color != active_color:
            if active_color is not None:
                out.append("[/]")
            out.append(f"[{color}]")
            active_color = color
        out.append("█")
    if active_color is not None:
        out.append("[/]")
    return "".join(out)


def _render_drip_line(width: int, tokens: tuple[tuple[int, str, str], ...]) -> str:
    line: list[str] = []
    active_color: str | None = None
    lookup = {index: (color, glyph) for index, color, glyph in tokens}
    for idx in range(width):
        token = lookup.get(idx)
        if token is None:
            if active_color is not None:
                line.append("[/]")
                active_color = None
            line.append(" ")
            continue
        color, glyph = token
        if color != active_color:
            if active_color is not None:
                line.append("[/]")
            line.append(f"[{color}]")
            active_color = color
        line.append(glyph)
    if active_color is not None:
        line.append("[/]")
    return "".join(line)


def _render_drip_lines(width: int, phase: int = 0) -> list[str]:
    frames = (
        (
            ((7, ARES_BLOOD, "╻"), (18, ARES_CRIMSON, "╻")),
            ((7, ARES_EMBER, "╹"), (18, ARES_EMBER, "╹")),
        ),
        (
            ((8, ARES_BLOOD, "╻"), (14, ARES_CRIMSON, "╻"), (19, ARES_BLOOD, "╻")),
            ((8, ARES_CRIMSON, "│"), (14, ARES_EMBER, "╹"), (19, ARES_CRIMSON, "╹")),
            ((8, ARES_EMBER, "╹"),),
        ),
        (
            ((9, ARES_CRIMSON, "╻"), (16, ARES_BLOOD, "╻")),
            ((9, ARES_EMBER, "│"), (16, ARES_CRIMSON, "│")),
            ((9, ARES_EMBER, "╹"), (16, ARES_EMBER, "╹")),
        ),
    )
    frame = frames[phase % len(frames)]
    return [_render_drip_line(width, line) for line in frame]


def _render_mask_row(row: str, color: str, char: str = "█") -> str:
    out: list[str] = []
    active = False
    for pixel in row:
        if pixel == ".":
            if active:
                out.append("[/]")
                active = False
            out.append(" ")
            continue
        if not active:
            out.append(f"[{color}]")
            active = True
        out.append(char)
    if active:
        out.append("[/]")
    return "".join(out)


def build_war_wall(width: int = 28, height: int = 18, phase: int = 0) -> list[str]:
    """Render a subdued mural for the right side of the dossier banner."""
    width = max(width, 18)
    height = max(height, 10)
    mural = _render_mural_asset("ascii-art.png", width, height)
    if mural:
        return list(mural)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col in range(2, width, 6):
        offset = (phase + col) % 4
        for row in range(offset + 2, height, 4):
            grid[row][col] = "▲"
            if row + 1 < height:
                grid[row + 1][col] = "│"

    lines: list[str] = []
    for row in grid:
        lines.append("".join(f"[{ARES_INK}]{char}[/]" if char.strip() else " " for char in row))
    return lines


def get_caduceus_frame(
    lore: HermesLoreState,
    phase: int = 0,
    *,
    width: int = 40,
    height: int = 22,
) -> str:
    """Select the active hero art for the banner."""
    payload = _call_mod("get_hero_art", width, height, phase, lore=lore)
    if payload:
        return str(payload)

    del lore, phase
    hero_assets = tuple(_mod_attr("HERO_ASSETS", ("pixel_art_large-2.png", "pixel_art_large.png")))
    for asset_name in hero_assets:
        hero_rows = _render_sprite_asset(asset_name, width=width, height=height)
        if hero_rows:
            return "\n".join(hero_rows)

    emblem_rows = _load_spartan_rows(2, 3)
    if emblem_rows:
        rendered_rows = [_render_emblem_row(row) for row in emblem_rows]
        return "\n".join(rendered_rows)
    return CADUCEUS_FRAMES[0]


def get_banner_title(lore: HermesLoreState) -> str:
    """Banner title line for the Hermes skin."""
    payload = _call_mod("get_banner_title", lore.glow_enabled)
    if payload:
        return payload
    if lore.glow_enabled:
        return f"[bold {ARES_SAND}]Ares Agent · Ember Command Core[/]"
    return f"[bold {ARES_SAND}]Ares Agent · Spartan Terminal Core[/]"


def get_lore_lines(lore: HermesLoreState) -> list[str]:
    """Right-panel lore lines for the Hermes banner."""
    payload = _call_mod("get_lore_lines", lore)
    if payload:
        return list(payload)
    lore_heading = _mod_attr("LORE_HEADING", "Ares Lore")
    empty_orbit = _mod_attr("EMPTY_ORBITING_SCROLLS", "none yet")
    lines = [
        "",
        f"[bold {ARES_BRONZE}]{lore_heading}[/]",
        f"[dim {ARES_ASH}]Sessions:[/] [{ARES_SAND}]{lore.sessions}[/] [dim {ARES_ASH}]· Clever replies:[/] [{ARES_SAND}]{lore.clever_replies}[/]",
        f"[dim {ARES_ASH}]Visual tier:[/] [{ARES_SAND}]{lore.summary()}[/]",
    ]
    if lore.orbiting_skills:
        orbiting = " · ".join(lore.orbiting_skills)
        lines.append(f"[dim {ARES_ASH}]Orbiting scrolls:[/] [{ARES_SAND}]{orbiting}[/]")
    else:
        lines.append(f"[dim {ARES_ASH}]Orbiting scrolls:[/] [{ARES_SAND}]{empty_orbit}[/]")
    return lines


def build_speed_line(width: int, phase: int = 0) -> str:
    """Ambient separator rendered before and after scroll content."""
    payload = _call_mod("build_speed_line", width, phase)
    if payload:
        return str(payload)
    width = max(width, 24)
    trails = ("▲┄┄", "┄▲┄", "┄┄▲", "▼┄┄")
    trail = trails[phase % len(trails)]
    body_width = max(width - len(trail) * 2 - 2, 0)
    return f"{trail}{'─' * body_width}{trail}"


def build_scroll_frame(width: int, lore: HermesLoreState, phase: int = 0) -> tuple[str, str, str]:
    """Return (top, subtitle, bottom) frame strings for Hermes responses."""
    payload = _call_mod("build_scroll_frame", width, lore, phase)
    if payload and len(payload) == 3:
        return tuple(payload)
    width = max(width, 36)
    title = MESSENGER_TITLES[phase % len(MESSENGER_TITLES)]
    accent = "bronze seal"
    if lore.glow_enabled:
        accent += " · ember wake"
    fill = max(width - len(title) - len(accent) - 10, 0)
    top = f"╭═▲ {title} · {accent} {'═' * fill}╮"
    subtitle = build_relay_telemetry(lore, phase, width - 4, active=True)
    bottom_fill = max(width - 34, 0)
    bottom = f"╰═ dispatch returns to the war room {'═' * bottom_fill}╯"
    return top, subtitle, bottom


def maybe_create_trickster_note(message: str, enabled: bool = True, chance: float = 0.01) -> str | None:
    """Rare Hermes-only note that steals a letter or fixes a typo."""
    payload = _call_mod("maybe_create_trickster_note", message, enabled=enabled, chance=chance)
    if payload is not None:
        return str(payload) if payload else None
    if not enabled or not message or random.random() > chance:
        return None
    lowered = message.lower()
    for typo, correction in TRICKSTER_CORRECTIONS.items():
        if typo in lowered:
            return f"Ares struck '{typo}' and reforged it as '{correction}'."

    letters = [char for char in message if char.isalpha()]
    if not letters:
        return None
    stolen = random.choice(letters)
    return f"Ares knocked '{stolen}' from your last message and left it on the shield rim."


def format_flip_result(result: str) -> str:
    """Narration for the coin flip easter egg."""
    payload = _call_mod("format_flip_result", result)
    if payload:
        return str(payload)
    if result == "heads":
        return "heads · the shield face lands forward"
    if result == "tails":
        return "tails · the spear butt hits first, but Ares keeps the edge"
    return result


def parse_dice_spec(spec: str | None) -> int:
    """Parse a dice specifier like d20 or 20 into a positive number of sides."""
    if not spec:
        return 6
    cleaned = spec.strip().lower()
    if cleaned.startswith("d"):
        cleaned = cleaned[1:]
    try:
        sides = int(cleaned)
    except ValueError:
        return 6
    return max(2, min(sides, 100))
