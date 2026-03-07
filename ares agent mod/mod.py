"""Ares Agent mod payload.

This folder is the portable payload for the Ares visual skin: palette,
branding copy, prompt/spinner assets, and image paths all live here.
"""

from __future__ import annotations

from pathlib import Path


MOD_NAME = "ares-agent-mod"
MOD_VERSION = "1.0.0"
BRAND_NAME = "Ares Agent"
OMENS_TITLE = "Ares Omens"
LORE_HEADING = "Ares Lore"
UNIT_DESIGNATION = "UNIT DESIGNATION: MILITARY INTELLIGENCE // WAR DEPARTMENT // Ares-001"
WELCOME_MESSAGE = "Welcome to Ares Agent! Type your message or /help for commands."
PLACEHOLDER_TEXT = "ask Ares, or try /flip, /roll d20, /skin hermes"
ACTIVE_HINT_TEMPLATE = "  {glyph} warpath in flight · type to interrupt · Ctrl+C to break"
IDLE_HINT_TEMPLATE = "  {glyph} shield line ready · rituals /flip /roll d20 · orbit {orbit_count}"
HELP_SUFFIX = "/help - for Sparta"
SKIN_STATUS_LABEL = "Skin"
EMBER_CORE_TITLE = "Ares Agent · Ember Command Core"
SPARTAN_CORE_TITLE = "Ares Agent · Spartan Terminal Core"
EMPTY_ORBITING_SCROLLS = "none yet"
PLAIN_EMPTY_ORBITING = "awaiting published scrolls"

ARES_CRIMSON = "#9F1C1C"
ARES_BLOOD = "#6B1717"
ARES_EMBER = "#DD4A3A"
ARES_BRONZE = "#C7A96B"
ARES_SAND = "#F1E6CF"
ARES_ASH = "#6E584B"
ARES_STEEL = "#51433B"
ARES_OBSIDIAN = "#1A1513"
ARES_INK = "#241B18"
ARES_PATINA = "#8E6A42"

COIN_SPIN_FRAMES = ("◐", "◓", "◑", "◒", "◐", "◎")
DI20_GLYPHS = ("◢", "◣", "◤", "◥", "⬢", "⬡")
MESSENGER_TITLES = (
    "Ares Dispatch",
    "War Scroll",
    "Iron Decree",
)
TRICKSTER_CORRECTIONS = {
    "teh": "the",
    "adn": "and",
    "heremes": "Ares",
    "definately": "definitely",
    "wierd": "weird",
}
SPINNER_WINGS = (
    ("⟪⚔", "⚔⟫"),
    ("⟪▲", "▲⟫"),
    ("⟪╸", "╺⟫"),
    ("⟪⛨", "⛨⟫"),
)
WAITING_FACES = (
    "(⚔)",
    "(⛨)",
    "(▲)",
    "(<> )",
    "(/)",
)
THINKING_FACES = (
    "(⚔)",
    "(⛨)",
    "(▲)",
    "(⌁)",
    "(<> )",
)
THINKING_VERBS = (
    "forging",
    "marching",
    "sizing the field",
    "holding the line",
    "hammering plans",
    "tempering steel",
    "plotting impact",
    "raising the shield",
)
ACTIVE_PROMPT_FRAMES = (
    "⟪⚔⟫ ",
    "⟪▲⟫ ",
    "⟪⛨⟫ ",
    "⟪⚔⟫ ",
)
IDLE_PROMPT_FRAMES = (
    "⚔ ",
    "⛨ ",
    "▲ ",
    "⚔ ",
)
RITUALS = (
    ("/flip", "shield omen"),
    ("/roll d20", "weighted Ares dice"),
    ("flip coin", "local shortcut"),
    ("roll dice", "local shortcut"),
)

PIXEL_FONT = {
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


def get_asset_dir() -> Path:
    return Path(__file__).resolve().parent


def build_ares_masthead() -> str:
    return """[bold #A3261F] █████╗ ██████╗ ███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #B73122]██╔══██╗██╔══██╗██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#C93C24]███████║██████╔╝█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#D84A28]██╔══██║██╔══██╗██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#E15A2D]██║  ██║██║  ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#EB6C32]╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""


def get_banner_title(glow_enabled: bool) -> str:
    title = EMBER_CORE_TITLE if glow_enabled else SPARTAN_CORE_TITLE
    return f"[bold {ARES_SAND}]{title}[/]"


def get_help_footer(tool_count: int, skill_count: int) -> str:
    return f"{tool_count} tools · {skill_count} skills · {HELP_SUFFIX}"


def get_welcome_message() -> str:
    return WELCOME_MESSAGE


def get_placeholder_text() -> str:
    return PLACEHOLDER_TEXT


def get_hint_bar(agent_running: bool, glyph: str, orbit_count: int) -> str:
    if agent_running:
        return ACTIVE_HINT_TEMPLATE.format(glyph=glyph, orbit_count=orbit_count)
    return IDLE_HINT_TEMPLATE.format(glyph=glyph, orbit_count=orbit_count)


def get_version_title(version: str) -> str:
    return f"{BRAND_NAME} {version}"
