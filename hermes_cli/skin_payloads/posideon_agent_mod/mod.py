"""Posideon Agent mod payload.

Portable theme data for the Posideon visual skin.
"""

from __future__ import annotations

from pathlib import Path


MOD_NAME = "posideon-agent-mod"
MOD_VERSION = "1.0.0"
BRAND_NAME = "Posideon Agent"
ASSISTANT_NAME = "Posideon"
AGENT_GLYPH = "Ψ"
OMENS_TITLE = "Posideon Signals"
LORE_HEADING = "Tide Readings"
UNIT_DESIGNATION = "UNIT DESIGNATION: OCEAN INTELLIGENCE // DEEPWATER DESK // Posideon-001"
WELCOME_MESSAGE = "Welcome to Posideon Agent! Type your message or /help for commands."
PLACEHOLDER_TEXT = "ask Posideon, or try /flip, /roll d20, /skin hermes"
ACTIVE_HINT_TEMPLATE = "  {glyph} tidewatch active · type to interrupt · Ctrl+C to break"
IDLE_HINT_TEMPLATE = "  {glyph} harbor steady · rituals /flip /roll d20 · orbit {orbit_count}"
HELP_SUFFIX = "/help - sound the horn"
SKIN_STATUS_LABEL = "Current"
COMPACT_TAGLINE = "Oceanic CLI Skin"
COMPACT_DESCRIPTION = "Deepwater routing, trident lore, and tide telemetry"
EMPTY_ORBITING_SCROLLS = "no drifting signals"
PLAIN_EMPTY_ORBITING = "waiting for the next tide"
ACTIVE_STATUS = "tidewatch active"
IDLE_STATUS = "harbor steady"
PROGRESS_LABELS = ("Current pressure", "Foam line", "Signal orbit")
NEXT_PROGRESS_LABELS = ("Next swell", "Next harbor mark")

ARES_CRIMSON = "#2A6FB9"
ARES_BLOOD = "#153C73"
ARES_EMBER = "#5DB8F5"
ARES_BRONZE = "#A9DFFF"
ARES_SAND = "#EAF7FF"
ARES_ASH = "#6FA6C8"
ARES_STEEL = "#496884"
ARES_OBSIDIAN = "#091320"
ARES_INK = "#10263A"
ARES_PATINA = "#3C93D1"

COIN_SPIN_FRAMES = ("◜", "◠", "◝", "◞", "◡", "◟")
DI20_GLYPHS = ("◈", "⬡", "⬢", "◉", "◇", "◆")
MESSENGER_TITLES = (
    "Tide Report",
    "Deepwater Dispatch",
    "Foam Ledger",
)
TRICKSTER_CORRECTIONS = {
    "teh": "the",
    "adn": "and",
    "posiedon": "Posideon",
    "definately": "definitely",
    "wierd": "weird",
}
SPINNER_WINGS = (
    ("⟪≈", "≈⟫"),
    ("⟪Ψ", "Ψ⟫"),
    ("⟪∿", "∿⟫"),
    ("⟪◌", "◌⟫"),
)
WAITING_FACES = (
    "(≈)",
    "(Ψ)",
    "(∿)",
    "(◌)",
    "(◠)",
)
THINKING_FACES = (
    "(Ψ)",
    "(≈)",
    "(∿)",
    "(◉)",
    "(◌)",
)
THINKING_VERBS = (
    "charting currents",
    "sounding the depth",
    "reading foam lines",
    "steering the trident",
    "tracking undertow",
    "plotting sea lanes",
    "calling the swell",
    "measuring pressure",
)
ACTIVE_PROMPT_FRAMES = (
    "⟪Ψ⟫ ",
    "⟪≈⟫ ",
    "⟪∿⟫ ",
    "⟪◌⟫ ",
)
IDLE_PROMPT_FRAMES = (
    "Ψ ",
    "≈ ",
    "∿ ",
    "Ψ ",
)
RITUALS = (
    ("/flip", "tide coin"),
    ("/roll d20", "storm roll"),
    ("flip coin", "local shortcut"),
    ("roll dice", "local shortcut"),
)
HERO_ASSETS = ("posideon.png",)


def get_asset_dir() -> Path:
    return Path(__file__).resolve().parent


def build_masthead() -> str:
    return """[bold #B8E8FF]██████╗  ██████╗ ███████╗██╗██████╗ ███████╗ ██████╗ ███╗   ██╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #97D6FF]██╔══██╗██╔═══██╗██╔════╝██║██╔══██╗██╔════╝██╔═══██╗████╗  ██║      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#75C1F6]██████╔╝██║   ██║███████╗██║██║  ██║█████╗  ██║   ██║██╔██╗ ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#4FA2E0]██╔═══╝ ██║   ██║╚════██║██║██║  ██║██╔══╝  ██║   ██║██║╚██╗██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#2E7CC7]██║     ╚██████╔╝███████║██║██████╔╝███████╗╚██████╔╝██║ ╚████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#1B4F95]╚═╝      ╚═════╝ ╚══════╝╚═╝╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""


def get_banner_title(glow_enabled: bool) -> str:
    title = "Posideon Agent · Tidewatch Command Deck" if glow_enabled else "Posideon Agent · Deepwater Relay Desk"
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


def build_relay_telemetry(lore, phase: int, width: int, *, active: bool = False) -> str:
    width = max(width, 28)
    beacon = "Ψ" if active else "◌"
    track_width = max(width - 24, 10)
    track = ["·"] * track_width
    marker = phase % track_width
    track[marker] = beacon
    if track_width > 8:
        track[(marker + 5) % track_width] = "╾"
        track[(marker - 5) % track_width] = "╼"
        track[track_width // 3] = "◦"
        track[(2 * track_width) // 3] = "•"
    status = ACTIVE_STATUS if active else IDLE_STATUS
    return f"{''.join(track)}  {status}  orbit {len(lore.orbiting_skills)}"


def build_speed_line(width: int, phase: int = 0) -> str:
    width = max(width, 24)
    trails = ("∿∿Ψ", "≈Ψ≈", "Ψ∿∿", "≋≈Ψ")
    trail = trails[phase % len(trails)]
    body_width = max(width - len(trail) * 2 - 2, 0)
    return f"{trail}{'─' * body_width}{trail}"


def build_scroll_frame(width: int, lore, phase: int = 0) -> tuple[str, str, str]:
    width = max(width, 36)
    title = MESSENGER_TITLES[phase % len(MESSENGER_TITLES)]
    accent = "tide ledger"
    if lore.glow_enabled:
        accent += " · stormglass lit"
    fill = max(width - len(title) - len(accent) - 10, 0)
    top = f"╭═Ψ {title} · {accent} {'═' * fill}╮"
    subtitle = build_relay_telemetry(lore, phase, width - 4, active=True)
    bottom_fill = max(width - 34, 0)
    bottom = f"╰═ dispatch settles beneath the swell {'═' * bottom_fill}╯"
    return top, subtitle, bottom


def get_lore_lines(lore) -> list[str]:
    lines = [
        "",
        f"[bold {ARES_BRONZE}]{LORE_HEADING}[/]",
        f"[dim {ARES_ASH}]Sessions:[/] [{ARES_SAND}]{lore.sessions}[/] [dim {ARES_ASH}]· Clever replies:[/] [{ARES_SAND}]{lore.clever_replies}[/]",
        f"[dim {ARES_ASH}]Deck:[/] [{ARES_SAND}]{'tidewatch' if lore.glow_enabled else 'harbor desk'}[/]",
    ]
    if lore.orbiting_skills:
        orbiting = " · ".join(lore.orbiting_skills)
        lines.append(f"[dim {ARES_ASH}]Signals:[/] [{ARES_SAND}]{orbiting}[/]")
    else:
        lines.append(f"[dim {ARES_ASH}]Signals:[/] [{ARES_SAND}]{EMPTY_ORBITING_SCROLLS}[/]")
    return lines


def maybe_create_trickster_note(message: str, enabled: bool = True, chance: float = 0.01) -> str | None:
    if not enabled or not message:
        return ""
    lowered = message.lower()
    for typo, correction in TRICKSTER_CORRECTIONS.items():
        if typo in lowered:
            return f"Posideon caught '{typo}' in the undertow and returned '{correction}'."
    return ""


def format_flip_result(result: str) -> str:
    if result == "heads":
        return "heads · the tide rises in your favor"
    if result == "tails":
        return "tails · undertow says hold position"
    return result
