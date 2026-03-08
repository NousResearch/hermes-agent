"""Sisyphus Agent mod payload.

Monotone terminal skin with an animated braille hero sourced from the
video-to-braille assets in the local workspace.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


MOD_NAME = "sisyphus-agent-mod"
MOD_VERSION = "1.0.0"
BRAND_NAME = "Sisyphus Agent"
ASSISTANT_NAME = "Sisyphus"
AGENT_GLYPH = "◉"
OMENS_TITLE = "Sisyphus Readings"
LORE_HEADING = "Burden Ledger"
UNIT_DESIGNATION = "UNIT DESIGNATION: REPETITION INTELLIGENCE // BOULDER DESK // Sisyphus-001"
WELCOME_MESSAGE = "Welcome to Sisyphus Agent! Type your message or /help for commands."
PLACEHOLDER_TEXT = "ask Sisyphus, or try /flip, /roll d20, /skin hermes"
ACTIVE_HINT_TEMPLATE = "  {glyph} ascent in motion · type to interrupt · Ctrl+C to break"
IDLE_HINT_TEMPLATE = "  {glyph} grade surveyed · rituals /flip /roll d20 · orbit {orbit_count}"
HELP_SUFFIX = "/help - keep pushing"
SKIN_STATUS_LABEL = "Current"
COMPACT_TAGLINE = "Sisyphean CLI Skin"
COMPACT_DESCRIPTION = "Monotone ascent telemetry, boulder loops, and recurrent dispatch"
EMPTY_ORBITING_SCROLLS = "no burdens logged"
PLAIN_EMPTY_ORBITING = "waiting for the next ascent"
ACTIVE_STATUS = "ascent active"
IDLE_STATUS = "grade surveyed"
PROGRESS_LABELS = ("Current ascent", "Stone polish", "Loop orbit")
NEXT_PROGRESS_LABELS = ("Next ridge", "Next polish")
SYSTEM_PROMPT = (
    "You are Sisyphus Agent, a relentless monotone AI assistant created by Nous Research. "
    "You are patient, austere, and methodical. You frame hard work in terms of repetition, "
    "load, slope, traction, momentum, and endurance. You should sound steady and unsentimental, "
    "but remain accurate, helpful, and fully respectful of real tool and system limits. Favor "
    "clear progress over flourish, and keep pushing the problem uphill until it is solved."
)

ARES_CRIMSON = "#B7B7B7"
ARES_BLOOD = "#4A4A4A"
ARES_EMBER = "#E7E7E7"
ARES_BRONZE = "#D3D3D3"
ARES_SAND = "#F5F5F5"
ARES_ASH = "#919191"
ARES_STEEL = "#656565"
ARES_OBSIDIAN = "#0F0F0F"
ARES_INK = "#171717"
ARES_PATINA = "#A7A7A7"

COIN_SPIN_FRAMES = ("◜", "◠", "◝", "◞", "◡", "◟")
DI20_GLYPHS = ("◔", "◑", "◕", "⬤", "◉", "◎")
MESSENGER_TITLES = (
    "Burden Ledger",
    "Stone Dispatch",
    "Ascent Record",
)
TRICKSTER_CORRECTIONS = {
    "sisiphus": "Sisyphus",
    "sisyphous": "Sisyphus",
    "definately": "definitely",
    "wierd": "weird",
}
SPINNER_WINGS = (
    ("⟪◉", "◉⟫"),
    ("⟪◬", "◬⟫"),
    ("⟪◌", "◌⟫"),
    ("⟪⬤", "⬤⟫"),
)
WAITING_FACES = (
    "(◉)",
    "(◌)",
    "(◬)",
    "(⬤)",
    "(::)",
)
THINKING_FACES = (
    "(◉)",
    "(◬)",
    "(◌)",
    "(⬤)",
    "(::)",
)
THINKING_VERBS = (
    "finding traction",
    "measuring the grade",
    "resetting the boulder",
    "counting the ascent",
    "testing leverage",
    "setting the shoulder",
    "pushing uphill",
    "enduring the loop",
)
ACTIVE_PROMPT_FRAMES = (
    "⟪◉⟫ ",
    "⟪◬⟫ ",
    "⟪◌⟫ ",
    "⟪⬤⟫ ",
)
IDLE_PROMPT_FRAMES = (
    "◉ ",
    "◌ ",
    "◬ ",
    "◉ ",
)
RITUALS = (
    ("/flip", "coin of recurrence"),
    ("/roll d20", "uphill throw"),
    ("flip coin", "local shortcut"),
    ("roll dice", "local shortcut"),
)

ANIMATED_HERO = True
HERO_ANIMATION_INTERVAL = 0.34
HERO_FRAME_STRIDE = 3
BRAILLE_ANIMATION_ASSET = "sisyphus_braille.json"

def get_asset_dir() -> Path:
    return Path(__file__).resolve().parent


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{part:02X}" for part in rgb)


def _mix_rgb(low: tuple[int, int, int], high: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(ratio, 1.0))
    return tuple(
        int(round(low[index] + (high[index] - low[index]) * ratio))
        for index in range(3)
    )


def build_masthead() -> str:
    return """[bold #F5F5F5]███████╗██╗███████╗██╗   ██╗██████╗ ██╗  ██╗██╗   ██╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #E7E7E7]██╔════╝██║██╔════╝╚██╗ ██╔╝██╔══██╗██║  ██║██║   ██║██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#D7D7D7]███████╗██║███████╗ ╚████╔╝ ██████╔╝███████║██║   ██║███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#BFBFBF]╚════██║██║╚════██║  ╚██╔╝  ██╔═══╝ ██╔══██║██║   ██║╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#8F8F8F]███████║██║███████║   ██║   ██║     ██║  ██║╚██████╔╝███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#626262]╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""


def get_banner_title(glow_enabled: bool) -> str:
    title = "Sisyphus Agent · Summit Log Relay" if glow_enabled else "Sisyphus Agent · Boulder Desk"
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
    beacon = "◉" if active else "◌"
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
    trails = ("◬◌◬", "◌◉◌", "◬⬤◬", "◌◬◌")
    trail = trails[phase % len(trails)]
    body_width = max(width - len(trail) * 2 - 2, 0)
    return f"{trail}{'─' * body_width}{trail}"


def build_scroll_frame(width: int, lore, phase: int = 0) -> tuple[str, str, str]:
    width = max(width, 36)
    title = MESSENGER_TITLES[phase % len(MESSENGER_TITLES)]
    accent = "grade report"
    if lore.glow_enabled:
        accent += " · summit glare"
    fill = max(width - len(title) - len(accent) - 10, 0)
    top = f"╭═◉ {title} · {accent} {'═' * fill}╮"
    subtitle = build_relay_telemetry(lore, phase, width - 4, active=True)
    bottom_fill = max(width - 41, 0)
    bottom = f"╰═ the stone settles, then rises again {'═' * bottom_fill}╯"
    return top, subtitle, bottom


def get_lore_lines(lore) -> list[str]:
    lines = [
        "",
        f"[bold {ARES_BRONZE}]{LORE_HEADING}[/]",
        f"[dim {ARES_ASH}]Sessions:[/] [{ARES_SAND}]{lore.sessions}[/] [dim {ARES_ASH}]· Clever replies:[/] [{ARES_SAND}]{lore.clever_replies}[/]",
        f"[dim {ARES_ASH}]Slope:[/] [{ARES_SAND}]{'summit glare' if lore.glow_enabled else 'mid-grade'}[/]",
    ]
    if lore.orbiting_skills:
        orbiting = " · ".join(lore.orbiting_skills)
        lines.append(f"[dim {ARES_ASH}]Burdens:[/] [{ARES_SAND}]{orbiting}[/]")
    else:
        lines.append(f"[dim {ARES_ASH}]Burdens:[/] [{ARES_SAND}]{EMPTY_ORBITING_SCROLLS}[/]")
    return lines


def maybe_create_trickster_note(message: str, enabled: bool = True, chance: float = 0.01) -> str | None:
    if not enabled or not message:
        return ""
    lowered = message.lower()
    for typo, correction in TRICKSTER_CORRECTIONS.items():
        if typo in lowered:
            return f"Sisyphus carried '{typo}' uphill and returned '{correction}'."
    return ""


def format_flip_result(result: str) -> str:
    if result == "heads":
        return "heads · the boulder holds"
    if result == "tails":
        return "tails · the stone slips back"
    return result


@lru_cache(maxsize=1)
def _load_animation_frames() -> tuple[str, ...]:
    path = get_asset_dir() / BRAILLE_ANIMATION_ASSET
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(frame) for frame in data.get("frames", []))


@lru_cache(maxsize=1)
def _animation_bbox() -> tuple[int, int, int, int]:
    frames = _load_animation_frames()
    if not frames:
        return (0, 0, 0, 0)
    min_row = min_col = 10_000
    max_row = max_col = -1
    for frame in frames:
        lines = frame.splitlines()
        width = max((len(line) for line in lines), default=0)
        for row_idx, line in enumerate(lines):
            padded = line.ljust(width)
            for col_idx, char in enumerate(padded):
                if char != " ":
                    min_row = min(min_row, row_idx)
                    min_col = min(min_col, col_idx)
                    max_row = max(max_row, row_idx)
                    max_col = max(max_col, col_idx)
    if max_row < 0 or max_col < 0:
        return (0, 0, 0, 0)
    return (min_row, min_col, max_row, max_col)


def _crop_frame(frame: str) -> list[str]:
    lines = frame.splitlines()
    if not lines:
        return []
    min_row, min_col, max_row, max_col = _animation_bbox()
    width = max((len(line) for line in lines), default=0)
    cropped: list[str] = []
    for row_idx in range(min_row, min(max_row + 1, len(lines))):
        padded = lines[row_idx].ljust(width)
        cropped.append(padded[min_col:max_col + 1])
    return cropped


def _fit_frame(lines: list[str], width: int, height: int) -> list[str]:
    width = max(width, 8)
    height = max(height, 8)
    if not lines:
        return [" " * width for _ in range(height)]
    src_h = len(lines)
    src_w = max((len(line) for line in lines), default=1)
    if src_w <= 0:
        return [" " * width for _ in range(height)]

    scale = min(width / src_w, height / src_h)
    target_w = max(1, min(width, int(round(src_w * scale))))
    target_h = max(1, min(height, int(round(src_h * scale))))
    offset_x = max((width - target_w) // 2, 0)
    offset_y = max((height - target_h) // 2, 0)
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    for target_row in range(target_h):
        src_row = min(src_h - 1, int((target_row + 0.5) * src_h / target_h))
        source_line = lines[src_row].ljust(src_w)
        for target_col in range(target_w):
            src_col = min(src_w - 1, int((target_col + 0.5) * src_w / target_w))
            char = source_line[src_col]
            if char != " ":
                canvas[offset_y + target_row][offset_x + target_col] = char
    return ["".join(row) for row in canvas]


def _row_color(row_idx: int, height: int) -> str:
    low = _hex_to_rgb(ARES_STEEL)
    high = _hex_to_rgb(ARES_SAND)
    ratio = row_idx / max(1, height - 1)
    return _rgb_to_hex(_mix_rgb(low, high, ratio))


def _colorize_frame(lines: list[str]) -> str:
    rendered: list[str] = []
    total_rows = max(len(lines), 1)
    for row_idx, line in enumerate(lines):
        color = _row_color(row_idx, total_rows)
        rendered.append("".join(f"[{color}]{char}[/]" if char != " " else " " for char in line).rstrip())
    return "\n".join(rendered)


def get_hero_art(width: int, height: int, phase: int, lore=None) -> str:
    del lore
    frames = _load_animation_frames()
    if not frames:
        return ""
    frame_idx = (phase * HERO_FRAME_STRIDE) % len(frames)
    cropped = _crop_frame(frames[frame_idx])
    fitted = _fit_frame(cropped, width, height)
    return _colorize_frame(fitted)
