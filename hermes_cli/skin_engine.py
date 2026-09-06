"""Hermes skin/theme engine — the theme SDK for every surface."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


@dataclass
class SkinConfig:
    """Complete skin configuration."""
    name: str
    description: str = ""
    colors: Dict[str, str] = field(default_factory=dict)
    # Paired palettes for the opposite background polarity (mirrors the desktop app's
    # colors/darkColors pairing): a light terminal prefers `light_colors` (falling back to
    # `colors`), and vice versa for `dark_colors`.
    light_colors: Dict[str, str] = field(default_factory=dict)
    dark_colors: Dict[str, str] = field(default_factory=dict)
    spinner: Dict[str, Any] = field(default_factory=dict)
    branding: Dict[str, str] = field(default_factory=dict)
    tool_prefix: str = "┊"
    tool_emojis: Dict[str, str] = field(default_factory=dict)  # per-tool emoji overrides
    banner_logo: str = ""    # Rich-markup ASCII art logo (replaces HERMES_AGENT_LOGO)
    banner_hero: str = ""    # Rich-markup hero art (replaces HERMES_CADUCEUS)

    def get_color(self, key: str, fallback: str = "") -> str:
        return self.colors.get(key, fallback)

    def get_branding(self, key: str, fallback: str = "") -> str:
        return self.branding.get(key, fallback)

    def get_spinner_wings(self) -> List[Tuple[str, str]]:
        """Spinner wing pairs, or empty list if none."""
        return [(str(pair[0]), str(pair[1])) for pair in self.spinner.get("wings", [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2]


def _branding(who: str, symbol: str, goodbye: str, prompt: str = "", help_header: str = "") -> Dict[str, str]:
    """Branding block for a "<who> Agent" persona keyed by its glyph."""
    return {
        "agent_name": f"{who} Agent",
        "welcome": f"Welcome to {who} Agent! Type your message or /help for commands.",
        "goodbye": goodbye, "response_label": f" {symbol} {who} ", "prompt_symbol": prompt or symbol,
        "help_header": help_header or f"({symbol}) Available Commands"}


def _wings(*glyphs) -> List[List[str]]:
    """Spinner wing pairs `⟪g` / `g⟫`; a (left, right) tuple gives asymmetric glyphs."""
    return [[f"⟪{g[0] if isinstance(g, tuple) else g}", f"{g[1] if isinstance(g, tuple) else g}⟫"]
            for g in glyphs]


# Branding shared by every Hermes-named built-in (mono/daylight override help_header).
_HERMES_BRANDING: Dict[str, str] = _branding(
    "Hermes", "⚕", "Goodbye! ⚕", prompt="❯", help_header="(^_^)? Available Commands")

_BUILTIN_SKINS: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "default", "description": "Classic Hermes — gold and kawaii",
        # Dark-authored; values match the TUI's DARK_THEME so both render the same gold.
        "colors": {
            "banner_border": "#CD7F32", "banner_title": "#FFD700", "banner_accent": "#FFBF00",
            "banner_dim": "#B8860B", "banner_text": "#FFF8DC", "ui_accent": "#FFBF00",
            "ui_label": "#DAA520", "ui_ok": "#4caf50", "ui_error": "#ef5350", "ui_warn": "#ffa726",
            "prompt": "#FFF8DC", "input_rule": "#CD7F32", "response_border": "#FFD700",
            "status_bar_bg": "#1a1a2e", "status_bar_text": "#C0C0C0",
            "status_bar_strong": "#FFD700", "status_bar_dim": "#8A7A4A",
            "status_bar_good": "#8FBC8F", "status_bar_warn": "#FFD700", "status_bar_bad": "#FF8C00",
            "status_bar_critical": "#FF6B6B", "session_label": "#DAA520",
            "session_border": "#8B8682", "completion_menu_bg": "#1a1a2e",
            "completion_menu_current_bg": "#333355", "selection_bg": "#3a3a55",
            "shell_dollar": "#4dabf7", "voice_status_bg": "#1a1a2e"},
        # Light overlay (merged onto `colors`). Goldenrod ladder: on white the vivid
        # #FFD700/#FFBF00 read as glare and WCAG-darkened mustard (#867000) as mud; the
        # statusbar's goldenrod family (#B8860B/#DAA520) keeps the hue, tames saturation.
        # Hierarchy on white: ink body 8.9:1 > fade 5.2 > label 3.7 > muted 3.3 > title 2.7 >
        # headers 2.4. Fills (*_bg) flip the dark navy surfaces to light polarity.
        "light_colors": {
            "banner_title": "#C8961E", "banner_accent": "#D89B04", "banner_dim": "#B8860B",
            "banner_text": "#5C4718", "ui_accent": "#D89B04", "ui_label": "#A97E10",
            "ui_ok": "#2E7D32", "ui_error": "#C62828", "ui_warn": "#D97706", "prompt": "#5C4718",
            "response_border": "#C8961E", "session_label": "#A97E10", "status_bar_text": "#6F6F6F",
            "status_bar_strong": "#C8961E", "status_bar_dim": "#9A8A5A",
            "status_bar_good": "#2E7D32", "status_bar_warn": "#C8961E", "status_bar_bad": "#C2410C",
            "status_bar_critical": "#B91C1C", "shell_dollar": "#1E6FC0",
            "completion_menu_bg": "#F5F5F5", "completion_menu_current_bg": "#E0D1BF",
            "selection_bg": "#D4E4F7", "status_bar_bg": "#F5F5F5", "voice_status_bg": "#F5F5F5"},
        "spinner": {},  # empty = hardcoded defaults in display.py
        "branding": _HERMES_BRANDING,
        "tool_prefix": "┊"},
    "ares": {
        "name": "ares", "description": "War-god theme — crimson and bronze",
        "colors": {
            "banner_border": "#A93333", "banner_title": "#C7A96B", "banner_accent": "#DD4A3A",
            "banner_dim": "#905151", "banner_text": "#F1E6CF", "ui_accent": "#DD4A3A",
            "ui_label": "#C7A96B", "ui_ok": "#4caf50", "ui_error": "#ef5350", "ui_warn": "#ffa726",
            "prompt": "#F1E6CF", "input_rule": "#A93333", "response_border": "#C7A96B",
            "status_bar_bg": "#2A1212", "status_bar_text": "#F1E6CF",
            "status_bar_strong": "#C7A96B", "status_bar_dim": "#756054",
            "status_bar_good": "#7BC96F", "status_bar_warn": "#C7A96B", "status_bar_bad": "#DD4A3A",
            "status_bar_critical": "#EF5350", "session_label": "#C7A96B",
            "session_border": "#6E584B", "completion_menu_bg": "#2A1212",
            "completion_menu_current_bg": "#5C221D", "selection_bg": "#692620",
            "shell_dollar": "#DD4A3A", "voice_status_bg": "#2A1212"},
        "spinner": {
            "waiting_faces": ["(⚔)", "(⛨)", "(▲)", "(<>)", "(/)"],
            "thinking_faces": ["(⚔)", "(⛨)", "(▲)", "(⌁)", "(<>)"],
            "thinking_verbs": [
                "forging", "marching", "sizing the field", "holding the line",
                "hammering plans", "tempering steel", "plotting impact", "raising the shield"],
            "wings": _wings("⚔", "▲", ("╸", "╺"), "⛨")},
        "branding": _branding("Ares", "⚔", "Farewell, warrior! ⚔"),
        "tool_prefix": "╎",
        "banner_logo": """[bold #A3261F] █████╗ ██████╗ ███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #B73122]██╔══██╗██╔══██╗██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#C93C24]███████║██████╔╝█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#D84A28]██╔══██║██╔══██╗██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#E15A2D]██║  ██║██║  ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#EB6C32]╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]""",
        "banner_hero": """[#9F1C1C]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#9F1C1C]⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⠟⠻⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#C7A96B]⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠋⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀[/]
[#C7A96B]⠀⠀⠀⠀⠀⢀⣾⡿⠋⠀⠀⢠⡄⠀⠀⠙⢿⣷⡀⠀⠀⠀⠀⠀[/]
[#DD4A3A]⠀⠀⠀⠀⣰⣿⠟⠀⠀⠀⣰⣿⣿⣆⠀⠀⠀⠻⣿⣆⠀⠀⠀⠀[/]
[#DD4A3A]⠀⠀⠀⢰⣿⠏⠀⠀⢀⣾⡿⠉⢿⣷⡀⠀⠀⠹⣿⡆⠀⠀⠀[/]
[#9F1C1C]⠀⠀⠀⣿⡟⠀⠀⣠⣿⠟⠀⠀⠀⠻⣿⣄⠀⠀⢻⣿⠀⠀⠀[/]
[#9F1C1C]⠀⠀⠀⣿⡇⠀⠀⠙⠋⠀⠀⚔⠀⠀⠙⠋⠀⠀⢸⣿⠀⠀⠀[/]
[#6B1717]⠀⠀⠀⢿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡿⠀⠀⠀[/]
[#6B1717]⠀⠀⠀⠘⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠃⠀⠀⠀[/]
[#C7A96B]⠀⠀⠀⠀⠈⠻⣿⣷⣦⣤⣀⣀⣤⣤⣶⣿⠿⠋⠀⠀⠀⠀[/]
[#C7A96B]⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⠿⠿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀[/]
[#DD4A3A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⚔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[dim #6B1717]⠀⠀⠀⠀⠀⠀⠀⠀war god online⠀⠀⠀⠀⠀⠀⠀⠀[/]""",
    },
    "mono": {
        "name": "mono", "description": "Monochrome — clean grayscale",
        "colors": {
            "banner_border": "#5E5E5E", "banner_title": "#e6edf3", "banner_accent": "#aaaaaa",
            "banner_dim": "#606060", "banner_text": "#c9d1d9", "ui_accent": "#aaaaaa",
            "ui_label": "#888888", "ui_ok": "#888888", "ui_error": "#cccccc", "ui_warn": "#999999",
            "prompt": "#c9d1d9", "input_rule": "#606060", "response_border": "#aaaaaa",
            "status_bar_bg": "#1F1F1F", "status_bar_text": "#C9D1D9",
            "status_bar_strong": "#E6EDF3", "status_bar_dim": "#777777",
            "status_bar_good": "#B5B5B5", "status_bar_warn": "#AAAAAA", "status_bar_bad": "#D0D0D0",
            "status_bar_critical": "#F0F0F0", "session_label": "#888888",
            "session_border": "#5E5E5E", "completion_menu_bg": "#1F1F1F",
            "completion_menu_current_bg": "#464646", "selection_bg": "#505050",
            "shell_dollar": "#aaaaaa", "voice_status_bg": "#1F1F1F"},
        "spinner": {},
        "branding": {**_HERMES_BRANDING, "help_header": "[?] Available Commands"},
        "tool_prefix": "┊"},
    "slate": {
        "name": "slate", "description": "Cool blue — developer-focused",
        "colors": {
            "banner_border": "#4169e1", "banner_title": "#7eb8f6", "banner_accent": "#8EA8FF",
            "banner_dim": "#545E6B", "banner_text": "#c9d1d9", "ui_accent": "#7eb8f6",
            "ui_label": "#8EA8FF", "ui_ok": "#63D0A6", "ui_error": "#F7A072", "ui_warn": "#e6a855",
            "prompt": "#c9d1d9", "input_rule": "#4169e1", "response_border": "#7eb8f6",
            "status_bar_bg": "#151C2F", "status_bar_text": "#C9D1D9",
            "status_bar_strong": "#7EB8F6", "status_bar_dim": "#5D6672",
            "status_bar_good": "#63D0A6", "status_bar_warn": "#E6A855", "status_bar_bad": "#F7A072",
            "status_bar_critical": "#FF7A7A", "session_label": "#7eb8f6",
            "session_border": "#545E6B", "completion_menu_bg": "#151C2F",
            "completion_menu_current_bg": "#324867", "selection_bg": "#3A5375",
            "shell_dollar": "#7eb8f6", "voice_status_bg": "#151C2F"},
        "spinner": {}, "branding": _HERMES_BRANDING, "tool_prefix": "┊"},
    "daylight": {
        "name": "daylight",
        "description": "Light theme for bright terminals with dark text and cool blue accents",
        "colors": {
            "banner_border": "#2563EB", "banner_title": "#0F172A", "banner_accent": "#1D4ED8",
            "banner_dim": "#475569", "banner_text": "#111827", "ui_accent": "#2563EB",
            "ui_label": "#0F766E", "ui_ok": "#15803D", "ui_error": "#B91C1C", "ui_warn": "#B45309",
            "prompt": "#111827", "input_rule": "#6E94BE", "response_border": "#2563EB",
            "status_bar_bg": "#E5EDF8", "status_bar_text": "#111827",
            "status_bar_strong": "#2563EB", "status_bar_dim": "#838890",
            "status_bar_good": "#15803D", "status_bar_warn": "#B45309", "status_bar_bad": "#B45309",
            "status_bar_critical": "#B91C1C", "session_label": "#1D4ED8",
            "session_border": "#64748B", "completion_menu_bg": "#F8FAFC",
            "completion_menu_current_bg": "#DBEAFE", "completion_menu_meta_bg": "#EEF2FF",
            "completion_menu_meta_current_bg": "#BFDBFE", "selection_bg": "#D3E0FB",
            "shell_dollar": "#2563EB", "voice_status_bg": "#E5EDF8"},
        "spinner": {},
        "branding": {**_HERMES_BRANDING, "help_header": "[?] Available Commands"},
        "tool_prefix": "│"},
    "warm-lightmode": {
        "name": "warm-lightmode",
        "description": "Warm light mode — dark brown/gold text for light terminal backgrounds",
        "colors": {
            "banner_border": "#8B6914", "banner_title": "#5C3D11", "banner_accent": "#8B4513",
            "banner_dim": "#8B7355", "banner_text": "#2C1810", "ui_accent": "#8B4513",
            "ui_label": "#5C3D11", "ui_ok": "#2E7D32", "ui_error": "#C62828", "ui_warn": "#E65100",
            "prompt": "#2C1810", "input_rule": "#8B6914", "response_border": "#8B6914",
            "status_bar_bg": "#F5F0E8", "status_bar_text": "#2C1810",
            "status_bar_strong": "#8B4513", "status_bar_dim": "#8A8F98",
            "status_bar_good": "#2E7D32", "status_bar_warn": "#E65100", "status_bar_bad": "#DA4D00",
            "status_bar_critical": "#C62828", "session_label": "#5C3D11",
            "session_border": "#A0845C", "completion_menu_bg": "#F5EFE0",
            "completion_menu_current_bg": "#E8DCC8", "completion_menu_meta_bg": "#F0E8D8",
            "completion_menu_meta_current_bg": "#DFCFB0", "selection_bg": "#E8DAD0",
            "shell_dollar": "#8B4513", "voice_status_bg": "#F5F0E8"},
        "spinner": {}, "branding": _HERMES_BRANDING, "tool_prefix": "┊"},
    "poseidon": {
        "name": "poseidon", "description": "Ocean-god theme — deep blue and seafoam",
        "colors": {
            "banner_border": "#2A6FB9", "banner_title": "#A9DFFF", "banner_accent": "#5DB8F5",
            "banner_dim": "#44638F", "banner_text": "#EAF7FF", "ui_accent": "#5DB8F5",
            "ui_label": "#A9DFFF", "ui_ok": "#4caf50", "ui_error": "#ef5350", "ui_warn": "#ffa726",
            "prompt": "#EAF7FF", "input_rule": "#2A6FB9", "response_border": "#5DB8F5",
            "status_bar_bg": "#0F2440", "status_bar_text": "#EAF7FF",
            "status_bar_strong": "#A9DFFF", "status_bar_dim": "#52708A",
            "status_bar_good": "#6ED7B0", "status_bar_warn": "#5DB8F5", "status_bar_bad": "#3576BC",
            "status_bar_critical": "#D94F4F", "session_label": "#A9DFFF",
            "session_border": "#496884", "completion_menu_bg": "#0F2440",
            "completion_menu_current_bg": "#254D73", "selection_bg": "#2A587F",
            "shell_dollar": "#5DB8F5", "voice_status_bg": "#0F2440"},
        "spinner": {
            "waiting_faces": ["(≈)", "(Ψ)", "(∿)", "(◌)", "(◠)"],
            "thinking_faces": ["(Ψ)", "(∿)", "(≈)", "(⌁)", "(◌)"],
            "thinking_verbs": [
                "charting currents", "sounding the depth", "reading foam lines",
                "steering the trident", "tracking undertow", "plotting sea lanes",
                "calling the swell", "measuring pressure"],
            "wings": _wings("≈", "Ψ", "∿", "◌")},
        "branding": _branding("Poseidon", "Ψ", "Fair winds! Ψ"),
        "tool_prefix": "│",
        "banner_logo": """[bold #B8E8FF]██████╗  ██████╗ ███████╗███████╗██╗██████╗  ██████╗ ███╗   ██╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #97D6FF]██╔══██╗██╔═══██╗██╔════╝██╔════╝██║██╔══██╗██╔═══██╗████╗  ██║      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#75C1F6]██████╔╝██║   ██║███████╗█████╗  ██║██║  ██║██║   ██║██╔██╗ ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#4FA2E0]██╔═══╝ ██║   ██║╚════██║██╔══╝  ██║██║  ██║██║   ██║██║╚██╗██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#2E7CC7]██║     ╚██████╔╝███████║███████╗██║██████╔╝╚██████╔╝██║ ╚████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#1B4F95]╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]""",
        "banner_hero": """[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⢠⣿⠏⠀Ψ⠀⠹⣿⡄⠀⠀⠀⠀⠀⠀⠀[/]
[#A9DFFF]⠀⠀⠀⠀⠀⠀⠀⣿⡟⠀⠀⠀⠀⠀⢻⣿⠀⠀⠀⠀⠀⠀⠀[/]
[#A9DFFF]⠀⠀⠀≈≈≈≈≈⣿⡇⠀⠀⠀⠀⠀⢸⣿≈≈≈≈≈⠀⠀⠀[/]
[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀[/]
[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⢿⣧⠀⠀⠀⠀⠀⣼⡿⠀⠀⠀⠀⠀⠀⠀[/]
[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⠘⢿⣷⣄⣀⣠⣾⡿⠃⠀⠀⠀⠀⠀⠀⠀[/]
[#153C73]⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#153C73]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#5DB8F5]⠀⠀⠀⠀⠀≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈⠀⠀⠀⠀⠀[/]
[#A9DFFF]⠀⠀⠀⠀⠀⠀≈≈≈≈≈≈≈≈≈≈≈≈≈⠀⠀⠀⠀⠀⠀[/]
[dim #153C73]⠀⠀⠀⠀⠀⠀⠀deep waters hold⠀⠀⠀⠀⠀⠀⠀[/]""",
    },
    "sisyphus": {
        "name": "sisyphus", "description": "Sisyphean theme — austere grayscale with persistence",
        "colors": {
            "banner_border": "#B7B7B7", "banner_title": "#F5F5F5", "banner_accent": "#E7E7E7",
            "banner_dim": "#5C5C5C", "banner_text": "#D3D3D3", "ui_accent": "#E7E7E7",
            "ui_label": "#D3D3D3", "ui_ok": "#919191", "ui_error": "#E7E7E7", "ui_warn": "#B7B7B7",
            "prompt": "#F5F5F5", "input_rule": "#656565", "response_border": "#B7B7B7",
            "status_bar_bg": "#202020", "status_bar_text": "#D3D3D3",
            "status_bar_strong": "#F5F5F5", "status_bar_dim": "#6D6D6D",
            "status_bar_good": "#B7B7B7", "status_bar_warn": "#D3D3D3", "status_bar_bad": "#E7E7E7",
            "status_bar_critical": "#F5F5F5", "session_label": "#919191",
            "session_border": "#656565", "completion_menu_bg": "#202020",
            "completion_menu_current_bg": "#585858", "selection_bg": "#666666",
            "shell_dollar": "#E7E7E7", "voice_status_bg": "#202020"},
        "spinner": {
            "waiting_faces": ["(◉)", "(◌)", "(◬)", "(⬤)", "(::)"],
            "thinking_faces": ["(◉)", "(◬)", "(◌)", "(○)", "(●)"],
            "thinking_verbs": [
                "finding traction", "measuring the grade", "resetting the boulder",
                "counting the ascent", "testing leverage", "setting the shoulder",
                "pushing uphill", "enduring the loop"],
            "wings": _wings("◉", "◬", "◌", "⬤")},
        "branding": _branding("Sisyphus", "◉", "The boulder waits. ◉"),
        "tool_prefix": "│",
        "banner_logo": """[bold #F5F5F5]███████╗██╗███████╗██╗   ██╗██████╗ ██╗  ██╗██╗   ██╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #E7E7E7]██╔════╝██║██╔════╝╚██╗ ██╔╝██╔══██╗██║  ██║██║   ██║██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#D7D7D7]███████╗██║███████╗ ╚████╔╝ ██████╔╝███████║██║   ██║███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#BFBFBF]╚════██║██║╚════██║  ╚██╔╝  ██╔═══╝ ██╔══██║██║   ██║╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#8F8F8F]███████║██║███████║   ██║   ██║     ██║  ██║╚██████╔╝███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#626262]╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]""",
        "banner_hero": """[#B7B7B7]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#D3D3D3]⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#E7E7E7]⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀[/]
[#F5F5F5]⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀[/]
[#E7E7E7]⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀[/]
[#D3D3D3]⠀⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀[/]
[#B7B7B7]⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#919191]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#656565]⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#656565]⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#4A4A4A]⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#4A4A4A]⠀⠀⠀⠀⠀⣀⣴⣿⣿⣿⣿⣿⣿⣦⣀⠀⠀⠀⠀⠀⠀[/]
[#656565]⠀⠀⠀━━━━━━━━━━━━━━━━━━━━━━━⠀⠀⠀[/]
[dim #4A4A4A]⠀⠀⠀⠀⠀⠀⠀⠀⠀the boulder⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]""",
    },
    "charizard": {
        "name": "charizard", "description": "Volcanic theme — burnt orange and ember",
        "colors": {
            "banner_border": "#C75B1D", "banner_title": "#FFD39A", "banner_accent": "#F29C38",
            "banner_dim": "#C58A45", "banner_text": "#FFF0D4", "ui_accent": "#F29C38",
            "ui_label": "#FFD39A", "ui_ok": "#4caf50", "ui_error": "#ef5350", "ui_warn": "#ffa726",
            "prompt": "#FFF0D4", "input_rule": "#C75B1D", "response_border": "#F29C38",
            "status_bar_bg": "#2B160E", "status_bar_text": "#FFF0D4",
            "status_bar_strong": "#FFD39A", "status_bar_dim": "#826144",
            "status_bar_good": "#6BCB77", "status_bar_warn": "#F29C38", "status_bar_bad": "#E2832B",
            "status_bar_critical": "#EF5350", "session_label": "#FFD39A",
            "session_border": "#7B593A", "completion_menu_bg": "#0B0503",
            "completion_menu_current_bg": "#4A1B07", "completion_menu_meta_bg": "#120806",
            "completion_menu_meta_current_bg": "#5A260D", "selection_bg": "#5A260D",
            "shell_dollar": "#F29C38", "voice_status_bg": "#2B160E"},
        "spinner": {
            "waiting_faces": ["(✦)", "(▲)", "(◇)", "(<>)", "(🔥)"],
            "thinking_faces": ["(✦)", "(▲)", "(◇)", "(⌁)", "(🔥)"],
            "thinking_verbs": [
                "banking into the draft", "measuring burn", "reading the updraft",
                "tracking ember fall", "setting wing angle", "holding the flame core",
                "plotting a hot landing", "coiling for lift"],
            "wings": _wings("✦", "▲", "◌", "◇")},
        "branding": _branding("Charizard", "✦", "Flame out! ✦"),
        "tool_prefix": "│",
        "banner_logo": """[bold #FFF0D4] ██████╗██╗  ██╗ █████╗ ██████╗ ██╗███████╗ █████╗ ██████╗ ██████╗        █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD39A]██╔════╝██║  ██║██╔══██╗██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗██╔══██╗      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#F29C38]██║     ███████║███████║██████╔╝██║  ███╔╝ ███████║██████╔╝██║  ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#E2832B]██║     ██╔══██║██╔══██║██╔══██╗██║ ███╔╝  ██╔══██║██╔══██╗██║  ██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#C75B1D]╚██████╗██║  ██║██║  ██║██║  ██║██║███████╗██║  ██║██║  ██║██████╔╝      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#7A3511] ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]""",
        "banner_hero": """[#FFD39A]⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⠶⠶⠶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#F29C38]⠀⠀⠀⠀⠀⠀⣴⠟⠁⠀⠀⠀⠀⠈⠻⣦⠀⠀⠀⠀⠀⠀[/]
[#F29C38]⠀⠀⠀⠀⠀⣼⠏⠀⠀⠀✦⠀⠀⠀⠀⠹⣧⠀⠀⠀⠀⠀[/]
[#E2832B]⠀⠀⠀⠀⢰⡟⠀⠀⣀⣤⣤⣤⣀⠀⠀⠀⢻⡆⠀⠀⠀⠀[/]
[#E2832B]⠀⠀⣠⡾⠛⠁⣠⣾⠟⠉⠀⠉⠻⣷⣄⠀⠈⠛⢷⣄⠀⠀[/]
[#C75B1D]⠀⣼⠟⠀⢀⣾⠟⠁⠀⠀⠀⠀⠀⠈⠻⣷⡀⠀⠻⣧⠀[/]
[#C75B1D]⢸⡟⠀⠀⣿⡟⠀⠀⠀🔥⠀⠀⠀⠀⢻⣿⠀⠀⢻⡇[/]
[#7A3511]⠀⠻⣦⡀⠘⢿⣧⡀⠀⠀⠀⠀⠀⢀⣼⡿⠃⢀⣴⠟⠀[/]
[#7A3511]⠀⠀⠈⠻⣦⣀⠙⢿⣷⣤⣤⣤⣾⡿⠋⣀⣴⠟⠁⠀⠀[/]
[#C75B1D]⠀⠀⠀⠀⠈⠙⠛⠶⠤⠭⠭⠤⠶⠛⠋⠁⠀⠀⠀⠀[/]
[#F29C38]⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#F29C38]⠀⠀⠀⠀⠀⠀⠀⣼⡟⠀⠀⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀[/]
[dim #7A3511]⠀⠀⠀⠀⠀⠀⠀tail flame lit⠀⠀⠀⠀⠀⠀⠀⠀[/]""",
    }}

_active_skin: Optional[SkinConfig] = None
_active_skin_name: str = "default"


def _skins_dir() -> Path:
    return get_hermes_home() / "skins"


def _load_skin_from_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a skin definition from a YAML file; None on any failure."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "name" in data:
            return data
    except Exception as e:
        logger.debug("Failed to load skin from %s: %s", path, e)
    return None


def _build_skin_config(data: Dict[str, Any]) -> SkinConfig:
    """Build a SkinConfig from a raw dict (built-in or loaded from YAML)."""
    default = _BUILTIN_SKINS["default"]
    skin_name = str(data.get("name", "unknown"))

    def section(key: str) -> Dict[str, Any]:
        value = data.get(key)
        if isinstance(value, dict):
            return value
        if value is not None:
            logger.warning("Skin '%s' has invalid '%s' section type (%s); ignoring section",
                           skin_name, key, type(value).__name__)
        return {}

    def merged(key: str) -> Dict[str, Any]:
        return {**default.get(key, {}), **section(key)}
    # Paired palettes are NOT merged over the default skin's blocks: an empty block means
    # "no hand-tuned variant for that polarity" and consumers (the TUI) fall back to `colors`
    # + automatic adaptation, which beats the default's gold light palette under a crimson skin.
    return SkinConfig(
        name=skin_name, description=data.get("description", ""), colors=merged("colors"),
        light_colors=section("light_colors"), dark_colors=section("dark_colors"),
        spinner=merged("spinner"), branding=merged("branding"),
        tool_prefix=data.get("tool_prefix", default.get("tool_prefix", "┊")),
        tool_emojis=section("tool_emojis"), banner_logo=data.get("banner_logo", ""),
        banner_hero=data.get("banner_hero", ""))


def list_skins() -> List[Dict[str, str]]:
    """List all available skins (built-in + user-installed); user skins never shadow built-ins."""
    result = [{"name": name, "description": data.get("description", ""), "source": "builtin"}
              for name, data in _BUILTIN_SKINS.items()]
    skins_path = _skins_dir()
    for f in sorted(skins_path.glob("*.yaml")) if skins_path.is_dir() else ():
        data = _load_skin_from_yaml(f)
        if data and not any(s["name"] == data.get("name", f.stem) for s in result):
            result.append({"name": data.get("name", f.stem), "description": data.get("description", ""),
                           "source": "user"})
    return result


def load_skin(name: str) -> SkinConfig:
    """Load a skin by name: user skins first, then built-in, then default."""
    user_file = _skins_dir() / f"{name}.yaml"
    data = _load_skin_from_yaml(user_file) if user_file.is_file() else None
    if not data and name not in _BUILTIN_SKINS:
        logger.warning("Skin '%s' not found, using default", name)
    return _build_skin_config(data or _BUILTIN_SKINS.get(name) or _BUILTIN_SKINS["default"])


def get_active_skin() -> SkinConfig:
    """Currently active skin config (cached)."""
    global _active_skin
    if _active_skin is None:
        _active_skin = load_skin(_active_skin_name)
    return _active_skin


def set_active_skin(name: str) -> SkinConfig:
    """Switch the active skin. Returns the new SkinConfig."""
    global _active_skin, _active_skin_name
    _active_skin_name = name
    _active_skin = load_skin(name)
    return _active_skin


def get_active_skin_name() -> str:
    return _active_skin_name


def init_skin_from_config(config: dict) -> None:
    """Initialize the active skin from CLI config at startup."""
    display = config.get("display") or {}
    skin_name = display.get("skin", "default") if isinstance(display, dict) else "default"
    set_active_skin(skin_name.strip() if isinstance(skin_name, str) and skin_name.strip() else "default")


def _active_branding(key: str, fallback: str) -> str:
    try:
        return get_active_skin().get_branding(key, fallback)
    except Exception:
        return fallback


def get_active_prompt_symbol(fallback: str = "❯") -> str:
    """Interactive prompt symbol (skins store a bare token) plus a single trailing space."""
    cleaned = (_active_branding("prompt_symbol", fallback) or fallback).strip()
    return f"{cleaned or fallback.strip()} "


def get_active_help_header(fallback: str = "(^_^)? Available Commands") -> str:
    return _active_branding("help_header", fallback)


def get_active_goodbye(fallback: str = "Goodbye! ⚕") -> str:
    return _active_branding("goodbye", fallback)


# Palette resolution order for prompt_toolkit styles: (name, skin color key, fallback). A
# fallback starting with "@" names an earlier entry (so a missing key inherits its remapped value).
_STYLE_PALETTE = (
    ("prompt", "prompt", ""), ("input_rule", "input_rule", "#CD7F32"),
    ("title", "banner_title", "#FFD700"), ("text", "banner_text", "#FFF8DC"),
    ("dim", "banner_dim", "#555555"), ("label", "ui_label", "@title"), ("warn", "ui_warn", "#FF8C00"),
    ("error", "ui_error", "#FF6B6B"), ("status_bg", "status_bar_bg", "#1a1a2e"),
    ("status_text", "status_bar_text", "@text"), ("status_strong", "status_bar_strong", "@title"),
    ("status_dim", "status_bar_dim", "@dim"), ("ok", "ui_ok", "#8FBC8F"),
    ("status_good", "status_bar_good", "@ok"), ("status_warn", "status_bar_warn", "@warn"),
    ("accent", "banner_accent", "@warn"), ("status_bad", "status_bar_bad", "@accent"),
    ("status_critical", "status_bar_critical", "@error"), ("voice_bg", "voice_status_bg", "@status_bg"),
    ("menu_bg", "completion_menu_bg", "#1a1a2e"), ("menu_current_bg", "completion_menu_current_bg", "#333355"),
    ("menu_meta_bg", "completion_menu_meta_bg", "@menu_bg"),
    ("menu_meta_current_bg", "completion_menu_meta_current_bg", "@menu_current_bg"))

# prompt_toolkit style class -> format template over the resolved palette names.
_STYLE_TEMPLATES = {
    "input-area": "",  # terminal default fg/bg — `prompt` styles the symbol, NOT typed text
    "placeholder": "{dim} italic", "prompt": "{prompt}", "prompt-working": "{dim} italic",
    "hint": "{dim} italic",
    "status-bar": "bg:{status_bg} {status_text}", "status-bar-strong": "bg:{status_bg} {status_strong} bold",
    "status-bar-dim": "bg:{status_bg} {status_dim}", "status-bar-good": "bg:{status_bg} {status_good} bold",
    "status-bar-warn": "bg:{status_bg} {status_warn} bold", "status-bar-bad": "bg:{status_bg} {status_bad} bold",
    "status-bar-critical": "bg:{status_bg} {status_critical} bold",
    "input-rule": "{input_rule}", "image-badge": "{label} bold",
    "completion-menu": "bg:{menu_bg} {text}", "completion-menu.completion": "bg:{menu_bg} {text}",
    "completion-menu.completion.current": "bg:{menu_current_bg} {title}",
    "completion-menu.meta.completion": "bg:{menu_meta_bg} {dim}",
    "completion-menu.meta.completion.current": "bg:{menu_meta_current_bg} {label}",
    "clarify-border": "{input_rule}", "clarify-title": "{title} bold", "clarify-question": "{text} bold",
    "clarify-choice": "{dim}", "clarify-selected": "{title} bold", "clarify-active-other": "{title} italic",
    "clarify-countdown": "{input_rule}",
    "sudo-prompt": "{error} bold", "sudo-border": "{input_rule}", "sudo-title": "{error} bold",
    "sudo-text": "{text}",
    "approval-border": "{input_rule}", "approval-title": "{warn} bold", "approval-desc": "{text} bold",
    "approval-cmd": "{dim} italic", "approval-choice": "{dim}", "approval-selected": "{title} bold",
    "voice-status": "bg:{voice_bg} {label}", "voice-status-recording": "bg:{voice_bg} {error} bold"}


def get_prompt_toolkit_style_overrides() -> Dict[str, str]:
    """Return prompt_toolkit style overrides derived from the active skin."""
    try:
        skin = get_active_skin()
    except Exception:
        return {}
    # `prompt` is unset by default so typed text inherits the terminal's foreground (readable
    # on light and dark schemes); skins opt into a colored prompt symbol via `prompt` in YAML.
    # Every read goes through skin.get_color (cli.py wraps it for light-mode remapping).
    palette: Dict[str, str] = {}
    for name, key, fallback in _STYLE_PALETTE:
        palette[name] = skin.get_color(key, palette[fallback[1:]] if fallback.startswith("@") else fallback)
    return {cls: tpl.format(**palette) for cls, tpl in _STYLE_TEMPLATES.items()}
