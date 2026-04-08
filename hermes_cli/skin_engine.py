"""Hermes CLI skin/theme engine.

A data-driven skin system that lets users customize the CLI's visual appearance.
Skins are defined as YAML files in ~/.hermes/skins/ or as built-in presets.
No code changes are needed to add a new skin.

SKIN YAML SCHEMA
================

All fields are optional. Missing values inherit from the ``default`` skin.

.. code-block:: yaml

    # Required: skin identity
    name: mytheme                         # Unique skin name (lowercase, hyphens ok)
    description: Short description        # Shown in /skin listing

    # Colors: hex values for Rich markup (banner, UI, response box)
    colors:
      banner_border: "#CD7F32"            # Panel border color
      banner_title: "#FFD700"             # Panel title text color
      banner_accent: "#FFBF00"            # Section headers (Available Tools, etc.)
      banner_dim: "#B8860B"               # Dim/muted text (separators, labels)
      banner_text: "#FFF8DC"              # Body text (tool names, skill names)
      ui_accent: "#FFBF00"               # General UI accent
      ui_label: "#4dd0e1"                # UI labels
      ui_ok: "#4caf50"                   # Success indicators
      ui_error: "#ef5350"                # Error indicators
      ui_warn: "#ffa726"                 # Warning indicators
      prompt: "#FFF8DC"                  # Prompt text color
      input_rule: "#CD7F32"              # Input area horizontal rule
      response_border: "#FFD700"         # Response box border (ANSI)
      session_label: "#DAA520"           # Session label color
      session_border: "#8B8682"          # Session ID dim color

    # Spinner: customize the animated spinner during API calls
    spinner:
      style: dots                         # TUI prompt spinner: dots|bounce|grow|arrows|star|moon|pulse|clock|none
      waiting_faces:                      # Faces shown while waiting for API
        - "(⚔)"
        - "(⛨)"
      thinking_faces:                     # Faces shown during reasoning
        - "(⌁)"
        - "(<>)"
      thinking_verbs:                     # Verbs for spinner messages
        - "forging"
        - "plotting"
      wings:                              # Optional left/right spinner decorations
        - ["⟪⚔", "⚔⟫"]                  # Each entry is [left, right] pair
        - ["⟪▲", "▲⟫"]

    # Branding: text strings used throughout the CLI
    branding:
      agent_name: "Hermes Agent"          # Banner title, status display
      welcome: "Welcome message"          # Shown at CLI startup
      goodbye: "Goodbye! ⚕"              # Shown on exit
      response_label: " ⚕ Hermes "       # Response box header label
      prompt_symbol: "❯ "                # Input prompt symbol
      help_header: "(^_^)? Commands"      # /help header text

    # Tool prefix: character for tool output lines (default: ┊)
    tool_prefix: "┊"

    # Tool emojis: override the default emoji for any tool (used in spinners & progress)
    tool_emojis:
      terminal: "⚔"           # Override terminal tool emoji
      web_search: "🔮"        # Override web_search tool emoji
      # Any tool not listed here uses its registry default

    # Syntax highlighting color scheme
    syntax_scheme: monokai      # Named scheme from built-in list
                                # Options: hermes (default), monokai, dracula,
                                #   one-dark, github-dark, nord, catppuccin,
                                #   tokyo-night, gruvbox, solarized-dark

    # Token-level overrides on top of the named scheme (optional)
    syntax:
      keyword: "bold #FF79C6"  # Any logical token name from SYNTAX_SCHEMES keys
      comment: "italic dim green"

    # Diff renderer colors (hex for bg/fg; Rich style strings for line_number etc.)
    # All *_bg/*_fg values MUST be 6-digit hex (#RRGGBB)
    diff:
      deletion_bg: "#781414"
      addition_bg: "#145a14"
      deletion_fg: "#ffffff"
      addition_fg: "#ffffff"
      intra_del_bg: "#9b1c1c"
      intra_add_bg: "#166534"
      intra_del_fg: "#ff8080"
      intra_add_fg: "#80ff80"
      line_number: "dim"        # Rich style string
      hunk_header: "bold cyan"  # Rich style string
      filename: "bold bright_white"
      file_path_fg: "#B4A0FF"   # inline diff only (display.py)
      hunk_fg: "#787882"        # inline diff only (display.py)
      context_fg: "#969696"     # inline diff only (display.py)

    # Markdown rendering styles (Rich style strings, except bullets/blockquote_marker)
    markdown:
      link: "#58A6FF underline"
      code_span: "bright_white"
      heading_1: "bold bright_white"
      blockquote_marker: "▌"   # Unicode character, not a style
      bullets: ["•", "◦", "▸", "·"]  # List, not a style
      strike: "strike"          # Rich style name for ANSI SGR 9

    # Extended UI colors
    ui_ext:
      context_bar_normal: "#5f87d7"
      context_bar_warn: "#ffa726"
      context_bar_crit: "#ef5350"
      menu_cursor: ["fg_green", "bold"]   # List — prompt_toolkit style tuple
      menu_highlight: ["fg_green"]
      table_col_accent: "bold cyan"
      panel_border: "cyan"

USAGE
=====

.. code-block:: python

    from hermes_cli.skin_engine import get_active_skin, list_skins, set_active_skin

    skin = get_active_skin()
    print(skin.colors["banner_title"])    # "#FFD700"
    print(skin.get_branding("agent_name"))  # "Hermes Agent"

    set_active_skin("ares")               # Switch to built-in ares skin
    set_active_skin("mytheme")            # Switch to user skin from ~/.hermes/skins/

BUILT-IN SKINS
==============

- ``default`` — Classic Hermes gold/kawaii (the current look)
- ``ares``    — Crimson/bronze war-god theme with custom spinner wings
- ``mono``    — Clean grayscale monochrome
- ``slate``   — Cool blue developer-focused theme

USER SKINS
==========

Drop a YAML file in ``~/.hermes/skins/<name>.yaml`` following the schema above.
Activate with ``/skin <name>`` in the CLI or ``display.skin: <name>`` in config.yaml.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# =============================================================================
# Syntax color schemes (logical token name → Rich style string)
# Keeping these in skin_engine.py avoids a circular import:
#   rich_output imports skin_engine; skin_engine must not import rich_output.
# =============================================================================

# MIT-licensed originals credited inline.
SYNTAX_SCHEMES: Dict[str, Dict[str, str]] = {
    "hermes": {
        # Current hardcoded palette — unchanged for backward compat.
        # "name" intentionally omitted: plain identifiers render as terminal default.
        "keyword":             "bold blue",
        "keyword_type":        "bold cyan",
        "name_builtin":        "cyan",
        "name_class":          "bold yellow",
        "name_function":       "bold yellow",
        "name_function_magic": "cyan",
        "name_decorator":      "bright_cyan",
        "name_exception":      "red",
        "comment":             "dim green",
        "string":              "green",
        "string_doc":          "dim green",
        "string_escape":       "green",
        "string_regex":        "magenta",
        "number":              "magenta",
        "operator":            "white",
        "operator_word":       "bold blue",
        "error":               "red",
        "diff_deleted":        "red",
        "diff_inserted":       "green",
    },
    "monokai": {
        # Adapted from Wimer Hazenberg's Monokai (MIT). Background ref: #272822
        "keyword":             "bold #F92672",
        "keyword_type":        "#66D9EF",
        "name":                "#F8F8F2",
        "name_builtin":        "#66D9EF",
        "name_class":          "bold #A6E22E",
        "name_function":       "bold #A6E22E",
        "name_function_magic": "#66D9EF",
        "name_decorator":      "#FD971F",
        "name_exception":      "#F92672",
        "comment":             "#75715E",
        "string":              "#E6DB74",
        "string_doc":          "#E6DB74",
        "string_escape":       "#AE81FF",
        "string_regex":        "#E6DB74",
        "number":              "#AE81FF",
        "operator":            "#F92672",
        "operator_word":       "#F92672",
        "error":               "#F44747",
        "diff_deleted":        "#F92672",
        "diff_inserted":       "#A6E22E",
    },
    "dracula": {
        # Adapted from Zeno Rocha's Dracula (MIT). Background ref: #282A36
        "keyword":             "bold #FF79C6",
        "keyword_type":        "#8BE9FD",
        "name":                "#F8F8F2",
        "name_builtin":        "#50FA7B",
        "name_class":          "#50FA7B",
        "name_function":       "#50FA7B",
        "name_function_magic": "#50FA7B",
        "name_decorator":      "#FFB86C",
        "name_exception":      "#FF5555",
        "comment":             "italic #6272A4",
        "string":              "#F1FA8C",
        "string_doc":          "#F1FA8C",
        "string_escape":       "#FFB86C",
        "string_regex":        "#F1FA8C",
        "number":              "#BD93F9",
        "operator":            "#FF79C6",
        "operator_word":       "#FF79C6",
        "error":               "#FF5555",
        "diff_deleted":        "#FF5555",
        "diff_inserted":       "#50FA7B",
    },
    "one-dark": {
        # Adapted from Atom One Dark / One Dark Pro (MIT). Background ref: #282C34
        "keyword":             "bold #C678DD",
        "keyword_type":        "#E5C07B",
        "name":                "#ABB2BF",
        "name_builtin":        "#61AFEF",
        "name_class":          "bold #E5C07B",
        "name_function":       "bold #61AFEF",
        "name_function_magic": "#61AFEF",
        "name_decorator":      "#D19A66",
        "name_exception":      "#E06C75",
        "comment":             "italic #7F848E",
        "string":              "#98C379",
        "string_doc":          "#98C379",
        "string_escape":       "#D19A66",
        "string_regex":        "#98C379",
        "number":              "#D19A66",
        "operator":            "#56B6C2",
        "operator_word":       "#C678DD",
        "error":               "#E06C75",
        "diff_deleted":        "#E06C75",
        "diff_inserted":       "#98C379",
    },
    "github-dark": {
        # Adapted from GitHub Primer VSCode theme (MIT). Background ref: #0D1117
        "keyword":             "bold #FF7B72",
        "keyword_type":        "#79C0FF",
        "name":                "#C9D1D9",
        "name_builtin":        "#79C0FF",
        "name_class":          "bold #D0883B",
        "name_function":       "bold #79C0FF",
        "name_function_magic": "#79C0FF",
        "name_decorator":      "#D0883B",
        "name_exception":      "#FF7B72",
        "comment":             "italic #8B949E",
        "string":              "#A5D6FF",
        "string_doc":          "#A5D6FF",
        "string_escape":       "#79C0FF",
        "string_regex":        "#A5D6FF",
        "number":              "#79C0FF",
        "operator":            "#FF7B72",
        "operator_word":       "#FF7B72",
        "error":               "#FF7B72",
        "diff_deleted":        "#FF7B72",
        "diff_inserted":       "#3FB950",
    },
    "nord": {
        # Adapted from Arctic Ice Studio Nord (MIT). Background ref: #2E3440
        "keyword":             "#81A1C1",
        "keyword_type":        "#8FBCBB",
        "name":                "#D8DEE9",
        "name_builtin":        "#88C0D0",
        "name_class":          "bold #8FBCBB",
        "name_function":       "bold #88C0D0",
        "name_function_magic": "#88C0D0",
        "name_decorator":      "#D08770",
        "name_exception":      "#BF616A",
        "comment":             "italic #4C566A",
        "string":              "#A3BE8C",
        "string_doc":          "#A3BE8C",
        "string_escape":       "#EBCB8B",
        "string_regex":        "#A3BE8C",
        "number":              "#B48EAD",
        "operator":            "#81A1C1",
        "operator_word":       "#81A1C1",
        "error":               "#BF616A",
        "diff_deleted":        "#BF616A",
        "diff_inserted":       "#A3BE8C",
    },
    "catppuccin": {
        # Adapted from Catppuccin Mocha (MIT). Background ref: #1E1E2E
        "keyword":             "bold #CBA6F7",
        "keyword_type":        "#89B4FA",
        "name":                "#CDD6F4",
        "name_builtin":        "#89DCEB",
        "name_class":          "bold #A6E3A1",
        "name_function":       "bold #89B4FA",
        "name_function_magic": "#89DCEB",
        "name_decorator":      "#F9E2AF",
        "name_exception":      "#F38BA8",
        "comment":             "italic #6C7086",
        "string":              "#A6E3A1",
        "string_doc":          "#A6E3A1",
        "string_escape":       "#F9E2AF",
        "string_regex":        "#A6E3A1",
        "number":              "#FAB387",
        "operator":            "#89DCEB",
        "operator_word":       "#CBA6F7",
        "error":               "#F38BA8",
        "diff_deleted":        "#F38BA8",
        "diff_inserted":       "#A6E3A1",
    },
    "tokyo-night": {
        # Adapted from enkia/tokyo-night-vscode-theme (MIT). Background ref: #1A1B26
        "keyword":             "bold #BB9AF7",
        "keyword_type":        "#7AA2F7",
        "name":                "#C0CAF5",
        "name_builtin":        "#7AA2F7",
        "name_class":          "bold #0DB9D7",
        "name_function":       "bold #7AA2F7",
        "name_function_magic": "#7AA2F7",
        "name_decorator":      "#FF9E64",
        "name_exception":      "#F7768E",
        "comment":             "italic #51597D",
        "string":              "#9ECE6A",
        "string_doc":          "#9ECE6A",
        "string_escape":       "#89DDFF",
        "string_regex":        "#9ECE6A",
        "number":              "#FF9E64",
        "operator":            "#89DDFF",
        "operator_word":       "#BB9AF7",
        "error":               "#F7768E",
        "diff_deleted":        "#F7768E",
        "diff_inserted":       "#9ECE6A",
    },
    "gruvbox": {
        # Adapted from morhetz/gruvbox (MIT). Background ref: #282828
        "keyword":             "bold #FB4934",
        "keyword_type":        "#83A598",
        "name":                "#EBDBB2",
        "name_builtin":        "#83A598",
        "name_class":          "bold #B8BB26",
        "name_function":       "bold #B8BB26",
        "name_function_magic": "#83A598",
        "name_decorator":      "#FABD2F",
        "name_exception":      "#FB4934",
        "comment":             "italic #928374",
        "string":              "#B8BB26",
        "string_doc":          "#B8BB26",
        "string_escape":       "#FABD2F",
        "string_regex":        "#B8BB26",
        "number":              "#D3869B",
        "operator":            "#8EC07C",
        "operator_word":       "#FB4934",
        "error":               "#FB4934",
        "diff_deleted":        "#FB4934",
        "diff_inserted":       "#B8BB26",
    },
    "solarized-dark": {
        # Adapted from Ethan Schoonover's Solarized (MIT). Background ref: #002B36
        "keyword":             "bold #268BD2",
        "keyword_type":        "#268BD2",
        "name":                "#839496",
        "name_builtin":        "#2AA198",
        "name_class":          "bold #859900",
        "name_function":       "bold #859900",
        "name_function_magic": "#2AA198",
        "name_decorator":      "#CB4B16",
        "name_exception":      "#DC322F",
        "comment":             "italic #586E75",
        "string":              "#859900",
        "string_doc":          "#859900",
        "string_escape":       "#CB4B16",
        "string_regex":        "#2AA198",
        "number":              "#D33682",
        "operator":            "#268BD2",
        "operator_word":       "#268BD2",
        "error":               "#DC322F",
        "diff_deleted":        "#DC322F",
        "diff_inserted":       "#859900",
    },
}

# =============================================================================
# Default values for new skin sections
# =============================================================================

_DIFF_DEFAULTS: Dict[str, str] = {
    "deletion_bg":  "#781414",
    "addition_bg":  "#145a14",
    "deletion_fg":  "#ffffff",
    "addition_fg":  "#ffffff",
    "deletion_marker_fg": "#FF7B72",
    "addition_marker_fg": "#56D364",
    "intra_del_bg": "#9b1c1c",
    "intra_add_bg": "#166534",
    "intra_del_fg": "#ff8080",
    "intra_add_fg": "#80ff80",
    "line_number":  "dim",
    "separator":    "dim",
    "hunk_header":  "bold cyan",
    "filename":     "bold bright_white",
    "file_path_fg": "#B4A0FF",
    "hunk_fg":      "#787882",
    "context_fg":   "#969696",
}

_MARKDOWN_DEFAULTS: Dict[str, Any] = {
    "link":               "#58A6FF underline",
    "code_span":          "bright_white",
    "heading_1":          "bold bright_white",
    "heading_2":          "bold white",
    "heading_3":          "bold",
    "heading_4_6":        "bold dim",
    "blockquote":         "dim",
    "blockquote_marker":  "▌",
    "hr":                 "dim",
    "task_checked":       "bold #4caf50",
    "task_unchecked":     "dim",
    "strike":             "strike",
    "image_alt":          "dim",
    "bullets":            ["•", "◦", "▸", "·"],
    "ol_numeral":         "dim",
}

_UI_EXT_DEFAULTS: Dict[str, Any] = {
    "context_bar_normal": "#5f87d7",
    "context_bar_warn":   "#ffa726",
    "context_bar_crit":   "#ef5350",
    "tool_error_prefix":  "red",
    "tool_disabled":      "red",
    "tool_lazy":          "yellow",
    "menu_cursor":        ["fg_green", "bold"],
    "menu_highlight":     ["fg_green"],
    "table_header":       "bold",
    "table_col_accent":   "bold cyan",
    "table_col_dim":      "dim",
    "panel_border":       "cyan",
}

# =============================================================================
# Skin-switch invalidation callbacks
# =============================================================================

_invalidation_callbacks: List[Callable[[], None]] = []


def register_skin_callback(fn: Callable[[], None]) -> None:
    """Register a callable to be invoked after every skin switch.

    Use this to invalidate caches in modules that read skin values at startup
    (e.g. ANSI-string caches in rich_output.py, syntax formatter in display.py).
    skin_engine never imports those modules — callers register themselves.
    """
    _invalidation_callbacks.append(fn)


def _syntax_bold_enabled() -> bool:
    """Return whether syntax token styles should keep bold emphasis."""
    try:
        from hermes_cli.config import load_config

        display = load_config().get("display", {})
        value = display.get("syntax_bold", True)
    except Exception:
        return True

    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _strip_bold(style: str) -> str:
    """Remove the standalone ``bold`` token from a Rich style string."""
    parts = [part for part in style.split() if part.lower() != "bold"]
    return " ".join(parts)


# =============================================================================
# Skin data structure
# =============================================================================

@dataclass
class SkinConfig:
    """Complete skin configuration."""
    name: str
    description: str = ""
    colors: Dict[str, str] = field(default_factory=dict)
    spinner: Dict[str, Any] = field(default_factory=dict)
    branding: Dict[str, str] = field(default_factory=dict)
    tool_prefix: str = "┊"
    tool_emojis: Dict[str, str] = field(default_factory=dict)  # per-tool emoji overrides
    banner_logo: str = ""    # Rich-markup ASCII art logo (replaces HERMES_AGENT_LOGO)
    banner_hero: str = ""    # Rich-markup hero art (replaces HERMES_CADUCEUS)
    # New in theme-integration: syntax, diff, markdown, ui_ext
    syntax_scheme: str = "hermes"
    syntax: Dict[str, str] = field(default_factory=dict)       # per-token overrides
    diff: Dict[str, str] = field(default_factory=dict)
    markdown: Dict[str, Any] = field(default_factory=dict)     # Any: lists + strings
    ui_ext: Dict[str, Any] = field(default_factory=dict)       # Any: lists + strings

    def get_color(self, key: str, fallback: str = "") -> str:
        """Get a color value with fallback."""
        return self.colors.get(key, fallback)

    def get_spinner_list(self, key: str) -> List[str]:
        """Get a spinner list (faces, verbs, etc.)."""
        return self.spinner.get(key, [])

    def get_spinner_style(self) -> Optional[str]:
        """Return the TUI spinner style key for this skin, or None to use config default."""
        return self.spinner.get("style") or None

    def get_spinner_wings(self) -> List[Tuple[str, str]]:
        """Get spinner wing pairs, or empty list if none."""
        raw = self.spinner.get("wings", [])
        result = []
        for pair in raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                result.append((str(pair[0]), str(pair[1])))
        return result

    def get_branding(self, key: str, fallback: str = "") -> str:
        """Get a branding value with fallback."""
        return self.branding.get(key, fallback)

    def get_syntax_styles(self) -> Dict[str, str]:
        """Return merged syntax styles: named scheme + per-skin token overrides."""
        base = dict(SYNTAX_SCHEMES.get(self.syntax_scheme, SYNTAX_SCHEMES["hermes"]))
        base.update(self.syntax)  # per-skin overrides win
        if not _syntax_bold_enabled():
            base = {
                token: (_strip_bold(style) if isinstance(style, str) else style)
                for token, style in base.items()
            }
        return base

    def get_diff(self, key: str, fallback: str = "") -> str:
        """Return a diff color/style, falling back to _DIFF_DEFAULTS then fallback."""
        return self.diff.get(key, _DIFF_DEFAULTS.get(key, fallback))

    def get_markdown(self, key: str, fallback: Any = None) -> Any:
        """Return a markdown style/value, falling back to _MARKDOWN_DEFAULTS."""
        return self.markdown.get(key, _MARKDOWN_DEFAULTS.get(key, fallback))

    def get_ui_ext(self, key: str, fallback: Any = None) -> Any:
        """Return an extended UI style/value, falling back to _UI_EXT_DEFAULTS."""
        return self.ui_ext.get(key, _UI_EXT_DEFAULTS.get(key, fallback))


# =============================================================================
# Built-in skin definitions
# =============================================================================

_BUILTIN_SKINS: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "default",
        "description": "Classic Hermes — gold and kawaii",
        "syntax_scheme": "hermes",
        "diff": {
            "deletion_bg": "#781414",
            "addition_bg": "#145a14",
            "intra_del_bg": "#9b1c1c",
            "intra_add_bg": "#166534",
        },
        "colors": {
            "banner_border": "#CD7F32",
            "banner_title": "#FFD700",
            "banner_accent": "#FFBF00",
            "banner_dim": "#B8860B",
            "banner_text": "#FFF8DC",
            "ui_accent": "#FFBF00",
            "ui_label": "#4dd0e1",
            "ui_ok": "#4caf50",
            "ui_error": "#ef5350",
            "ui_warn": "#ffa726",
            "prompt": "#FFF8DC",
            "input_rule": "#CD7F32",
            "response_border": "#FFD700",
            "session_label": "#DAA520",
            "session_border": "#8B8682",
        },
        "spinner": {
            "style": "dots",
            # Empty = use hardcoded defaults in display.py
        },
        "branding": {
            "agent_name": "Hermes Agent",
            "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
            "goodbye": "Goodbye! ⚕",
            "response_label": " ⚕ Hermes ",
            "prompt_symbol": "❯ ",
            "help_header": "(^_^)? Available Commands",
        },
        "tool_prefix": "┊",
    },
    "ares": {
        "name": "ares",
        "description": "War-god theme — crimson and bronze",
        "syntax_scheme": "gruvbox",
        "diff": {
            "deletion_bg": "#6F1D1B",
            "addition_bg": "#3F5A2A",
            "intra_del_bg": "#8C2F26",
            "intra_add_bg": "#557A34",
        },
        "colors": {
            "banner_border": "#9F1C1C",
            "banner_title": "#C7A96B",
            "banner_accent": "#DD4A3A",
            "banner_dim": "#6B1717",
            "banner_text": "#F1E6CF",
            "ui_accent": "#DD4A3A",
            "ui_label": "#C7A96B",
            "ui_ok": "#4caf50",
            "ui_error": "#ef5350",
            "ui_warn": "#ffa726",
            "prompt": "#F1E6CF",
            "input_rule": "#9F1C1C",
            "response_border": "#C7A96B",
            "session_label": "#C7A96B",
            "session_border": "#6E584B",
        },
        "spinner": {
            "style": "arrows",
            "waiting_faces": ["(⚔)", "(⛨)", "(▲)", "(<>)", "(/)"],
            "thinking_faces": ["(⚔)", "(⛨)", "(▲)", "(⌁)", "(<>)"],
            "thinking_verbs": [
                "forging", "marching", "sizing the field", "holding the line",
                "hammering plans", "tempering steel", "plotting impact", "raising the shield",
            ],
            "wings": [
                ["⟪⚔", "⚔⟫"],
                ["⟪▲", "▲⟫"],
                ["⟪╸", "╺⟫"],
                ["⟪⛨", "⛨⟫"],
            ],
        },
        "branding": {
            "agent_name": "Ares Agent",
            "welcome": "Welcome to Ares Agent! Type your message or /help for commands.",
            "goodbye": "Farewell, warrior! ⚔",
            "response_label": " ⚔ Ares ",
            "prompt_symbol": "⚔ ❯ ",
            "help_header": "(⚔) Available Commands",
        },
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
        "name": "mono",
        "description": "Monochrome — clean grayscale",
        "syntax_scheme": "solarized-dark",
        "diff": {
            "deletion_bg": "#3A3030",
            "addition_bg": "#2F3A30",
            "intra_del_bg": "#4A3A3A",
            "intra_add_bg": "#3A4A3A",
            "deletion_marker_fg": "#D0D0D0",
            "addition_marker_fg": "#F0F0F0",
        },
        "colors": {
            "banner_border": "#555555",
            "banner_title": "#e6edf3",
            "banner_accent": "#aaaaaa",
            "banner_dim": "#444444",
            "banner_text": "#c9d1d9",
            "ui_accent": "#aaaaaa",
            "ui_label": "#888888",
            "ui_ok": "#888888",
            "ui_error": "#cccccc",
            "ui_warn": "#999999",
            "prompt": "#c9d1d9",
            "input_rule": "#444444",
            "response_border": "#aaaaaa",
            "session_label": "#888888",
            "session_border": "#555555",
        },
        "spinner": {"style": "none"},
        "branding": {
            "agent_name": "Hermes Agent",
            "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
            "goodbye": "Goodbye! ⚕",
            "response_label": " ⚕ Hermes ",
            "prompt_symbol": "❯ ",
            "help_header": "[?] Available Commands",
        },
        "tool_prefix": "┊",
    },
    "slate": {
        "name": "slate",
        "description": "Cool blue — developer-focused",
        "syntax_scheme": "one-dark",
        "diff": {
            "deletion_bg": "#3F2630",
            "addition_bg": "#203D36",
            "intra_del_bg": "#5A3240",
            "intra_add_bg": "#2A544A",
        },
        "colors": {
            "banner_border": "#4169e1",
            "banner_title": "#7eb8f6",
            "banner_accent": "#8EA8FF",
            "banner_dim": "#4b5563",
            "banner_text": "#c9d1d9",
            "ui_accent": "#7eb8f6",
            "ui_label": "#8EA8FF",
            "ui_ok": "#63D0A6",
            "ui_error": "#F7A072",
            "ui_warn": "#e6a855",
            "prompt": "#c9d1d9",
            "input_rule": "#4169e1",
            "response_border": "#7eb8f6",
            "session_label": "#7eb8f6",
            "session_border": "#4b5563",
        },
        "spinner": {"style": "pulse"},
        "branding": {
            "agent_name": "Hermes Agent",
            "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
            "goodbye": "Goodbye! ⚕",
            "response_label": " ⚕ Hermes ",
            "prompt_symbol": "❯ ",
            "help_header": "(^_^)? Available Commands",
        },
        "tool_prefix": "┊",
    },
    "poseidon": {
        "name": "poseidon",
        "description": "Ocean-god theme — deep blue and seafoam",
        "syntax_scheme": "nord",
        "diff": {
            "deletion_bg": "#433047",
            "addition_bg": "#244A44",
            "intra_del_bg": "#5A4060",
            "intra_add_bg": "#2F6259",
        },
        "colors": {
            "banner_border": "#2A6FB9",
            "banner_title": "#A9DFFF",
            "banner_accent": "#5DB8F5",
            "banner_dim": "#153C73",
            "banner_text": "#EAF7FF",
            "ui_accent": "#5DB8F5",
            "ui_label": "#A9DFFF",
            "ui_ok": "#4caf50",
            "ui_error": "#ef5350",
            "ui_warn": "#ffa726",
            "prompt": "#EAF7FF",
            "input_rule": "#2A6FB9",
            "response_border": "#5DB8F5",
            "session_label": "#A9DFFF",
            "session_border": "#496884",
        },
        "spinner": {
            "style": "bounce",
            "waiting_faces": ["(≈)", "(Ψ)", "(∿)", "(◌)", "(◠)"],
            "thinking_faces": ["(Ψ)", "(∿)", "(≈)", "(⌁)", "(◌)"],
            "thinking_verbs": [
                "charting currents", "sounding the depth", "reading foam lines",
                "steering the trident", "tracking undertow", "plotting sea lanes",
                "calling the swell", "measuring pressure",
            ],
            "wings": [
                ["⟪≈", "≈⟫"],
                ["⟪Ψ", "Ψ⟫"],
                ["⟪∿", "∿⟫"],
                ["⟪◌", "◌⟫"],
            ],
        },
        "branding": {
            "agent_name": "Poseidon Agent",
            "welcome": "Welcome to Poseidon Agent! Type your message or /help for commands.",
            "goodbye": "Fair winds! Ψ",
            "response_label": " Ψ Poseidon ",
            "prompt_symbol": "Ψ ❯ ",
            "help_header": "(Ψ) Available Commands",
        },
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
        "name": "sisyphus",
        "description": "Sisyphean theme — austere grayscale with persistence",
        "syntax_scheme": "hermes",
        "diff": {
            "deletion_bg": "#3E3E3E",
            "addition_bg": "#303030",
            "intra_del_bg": "#555555",
            "intra_add_bg": "#464646",
            "deletion_marker_fg": "#D6D6D6",
            "addition_marker_fg": "#F5F5F5",
        },
        "colors": {
            "banner_border": "#B7B7B7",
            "banner_title": "#F5F5F5",
            "banner_accent": "#E7E7E7",
            "banner_dim": "#4A4A4A",
            "banner_text": "#D3D3D3",
            "ui_accent": "#E7E7E7",
            "ui_label": "#D3D3D3",
            "ui_ok": "#919191",
            "ui_error": "#E7E7E7",
            "ui_warn": "#B7B7B7",
            "prompt": "#F5F5F5",
            "input_rule": "#656565",
            "response_border": "#B7B7B7",
            "session_label": "#919191",
            "session_border": "#656565",
        },
        "spinner": {
            "style": "grow",
            "waiting_faces": ["(◉)", "(◌)", "(◬)", "(⬤)", "(::)"],
            "thinking_faces": ["(◉)", "(◬)", "(◌)", "(○)", "(●)"],
            "thinking_verbs": [
                "finding traction", "measuring the grade", "resetting the boulder",
                "counting the ascent", "testing leverage", "setting the shoulder",
                "pushing uphill", "enduring the loop",
            ],
            "wings": [
                ["⟪◉", "◉⟫"],
                ["⟪◬", "◬⟫"],
                ["⟪◌", "◌⟫"],
                ["⟪⬤", "⬤⟫"],
            ],
        },
        "branding": {
            "agent_name": "Sisyphus Agent",
            "welcome": "Welcome to Sisyphus Agent! Type your message or /help for commands.",
            "goodbye": "The boulder waits. ◉",
            "response_label": " ◉ Sisyphus ",
            "prompt_symbol": "◉ ❯ ",
            "help_header": "(◉) Available Commands",
        },
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
        "name": "charizard",
        "description": "Volcanic theme — burnt orange and ember",
        "syntax_scheme": "monokai",
        "diff": {
            "deletion_bg": "#5A2317",
            "addition_bg": "#2E4A24",
            "intra_del_bg": "#7A2E1D",
            "intra_add_bg": "#3F6530",
        },
        "colors": {
            "banner_border": "#C75B1D",
            "banner_title": "#FFD39A",
            "banner_accent": "#F29C38",
            "banner_dim": "#7A3511",
            "banner_text": "#FFF0D4",
            "ui_accent": "#F29C38",
            "ui_label": "#FFD39A",
            "ui_ok": "#4caf50",
            "ui_error": "#ef5350",
            "ui_warn": "#ffa726",
            "prompt": "#FFF0D4",
            "input_rule": "#C75B1D",
            "response_border": "#F29C38",
            "session_label": "#FFD39A",
            "session_border": "#6C4724",
        },
        "spinner": {
            "waiting_faces": ["(✦)", "(▲)", "(◇)", "(<>)", "(🔥)"],
            "thinking_faces": ["(✦)", "(▲)", "(◇)", "(⌁)", "(🔥)"],
            "thinking_verbs": [
                "banking into the draft", "measuring burn", "reading the updraft",
                "tracking ember fall", "setting wing angle", "holding the flame core",
                "plotting a hot landing", "coiling for lift",
            ],
            "wings": [
                ["⟪✦", "✦⟫"],
                ["⟪▲", "▲⟫"],
                ["⟪◌", "◌⟫"],
                ["⟪◇", "◇⟫"],
            ],
        },
        "branding": {
            "agent_name": "Charizard Agent",
            "welcome": "Welcome to Charizard Agent! Type your message or /help for commands.",
            "goodbye": "Flame out! ✦",
            "response_label": " ✦ Charizard ",
            "prompt_symbol": "✦ ❯ ",
            "help_header": "(✦) Available Commands",
        },
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
    },
}


# =============================================================================
# Skin loading and management
# =============================================================================

_active_skin: Optional[SkinConfig] = None
_active_skin_name: str = "default"


def _skins_dir() -> Path:
    """User skins directory."""
    return get_hermes_home() / "skins"


def _load_skin_from_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a skin definition from a YAML file."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "name" in data:
            return data
    except Exception as e:
        logger.debug("Failed to load skin from %s: %s", path, e)
    return None


_HEX_RE = re.compile(r'^#[0-9a-fA-F]{6}$')


def _validate_hex(value: str, key: str, default: str) -> str:
    """Return value if it's a valid 6-digit hex color, else fall back to default."""
    if isinstance(value, str) and _HEX_RE.match(value):
        return value
    logger.warning("skin: invalid hex color for diff.%s=%r, using default %r", key, value, default)
    return default


def _build_skin_config(data: Dict[str, Any]) -> SkinConfig:
    """Build a SkinConfig from a raw dict (built-in or loaded from YAML)."""
    # Start with default values as base for missing keys
    default = _BUILTIN_SKINS["default"]
    colors = dict(default.get("colors", {}))
    colors.update(data.get("colors", {}))
    spinner = dict(default.get("spinner", {}))
    spinner.update(data.get("spinner", {}))
    branding = dict(default.get("branding", {}))
    branding.update(data.get("branding", {}))

    # --- syntax_scheme: validate against known schemes ---
    syntax_scheme = data.get("syntax_scheme", "hermes")
    if not isinstance(syntax_scheme, str) or syntax_scheme not in SYNTAX_SCHEMES:
        logger.warning("skin: unknown syntax_scheme=%r, falling back to 'hermes'", syntax_scheme)
        syntax_scheme = "hermes"

    # --- syntax: unknown token keys are silently ignored ---
    raw_syntax = data.get("syntax", {})
    syntax: Dict[str, str] = {k: v for k, v in raw_syntax.items() if isinstance(k, str)} if isinstance(raw_syntax, dict) else {}

    # --- diff: validate hex color values ---
    raw_diff = data.get("diff", {})
    diff: Dict[str, str] = {}
    if isinstance(raw_diff, dict):
        for k, v in raw_diff.items():
            if not isinstance(k, str):
                continue
            # bg/fg keys must be valid hex; style keys (line_number, hunk_header…) pass through
            if k.endswith(("_bg", "_fg")):
                default_val = _DIFF_DEFAULTS.get(k, "")
                diff[k] = _validate_hex(str(v), k, default_val)
            else:
                diff[k] = str(v) if v is not None else ""

    # --- markdown: accept as-is (Rich validates style strings at render time) ---
    raw_md = data.get("markdown", {})
    markdown: Dict[str, Any] = dict(raw_md) if isinstance(raw_md, dict) else {}

    # --- ui_ext: menu_cursor / menu_highlight accepted as list or space-split string ---
    raw_ui_ext = data.get("ui_ext", {})
    ui_ext: Dict[str, Any] = {}
    if isinstance(raw_ui_ext, dict):
        for k, v in raw_ui_ext.items():
            if k in ("menu_cursor", "menu_highlight"):
                if isinstance(v, str):
                    v = v.split() or _UI_EXT_DEFAULTS.get(k, [])
                elif isinstance(v, list) and not v:
                    v = _UI_EXT_DEFAULTS.get(k, [])
            ui_ext[k] = v

    return SkinConfig(
        name=data.get("name", "unknown"),
        description=data.get("description", ""),
        colors=colors,
        spinner=spinner,
        branding=branding,
        tool_prefix=data.get("tool_prefix", default.get("tool_prefix", "┊")),
        tool_emojis=data.get("tool_emojis", {}),
        banner_logo=data.get("banner_logo", ""),
        banner_hero=data.get("banner_hero", ""),
        syntax_scheme=syntax_scheme,
        syntax=syntax,
        diff=diff,
        markdown=markdown,
        ui_ext=ui_ext,
    )


def list_skins() -> List[Dict[str, str]]:
    """List all available skins (built-in + user-installed).

    Returns list of {"name": ..., "description": ..., "source": "builtin"|"user"}.
    """
    result = []
    for name, data in _BUILTIN_SKINS.items():
        result.append({
            "name": name,
            "description": data.get("description", ""),
            "source": "builtin",
        })

    skins_path = _skins_dir()
    if skins_path.is_dir():
        for f in sorted(skins_path.glob("*.yaml")):
            data = _load_skin_from_yaml(f)
            if data:
                skin_name = data.get("name", f.stem)
                # Skip if it shadows a built-in
                if any(s["name"] == skin_name for s in result):
                    continue
                result.append({
                    "name": skin_name,
                    "description": data.get("description", ""),
                    "source": "user",
                })

    return result


def load_skin(name: str) -> SkinConfig:
    """Load a skin by name. Checks user skins first, then built-in."""
    # Check user skins directory
    skins_path = _skins_dir()
    user_file = skins_path / f"{name}.yaml"
    if user_file.is_file():
        data = _load_skin_from_yaml(user_file)
        if data:
            return _build_skin_config(data)

    # Check built-in skins
    if name in _BUILTIN_SKINS:
        return _build_skin_config(_BUILTIN_SKINS[name])

    # Fallback to default
    logger.warning("Skin '%s' not found, using default", name)
    return _build_skin_config(_BUILTIN_SKINS["default"])


def get_active_skin() -> SkinConfig:
    """Get the currently active skin config (cached)."""
    global _active_skin
    if _active_skin is None:
        _active_skin = load_skin(_active_skin_name)
    return _active_skin


def set_active_skin(name: str) -> SkinConfig:
    """Switch the active skin. Returns the new SkinConfig."""
    global _active_skin, _active_skin_name
    _active_skin_name = name
    _active_skin = load_skin(name)
    for fn in _invalidation_callbacks:
        try:
            fn()
        except Exception:
            pass
    return _active_skin


def get_active_skin_name() -> str:
    """Get the name of the currently active skin."""
    return _active_skin_name


def init_skin_from_config(config: dict) -> None:
    """Initialize the active skin from CLI config at startup.

    Call this once during CLI init with the loaded config dict.
    """
    display = config.get("display", {})
    skin_name = display.get("skin", "default")
    if isinstance(skin_name, str) and skin_name.strip():
        set_active_skin(skin_name.strip())
    else:
        set_active_skin("default")


# =============================================================================
# Convenience helpers for CLI modules
# =============================================================================


def get_active_prompt_symbol(fallback: str = "❯ ") -> str:
    """Get the interactive prompt symbol from the active skin."""
    try:
        return get_active_skin().get_branding("prompt_symbol", fallback)
    except Exception:
        return fallback



def get_active_help_header(fallback: str = "(^_^)? Available Commands") -> str:
    """Get the /help header from the active skin."""
    try:
        return get_active_skin().get_branding("help_header", fallback)
    except Exception:
        return fallback



def get_active_goodbye(fallback: str = "Goodbye! ⚕") -> str:
    """Get the goodbye line from the active skin."""
    try:
        return get_active_skin().get_branding("goodbye", fallback)
    except Exception:
        return fallback



def get_prompt_toolkit_style_overrides() -> Dict[str, str]:
    """Return prompt_toolkit style overrides derived from the active skin.

    These are layered on top of the CLI's base TUI style so /skin can refresh
    the live prompt_toolkit UI immediately without rebuilding the app.
    """
    try:
        skin = get_active_skin()
    except Exception:
        return {}

    prompt = skin.get_color("prompt", "#FFF8DC")
    input_rule = skin.get_color("input_rule", "#CD7F32")
    title = skin.get_color("banner_title", "#FFD700")
    text = skin.get_color("banner_text", prompt)
    dim = skin.get_color("banner_dim", "#555555")
    label = skin.get_color("ui_label", title)
    warn = skin.get_color("ui_warn", "#FF8C00")
    error = skin.get_color("ui_error", "#FF6B6B")

    accent = skin.get_color("ui_accent", title)
    ok = skin.get_color("ui_ok", "#8FBC8F")
    sb_bg = skin.get_color("statusbar_bg", "#1a1a2e")

    return {
        "input-area": prompt,
        "placeholder": f"{dim} italic",
        "prompt": prompt,
        "prompt-working": f"{dim} italic",
        "hint": f"{dim} italic",
        "input-rule": input_rule,
        "image-badge": f"{label} bold",
        "status-bar": f"bg:{sb_bg} {text}",
        "status-bar-strong": f"bg:{sb_bg} {accent} bold",
        "status-bar-dim": f"bg:{sb_bg} {dim}",
        "status-bar-good": f"bg:{sb_bg} {ok} bold",
        "status-bar-warn": f"bg:{sb_bg} {warn} bold",
        "status-bar-bad": f"bg:{sb_bg} {warn} bold",
        "status-bar-critical": f"bg:{sb_bg} {error} bold",
        "completion-menu": f"bg:#1a1a2e {text}",
        "completion-menu.completion": f"bg:#1a1a2e {text}",
        "completion-menu.completion.current": f"bg:#333355 {title}",
        "completion-menu.meta.completion": f"bg:#1a1a2e {dim}",
        "completion-menu.meta.completion.current": f"bg:#333355 {label}",
        "clarify-border": input_rule,
        "clarify-title": f"{title} bold",
        "clarify-question": f"{text} bold",
        "clarify-choice": dim,
        "clarify-selected": f"{title} bold",
        "clarify-active-other": f"{title} italic",
        "clarify-countdown": input_rule,
        "sudo-prompt": f"{error} bold",
        "sudo-border": input_rule,
        "sudo-title": f"{error} bold",
        "sudo-text": text,
        "approval-border": input_rule,
        "approval-title": f"{warn} bold",
        "approval-desc": f"{text} bold",
        "approval-cmd": f"{dim} italic",
        "approval-choice": dim,
        "approval-selected": f"{title} bold",
    }
