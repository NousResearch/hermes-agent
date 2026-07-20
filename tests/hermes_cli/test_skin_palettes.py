"""Built-in skin palette audit: completeness + WCAG contrast, per polarity.

Every built-in skin must be a complete, coherent palette with no accidental
fallbacks (a partial skin inherits the default skin's gold, which is how
"slate feels all over the place" happened), and every palette must be a
fixed point of the TUI's runtime readability adaptation — hand-tuned values
that already pass the same contrast floors the TUI enforces (strong >= 3.9,
soft >= 2.8, fills matching the background polarity). Mirrors the desktop
app's paired colors/darkColors contract.
"""

import pytest

from hermes_cli.skin_engine import _BUILTIN_SKINS

# Union of the color keys consumed by the TUI (fromSkin) and the classic CLI
# (banner.py / display.py / prompt_toolkit overrides). completion_menu_meta_*
# intentionally excluded: they default to the base menu keys.
REQUIRED_KEYS = frozenset(
    {
        "banner_border",
        "banner_title",
        "banner_accent",
        "banner_dim",
        "banner_text",
        "ui_accent",
        "ui_label",
        "ui_ok",
        "ui_error",
        "ui_warn",
        "prompt",
        "input_rule",
        "response_border",
        "status_bar_bg",
        "status_bar_text",
        "status_bar_strong",
        "status_bar_dim",
        "status_bar_good",
        "status_bar_warn",
        "status_bar_bad",
        "status_bar_critical",
        "session_label",
        "session_border",
        "completion_menu_bg",
        "completion_menu_current_bg",
        "selection_bg",
        "shell_dollar",
        "voice_status_bg",
    }
)

# Foreground roles and their minimum contrast against the palette's pole.
# Matches ui-tui/src/theme.ts STRONG/SOFT tiers.
STRONG_FG = (
    "banner_title",
    "banner_accent",
    "banner_text",
    "ui_accent",
    "ui_label",
    "ui_ok",
    "ui_error",
    "prompt",
    "status_bar_strong",
    "status_bar_good",
    "status_bar_bad",
    "status_bar_critical",
    "shell_dollar",
)
SOFT_FG = (
    "banner_dim",
    "banner_border",
    "ui_warn",
    "input_rule",
    "response_border",
    "status_bar_dim",
    "status_bar_warn",
    "session_label",
    "session_border",
)
# status_bar_text renders on status_bar_bg, not the terminal background.
ON_STATUS_BAR = ("status_bar_text", "status_bar_strong", "status_bar_dim")

FILLS = (
    "status_bar_bg",
    "completion_menu_bg",
    "completion_menu_current_bg",
    "selection_bg",
    "voice_status_bg",
)

STRONG_MIN = 3.9
SOFT_MIN = 2.8
# Assumed terminal poles, matching ui-tui/src/theme.ts referenceBackground().
DARK_POLE = "#101014"
LIGHT_POLE = "#ffffff"


def _rgb(hex_color: str):
    h = hex_color.lstrip("#")
    assert len(h) == 6, f"not a 6-digit hex: {hex_color!r}"
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _channel(v: float) -> float:
    c = v / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    r, g, b = _rgb(hex_color)
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _palettes():
    """Yield (skin, palette_name, palette, is_light) for every built-in."""
    for name, skin in _BUILTIN_SKINS.items():
        colors = skin.get("colors", {})
        light = skin.get("light_colors", {})
        dark = skin.get("dark_colors", {})

        # `colors` polarity is declared by which paired block the skin ships.
        colors_are_light = bool(dark) and not light
        yield name, "colors", colors, colors_are_light

        if light:
            yield name, "light_colors", light, True
        if dark:
            yield name, "dark_colors", dark, False


ALL_PALETTES = list(_palettes())
PALETTE_IDS = [f"{skin}:{block}" for skin, block, _, _ in ALL_PALETTES]


def test_every_builtin_ships_a_paired_palette():
    for name, skin in _BUILTIN_SKINS.items():
        assert skin.get("light_colors") or skin.get("dark_colors"), (
            f"skin {name!r} has no paired palette: dark-authored skins need "
            f"light_colors, light-authored skins need dark_colors"
        )
        assert not (skin.get("light_colors") and skin.get("dark_colors")), (
            f"skin {name!r} declares both paired blocks; `colors` polarity "
            f"would be ambiguous"
        )


@pytest.mark.parametrize(("skin", "block", "palette", "is_light"), ALL_PALETTES, ids=PALETTE_IDS)
def test_palette_is_complete(skin, block, palette, is_light):
    missing = REQUIRED_KEYS - palette.keys()
    assert not missing, f"{skin}.{block} missing keys: {sorted(missing)}"


@pytest.mark.parametrize(("skin", "block", "palette", "is_light"), ALL_PALETTES, ids=PALETTE_IDS)
def test_palette_contrast_and_polarity(skin, block, palette, is_light):
    pole = LIGHT_POLE if is_light else DARK_POLE
    problems = []

    for key in STRONG_FG:
        ratio = contrast(palette[key], pole)
        if ratio < STRONG_MIN:
            problems.append(f"{key}={palette[key]} contrast {ratio:.2f} < {STRONG_MIN} vs {pole}")

    for key in SOFT_FG:
        ratio = contrast(palette[key], pole)
        if ratio < SOFT_MIN:
            problems.append(f"{key}={palette[key]} contrast {ratio:.2f} < {SOFT_MIN} vs {pole}")

    status_bg = palette["status_bar_bg"]
    for key in ON_STATUS_BAR:
        floor = STRONG_MIN if key == "status_bar_strong" else SOFT_MIN
        ratio = contrast(palette[key], status_bg)
        if ratio < floor:
            problems.append(f"{key}={palette[key]} contrast {ratio:.2f} < {floor} vs status_bar_bg {status_bg}")

    for key in FILLS:
        lum = luminance(palette[key])
        if is_light and lum < 0.4:
            problems.append(f"{key}={palette[key]} is a dark fill (lum {lum:.2f}) in a light palette")
        if not is_light and lum > 0.35:
            problems.append(f"{key}={palette[key]} is a light fill (lum {lum:.2f}) in a dark palette")

    # The selection chip must remain distinguishable from the menu surface.
    chip = contrast(palette["completion_menu_current_bg"], palette["completion_menu_bg"])
    if chip < 1.15:
        problems.append(
            f"completion_menu_current_bg={palette['completion_menu_current_bg']} "
            f"indistinguishable from completion_menu_bg (contrast {chip:.2f})"
        )

    assert not problems, f"{skin}.{block}:\n  " + "\n  ".join(problems)
