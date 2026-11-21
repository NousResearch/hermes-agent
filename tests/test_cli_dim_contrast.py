"""T2.6: dim-text fallback colors must meet the contrast floor (Rec.709 luminance >= 0.30).

Premise check: upstream ``#B8860B`` (banner_dim fallback) and ``#6b7684`` (dim chrome)
already measure ~0.53 / ~0.46 — above the floor. This test pins the floor as a
regression guard so a future darker dim fallback cannot slip in.
"""
from hermes_cli.cli_render import _luminance_from_hex

# Dim-text fallback colors actually shipped: cli_render banner_dim, cli_tui_runtime_mixin chrome.
DIM_FALLBACKS = ["#B8860B", "#6b7684"]


def test_dim_colors_meet_contrast_floor():
    for hex_color in DIM_FALLBACKS:
        lum = _luminance_from_hex(hex_color)
        assert lum is not None, hex_color
        assert lum >= 0.30, f"{hex_color} luminance {lum:.3f} below floor 0.30"
