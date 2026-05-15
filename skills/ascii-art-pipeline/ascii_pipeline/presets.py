from __future__ import annotations

from dataclasses import dataclass

from .eikon import D30_CHARSET, DENSE_REF_CHARSET


@dataclass(frozen=True)
class Preset:
    name: str
    backend: str
    target: tuple[int, int]
    source_scale: tuple[int, int] | None
    preprocess: tuple[str, ...]
    preview_font_size: int
    quality_thresholds: dict[str, float]
    notes: str
    charset: str | None = None
    braille: bool = False
    intermediate_grid: tuple[int, int] | None = None
    downsample: str = "majority"  # "majority" | "edge-aware" | "clahe"


PRESETS = {
    "stroke-clarity": Preset(
        name="stroke-clarity",
        backend="python",
        target=(48, 24),
        source_scale=(96, 48),
        preprocess=("grayscale", "autocontrast", "contrast+10%"),
        preview_font_size=18,
        quality_thresholds={"heavy_ratio_min": 0.22, "light_ratio_max": 0.24, "fill_ratio_min": 0.22},
        notes="Readable stroke-first ASCII using the dense-ref palette with pure-Python rendering.",
        charset=DENSE_REF_CHARSET,
    ),
    "d30-dense": Preset(
        name="d30-dense",
        backend="python",
        target=(48, 24),
        source_scale=(384, 192),
        preprocess=("grayscale", "autocontrast", "contrast+10%", "integer-downsample"),
        preview_font_size=18,
        quality_thresholds={"heavy_ratio_min": 0.30, "light_ratio_max": 0.20, "fill_ratio_min": 0.30},
        notes="High-density D30-style pipeline using pure-Python Lanczos resampling plus edge-aware downsampling for crisp edges.",
        charset=D30_CHARSET,
        intermediate_grid=(384, 192),
        downsample="edge-aware",
    ),
    "braille-detail": Preset(
        name="braille-detail",
        backend="python",
        target=(48, 24),
        source_scale=(96, 48),
        preprocess=("grayscale", "autocontrast", "floyd-steinberg"),
        preview_font_size=20,
        quality_thresholds={"braille_ratio_min": 0.80, "fill_ratio_min": 0.20},
        notes="Maximum effective detail via Braille block encoding with pure-Python Floyd–Steinberg dithering.",
        braille=True,
    ),
    "eikon-motion": Preset(
        name="eikon-motion",
        backend="python-video",
        target=(48, 24),
        source_scale=(384, 192),
        preprocess=("frame-sampling", "integer-downsample"),
        preview_font_size=18,
        quality_thresholds={"motion_char_diff_min": 0.05, "heavy_ratio_min": 0.22, "light_ratio_max": 0.24},
        notes="Video-first animated eikon pipeline preserving motion richness.",
        charset=D30_CHARSET,
        intermediate_grid=(384, 192),
    ),
}
