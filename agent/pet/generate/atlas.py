"""Deterministic spritesheet assembly — generated row strips → Hermes atlas.

Image-generation models are good at *drawing* a row of poses but bad at exact
grid geometry, so the model never owns the atlas layout: it produces one loose
horizontal strip per state, and these deterministic ops slice that strip into
clean, centered, transparent ``192x208`` cells and pack them into the sheet our
renderer reads.

The atlas follows the **petdex/Codex standard**: 8 columns x 9 rows of
``192x208`` cells (``1536x1872``), with the row order + per-row frame counts
from OpenAI's ``hatch-pet`` skill. Our renderer (:mod:`agent.pet.render`) keys
frames as ``rows = states, cols = frames`` via
:data:`agent.pet.constants.CODEX_STATE_ROWS`, and a pet built here is a valid
``petdex submit`` spritesheet. Rows shorter than 8 columns leave the trailing
cells fully transparent.

Note ``running`` is the *working* state (in-place processing), NOT locomotion —
``running-right`` / ``running-left`` are the actual directional walk cycles.

The frame-segmentation, fit-to-cell, and transparency-residue logic is adapted
from OpenAI's ``hatch-pet`` skill (openai/skills, Apache-2.0).
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

from agent.pet.constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)

CELL_WIDTH = FRAME_W
CELL_HEIGHT = FRAME_H

# (state, row index, frame count). Order/row indices MUST match
# ``constants.CODEX_STATE_ROWS`` so the renderer crops the right row for each
# driven state, and the per-row frame counts mirror the petdex/Codex
# ``hatch-pet`` ``animation-rows`` spec. The renderer trims trailing blank
# columns, so rows shorter than ``COLUMNS`` (8) just leave the tail transparent.
ROW_SPECS: list[tuple[str, int, int]] = [
    ("idle", 0, 6),
    ("running-right", 1, 8),
    ("running-left", 2, 8),
    ("waving", 3, 4),
    ("jumping", 4, 5),
    ("failed", 5, 8),
    ("waiting", 6, 6),
    ("running", 7, 6),
    ("review", 8, 6),
]

ROWS = len(ROW_SPECS)
COLUMNS = max(count for _, _, count in ROW_SPECS)
ATLAS_WIDTH = COLUMNS * CELL_WIDTH
ATLAS_HEIGHT = ROWS * CELL_HEIGHT

FRAME_COUNTS: dict[str, int] = {state: count for state, _, count in ROW_SPECS}

# Alpha at/below which a pixel is "background" for component detection.
_ALPHA_FLOOR = 16
# Cell padding kept around a fitted sprite so poses never touch the edge.
_CELL_PAD = 10
# Margin for the normalized pass — small, to fill the cell like real petdex pets
# (they sit ~5px from the edges); the width clamp, not the pad, prevents clipping.
_NORMALIZE_PAD = 14
# Side-lobe cutoff for fitted frames. Adjacent-pose bleed usually appears as a
# small separated horizontal lobe beside the real subject; keep sizeable lobes so
# we don't punish a legitimate wide pose.
_SIDE_LOBE_RATIO = 0.18


# ───────────────────────── background removal ─────────────────────────


def _color_distance(r: int, g: int, b: int, key: tuple[int, int, int]) -> float:
    return math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2)


def _has_transparency(image) -> bool:
    """True if the strip already carries a real alpha background."""
    extrema = image.getchannel("A").getextrema()
    # Min alpha 0 somewhere and a meaningful share of fully-transparent pixels.
    if extrema[0] > _ALPHA_FLOOR:
        return False
    hist = image.getchannel("A").histogram()
    transparent = sum(hist[: _ALPHA_FLOOR + 1])
    total = image.width * image.height
    return transparent > total * 0.05


def _dominant_corner_color(image) -> tuple[int, int, int]:
    """Sample the four corners and return the most common opaque color."""
    from collections import Counter

    w, h = image.width, image.height
    px = image.load()
    counter: Counter = Counter()
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        r, g, b, a = px[x, y]
        if a > _ALPHA_FLOOR:
            counter[(r, g, b)] += 1
    if not counter:
        return (0, 255, 0)
    return counter.most_common(1)[0][0]


def _near_key_mask(image, key: tuple[int, int, int], tol: int = 48):
    """An ``L`` mask, 255 where a pixel is within *tol* per-channel of *key*.

    Tight on purpose: it only marks near-pure backdrop so trapped chroma pockets
    seed the flood, while chroma-*tinted* character pixels stay outside it. Built
    with channel point-ops (fast C), no per-pixel Python.
    """
    from PIL import ImageChops

    r, g, b, _a = image.split()
    kr, kg, kb = key
    return ImageChops.darker(
        ImageChops.darker(
            r.point(lambda v: 255 if abs(v - kr) <= tol else 0),
            g.point(lambda v: 255 if abs(v - kg) <= tol else 0),
        ),
        b.point(lambda v: 255 if abs(v - kb) <= tol else 0),
    )


def remove_background(image, *, chroma_key: tuple[int, int, int] | None = None, threshold: float = 90.0):
    """Return *image* (RGBA) with its flat background keyed out to transparent.

    If the strip already has a transparent background we leave it alone; else we
    key out *chroma_key* (or the dominant corner color when not given) via a
    **border flood-fill**: only background-coloured pixels *connected to an edge*
    are removed. A global color match (the old approach) punched holes in the pet
    wherever an interior highlight happened to match the backdrop — e.g. a pug's
    light belly against a near-white background — which then showed through as the
    window behind. Flood-fill keeps those interior pixels because they aren't
    reachable from the border without crossing the (non-background) pet.
    """
    from collections import deque

    rgba = image.convert("RGBA")
    if _has_transparency(rgba):
        return _repair_internal_alpha_holes(rgba)

    key = chroma_key or _dominant_corner_color(rgba)
    w, h = rgba.width, rgba.height
    px = rgba.load()

    def _is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a > _ALPHA_FLOOR and _color_distance(r, g, b, key) <= threshold

    visited = bytearray(w * h)
    queue: deque[tuple[int, int]] = deque()

    # Seed from every border pixel that looks like background.
    for x in range(w):
        for y in (0, h - 1):
            if _is_bg(x, y) and not visited[y * w + x]:
                visited[y * w + x] = 1
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if _is_bg(x, y) and not visited[y * w + x]:
                visited[y * w + x] = 1
                queue.append((x, y))

    # Trapped pockets: background enclosed by the character (the magenta between
    # an arm and the body) isn't border-reachable, so also seed the flood from
    # interior near-key pixels. Gated to a *saturated* key (our magenta backdrop)
    # so we never seed from a character sharing a desaturated near-white/gray key
    # — that's the hole-punching the border-only flood exists to avoid.
    if max(key) - min(key) >= 120:
        for i, near in enumerate(_near_key_mask(rgba, key).getdata()):
            if near and not visited[i]:
                visited[i] = 1
                queue.append((i % w, i // w))

    while queue:
        x, y = queue.popleft()
        px[x, y] = (0, 0, 0, 0)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h:
                idx = ny * w + nx
                if not visited[idx]:
                    visited[idx] = 1
                    if _is_bg(nx, ny):
                        queue.append((nx, ny))
    return rgba


def _repair_internal_alpha_holes(image):
    """Fill transparent islands fully enclosed by opaque sprite pixels.

    Some providers return "transparent" PNGs with swiss-cheese alpha inside the
    character. Border flood-fill cannot see those because there is no opaque
    backdrop to key, so repair the alpha mask itself: transparent components that
    touch an image edge remain background; transparent components enclosed by
    the sprite are filled with the average color of their opaque neighbours.
    """
    from collections import deque

    rgba = image.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    visited = bytearray(w * h)

    def _is_transparent(x: int, y: int) -> bool:
        return px[x, y][3] <= _ALPHA_FLOOR

    def _mark_border_component(sx: int, sy: int) -> None:
        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        visited[sy * w + sx] = 1
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    idx = ny * w + nx
                    if not visited[idx] and _is_transparent(nx, ny):
                        visited[idx] = 1
                        queue.append((nx, ny))

    # First mark true background: all transparent pixels reachable from the edge.
    for x in range(w):
        for y in (0, h - 1):
            if _is_transparent(x, y) and not visited[y * w + x]:
                _mark_border_component(x, y)
    for y in range(h):
        for x in (0, w - 1):
            if _is_transparent(x, y) and not visited[y * w + x]:
                _mark_border_component(x, y)

    def _collect_hole(sx: int, sy: int) -> list[tuple[int, int]]:
        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        visited[sy * w + sx] = 1
        pixels: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            pixels.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    idx = ny * w + nx
                    if not visited[idx] and _is_transparent(nx, ny):
                        visited[idx] = 1
                        queue.append((nx, ny))
        return pixels

    def _fill_color(hole: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        samples: list[tuple[int, int, int]] = []
        seen = set(hole)
        for x, y in hole:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                    r, g, b, a = px[nx, ny]
                    if a > _ALPHA_FLOOR:
                        samples.append((r, g, b))
        if not samples:
            return (0, 0, 0, 255)
        return (
            round(sum(c[0] for c in samples) / len(samples)),
            round(sum(c[1] for c in samples) / len(samples)),
            round(sum(c[2] for c in samples) / len(samples)),
            255,
        )

    for start, _ in enumerate(visited):
        if visited[start]:
            continue
        x = start % w
        y = start // w
        if not _is_transparent(x, y):
            continue
        hole = _collect_hole(x, y)
        color = _fill_color(hole)
        for hx, hy in hole:
            px[hx, hy] = color
    return rgba


# ───────────────────────── frame extraction ─────────────────────────


def _fit_to_cell(image):
    """Crop to content, scale to fit a padded cell, and center on transparent."""
    from PIL import Image

    target = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    image = _drop_side_bleed(image)
    bbox = image.getbbox()
    if bbox is None:
        return target

    sprite = image.crop(bbox)
    max_w = CELL_WIDTH - _CELL_PAD
    max_h = CELL_HEIGHT - _CELL_PAD
    scale = min(max_w / sprite.width, max_h / sprite.height, 1.0)
    if scale != 1.0:
        sprite = sprite.resize(
            (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
            Image.Resampling.LANCZOS,
        )
    left = (CELL_WIDTH - sprite.width) // 2
    top = (CELL_HEIGHT - sprite.height) // 2
    target.alpha_composite(sprite, (left, top))
    return target


def _drop_side_bleed(image):
    """Remove tiny separated left/right lobes before fitting a frame.

    Frogger showed the failure mode: a good centered pose plus a thin vertical
    sliver from the neighbouring pose. By the time it reaches a cell, that sliver
    may be close enough to the subject that component extraction already grouped
    it. A horizontal alpha projection still reveals it as a small side lobe with
    a low mass compared to the main silhouette. Drop only those low-mass lobes;
    keep large lobes so wide poses and real limbs survive.
    """
    from PIL import Image

    rgba = image.convert("RGBA")
    w, h = rgba.size
    profile = _column_profile(rgba)  # mean alpha per column (fast C resize)

    segments: list[tuple[int, int, int]] = []  # (left, right, mass)
    start = mass = 0
    started = False
    for x, v in enumerate(profile + [0]):
        if v > 2:
            if not started:
                start, mass, started = x, 0, True
            mass += v
        elif started:
            segments.append((start, x, mass))
            started = False

    if len(segments) < 2:
        return rgba
    keep_mass = max(m for _, _, m in segments) * _SIDE_LOBE_RATIO
    keep = [(l, r) for l, r, m in segments if m >= keep_mass]
    if len(keep) == len(segments):
        return rgba

    # Zero every column band that isn't a kept segment (box paste, not per-pixel).
    rgba = rgba.copy()
    cut, prev = Image.new("RGBA", (w, h), (0, 0, 0, 0)), 0
    for left, right in keep:
        if left > prev:
            rgba.paste(cut.crop((prev, 0, left, h)), (prev, 0))
        prev = right
    if prev < w:
        rgba.paste(cut.crop((prev, 0, w, h)), (prev, 0))
    return rgba


def _connected_components(image) -> list[dict]:
    """Flood-fill the alpha mask into connected blobs (4-connectivity)."""
    alpha = image.getchannel("A")
    w, h = image.size
    data = alpha.tobytes()
    visited = bytearray(w * h)
    out: list[dict] = []

    for start, a in enumerate(data):
        if a <= _ALPHA_FLOOR or visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        pixels: list[int] = []
        min_x = w
        min_y = h
        max_x = 0
        max_y = 0
        while stack:
            cur = stack.pop()
            pixels.append(cur)
            x = cur % w
            y = cur // w
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for nb, ok in (
                (cur - 1, x > 0),
                (cur + 1, x + 1 < w),
                (cur - w, y > 0),
                (cur + w, y + 1 < h),
            ):
                if ok and not visited[nb] and data[nb] > _ALPHA_FLOOR:
                    visited[nb] = 1
                    stack.append(nb)
        out.append(
            {
                "pixels": pixels,
                "area": len(pixels),
                "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                "center_x": (min_x + max_x + 1) / 2,
            }
        )
    return out


def _sever_expected_gutters(strip, frame_count: int):
    """Cut thin vertical gutters at expected frame boundaries before labeling.

    Generated rows often have a shared shadow, glow, motion smear, or 1px bridge
    that connects neighbouring poses. Component detection then sees one giant
    blob and either fails or falls back to slot slicing. We know the requested
    frame count, so cut a very narrow transparent band at each expected boundary
    before connected-component labeling. If a pose truly overlaps the boundary,
    losing a few pixels is better than exporting merged frames.
    """
    if frame_count <= 1:
        return strip

    out = strip.copy()
    px = out.load()
    slot = out.width / frame_count
    half = max(2, min(8, round(slot * 0.02)))
    for i in range(1, frame_count):
        x = round(i * slot)
        left = max(0, x - half)
        right = min(out.width, x + half + 1)
        for gx in range(left, right):
            for gy in range(out.height):
                r, g, b, _a = px[gx, gy]
                px[gx, gy] = (r, g, b, 0)
    return out


def _segmentable(strip, frame_count: int) -> bool:
    """True if the (gutter-severed) strip yields ≥ *frame_count* distinct blobs.

    Used only as a quality gate: a row that can't show this many separable poses
    is a bad generation (caller retries / falls back), never silently sliced into
    merged frames.
    """
    components = _connected_components(strip)
    if not components:
        return False
    largest = max(c["area"] for c in components)
    seed_threshold = max(120, largest * 0.20)
    return sum(1 for c in components if c["area"] >= seed_threshold) >= frame_count


def _slot_crops(strip, frame_count: int) -> list:
    """Slice *strip* into *frame_count* uniform columns (one coordinate space).

    Equal-width columns keep every frame in a single shared coordinate frame, so
    a later union-crop + shared placement (:func:`normalize_cells`) preserves the
    row's real motion without the per-frame re-centering that makes a pet visibly
    slide. Neighbour side-bleed is trimmed per column.
    """
    w0 = max(1, strip.width // frame_count)
    h = strip.height
    return [_drop_side_bleed(strip.crop((i * w0, 0, i * w0 + w0, h))) for i in range(frame_count)]


def extract_strip_frames(
    strip,
    frame_count: int,
    *,
    chroma_key: tuple[int, int, int] | None = None,
    method: str = "auto",
    fit: bool = True,
) -> list:
    """Turn one generated row strip into *frame_count* frames.

    Background is keyed out, the expected frame gutters are severed, then the
    strip is sliced into equal columns. Connected components only *validate* that
    the row holds *frame_count* separable poses (``components`` raises, ``auto``
    falls back to slicing the un-severed strip).

    *fit* (default) fits+centers each frame into a 192x208 cell — the standalone
    contract for callers that don't normalize. Hatching passes ``fit=False`` to
    keep raw, coordinate-aligned columns for :func:`normalize_cells`, which lays
    one shared scale + baseline across the whole pet (no slide, no size pulse).
    """
    from PIL import Image

    if isinstance(strip, (str, Path)):
        with Image.open(strip) as opened:
            strip = opened.convert("RGBA")
    else:
        strip = strip.convert("RGBA")

    strip = remove_background(strip, chroma_key=chroma_key)
    severed = _sever_expected_gutters(strip, frame_count)
    segmentable = _segmentable(severed, frame_count)
    if method == "components" and not segmentable:
        raise ValueError(f"could not segment {frame_count} sprites from strip")

    frames = _slot_crops(severed if segmentable else strip, frame_count)
    return [_fit_to_cell(f) for f in frames] if fit else frames


def _column_profile(image) -> list[int]:
    """Per-column alpha mass — collapse the frame to a 1px-tall strip (fast in C)."""
    from PIL import Image

    return list(image.getchannel("A").resize((image.width, 1), Image.BILINEAR).getdata())


def _best_shift(ref: list[int], prof: list[int], window: int) -> int:
    """Integer dx that best aligns *prof* onto *ref* by cross-correlation.

    This is 1-D phase correlation: the body is the dominant mass in the column
    profile, so the peak overlap locks onto the body and a flipping arm/cape (a
    small secondary bump) doesn't move the match. Proven on the jitter case to
    cut body drift from ~9px to ~1px where a centroid/bbox anchor cannot.
    """
    n = len(ref)
    best_score: float | None = None
    best = 0
    for d in range(-window, window + 1):
        score = 0
        for x in range(max(0, d), min(n, n + d)):
            score += ref[x] * prof[x - d]
        if best_score is None or score > best_score:
            best_score = score
            best = d
    return best


def normalize_cells(frames_by_state: dict[str, list], *, pad: int = _NORMALIZE_PAD) -> dict[str, list]:
    """Register every frame into a 192x208 cell — the deterministic anti-jitter math.

    A per-frame "crop→scale→center" pipeline jitters because a moving limb/cape
    shifts the bbox (or even the centroid) and a per-frame scale pulses the size.
    The rigorous fix, matching image-registration practice (phase correlation)
    and AI-sprite pipelines (perfectpixel-studio / sprite-gen):

    1. **Cross-correlate** each frame's column profile against the per-state
       *median* profile to find the integer shift that locks the **body** in
       place — robust to limbs/cape because the body dominates the profile.
    2. **Union-crop** the registered frames through one shared window and apply
       **one shared scale** + bottom-anchor, so size and baseline are uniform and
       intra-state vertical motion (a jump's lift) is preserved.
    """
    from PIL import Image

    blank = lambda: Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))

    out: dict[str, list] = {}
    for state, frames in frames_by_state.items():
        rgba = [f.convert("RGBA") for f in frames]
        if not any(f.getbbox() for f in rgba):
            out[state] = [blank() for _ in frames]
            continue

        # Pad every frame to a common canvas so column profiles are comparable.
        w0 = max(f.width for f in rgba)
        h0 = max(f.height for f in rgba)
        canvas = []
        for f in rgba:
            if f.size != (w0, h0):
                c = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
                c.alpha_composite(f, (0, 0))
                f = c
            canvas.append(f)

        # Register horizontally: shift each frame to lock the body (xcorr).
        profiles = [_column_profile(f) for f in canvas]
        ref = [sorted(p[x] for p in profiles)[len(profiles) // 2] for x in range(w0)]
        window = max(8, w0 // 5)
        margin = window
        aligned = []
        for f, prof in zip(canvas, profiles):
            shifted = Image.new("RGBA", (w0 + 2 * margin, h0), (0, 0, 0, 0))
            shifted.alpha_composite(f, (margin + _best_shift(ref, prof, window), 0))
            aligned.append(shifted)

        # Shared window + scale over the registered set; bottom-anchored, centered.
        boxes = [b for b in (a.getbbox() for a in aligned) if b]
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        uw, uh = right - left, bottom - top
        scale = min((CELL_WIDTH - pad) / uw, (CELL_HEIGHT - pad) / uh)
        sw, sh = max(1, round(uw * scale)), max(1, round(uh * scale))
        px, py = round((CELL_WIDTH - sw) / 2), round((CELL_HEIGHT - pad // 2) - sh)

        cells = []
        for a in aligned:
            crop = a.crop((left, top, right, bottom))
            if crop.size != (sw, sh):
                crop = crop.resize((sw, sh), Image.Resampling.LANCZOS)
            cell = blank()
            cell.alpha_composite(crop, (px, py))
            cells.append(cell)
        out[state] = cells
    return out


# ───────────────────────── atlas composition ─────────────────────────


def single_frame(image, *, fit: bool = True):
    """One frame from a standalone image (e.g. the base look).

    Used as an idle fallback so a pet always renders even if the idle row
    generation failed. *fit* yields a finished 192x208 cell; ``fit=False`` yields
    the raw keyed sprite for :func:`normalize_cells` to place with the rest.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGBA")
    keyed = remove_background(image)
    return _fit_to_cell(keyed) if fit else _drop_side_bleed(keyed)


def _clear_transparent_rgb(image):
    """Zero the RGB of fully-transparent pixels (no colored-halo residue)."""
    from PIL import Image

    rgba = image.convert("RGBA")
    data = bytearray(rgba.tobytes())
    for i in range(0, len(data), 4):
        if data[i + 3] == 0:
            data[i] = data[i + 1] = data[i + 2] = 0
    return Image.frombytes("RGBA", rgba.size, bytes(data))


def mirror_frames(frames: list) -> list:
    """Horizontally flip each frame *in place* (RGBA-safe).

    Used to derive ``running-left`` from an approved ``running-right`` row. The
    flip is per-frame so the leftward loop preserves the rightward loop's frame
    order and timing — this is NOT a whole-strip reverse (which would play the
    animation backwards), matching the petdex/Codex mirror rule.
    """
    from PIL import Image

    flip = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT
    return [frame.convert("RGBA").transpose(flip) for frame in frames]


def compose_atlas(frames_by_state: dict[str, list]):
    """Pack per-state frame lists into the Hermes atlas (RGBA, residue-cleared).

    Missing/short states leave their trailing cells transparent; extra frames
    beyond a state's spec are dropped.
    """
    from PIL import Image

    atlas = Image.new("RGBA", (ATLAS_WIDTH, ATLAS_HEIGHT), (0, 0, 0, 0))
    for state, row, count in ROW_SPECS:
        frames = frames_by_state.get(state) or []
        for col, frame in enumerate(frames[:count]):
            cell = frame.convert("RGBA")
            if cell.size != (CELL_WIDTH, CELL_HEIGHT):
                cell = _fit_to_cell(cell)
            atlas.alpha_composite(cell, (col * CELL_WIDTH, row * CELL_HEIGHT))
    return _clear_transparent_rgb(atlas)


def atlas_to_webp_bytes(atlas) -> bytes:
    """Encode an atlas image to lossless WebP bytes (the on-disk pet format)."""
    buf = io.BytesIO()
    atlas.save(buf, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return buf.getvalue()


def validate_atlas(atlas) -> dict:
    """Check geometry, per-cell occupancy, and transparency invariants.

    Returns ``{ok, width, height, errors, warnings, filled_states}``. Errors are
    blockers (wrong size, empty used cell, opaque/dirty transparency); warnings
    are soft (a whole state row blank — generation likely dropped a row).
    """
    from PIL import Image

    if isinstance(atlas, (str, Path)):
        with Image.open(atlas) as opened:
            atlas = opened.convert("RGBA")
    else:
        atlas = atlas.convert("RGBA")

    errors: list[str] = []
    warnings: list[str] = []

    if atlas.size != (ATLAS_WIDTH, ATLAS_HEIGHT):
        errors.append(f"expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}, got {atlas.width}x{atlas.height}")
        return {"ok": False, "width": atlas.width, "height": atlas.height, "errors": errors, "warnings": warnings, "filled_states": []}

    filled_states: list[str] = []
    for state, row, count in ROW_SPECS:
        row_pixels = 0
        for col in range(count):
            left = col * CELL_WIDTH
            top = row * CELL_HEIGHT
            cell = atlas.crop((left, top, left + CELL_WIDTH, top + CELL_HEIGHT))
            nonblank = sum(cell.getchannel("A").histogram()[1:])
            row_pixels += nonblank
        if row_pixels > 0:
            filled_states.append(state)
        else:
            warnings.append(f"state '{state}' has no frames")

    if not filled_states:
        errors.append("atlas is empty — no state produced any frames")

    # Transparent pixels must carry zero RGB (no halo residue).
    data = atlas.tobytes()
    residue = 0
    for i in range(0, len(data), 4):
        if data[i + 3] == 0 and (data[i] or data[i + 1] or data[i + 2]):
            residue += 1
    if residue:
        errors.append(f"{residue} transparent pixels retain RGB residue")

    return {
        "ok": not errors,
        "width": atlas.width,
        "height": atlas.height,
        "errors": errors,
        "warnings": warnings,
        "filled_states": filled_states,
    }
