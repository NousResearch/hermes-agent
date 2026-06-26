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


def _defringe(rgba):
    """Shave the 1px antialiased edge ring left after keying.

    Chroma keying can't catch the antialiased band where the sprite meets the
    backdrop — those pixels are a key/sprite blend, too far from the key to be
    removed, so they ring the cutout in magenta/green. Erode the alpha by one
    pixel (a 3x3 min filter) to drop that contaminated ring; the sprite's own
    thick dark outline keeps the silhouette intact. Built on a C-level filter, no
    per-pixel Python.
    """
    from PIL import ImageFilter

    rgba.putalpha(rgba.getchannel("A").filter(ImageFilter.MinFilter(3)))
    return rgba


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

    from PIL import Image, ImageChops

    rgba = image.convert("RGBA")
    if _has_transparency(rgba):
        return _repair_internal_alpha_holes(rgba)

    key = chroma_key or _dominant_corner_color(rgba)
    w, h = rgba.width, rgba.height
    px = rgba.load()

    def _is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a > _ALPHA_FLOOR and _color_distance(r, g, b, key) <= threshold

    # Fast path for strongly-saturated chroma keys (our normal sprite prompts use
    # hot magenta): remove all near-key opaque pixels with C-level channel ops.
    # This clears both border-connected backdrop and enclosed triangular pockets
    # between connected limbs/capes, without a Python flood over ~1.5M pixels.
    if max(key) - min(key) >= 120:
        near = _near_key_mask(rgba, key)  # L mask, 255 where near key
        opaque = rgba.getchannel("A").point(lambda a: 255 if a > _ALPHA_FLOOR else 0)
        remove_mask = ImageChops.darker(near, opaque)
        keyed = Image.composite(Image.new("RGBA", rgba.size, (0, 0, 0, 0)), rgba, remove_mask)
        return _defringe(keyed)

    visited = bytearray(w * h)
    # Mark removals in a flat mask and apply them in one C composite at the end —
    # writing `px[x, y] = (0,0,0,0)` per pixel was ~3M PixelAccess calls (84% of
    # the whole pipeline) and pegged a core in pure Python, stalling the gateway.
    remove = bytearray(w * h)
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
        remove[y * w + x] = 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h:
                idx = ny * w + nx
                if not visited[idx]:
                    visited[idx] = 1
                    if _is_bg(nx, ny):
                        queue.append((nx, ny))

    # One C-level composite instead of millions of per-pixel writes: paint the
    # flooded pixels to (0,0,0,0) wherever the mask is set.
    mask = Image.frombytes("L", (w, h), bytes(remove)).point(lambda v: 255 if v else 0)
    return _defringe(Image.composite(Image.new("RGBA", rgba.size, (0, 0, 0, 0)), rgba, mask))


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
        # NEAREST, not LANCZOS: the generated "pixel art" has hard edges, and any
        # interpolating resample anti-aliases them into a blurry, washed-out
        # sprite once the renderer upscales the cell. Crisp blocky downscale reads
        # as real pixel art.
        sprite = sprite.resize(
            (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
            Image.Resampling.NEAREST,
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

    runs = _content_runs(profile)
    if len(runs) < 2:
        return rgba
    masses = [sum(profile[l:r]) for l, r in runs]
    keep_mass = max(masses) * _SIDE_LOBE_RATIO
    keep = [run for run, m in zip(runs, masses) if m >= keep_mass]
    if len(keep) == len(runs):
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


def _erase_long_axis_lines(image):
    """Remove thin slot-spanning guide/floor/divider lines.

    Gemini will sometimes satisfy "baseline" / "cell" language by drawing
    literal horizontal floors or vertical panel dividers. They survive chroma
    keying and connect otherwise clean poses. Drop only *thin* rows/columns that
    span nearly the whole slot; thick sprite body rows are left alone.
    """
    from PIL import Image

    rgba = image.convert("RGBA").copy()
    w, h = rgba.size
    alpha = rgba.getchannel("A")

    def _thin_groups(indices: list[int]) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        start: int | None = None
        prev: int | None = None
        for idx in indices:
            if start is None:
                start = prev = idx
                continue
            if prev is not None and idx == prev + 1:
                prev = idx
                continue
            if start is not None and prev is not None and prev - start + 1 <= 4:
                groups.append((start, prev + 1))
            start = prev = idx
        if start is not None and prev is not None and prev - start + 1 <= 4:
            groups.append((start, prev + 1))
        return groups

    wide_rows = [
        y
        for y in range(h)
        if sum(1 for x in range(w) if alpha.getpixel((x, y)) > _ALPHA_FLOOR) >= w * 0.85
    ]
    tall_cols = [
        x
        for x in range(w)
        if sum(1 for y in range(h) if alpha.getpixel((x, y)) > _ALPHA_FLOOR) >= h * 0.85
    ]

    clear = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    for top, bottom in _thin_groups(wide_rows):
        rgba.paste(clear.crop((0, top, w, bottom)), (0, top))
    for left, right in _thin_groups(tall_cols):
        rgba.paste(clear.crop((left, 0, right, h)), (left, 0))
    return rgba


def _component_boxes(image) -> list[tuple[tuple[int, int, int, int], int]]:
    """Connected opaque components as ``[(bbox, mass)]``.

    A full ML segmenter would be overkill here: after chroma keying, "the pet" is
    the dominant connected alpha component inside each known slot. Tiny detached
    sparkles, tears, UI dots, and neighbour slivers are separate components.
    """
    from collections import deque

    rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        return []
    l0, t0, r0, b0 = bbox
    w, h = r0 - l0, b0 - t0
    alpha = rgba.getchannel("A").load()
    visited = bytearray(w * h)
    out: list[tuple[tuple[int, int, int, int], int]] = []

    for start in range(w * h):
        if visited[start]:
            continue
        sx, sy = start % w, start // w
        ax, ay = l0 + sx, t0 + sy
        visited[start] = 1
        if alpha[ax, ay] <= _ALPHA_FLOOR:
            continue

        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        left = right = sx
        top = bottom = sy
        mass = 0
        while queue:
            x, y = queue.popleft()
            mass += 1
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    idx = ny * w + nx
                    if not visited[idx]:
                        visited[idx] = 1
                        if alpha[l0 + nx, t0 + ny] > _ALPHA_FLOOR:
                            queue.append((nx, ny))
        out.append(((l0 + left, t0 + top, l0 + right + 1, t0 + bottom + 1), mass))
    return out


def _isolate_slot_subject(image):
    """Keep the slot's real subject; drop detached effects/noise."""
    from PIL import Image

    rgba = _erase_long_axis_lines(image)
    comps = _component_boxes(rgba)
    if not comps:
        return rgba

    main_box, main_mass = max(comps, key=lambda item: item[1])
    ml, mt, mr, mb = main_box
    mw = max(1, mr - ml)
    keep: list[tuple[int, int, int, int]] = []
    for box, mass in comps:
        if box == main_box:
            keep.append(box)
            continue
        left, _top, right, _bottom = box
        overlap = max(0, min(right, mr) - max(left, ml))
        center_x = (left + right) / 2
        near_main = (ml - mw * 0.25) <= center_x <= (mr + mw * 0.25)
        # Keep meaningful attached-looking accessories such as halos; drop
        # sparkles/tears/noise that don't overlap the body column.
        if mass >= max(24, main_mass * 0.035) and (overlap >= mw * 0.3 or near_main):
            keep.append(box)

    out = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    for box in keep:
        out.alpha_composite(rgba.crop(box), (box[0], box[1]))
    return out


def _has_slot_padding(image) -> bool:
    """True when content has empty room on all four slot edges."""
    bbox = image.getbbox()
    if bbox is None:
        return False
    w, h = image.size
    left, top, right, bottom = bbox
    min_x = max(4, min(12, round(w * 0.025)))
    min_y = max(4, min(16, round(h * 0.02)))
    return left >= min_x and top >= min_y and w - right >= min_x and h - bottom >= min_y


def _slot_bounds(width: int, frame_count: int) -> list[tuple[int, int]]:
    return [
        (round(i * width / frame_count), round((i + 1) * width / frame_count))
        for i in range(frame_count)
    ]


def _group_component_rows(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group component boxes into visual rows, then sort left→right."""
    if not boxes:
        return []
    heights = sorted(max(1, b[3] - b[1]) for b in boxes)
    row_tol = max(12, heights[len(heights) // 2] * 0.55)
    rows: list[list[tuple[int, int, int, int]]] = []
    centers: list[float] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (box[1] + box[3]) / 2
        for i, center in enumerate(centers):
            if abs(cy - center) <= row_tol:
                rows[i].append(box)
                centers[i] = sum((b[1] + b[3]) / 2 for b in rows[i]) / len(rows[i])
                break
        else:
            rows.append([box])
            centers.append(cy)
    ordered = [row for _center, row in sorted(zip(centers, rows, strict=False), key=lambda item: item[0])]
    for row in ordered:
        row.sort(key=lambda b: (b[0] + b[2]) / 2)
    return ordered


def _merge_related_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge disconnected parts that clearly belong to one subject.

    Capes, tails, horns, and held props sometimes key as separate components.
    Merge components on the same visual row when their vertical spans overlap and
    the horizontal gap is tiny compared with the component size. Do not bridge the
    much larger gaps between separate poses.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            al, at, ar, ab = a
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bl, bt, br, bb = boxes[j]
                v_overlap = max(0, min(ab, bb) - max(at, bt))
                min_h = max(1, min(ab - at, bb - bt))
                gap = max(0, max(al, bl) - min(ar, br))
                min_w = max(1, min(ar - al, br - bl))
                if v_overlap >= min_h * 0.45 and gap <= max(14, min_w * 0.22):
                    al, at, ar, ab = min(al, bl), min(at, bt), max(ar, br), max(ab, bb)
                    used[j] = True
                    changed = True
            merged.append((al, at, ar, ab))
        boxes = merged
    return boxes


def _component_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Extract frame subjects as connected non-background objects.

    This is the robust path for models that ignore "one horizontal row" and emit a
    2D sprite grid. We count real opaque subject components, discard tiny
    detached effects, sort in reading order, and return exactly *frame_count*
    frames. Slot slicing is only a fallback when object detection can't satisfy
    the contract.
    """
    from PIL import Image

    def attempt(source) -> list | None:
        comps = _component_boxes(source)
        if not comps:
            return None

        max_mass = max(m for _box, m in comps)
        subjects = _merge_related_boxes([box for box, mass in comps if mass >= max(64, max_mass * 0.12)])
        if len(subjects) < frame_count:
            return None

        rows = _group_component_rows(subjects)
        ordered = [box for row in rows for box in row][:frame_count]
        if len(ordered) < frame_count:
            return None

        if require_padding:
            min_x = max(4, min(12, round(source.width * 0.01)))
            min_y = max(4, min(16, round(source.height * 0.015)))
            for left, top, right, bottom in ordered:
                if left < min_x or top < min_y or source.width - right < min_x or source.height - bottom < min_y:
                    return None

        multirow = len(rows) > 1
        frames = []
        for left, top, right, bottom in ordered:
            pad_x = max(8, round((right - left) * 0.08))
            pad_y = max(8, round((bottom - top) * 0.08))
            if multirow:
                crop_box = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    min(source.width, right + pad_x),
                    min(source.height, bottom + pad_y),
                )
            elif frame_count == 1:
                crop_box = (0, 0, source.width, source.height)
            else:
                # Preserve vertical motion for true one-row strips (jumping,
                # bobbing) while still narrowing X around the object.
                crop_box = (max(0, left - pad_x), 0, min(source.width, right + pad_x), source.height)
            frame = Image.new("RGBA", (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), (0, 0, 0, 0))
            rel = (left - crop_box[0], top - crop_box[1], right - crop_box[0], bottom - crop_box[1])
            frame.alpha_composite(source.crop((left, top, right, bottom)), (rel[0], rel[1]))
            # The global component pass already chose the subject box. Do not run
            # another component filter here: capes/tails can be legitimate
            # disconnected lobes inside the chosen subject box.
            frames.append(frame)
        return frames

    return attempt(strip) or attempt(_erase_long_axis_lines(strip))


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
    half = max(3, min(18, round(slot * 0.06)))
    for i in range(1, frame_count):
        x = round(i * slot)
        left = max(0, x - half)
        right = min(out.width, x + half + 1)
        for gx in range(left, right):
            for gy in range(out.height):
                r, g, b, _a = px[gx, gy]
                px[gx, gy] = (r, g, b, 0)
    return out


def _slot_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Slice *strip* into *frame_count* uniform columns (one coordinate space).

    Equal-width columns keep every frame in a single shared coordinate frame, so
    a later union-crop + shared placement (:func:`normalize_cells`) preserves the
    row's real motion without the per-frame re-centering that makes a pet visibly
    slide. Each slot is cleaned independently so detached effects, floors,
    dividers, and neighbour slivers do not become "frames".
    """
    h = strip.height
    frames = []
    for left, right in _slot_bounds(strip.width, frame_count):
        slot = _drop_side_bleed(_isolate_slot_subject(strip.crop((left, 0, right, h))))
        if require_padding and not _has_slot_padding(slot):
            return None
        frames.append(slot)
    return frames


def _content_runs(profile: list[int], *, threshold: int = 2) -> list[tuple[int, int]]:
    """Contiguous column spans whose alpha mass exceeds *threshold*.

    A column-projection of the alpha mask: empty (background) columns separate
    one pose from the next, so the runs ARE the candidate frames.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, v in enumerate(list(profile) + [0]):
        if v > threshold:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
    return runs


def _frame_x_ranges(strip, frame_count: int) -> list[tuple[int, int]] | None:
    """Per-frame ``(left, right)`` column ranges from the row's empty gutters.

    The standard sprite-sheet slice — once poses are separated by real gaps
    (which generation now enforces), splitting is just "find the empty columns":

    * spans == frames → one span per frame.
    * spans  > frames → merge across the smallest gaps. A detached halo/ear sits
      a tiny gap from its body, while the inter-pose gutter is the big gap that
      survives — so over-segmentation (and any over-eager gutter sever) repairs
      itself by collapsing only the small internal gaps.
    * spans  < frames → poses are touching; not separable by gutters (the caller
      raises for ``components`` or falls back to even slots for ``auto``).

    Ranges span content only; the caller crops full cell height, so tall ears /
    halos are never cut.
    """
    profile = _column_profile(strip)
    runs = _content_runs(profile)
    if not runs:
        return None

    # Drop trivial specks so stray noise never counts as a pose.
    masses = [sum(profile[l:r]) for l, r in runs]
    floor = max(masses) * 0.02
    runs = [run for run, m in zip(runs, masses) if m >= floor]
    if len(runs) < frame_count:
        return None

    groups = [[l, r] for l, r in runs]
    while len(groups) > frame_count:
        gi = min(range(len(groups) - 1), key=lambda i: groups[i + 1][0] - groups[i][1])
        groups[gi][1] = groups[gi + 1][1]
        del groups[gi + 1]
    return [(l, r) for l, r in groups]


def _significant_subject_boxes(image) -> list[tuple[int, int, int, int]]:
    comps = _component_boxes(image)
    if not comps:
        return []
    max_mass = max(mass for _box, mass in comps)
    return _merge_related_boxes([box for box, mass in comps if mass >= max(32, max_mass * 0.12)])


def _validate_extracted_frames(frames: list, frame_count: int) -> None:
    """Reject rows where one "frame" is really multiple poses.

    A bad provider roll can collapse a strip into tiny repeated poses. If we let
    that through, normalization sees a huge motion envelope and shrinks the
    entire pet to postage-stamp size. Catch the row here so hatch can regenerate
    it instead of saving a technically non-empty but visually broken atlas.
    """
    if len(frames) != frame_count:
        raise ValueError(f"expected {frame_count} frames, got {len(frames)}")

    boxes = []
    for i, frame in enumerate(frames):
        bbox = frame.getbbox()
        if bbox is None:
            raise ValueError(f"frame {i} is empty")
        subjects = _significant_subject_boxes(frame)
        if len(subjects) >= 3:
            raise ValueError(f"frame {i} contains multiple separated subjects")
        boxes.append(bbox)

    if frame_count <= 1:
        return

    widths = sorted(b[2] - b[0] for b in boxes)
    heights = sorted(b[3] - b[1] for b in boxes)
    med_w = max(1, widths[len(widths) // 2])
    med_h = max(1, heights[len(heights) // 2])
    for i, (left, top, right, bottom) in enumerate(boxes):
        width = right - left
        height = bottom - top
        # A legitimate wing/arm can be wider than the median pose. A frame that is
        # several times wider while not proportionally taller is usually multiple
        # mini-poses packed into one accepted frame.
        if width > max(med_w * 3.0, med_w + 96) and height <= med_h * 1.6:
            raise ValueError(f"frame {i} is a multi-pose width outlier")


def extract_strip_frames(
    strip,
    frame_count: int,
    *,
    chroma_key: tuple[int, int, int] | None = None,
    method: str = "auto",
    fit: bool = True,
) -> list:
    """Turn one generated row strip into *frame_count* frames.

    The background is keyed out, then strict extraction treats the requested
    frame count as the source of truth: slice known equal slots, isolate the real
    subject in each slot, and require empty padding on X and Y. Empty chroma
    gutters are only a lenient salvage fallback.

    Each frame is cropped at full cell height so tall ears / halos are never
    clipped; detached effects and neighbour slivers are dropped per slot. When a
    pose does not have required space around it, ``components`` raises and
    ``auto`` falls back to best-effort slicing.

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

    # Strict path: count actual non-background subjects first. This handles both
    # the intended one-row strip and model-cheated 2D grids without ever stacking
    # two visual rows into one frame.
    frames = _component_crops(strip, frame_count, require_padding=True)
    if frames is None:
        frames = _slot_crops(strip, frame_count, require_padding=True)
    if frames is None:
        if method == "components":
            raise ValueError(f"could not segment {frame_count} padded sprites from strip")

        # Lenient salvage for the final attempt: prefer real gutters when they
        # exist, then sever expected boundaries, then fall back to raw slots. Still
        # try object extraction first, just without edge-padding enforcement, so
        # cached/borderline model rolls can be inspected without stacking a 2D grid.
        frames = _component_crops(strip, frame_count, require_padding=False)
    if frames is None:
        source = strip
        ranges = _frame_x_ranges(source, frame_count)
        if ranges is None:
            source = _sever_expected_gutters(strip, frame_count)
            ranges = _frame_x_ranges(source, frame_count)

        if ranges is None:
            frames = _slot_crops(source, frame_count, require_padding=False) or []
        else:
            h = source.height
            pad = max(2, min(16, round((source.width / max(1, frame_count)) * 0.04)))
            frames = [
                _drop_side_bleed(_isolate_slot_subject(source.crop((max(0, left - pad), 0, min(source.width, right + pad), h))))
                for left, right in ranges
            ]
    _validate_extracted_frames(frames, frame_count)
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
    2. **Union-crop** through one shared state window, then scale every state by a
       single global factor keyed to its median pose height, so the character is
       the same on-screen size in every row while a jump's lift still fits.
    """
    from PIL import Image

    blank = lambda: Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    med = lambda vs: sorted(vs)[len(vs) // 2]  # robust center; ignores a limb/cape outlier

    out: dict[str, list] = {}
    prepared: dict[str, tuple[list, tuple[int, int, int, int], tuple[int, int]]] = {}
    # Fill the cell — real petdex pets sit ~pad from the edges; the K cap below
    # keeps a tall pose (a jump's lift) from clipping.
    target_w = CELL_WIDTH - pad
    target_h = CELL_HEIGHT - pad

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

        # Shared window over the registered set; scale is resolved against a
        # common apparent-character target below.
        boxes = [b for b in (a.getbbox() for a in aligned) if b]
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        prepared[state] = (
            aligned,
            (left, top, right, bottom),
            (med([b[2] - b[0] for b in boxes]), med([b[3] - b[1] for b in boxes])),
        )

    if not prepared:
        return out

    # Uniform apparent size: scale each state by K / pose_h, so a row the model
    # drew small renders as big as one it drew large. K is the one global cap that
    # keeps the tallest/widest motion envelope (a jump's lift) inside the cell —
    # for a still row union ≈ pose so its term ≈ target_h (full fill).
    K = target_h
    for (_aligned, (left, top, right, bottom), (_pose_w, pose_h)) in prepared.values():
        uw, uh = right - left, bottom - top
        K = min(K, target_h * pose_h / max(1, uh), target_w * pose_h / max(1, uw))

    for state, (aligned, (left, top, right, bottom), (_pose_w, pose_h)) in prepared.items():
        uw, uh = right - left, bottom - top
        scale = K / max(1, pose_h)
        sw, sh = max(1, round(uw * scale)), max(1, round(uh * scale))
        px, py = round((CELL_WIDTH - sw) / 2), round((CELL_HEIGHT - pad // 2) - sh)

        cells = []
        for a in aligned:
            crop = a.crop((left, top, right, bottom))
            if crop.size != (sw, sh):
                # NEAREST keeps the pixel-art edges crisp; LANCZOS blurred them.
                crop = crop.resize((sw, sh), Image.Resampling.NEAREST)
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
    cell_boxes_by_state: dict[str, list[tuple[int, int, int, int]]] = {}
    for state, row, count in ROW_SPECS:
        row_pixels = 0
        boxes: list[tuple[int, int, int, int]] = []
        for col in range(count):
            left = col * CELL_WIDTH
            top = row * CELL_HEIGHT
            cell = atlas.crop((left, top, left + CELL_WIDTH, top + CELL_HEIGHT))
            nonblank = sum(cell.getchannel("A").histogram()[1:])
            row_pixels += nonblank
            bbox = cell.getbbox()
            if bbox is not None:
                boxes.append(bbox)
        if row_pixels > 0:
            filled_states.append(state)
            cell_boxes_by_state[state] = boxes
        else:
            warnings.append(f"state '{state}' has no frames")

    if not filled_states:
        errors.append("atlas is empty — no state produced any frames")

    # A visually valid pet must occupy the cell. A single bad row can otherwise
    # poison global normalization and shrink every state to a tiny postage stamp
    # while still passing the old "non-empty cells" check.
    all_widths = sorted(
        right - left
        for boxes in cell_boxes_by_state.values()
        for left, _top, right, _bottom in boxes
    )
    all_heights = sorted(
        bottom - top
        for boxes in cell_boxes_by_state.values()
        for _left, top, _right, bottom in boxes
    )
    global_med_w = 0
    global_med_h = 0
    if all_widths and all_heights:
        global_med_w = all_widths[len(all_widths) // 2]
        median_h = all_heights[len(all_heights) // 2]
        global_med_h = median_h
        min_h = max(56, round(CELL_HEIGHT * 0.28))
        if median_h < min_h:
            errors.append(f"atlas sprites are too small after normalization (median frame height {median_h}px)")

    for state, boxes in cell_boxes_by_state.items():
        if len(boxes) <= 1:
            continue
        widths = sorted(right - left for left, _top, right, _bottom in boxes)
        heights = sorted(bottom - top for _left, top, _right, bottom in boxes)
        med_w = max(1, widths[len(widths) // 2])
        med_h = max(1, heights[len(heights) // 2])
        max_w = widths[-1]
        max_h = heights[-1]
        if max_w > max(med_w * 3.0, med_w + 96) and max_h <= med_h * 1.6:
            errors.append(f"state '{state}' contains a multi-pose frame outlier")
        # Per-state collapse guard: one malformed row (tiny slivers / chopped
        # fragments) should not pass because other rows are healthy.
        if global_med_w and global_med_h:
            min_state_w = max(32, round(global_med_w * 0.42))
            min_state_h = max(40, round(global_med_h * 0.50))
            if med_w < min_state_w or med_h < min_state_h:
                errors.append(
                    f"state '{state}' appears collapsed (median {med_w}x{med_h}px, global median {global_med_w}x{global_med_h}px)"
                )

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
