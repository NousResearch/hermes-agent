"""Deterministic spritesheet assembly — generated row strips → Hermes atlas.

Image models draw a row of poses well but can't do exact grid geometry, so the
model never owns the layout: it emits one loose strip per state and these ops
slice it into centered transparent 192x208 cells packed into the petdex/Codex
atlas (8 cols x 9 rows, 1536x1872; see ``constants.CODEX_STATE_ROWS``).
Segmentation/fit/residue logic adapted from OpenAI's ``hatch-pet`` skill (Apache-2.0).
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from itertools import groupby
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from agent.pet.constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)

CELL_WIDTH = FRAME_W
CELL_HEIGHT = FRAME_H

# (state, row index, frame count). Order/row indices MUST match
# ``constants.CODEX_STATE_ROWS``; frame counts mirror the petdex ``hatch-pet``
# spec. Rows shorter than 8 leave their tail transparent (renderer trims it).
# ``running`` is the in-place *working* state; ``running-right``/``-left`` walk.
ROW_SPECS: list[tuple[str, int, int]] = [
    ("idle", 0, 6), ("running-right", 1, 8), ("running-left", 2, 8), ("waving", 3, 4), ("jumping", 4, 5),
    ("failed", 5, 8), ("waiting", 6, 6), ("running", 7, 6), ("review", 8, 6),
]

ATLAS_WIDTH = max(count for _, _, count in ROW_SPECS) * CELL_WIDTH
ATLAS_HEIGHT = len(ROW_SPECS) * CELL_HEIGHT

_ALPHA_FLOOR = 16  # alpha at/below which a pixel is "background"
_CELL_PAD = 10  # padding kept around a fitted sprite
_NORMALIZE_PAD = 14  # normalized cells fill like real petdex pets (~5px from the edges)
_SIDE_LOBE_RATIO = 0.18  # adjacent-pose bleed is a small lobe; sizeable lobes (wide poses) survive
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEAREST = Image.Resampling.NEAREST  # interpolating resamples blur hard pixel-art edges


def _median(values) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _median_box_size(boxes) -> tuple[int, int]:
    """``(median width, median height)`` of ``(l, t, r, b)`` *boxes*, each at least 1."""
    return max(1, _median(r - l for l, _t, r, _b in boxes)), max(1, _median(b - t for _l, t, _r, b in boxes))


def _blank(size=(CELL_WIDTH, CELL_HEIGHT)):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _place(sprite, size: tuple[int, int], offset: tuple[int, int] = (0, 0)):
    """*sprite* alpha-composited at *offset* onto a new transparent canvas of *size*."""
    canvas = _blank(size)
    canvas.alpha_composite(sprite, offset)
    return canvas


def _clear_region(image, box: tuple[int, int, int, int]) -> None:
    """Make the ``(left, top, right, bottom)`` region of *image* fully transparent, in place."""
    image.paste(_blank((box[2] - box[0], box[3] - box[1])), (box[0], box[1]))


def _load_rgba(image):
    """Open a path (or take an image) as RGBA."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGBA")
    return image.convert("RGBA")


def _flood(w: int, h: int, visited: bytearray, seeds, accept) -> list[tuple[int, int]]:
    """4-connected BFS from *seeds* over pixels passing *accept*; returns the visited pixels."""
    queue = deque(seeds)
    pixels: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        pixels.append((x, y))
        for dx, dy in _NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny * w + nx]:
                visited[ny * w + nx] = 1
                if accept(nx, ny):
                    queue.append((nx, ny))
    return pixels


def _border_flood(w: int, h: int, visited: bytearray, accept) -> list[tuple[int, int]]:
    """Flood from every border pixel passing *accept* (edge-connected region only)."""
    border = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    seeds: list[tuple[int, int]] = []
    for x, y in border:  # corners appear twice in ``border``; the visited mark dedupes them
        if not visited[y * w + x] and accept(x, y):
            visited[y * w + x] = 1
            seeds.append((x, y))
    return _flood(w, h, visited, seeds, accept)


def _unvisited_components(w: int, h: int, visited: bytearray, accept):
    """Yield each 4-connected component of not-yet-visited pixels passing *accept*, in scan order."""
    for start in range(w * h):
        if visited[start]:
            continue
        visited[start] = 1
        x, y = start % w, start // w
        if accept(x, y):
            yield _flood(w, h, visited, [(x, y)], accept)


def _near_key_mask(image, key: tuple[int, int, int], tol: int = 48):
    """``L`` mask, 255 where a pixel is within *tol* per-channel of *key*.

    Tight on purpose: marks only near-pure backdrop so trapped chroma pockets seed the flood while chroma-tinted character pixels stay outside it.
    """
    r, g, b = (ch.point(lambda v, k=k: 255 if abs(v - k) <= tol else 0) for ch, k in zip(image.split()[:3], key))
    return ImageChops.darker(ImageChops.darker(r, g), b)


def _remove_masked(rgba, mask):
    """Clear the pixels *mask* (``L``, 255 = remove) selects, then erode alpha by 1px (3x3 min).

    The erosion drops the antialiased key/sprite blend ring (too far from the key to match); the sprite's thick outline keeps the silhouette.
    """
    out = Image.composite(_blank(rgba.size), rgba, mask)
    out.putalpha(out.getchannel("A").filter(ImageFilter.MinFilter(3)))
    return out


def remove_background(image, *, chroma_key: tuple[int, int, int] | None = None, threshold: float = 90.0):
    """Return *image* (RGBA) with its flat background keyed out to transparent.

    Strips already carrying a real alpha background (>5% of pixels at/below the floor)
    are left alone (holes repaired). Otherwise key out *chroma_key* (or the most common
    opaque corner color) from the border inward: a global color match punched holes
    wherever an interior highlight matched the backdrop.
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] <= _ALPHA_FLOOR and sum(alpha.histogram()[: _ALPHA_FLOOR + 1]) > w * h * 0.05:
        return _repair_internal_alpha_holes(rgba)
    px = rgba.load()
    if not (key := chroma_key):
        corners = Counter(tuple(px[x, y][:3]) for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)) if px[x, y][3] > _ALPHA_FLOOR)
        key = corners.most_common(1)[0][0] if corners else (0, 255, 0)
    # Fast path for saturated chroma keys (our prompts use hot magenta): C-level
    # channel ops clear border backdrop and enclosed pockets alike, no Python flood.
    if max(key) - min(key) >= 120:
        opaque = rgba.getchannel("A").point(lambda a: 255 if a > _ALPHA_FLOOR else 0)
        return _remove_masked(rgba, ImageChops.darker(_near_key_mask(rgba, key), opaque))

    def _is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a > _ALPHA_FLOOR and math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2) <= threshold

    # Border-only flood on purpose: a desaturated near-white/gray key must never
    # seed from the character's interior (that is the hole-punching case). Mark
    # removals in a flat mask and composite once in C — per-pixel writes stalled the gateway.
    remove = bytearray(w * h)
    for x, y in _border_flood(w, h, bytearray(w * h), _is_bg):
        remove[y * w + x] = 1
    return _remove_masked(rgba, Image.frombytes("L", (w, h), bytes(remove)).point(lambda v: 255 if v else 0))


def _repair_internal_alpha_holes(image):
    """Fill transparent islands fully enclosed by opaque sprite pixels.

    Some providers return "transparent" PNGs with swiss-cheese alpha inside the character; enclosed holes take the average opaque-neighbour color.
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    visited = bytearray(w * h)
    _is_transparent = lambda x, y: px[x, y][3] <= _ALPHA_FLOOR  # noqa: E731
    _border_flood(w, h, visited, _is_transparent)  # edge-connected transparency = background
    for hole in _unvisited_components(w, h, visited, _is_transparent):
        seen = set(hole)
        samples = [
            px[nx, ny][:3]
            for x, y in hole
            for dx, dy in _NEIGHBOURS
            if 0 <= (nx := x + dx) < w and 0 <= (ny := y + dy) < h and (nx, ny) not in seen and px[nx, ny][3] > _ALPHA_FLOOR
        ]
        color = (*(round(sum(c[i] for c in samples) / len(samples)) for i in range(3)), 255) if samples else (0, 0, 0, 255)
        for hx, hy in hole:
            px[hx, hy] = color
    return rgba


def _fit_to_cell(image):
    """Crop to content, scale to fit a padded cell, and center on transparent."""
    image = _drop_side_bleed(image)
    if (bbox := image.getbbox()) is None:
        return _blank()
    sprite = image.crop(bbox)
    scale = min((CELL_WIDTH - _CELL_PAD) / sprite.width, (CELL_HEIGHT - _CELL_PAD) / sprite.height, 1.0)
    if scale != 1.0:
        sprite = sprite.resize((max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))), _NEAREST)
    return _place(sprite, (CELL_WIDTH, CELL_HEIGHT), ((CELL_WIDTH - sprite.width) // 2, (CELL_HEIGHT - sprite.height) // 2))


def _drop_side_bleed(image):
    """Remove tiny separated left/right lobes (neighbour-pose slivers) before fitting.

    Component extraction may group a near sliver with the subject; the column profile still shows it as a low-mass lobe. Only those go (wide poses survive).
    """
    rgba = image.convert("RGBA")
    w, h = rgba.size
    runs = _content_runs(_column_profile(rgba))
    keep = [run for run, m in runs if m >= max(m for _run, m in runs) * _SIDE_LOBE_RATIO] if runs else []
    if len(runs) < 2 or len(keep) == len(runs):
        return rgba
    rgba = rgba.copy()
    prev = 0
    for left, right in [*keep, (w, w)]:  # (w, w) sentinel clears the trailing gap
        if left > prev:
            _clear_region(rgba, (prev, 0, left, h))
        prev = right
    return rgba


def _thin_groups(indices: list[int]) -> list[tuple[int, int]]:
    """``(start, end)`` for each run of consecutive *indices* at most 4 long."""
    runs = ([i for _, i in g] for _k, g in groupby(enumerate(indices), key=lambda p: p[1] - p[0]))
    return [(run[0], run[-1] + 1) for run in runs if len(run) <= 4]


def _erase_long_axis_lines(image):
    """Remove thin slot-spanning guide/floor/divider lines (they survive keying and bridge clean poses)."""
    rgba = image.convert("RGBA").copy()
    w, h = rgba.size
    alpha = rgba.getchannel("A")
    opaque = [[alpha.getpixel((x, y)) > _ALPHA_FLOOR for x in range(w)] for y in range(h)]
    for top, bottom in _thin_groups([y for y in range(h) if sum(opaque[y]) >= w * 0.85]):
        _clear_region(rgba, (0, top, w, bottom))
    for left, right in _thin_groups([x for x in range(w) if sum(row[x] for row in opaque) >= h * 0.85]):
        _clear_region(rgba, (left, 0, right, h))
    return rgba


def _component_boxes(image) -> list[tuple[tuple[int, int, int, int], int]]:
    """Connected opaque components as ``[(bbox, mass)]``."""
    rgba = image.convert("RGBA")
    if (bbox := rgba.getbbox()) is None:
        return []
    l0, t0, r0, b0 = bbox
    w, h = r0 - l0, b0 - t0
    alpha = rgba.getchannel("A").load()
    components = _unvisited_components(w, h, bytearray(w * h), lambda x, y: alpha[l0 + x, t0 + y] > _ALPHA_FLOOR)
    return [((l0 + min(xs), t0 + min(ys), l0 + max(xs) + 1, t0 + max(ys) + 1), len(xs)) for xs, ys in map(lambda p: zip(*p), components)]


def _isolate_slot_subject(image):
    """Keep the slot's real subject; drop detached effects/noise."""
    rgba = _erase_long_axis_lines(image)
    comps = _component_boxes(rgba)
    if not comps:
        return rgba
    main_box, main_mass = max(comps, key=lambda item: item[1])
    ml, _mt, mr, _mb = main_box
    mw = max(1, mr - ml)
    out = _blank(rgba.size)
    for box, mass in comps:
        # Keep attached-looking accessories (halos); drop sparkles/tears/noise.
        left, _top, right, _bottom = box
        overlap = max(0, min(right, mr) - max(left, ml))
        near_main = (ml - mw * 0.25) <= (left + right) / 2 <= (mr + mw * 0.25)
        if box == main_box or (mass >= max(24, main_mass * 0.035) and (overlap >= mw * 0.3 or near_main)):
            out.alpha_composite(rgba.crop(box), (box[0], box[1]))
    return out


def _has_margin(size: tuple[int, int], box: tuple[int, int, int, int], fx: float, fy: float) -> bool:
    """True when *box* leaves empty room on all four edges of an image of *size* (≥4px, ≤12/16px)."""
    w, h = size
    left, top, right, bottom = box
    min_x, min_y = max(4, min(12, round(w * fx))), max(4, min(16, round(h * fy)))
    return left >= min_x and top >= min_y and w - right >= min_x and h - bottom >= min_y


def _group_component_rows(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group component boxes into visual rows, then sort left→right."""
    if not boxes:
        return []
    cy = lambda b: (b[1] + b[3]) / 2  # noqa: E731
    row_tol = max(12, _median_box_size(boxes)[1] * 0.55)
    rows: list[list[tuple[int, int, int, int]]] = []
    centers: list[float] = []
    for box in sorted(boxes, key=cy):
        i = next((i for i, center in enumerate(centers) if abs(cy(box) - center) <= row_tol), None)
        if i is None:
            rows.append([box])
            centers.append(cy(box))
        else:
            rows[i].append(box)
            centers[i] = sum(cy(b) for b in rows[i]) / len(rows[i])
    ordered = [row for _center, row in sorted(zip(centers, rows, strict=False), key=lambda item: item[0])]
    for row in ordered:
        row.sort(key=lambda b: (b[0] + b[2]) / 2)
    return ordered


def _merge_related_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge disconnected parts of one subject (capes, tails, props) on the same row.

    Merges when vertical spans overlap and the horizontal gap is tiny relative to component size; never bridges the larger gaps between separate poses.
    """
    def related(a, b) -> bool:
        (al, at, ar, ab), (bl, bt, br, bb) = a, b
        v_overlap, min_h = max(0, min(ab, bb) - max(at, bt)), max(1, min(ab - at, bb - bt))
        gap, min_w = max(0, max(al, bl) - min(ar, br)), max(1, min(ar - al, br - bl))
        return v_overlap >= min_h * 0.45 and gap <= max(14, min_w * 0.22)

    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            used[i] = True
            for j, b in enumerate(boxes[i + 1 :], i + 1):
                if not used[j] and related(a, b):
                    a = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    used[j] = changed = True
            merged.append(a)
        boxes = merged
    return boxes


def _component_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Frames as connected subjects in reading order (robust to 2D grids); ``None`` when *frame_count* can't be met."""
    def attempt(source) -> list | None:
        subjects = _significant_subject_boxes(source, min_mass=64)
        if len(subjects) < frame_count:
            return None
        rows = _group_component_rows(subjects)
        ordered = [box for row in rows for box in row][:frame_count]  # rows hold every subject, so exactly frame_count
        if require_padding and not all(_has_margin(source.size, box, 0.01, 0.015) for box in ordered):
            return None
        multirow = len(rows) > 1
        frames = []
        for left, top, right, bottom in ordered:
            pad_x, pad_y = max(8, round((right - left) * 0.08)), max(8, round((bottom - top) * 0.08))
            # One-row strips keep full height so vertical motion survives; grids pad the box.
            cl, cr = (0, source.width) if frame_count == 1 else (max(0, left - pad_x), min(source.width, right + pad_x))
            ct, cb = (max(0, top - pad_y), min(source.height, bottom + pad_y)) if multirow else (0, source.height)
            # No second component filter here: capes/tails can be legitimate
            # disconnected lobes inside the chosen subject box.
            frames.append(_place(source.crop((left, top, right, bottom)), (cr - cl, cb - ct), (left - cl, top - ct)))
        return frames

    return attempt(strip) or attempt(_erase_long_axis_lines(strip))


def _sever_expected_gutters(strip, frame_count: int):
    """Cut narrow transparent bands at expected frame boundaries (shared shadows/1px bridges merge poses)."""
    if frame_count <= 1:
        return strip
    out = strip.copy()
    alpha = out.getchannel("A")  # zero alpha only; RGB is left untouched
    slot = out.width / frame_count
    half = max(3, min(18, round(slot * 0.06)))
    for x in (round(i * slot) for i in range(1, frame_count)):
        alpha.paste(0, (max(0, x - half), 0, min(out.width, x + half + 1), out.height))
    out.putalpha(alpha)
    return out


def _clean_slot(image):
    return _drop_side_bleed(_isolate_slot_subject(image))


def _slot_crops(strip, frame_count: int, *, require_padding: bool = False) -> list | None:
    """Slice *strip* into *frame_count* equal, independently cleaned columns (one shared coordinate frame)."""
    w, h = strip.size
    frames = [_clean_slot(strip.crop((round(i * w / frame_count), 0, round((i + 1) * w / frame_count), h))) for i in range(frame_count)]
    if require_padding and any((b := f.getbbox()) is None or not _has_margin(f.size, b, 0.025, 0.02) for f in frames):
        return None
    return frames


def _content_runs(profile: list[int], *, threshold: int = 2) -> list[tuple[tuple[int, int], int]]:
    """``[((left, right), mass)]`` column spans whose alpha exceeds *threshold* (candidate frames)."""
    runs: list[tuple[tuple[int, int], int]] = []
    start: int | None = None
    for x, v in enumerate(list(profile) + [0]):
        if v > threshold:
            if start is None:
                start = x
        elif start is not None:
            runs.append(((start, x), sum(profile[start:x])))
            start = None
    return runs


def _frame_x_ranges(strip, frame_count: int) -> list[tuple[int, int]] | None:
    """Per-frame ``(left, right)`` column ranges from the row's empty gutters.

    Extra spans merge across the smallest gaps (a detached halo sits closer to its body than to the next pose); fewer spans than frames → ``None``.
    """
    runs = _content_runs(_column_profile(strip))
    groups = [[l, r] for (l, r), m in runs if m >= max(m for _run, m in runs) * 0.02] if runs else []
    if len(groups) < frame_count:
        return None
    while len(groups) > frame_count:
        gi = min(range(len(groups) - 1), key=lambda i: groups[i + 1][0] - groups[i][1])
        groups[gi][1] = groups[gi + 1][1]
        del groups[gi + 1]
    return [tuple(g) for g in groups]


def _significant_subject_boxes(image, *, min_mass: int = 32) -> list[tuple[int, int, int, int]]:
    """Merged boxes of components carrying meaningful mass (≥12% of the largest)."""
    comps = _component_boxes(image)
    if not comps:
        return []
    max_mass = max(mass for _box, mass in comps)
    return _merge_related_boxes([box for box, mass in comps if mass >= max(min_mass, max_mass * 0.12)])


def _is_multi_pose_outlier(width: int, height: int, med_w: int, med_h: int) -> bool:
    """A frame several times wider than the median but not proportionally taller."""
    return width > max(med_w * 3.0, med_w + 96) and height <= med_h * 1.6


def _validate_extracted_frames(frames: list, frame_count: int) -> None:
    """Reject rows where one "frame" is really multiple poses (normalization would shrink the whole pet)."""
    if len(frames) != frame_count:
        raise ValueError(f"expected {frame_count} frames, got {len(frames)}")
    boxes = []
    for i, frame in enumerate(frames):
        if (bbox := frame.getbbox()) is None:
            raise ValueError(f"frame {i} is empty")
        if len(_significant_subject_boxes(frame)) >= 3:
            raise ValueError(f"frame {i} contains multiple separated subjects")
        boxes.append(bbox)
    if frame_count <= 1:
        return
    med_w, med_h = _median_box_size(boxes)
    for i, (left, top, right, bottom) in enumerate(boxes):
        if _is_multi_pose_outlier(right - left, bottom - top, med_w, med_h):
            raise ValueError(f"frame {i} is a multi-pose width outlier")


def extract_strip_frames(
    strip, frame_count: int, *, chroma_key: tuple[int, int, int] | None = None, method: str = "auto", fit: bool = True
) -> list:
    """Turn one generated row strip into *frame_count* frames.

    Keys out the background, then isolates padded subjects (components, then equal slots). When that fails ``components``
    raises while ``auto`` salvages leniently. *fit* centers each frame into a cell; hatching passes ``fit=False`` so
    :func:`normalize_cells` can register the whole pet with one shared scale.
    """
    strip = remove_background(_load_rgba(strip), chroma_key=chroma_key)
    frames = _component_crops(strip, frame_count, require_padding=True) or _slot_crops(strip, frame_count, require_padding=True)
    if frames is None:
        if method == "components":
            raise ValueError(f"could not segment {frame_count} padded sprites from strip")
        frames = _component_crops(strip, frame_count, require_padding=False)
    if frames is None:
        frames = _salvage_frames(strip, frame_count)
    _validate_extracted_frames(frames, frame_count)
    return [_fit_to_cell(f) for f in frames] if fit else frames


def _salvage_frames(strip, frame_count: int) -> list:
    """Lenient last resort: gutter ranges (severing expected gutters if needed), else raw slots."""
    source, ranges = strip, _frame_x_ranges(strip, frame_count)
    if ranges is None:
        source = _sever_expected_gutters(strip, frame_count)
        ranges = _frame_x_ranges(source, frame_count)
    if ranges is None:
        return _slot_crops(source, frame_count, require_padding=False) or []
    w, h = source.size
    pad = max(2, min(16, round((w / max(1, frame_count)) * 0.04)))
    return [_clean_slot(source.crop((max(0, left - pad), 0, min(w, right + pad), h))) for left, right in ranges]


def _column_profile(image) -> list[int]:
    """Per-column alpha mass — collapse to a 1px-tall strip (fast in C)."""
    return list(image.getchannel("A").resize((image.width, 1), Image.BILINEAR).getdata())


def _best_shift(ref: list[int], prof: list[int], window: int) -> int:
    """Integer dx that best aligns *prof* onto *ref* (1-D cross-correlation; the body dominates, limbs barely move it)."""
    n = len(ref)
    score = lambda d: sum(ref[x] * prof[x - d] for x in range(max(0, d), min(n, n + d)))  # noqa: E731
    return max(range(-window, window + 1), key=score)  # ties → smallest dx


def normalize_cells(frames_by_state: dict[str, list], *, pad: int = _NORMALIZE_PAD) -> dict[str, list]:
    """Register every frame into a 192x208 cell — the deterministic anti-jitter math.

    Per-frame crop→scale→center jitters. Instead: align each frame's column profile
    to the state's median (locks the body), union-crop through one shared window,
    then scale all states by one global factor keyed to median pose height.
    """
    out: dict[str, list] = {}
    prepared: dict[str, tuple[list, tuple[int, int, int, int], int]] = {}
    target_w, target_h = CELL_WIDTH - pad, CELL_HEIGHT - pad
    for state, frames in frames_by_state.items():
        rgba = [f.convert("RGBA") for f in frames]
        if not any(f.getbbox() for f in rgba):
            out[state] = [_blank() for _ in frames]
            continue
        # Pad every frame to a common canvas so column profiles are comparable.
        w0, h0 = max(f.width for f in rgba), max(f.height for f in rgba)
        canvas = [f if f.size == (w0, h0) else _place(f, (w0, h0)) for f in rgba]
        profiles = [_column_profile(f) for f in canvas]
        ref = [_median(p[x] for p in profiles) for x in range(w0)]
        window = max(8, w0 // 5)
        aligned = [_place(f, (w0 + 2 * window, h0), (window + _best_shift(ref, prof, window), 0)) for f, prof in zip(canvas, profiles)]
        boxes = [b for b in (a.getbbox() for a in aligned) if b]
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        prepared[state] = (aligned, union, _median(b[3] - b[1] for b in boxes))
    if not prepared:
        return out

    # K is the one global cap keeping the tallest/widest motion envelope (a
    # jump's lift) inside the cell; a still row's union ≈ pose so it fills fully.
    K = target_h
    for _aligned, (left, top, right, bottom), pose_h in prepared.values():
        K = min(K, target_h * pose_h / max(1, bottom - top), target_w * pose_h / max(1, right - left))
    for state, (aligned, box, pose_h) in prepared.items():
        scale = K / max(1, pose_h)
        sw, sh = max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))
        offset = round((CELL_WIDTH - sw) / 2), round((CELL_HEIGHT - pad // 2) - sh)
        crops = [a.crop(box) for a in aligned]
        out[state] = [_place(c if c.size == (sw, sh) else c.resize((sw, sh), _NEAREST), (CELL_WIDTH, CELL_HEIGHT), offset) for c in crops]
    return out


def single_frame(image, *, fit: bool = True):
    """One frame from a standalone image (idle fallback); ``fit=False`` yields the raw keyed sprite for :func:`normalize_cells`."""
    keyed = remove_background(_load_rgba(image))
    return _fit_to_cell(keyed) if fit else _drop_side_bleed(keyed)


def _clear_transparent_rgb(image):
    """Zero the RGB of fully-transparent pixels (no colored-halo residue)."""
    rgba = image.convert("RGBA")
    data = bytearray(rgba.tobytes())
    for i in range(3, len(data), 4):
        if data[i] == 0:
            data[i - 3 : i] = b"\0\0\0"
    return Image.frombytes("RGBA", rgba.size, bytes(data))


def mirror_frames(frames: list) -> list:
    """Flip each frame horizontally (``running-left`` from ``running-right``; per-frame, NOT a strip reverse)."""
    return [frame.convert("RGBA").transpose(Image.Transpose.FLIP_LEFT_RIGHT) for frame in frames]


def compose_atlas(frames_by_state: dict[str, list]):
    """Pack per-state frame lists into the atlas; short states leave trailing cells transparent."""
    atlas = _blank((ATLAS_WIDTH, ATLAS_HEIGHT))
    for state, row, count in ROW_SPECS:
        for col, frame in enumerate((frames_by_state.get(state) or [])[:count]):
            cell = cell if (cell := frame.convert("RGBA")).size == (CELL_WIDTH, CELL_HEIGHT) else _fit_to_cell(cell)
            atlas.alpha_composite(cell, (col * CELL_WIDTH, row * CELL_HEIGHT))
    return _clear_transparent_rgb(atlas)


def validate_atlas(atlas) -> dict:
    """Geometry/occupancy/transparency checks → ``{ok, width, height, errors, warnings, filled_states}``."""
    atlas = _load_rgba(atlas)
    if atlas.size == (ATLAS_WIDTH, ATLAS_HEIGHT):
        errors, warnings, filled = _check_atlas_cells(atlas)
    else:
        errors, warnings, filled = [f"expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}, got {atlas.width}x{atlas.height}"], [], []
    return {"ok": not errors, "width": atlas.width, "height": atlas.height, "errors": errors, "warnings": warnings, "filled_states": filled}


def _check_atlas_cells(atlas) -> tuple[list[str], list[str], list[str]]:
    """Occupancy/collapse/residue checks for a correctly-sized atlas → ``(errors, warnings, filled_states)``."""
    errors, warnings, filled_states = [], [], []
    cell_boxes_by_state: dict[str, list[tuple[int, int, int, int]]] = {}
    for state, row, count in ROW_SPECS:
        cells = [atlas.crop((c * CELL_WIDTH, row * CELL_HEIGHT, (c + 1) * CELL_WIDTH, (row + 1) * CELL_HEIGHT)) for c in range(count)]
        if any(sum(cell.getchannel("A").histogram()[1:]) for cell in cells):
            filled_states.append(state)
            cell_boxes_by_state[state] = [bbox for cell in cells if (bbox := cell.getbbox()) is not None]
        else:
            warnings.append(f"state '{state}' has no frames")

    if not filled_states:
        errors.append("atlas is empty — no state produced any frames")
    # A valid pet must occupy the cell: one bad row can poison global
    # normalization and shrink every state while still passing "non-empty".
    all_boxes = [b for boxes in cell_boxes_by_state.values() for b in boxes]
    global_med_w, global_med_h = _median_box_size(all_boxes) if all_boxes else (0, 0)
    if all_boxes and global_med_h < max(56, round(CELL_HEIGHT * 0.28)):
        errors.append(f"atlas sprites are too small after normalization (median frame height {global_med_h}px)")
    for state, boxes in cell_boxes_by_state.items():
        if len(boxes) <= 1:
            continue
        med_w, med_h = _median_box_size(boxes)
        if _is_multi_pose_outlier(max(r - l for l, _t, r, _b in boxes), max(b - t for _l, t, _r, b in boxes), med_w, med_h):
            errors.append(f"state '{state}' contains a multi-pose frame outlier")
        # Per-state collapse guard: one malformed row must not pass on the
        # strength of the healthy ones.
        collapsed = med_w < max(32, round(global_med_w * 0.42)) or med_h < max(40, round(global_med_h * 0.50))
        if (global_med_w and global_med_h) and collapsed:
            errors.append(f"state '{state}' appears collapsed (median {med_w}x{med_h}px, global median {global_med_w}x{global_med_h}px)")
    data = atlas.tobytes()
    residue = sum(1 for i in range(3, len(data), 4) if data[i] == 0 and any(data[i - 3 : i]))
    if residue:
        errors.append(f"{residue} transparent pixels retain RGB residue")
    return errors, warnings, filled_states


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import io  # noqa: F401,E402
import io  # noqa: F401,E402

COLUMNS = max(count for _, _, count in ROW_SPECS)

FRAME_COUNTS: dict[str, int] = {state: count for state, _, count in ROW_SPECS}

ROWS = len(ROW_SPECS)

def atlas_to_webp_bytes(atlas) -> bytes:
    """Encode an atlas image to lossless WebP bytes (the on-disk pet format)."""
    buf = io.BytesIO()
    atlas.save(buf, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return buf.getvalue()
# ---- END PLUGIN-COMPAT ----
