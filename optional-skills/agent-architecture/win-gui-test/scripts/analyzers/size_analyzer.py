"""Size & alignment analysis — measure button/input/viewport consistency."""

import logging
from typing import Any

log = logging.getLogger("win-gui-test.analyzers.size")


def collect_sizes(elements: list[dict]) -> dict[str, Any]:
    """Extract width/height statistics from a list of element dicts.

    Expected element shape::

        {"name": str, "type": str, "rect": "(left,top) WxH"}
    """
    sizes: dict[str, list[int]] = {"width": [], "height": []}
    for el in elements:
        rect_str = el.get("rect", "")
        if "x" not in rect_str:
            continue
        try:
            _, dims = rect_str.split(") ")
            w_str, h_str = dims.split("x")
            sizes["width"].append(int(w_str))
            sizes["height"].append(int(h_str))
        except (ValueError, IndexError):
            continue

    result: dict[str, Any] = {"count": len(sizes["width"])}
    for dim in ("width", "height"):
        vals = sizes[dim]
        if not vals:
            result[dim] = {"min": 0, "max": 0, "avg": 0, "all_same": True}
            continue
        result[dim] = {
            "min": min(vals),
            "max": max(vals),
            "avg": round(sum(vals) / len(vals)),
            "all_same": len(set(vals)) == 1,
        }
    return result


def detect_inconsistencies(elements: list[dict]) -> list[dict]:
    """Return entries where an element's height differs from the group median."""
    heights = []
    for el in elements:
        rect_str = el.get("rect", "")
        if "x" not in rect_str:
            continue
        try:
            _, dims = rect_str.split(") ")
            _, h_str = dims.split("x")
            heights.append((el, int(h_str)))
        except (ValueError, IndexError):
            continue

    if not heights:
        return []

    sorted_h = sorted(h for _, h in heights)
    median = sorted_h[len(sorted_h) // 2]
    tolerance = max(4, median * 0.1)

    outliers = []
    for el, h in heights:
        if abs(h - median) > tolerance:
            outliers.append(
                {"name": el.get("name", ""), "type": el.get("type", ""), "height": h, "median": median}
            )
    return outliers
