#!/usr/bin/env python3
"""Validate JSON Canvas (.canvas) files against the JSON Canvas 1.0 spec.

Usage:
    python validate_canvas.py FILE [FILE ...]

Exits 0 when every file is valid, 1 otherwise. Errors are printed one per
line, prefixed with the file path. Stdlib only.

Spec: https://jsoncanvas.org/spec/1.0/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NODE_TYPES = {"text", "file", "link", "group"}
SIDES = {"top", "right", "bottom", "left"}
ENDS = {"none", "arrow"}
BACKGROUND_STYLES = {"cover", "ratio", "repeat"}
COLOR_PRESETS = {"1", "2", "3", "4", "5", "6"}
# 3-, 4-, 6-, and 8-digit hex forms (#RGB, #RGBA, #RRGGBB, #RRGGBBAA); the
# spec says only "color in hex format", so accept all common CSS hex forms.
HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

# type -> attribute that must be present on nodes of that type
REQUIRED_BY_TYPE = {"text": "text", "file": "file", "link": "url"}
GEOMETRY_FIELDS = ("x", "y", "width", "height")


def _check_color(value: object, where: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not (
        value in COLOR_PRESETS or HEX_COLOR_RE.match(value)
    ):
        errors.append(
            f'{where}: color must be a preset "1"-"6" or hex string, got {value!r}'
        )


def _check_node(node: object, index: int, seen_ids: set[str], errors: list[str]) -> None:
    where = f"nodes[{index}]"
    if not isinstance(node, dict):
        errors.append(f"{where}: node must be an object")
        return

    node_id = node.get("id")
    if not isinstance(node_id, str) or not node_id:
        errors.append(f"{where}: missing or non-string required field 'id'")
    elif node_id in seen_ids:
        errors.append(f"{where}: duplicate id {node_id!r}")
    else:
        seen_ids.add(node_id)

    node_type = node.get("type")
    # isinstance guard first: non-string values (lists, dicts) are unhashable
    # and would crash the set-membership test.
    if not isinstance(node_type, str) or node_type not in NODE_TYPES:
        errors.append(
            f"{where}: 'type' must be one of {sorted(NODE_TYPES)}, got {node_type!r}"
        )
    else:
        required = REQUIRED_BY_TYPE.get(node_type)
        if required is not None and not isinstance(node.get(required), str):
            errors.append(
                f"{where}: {node_type} node requires string field {required!r}"
            )
        if node_type == "group":
            style = node.get("backgroundStyle")
            if style is not None and (
                not isinstance(style, str) or style not in BACKGROUND_STYLES
            ):
                errors.append(
                    f"{where}: 'backgroundStyle' must be one of "
                    f"{sorted(BACKGROUND_STYLES)}, got {style!r}"
                )
            for str_field in ("label", "background"):
                value = node.get(str_field)
                if value is not None and not isinstance(value, str):
                    errors.append(f"{where}: {str_field!r} must be a string")

    for field in GEOMETRY_FIELDS:
        value = node.get(field)
        # The spec calls for integers; bools are ints in Python, reject them.
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{where}: missing or non-integer required field {field!r}")

    if "color" in node:
        _check_color(node["color"], where, errors)

    subpath = node.get("subpath")
    if subpath is not None and (
        not isinstance(subpath, str) or not subpath.startswith("#")
    ):
        errors.append(f"{where}: 'subpath' must be a string starting with '#'")


def _check_edge(
    edge: object,
    index: int,
    node_ids: set[str],
    seen_ids: set[str],
    errors: list[str],
) -> None:
    where = f"edges[{index}]"
    if not isinstance(edge, dict):
        errors.append(f"{where}: edge must be an object")
        return

    edge_id = edge.get("id")
    if not isinstance(edge_id, str) or not edge_id:
        errors.append(f"{where}: missing or non-string required field 'id'")
    elif edge_id in seen_ids:
        errors.append(f"{where}: duplicate id {edge_id!r}")
    else:
        seen_ids.add(edge_id)

    for endpoint in ("fromNode", "toNode"):
        value = edge.get(endpoint)
        if not isinstance(value, str) or not value:
            errors.append(f"{where}: missing or non-string required field {endpoint!r}")
        elif value not in node_ids:
            errors.append(
                f"{where}: {endpoint} {value!r} does not reference an existing node"
            )

    for field, allowed in (
        ("fromSide", SIDES),
        ("toSide", SIDES),
        ("fromEnd", ENDS),
        ("toEnd", ENDS),
    ):
        value = edge.get(field)
        if value is not None and (
            not isinstance(value, str) or value not in allowed
        ):
            errors.append(
                f"{where}: {field!r} must be one of {sorted(allowed)}, got {value!r}"
            )

    label = edge.get("label")
    if label is not None and not isinstance(label, str):
        errors.append(f"{where}: 'label' must be a string")

    if "color" in edge:
        _check_color(edge["color"], where, errors)


def validate_canvas_text(text: str) -> list[str]:
    """Validate raw .canvas file content. Returns a list of error strings."""
    errors: list[str] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    if not isinstance(data, dict):
        return ["top level must be a JSON object"]

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list):
        errors.append("'nodes' must be an array")
        nodes = []
    if not isinstance(edges, list):
        errors.append("'edges' must be an array")
        edges = []

    seen_ids: set[str] = set()
    for index, node in enumerate(nodes):
        _check_node(node, index, seen_ids, errors)

    node_ids = set(seen_ids)
    for index, edge in enumerate(edges):
        _check_edge(edge, index, node_ids, seen_ids, errors)

    return errors


def validate_canvas_file(path: Path) -> list[str]:
    """Validate one .canvas file. Returns a list of error strings."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read file: {exc}"]
    return validate_canvas_text(text)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    failed = False
    for arg in argv:
        path = Path(arg)
        errors = validate_canvas_file(path)
        if errors:
            failed = True
            for error in errors:
                print(f"{path}: {error}")
        else:
            print(f"{path}: OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
