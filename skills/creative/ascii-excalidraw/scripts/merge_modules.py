#!/usr/bin/env python3
"""
merge_modules.py - Merge multiple Excalidraw module JSON files into a single .excalidraw file.

Usage:
    python merge_modules.py -o diagram.excalidraw module1.json module2.json [module3.json ...]

Each module file should contain a JSON array of Excalidraw element objects.
The script merges them into the standard .excalidraw envelope.

This ensures:
- All element IDs are unique across modules (renames collisions)
- Drawing order is preserved (module 1 elements first, then module 2, etc.)
- Container bindings (containerId, boundElements) are updated if IDs changed
- Arrow bindings (startBinding.elementId, endBinding.elementId) are updated
- frameId and groupIds references are updated
"""

import argparse
import json
import sys
import uuid


def resolve_id(id_map, element_id):
    """Resolve an element ID through the ID mapping."""
    return id_map.get(element_id, element_id)


def update_bindings(element, id_map):
    """Update all binding references in an element using the ID map."""
    if "boundElements" in element and element["boundElements"]:
        for be in element["boundElements"]:
            if "id" in be:
                be["id"] = resolve_id(id_map, be["id"])

    if "containerId" in element and element["containerId"]:
        element["containerId"] = resolve_id(id_map, element["containerId"])

    if "startBinding" in element and element["startBinding"]:
        if "elementId" in element["startBinding"]:
            element["startBinding"]["elementId"] = resolve_id(
                id_map, element["startBinding"]["elementId"]
            )

    if "endBinding" in element and element["endBinding"]:
        if "elementId" in element["endBinding"]:
            element["endBinding"]["elementId"] = resolve_id(
                id_map, element["endBinding"]["elementId"]
            )

    if "groupIds" in element and element["groupIds"]:
        element["groupIds"] = [
            resolve_id(id_map, gid) for gid in element["groupIds"]
        ]

    if "frameId" in element and element["frameId"]:
        element["frameId"] = resolve_id(id_map, element["frameId"])


def merge_modules(module_files, output_path):
    """Merge multiple module JSON files into a single .excalidraw file."""
    all_elements = []
    id_map = {}
    seen_ids = set()

    for module_path in module_files:
        try:
            with open(module_path, "r") as f:
                elements = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {module_path}: {e}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(elements, list):
            print(
                f"Error: {module_path} should contain a JSON array of elements, "
                f"got {type(elements).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

        for element in elements:
            old_id = element.get("id")
            if old_id is None:
                new_id = f"elem_{uuid.uuid4().hex[:8]}"
                element["id"] = new_id
            elif old_id in seen_ids:
                new_id = f"{old_id}_{uuid.uuid4().hex[:8]}"
                id_map[old_id] = new_id
                element["id"] = new_id
                print(
                    f"Warning: Duplicate ID '{old_id}' in {module_path}, "
                    f"renamed to '{new_id}'",
                    file=sys.stderr,
                )
            else:
                new_id = old_id

            seen_ids.add(new_id)
            update_bindings(element, id_map)
            all_elements.append(element)

    excalidraw_file = {
        "type": "excalidraw",
        "version": 2,
        "source": "ascii-excalidraw",
        "elements": all_elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "currentItemStrokeColor": "#1e1e1e",
            "currentItemBackgroundColor": "transparent",
            "currentItemFillStyle": "hachure",
            "currentItemStrokeWidth": 2,
            "currentItemRoughness": 1,
            "currentItemFontFamily": 5,
        },
        "files": {},
    }

    try:
        with open(output_path, "w") as f:
            json.dump(excalidraw_file, f, indent=2)
        print(f"Merged {len(module_files)} modules into {output_path}")
        print(f"  Total elements: {len(all_elements)}")
    except IOError as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Excalidraw module JSON files into a single .excalidraw file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output .excalidraw file path"
    )
    parser.add_argument(
        "modules", nargs="+", help="Module JSON files (arrays of elements)"
    )
    args = parser.parse_args()

    merge_modules(args.modules, args.output)


if __name__ == "__main__":
    main()
