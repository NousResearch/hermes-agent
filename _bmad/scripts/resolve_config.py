#!/usr/bin/env python3
"""
Resolve BMad's central config using four-layer TOML merge.

Reads from four layers (highest priority last):
  1. {project-root}/_bmad/config.toml              (installer-owned team)
  2. {project-root}/_bmad/config.user.toml         (installer-owned user)
  3. {project-root}/_bmad/custom/config.toml       (human-authored team, committed)
  4. {project-root}/_bmad/custom/config.user.toml  (human-authored user, gitignored)

Outputs merged JSON to stdout. Errors go to stderr.

Uses only the Python stdlib (`tomllib`) — no third-party dependencies.
BMad is standardizing on `uv run` to invoke scripts (uv provisions a suitable
interpreter for you); a plain `python3` on PATH still works during the
transition. Either runner needs Python 3.11+ for `tomllib`.

  uv run resolve_config.py --project-root /abs/path/to/project
  uv run resolve_config.py --project-root ... --key core
  uv run resolve_config.py --project-root ... --key agents

Merge rules (same as resolve_customization.py):
  - Scalars: override wins
  - Tables: deep merge
  - Arrays of tables where every item shares `code` or `id`: merge by that key
  - All other arrays: append
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    sys.stderr.write(
        "error: Python 3.11+ is required (stdlib `tomllib` not found).\n"
    )
    sys.exit(3)


_MISSING = object()
_KEYED_MERGE_FIELDS = ("code", "id")


def load_toml(file_path: Path, required: bool = False) -> dict:
    if not file_path.exists():
        if required:
            sys.stderr.write(f"error: required config file not found: {file_path}\n")
            sys.exit(1)
        return {}
    try:
        with file_path.open("rb") as f:
            parsed = tomllib.load(f)
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except tomllib.TOMLDecodeError as error:
        level = "error" if required else "warning"
        sys.stderr.write(f"{level}: failed to parse {file_path}: {error}\n")
        if required:
            sys.exit(1)
        return {}
    except OSError as error:
        level = "error" if required else "warning"
        sys.stderr.write(f"{level}: failed to read {file_path}: {error}\n")
        if required:
            sys.exit(1)
        return {}


def _detect_keyed_merge_field(items):
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    for candidate in _KEYED_MERGE_FIELDS:
        if all(item.get(candidate) is not None for item in items):
            return candidate
    return None


def _merge_by_key(base, override, key_name):
    result = []
    index_by_key = {}
    for item in base:
        if not isinstance(item, dict):
            continue
        if item.get(key_name) is not None:
            index_by_key[item[key_name]] = len(result)
        result.append(dict(item))
    for item in override:
        if not isinstance(item, dict):
            result.append(item)
            continue
        key = item.get(key_name)
        if key is not None and key in index_by_key:
            result[index_by_key[key]] = dict(item)
        else:
            if key is not None:
                index_by_key[key] = len(result)
            result.append(dict(item))
    return result


def _merge_arrays(base, override):
    base_arr = base if isinstance(base, list) else []
    override_arr = override if isinstance(override, list) else []
    keyed_field = _detect_keyed_merge_field(base_arr + override_arr)
    if keyed_field:
        return _merge_by_key(base_arr, override_arr, keyed_field)
    return base_arr + override_arr


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, over_val in override.items():
            if key in result:
                result[key] = deep_merge(result[key], over_val)
            else:
                result[key] = over_val
        return result
    if isinstance(base, list) and isinstance(override, list):
        return _merge_arrays(base, override)
    return override


def extract_key(data, dotted_key: str):
    parts = dotted_key.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def main():
    parser = argparse.ArgumentParser(
        description="Resolve BMad central config using four-layer TOML merge.",
    )
    parser.add_argument(
        "--project-root", "-p", required=True,
        help="Absolute path to the project root (contains _bmad/)",
    )
    parser.add_argument(
        "--key", "-k", action="append", default=[],
        help="Dotted field path to resolve (repeatable). Omit for full dump.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    bmad_dir = project_root / "_bmad"

    base_team = load_toml(bmad_dir / "config.toml", required=True)
    base_user = load_toml(bmad_dir / "config.user.toml")
    custom_team = load_toml(bmad_dir / "custom" / "config.toml")
    custom_user = load_toml(bmad_dir / "custom" / "config.user.toml")

    merged = deep_merge(base_team, base_user)
    merged = deep_merge(merged, custom_team)
    merged = deep_merge(merged, custom_user)

    if args.key:
        output = {}
        for key in args.key:
            value = extract_key(merged, key)
            if value is not _MISSING:
                output[key] = value
    else:
        output = merged

    sys.stdout.write(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
