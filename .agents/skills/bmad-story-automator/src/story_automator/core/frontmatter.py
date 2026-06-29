from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .utils import parse_string_list_literal, read_text, trim_lines, unquote_scalar, write_atomic


def extract_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return ""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return ""
    return parts[1].lstrip("\n")


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1].lstrip("\n"), parts[2].lstrip("\n")


def parse_simple_frontmatter(text: str) -> dict[str, Any]:
    front = extract_frontmatter(text)
    if not front:
        return {}
    fields: dict[str, Any] = {}
    current_key = ""
    for line in trim_lines(front):
        if line.strip().startswith("#"):
            continue
        if re.match(r"^\S[^:]*:", line):
            key, raw = line.split(":", 1)
            key = key.strip()
            raw = raw.strip()
            if raw == "":
                fields[key] = []
                current_key = key
                continue
            parsed_list = parse_string_list_literal(raw)
            if parsed_list is not None:
                fields[key] = parsed_list
            else:
                fields[key] = unquote_scalar(raw)
            current_key = ""
            continue
        if current_key and line.strip().startswith("-"):
            fields.setdefault(current_key, [])
            fields[current_key].append(unquote_scalar(line.strip()[1:].strip()))
    return fields


def parse_frontmatter(text: str) -> dict[str, Any]:
    return parse_simple_frontmatter(text)


def find_frontmatter_value(path: str | Path, key: str) -> str:
    fields = parse_simple_frontmatter(read_text(path))
    value = fields.get(key, "")
    if isinstance(value, list):
        return ""
    return str(value)


def find_frontmatter_value_case(path: str | Path, key: str) -> str:
    front = extract_frontmatter(read_text(path))
    for line in trim_lines(front):
        if ":" not in line:
            continue
        left, raw = line.split(":", 1)
        if left.strip().lower() == key.lower():
            return unquote_scalar(raw.strip())
    return ""


def extract_last_action(path: str | Path) -> str:
    lines = trim_lines(read_text(path))
    for index, line in enumerate(lines):
        if line.startswith("## Action Log") and index + 2 < len(lines):
            return lines[index + 2].strip().lstrip("* ").strip()
    return ""


def read_story_range_from_state(path: str | Path) -> list[str]:
    text = read_text(path)
    for block in (extract_frontmatter(text), text):
        if not block.strip():
            continue
        lines = trim_lines(block)
        in_range = False
        story_range: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("storyRange:"):
                raw = stripped.split(":", 1)[1].strip()
                parsed = parse_string_list_literal(raw)
                if parsed is not None:
                    return parsed
                in_range = True
                continue
            if in_range and stripped.startswith("-"):
                story_range.append(unquote_scalar(stripped[1:].strip()))
                continue
            if in_range and re.match(r"^\S[^:]*:", line):
                break
        if story_range:
            return story_range
    return []


def update_simple_frontmatter(path: str | Path, updates: dict[str, str]) -> list[str]:
    path = Path(path)
    lines = trim_lines(read_text(path))
    updated: list[str] = []
    for idx, line in enumerate(lines):
        for key, value in updates.items():
            if line.startswith(f"{key}:"):
                lines[idx] = f"{key}: {value}"
                updated.append(key)
    if updated:
        write_atomic(path, "\n".join(lines) + "\n")
    return updated


def extract_json_block(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return ""


def dump_json_pretty(payload: Any) -> str:
    return json.dumps(payload, indent=2) + "\n"
