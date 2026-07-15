#!/usr/bin/env python3
"""Live evidence evaluator for the 13-step Migrate Web flow.

The module derives every result from files under ``project_root``.  It never
creates or trusts a mutable state/cache file.  YAML loading is stdlib-only at
its core, with optional PyYAML and a fail-closed mini-YAML fallback.

Python 3.9+
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

_FORCE_MINI_YAML = False
_MENU_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised on machines without PyYAML
    _yaml = None


class ConfigError(ValueError):
    """Raised when rules, a menu slug, or a requested step is invalid."""


class MiniYamlError(ValueError):
    """Raised when fallback YAML cannot represent the input unambiguously."""


def _split_flow_items(inner: str) -> List[str]:
    parts: List[str] = []
    buf = ""
    quote: Optional[str] = None
    depth = 0
    for char in inner:
        if quote:
            buf += char
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            buf += char
        elif char in "[{":
            depth += 1
            buf += char
        elif char in "]}":
            depth = max(0, depth - 1)
            buf += char
        elif char == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += char
    if quote or depth:
        raise MiniYamlError("unterminated inline YAML value")
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(item) for item in _split_flow_items(inner)]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        result: Dict[str, Any] = {}
        if not inner:
            return result
        for item in _split_flow_items(inner):
            if ":" not in item:
                raise MiniYamlError(f"invalid inline mapping item: {item!r}")
            key, _, raw = item.partition(":")
            result[key.strip()] = _scalar(raw)
        return result
    lowered = value.lower()
    if lowered in ("null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _mini_yaml(text: str) -> Any:
    """Parse the small YAML subset used by flow-rules.yaml, failing closed."""
    lines = [
        line
        for line in (raw.rstrip() for raw in text.splitlines())
        if line.strip() and not line.strip().startswith("#")
    ]
    position = 0

    def indent_of(line: str) -> int:
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise MiniYamlError("tabs are not allowed for YAML indentation")
        return len(line) - len(line.lstrip(" "))

    def is_list_item(value: str) -> bool:
        return value == "-" or value.startswith("- ")

    def parse_mapping_tail(target: Dict[str, Any], parent_indent: int) -> None:
        nonlocal position
        while position < len(lines):
            line = lines[position]
            indent = indent_of(line)
            if indent <= parent_indent:
                break
            stripped = line.strip()
            if is_list_item(stripped) or ":" not in stripped:
                raise MiniYamlError(f"expected key: value near {stripped!r}")
            key, _, raw = stripped.partition(":")
            position += 1
            if raw.strip():
                target[key.strip()] = _scalar(raw)
            elif position < len(lines) and indent_of(lines[position]) > indent:
                target[key.strip()] = parse_block(indent_of(lines[position]))
            else:
                target[key.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal position
        result: Any = None
        while position < len(lines):
            line = lines[position]
            current = indent_of(line)
            if current < indent:
                break
            if current > indent:
                raise MiniYamlError(f"orphan indentation near {line.strip()!r}")
            stripped = line.strip()
            if is_list_item(stripped):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError("mixed mapping and list")
                item = "" if stripped == "-" else stripped[2:].strip()
                position += 1
                if not item:
                    if position >= len(lines) or indent_of(lines[position]) <= indent:
                        result.append(None)
                    else:
                        result.append(parse_block(indent_of(lines[position])))
                elif ":" in item and item[0] not in "\"'":
                    key, _, raw = item.partition(":")
                    mapping = {key.strip(): _scalar(raw) if raw.strip() else None}
                    parse_mapping_tail(mapping, indent)
                    result.append(mapping)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict) or ":" not in stripped:
                    raise MiniYamlError(f"expected key: value near {stripped!r}")
                key, _, raw = stripped.partition(":")
                position += 1
                if raw.strip():
                    result[key.strip()] = _scalar(raw)
                elif position < len(lines) and indent_of(lines[position]) > current:
                    result[key.strip()] = parse_block(indent_of(lines[position]))
                else:
                    result[key.strip()] = None
        return {} if result is None else result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML with PyYAML when available, otherwise the strict fallback."""
    try:
        if not (force_mini or _FORCE_MINI_YAML or _yaml is None):
            data = _yaml.safe_load(text)
            return {} if data is None else data
        return _mini_yaml(text)
    except Exception as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid YAML: {exc}") from exc


def validate_menu(menu: str) -> str:
    """Return a safe menu slug or raise ConfigError."""
    if not isinstance(menu, str) or not _MENU_RE.fullmatch(menu):
        raise ConfigError(
            "invalid menu slug: use 1-64 lowercase letters, digits, '_' or '-', "
            "starting with a letter or digit"
        )
    return menu


def _validate_rule_path(raw_path: Any, context: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfigError(f"{context}: path must be a non-empty string")
    normalized = raw_path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(raw_path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ConfigError(f"{context}: absolute path is forbidden: {raw_path}")
    if ".." in posix.parts:
        raise ConfigError(f"{context}: '..' path segment is forbidden: {raw_path}")
    unknown = re.findall(r"\{([^{}]+)\}", raw_path)
    if any(name != "menu" for name in unknown) or raw_path.count("{") != raw_path.count("}"):
        raise ConfigError(f"{context}: only the {{menu}} placeholder is allowed")
    return normalized


def _string_list(value: Any, context: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{context} must be a list of non-empty strings")
    return list(value)


def validate_rules(rules: Any) -> Dict[str, Any]:
    """Validate and normalize a complete rules document."""
    if not isinstance(rules, dict):
        raise ConfigError("rules must be a mapping")
    steps = rules.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ConfigError("rules must contain a non-empty steps list")

    seen = set()
    normalized_steps: List[Dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        context = f"steps[{step_index}]"
        if not isinstance(step, dict):
            raise ConfigError(f"{context} must be a mapping")
        step_id = step.get("id")
        title = step.get("title")
        outputs = step.get("outputs")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ConfigError(f"{context}: id must be a non-empty string")
        if step_id in seen:
            raise ConfigError(f"duplicate step id: {step_id}")
        seen.add(step_id)
        if not isinstance(title, str) or not title.strip():
            raise ConfigError(f"step {step_id}: title must be a non-empty string")
        if not isinstance(outputs, list) or not outputs:
            raise ConfigError(f"step {step_id}: outputs must be a non-empty list")

        normalized_outputs: List[Dict[str, Any]] = []
        for output_index, output in enumerate(outputs):
            output_context = f"step {step_id} output[{output_index}]"
            if not isinstance(output, dict):
                raise ConfigError(f"{output_context} must be a mapping")
            path = _validate_rule_path(output.get("path"), output_context)
            min_bytes = output.get("min_bytes", 0)
            if isinstance(min_bytes, bool) or not isinstance(min_bytes, int) or min_bytes < 0:
                raise ConfigError(f"{output_context}: min_bytes must be a non-negative integer")
            normalized_outputs.append(
                {
                    "path": path,
                    "min_bytes": min_bytes,
                    "must_contain": _string_list(
                        output.get("must_contain"), f"{output_context}.must_contain"
                    ),
                    "must_contain_any": _string_list(
                        output.get("must_contain_any"),
                        f"{output_context}.must_contain_any",
                    ),
                }
            )
        normalized_steps.append({"id": step_id, "title": title, "outputs": normalized_outputs})

    normalized = dict(rules)
    normalized["steps"] = normalized_steps
    return normalized


def load_rules(path: Path, force_mini: bool = False) -> Dict[str, Any]:
    """Read and validate rules from disk, converting all failures to ConfigError."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read rules file {path}: {exc}") from exc
    return validate_rules(load_yaml_text(text, force_mini=force_mini))


def _contained_output_path(project_root: Path, relative: str) -> Tuple[Optional[Path], Optional[str]]:
    root = Path(os.path.realpath(str(project_root)))
    candidate = Path(os.path.realpath(str(root / relative)))
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, f"path escapes project root: {relative}"
    return candidate, None


def _evaluate_output(project_root: Path, menu: str, output: Dict[str, Any]) -> List[str]:
    relative = output["path"].replace("{menu}", menu)
    path, path_error = _contained_output_path(project_root, relative)
    if path_error or path is None:
        return [path_error or f"unsafe path: {relative}"]
    try:
        if not path.is_file():
            return [f"{relative}: file missing"]
        size = path.stat().st_size
    except OSError as exc:
        return [f"{relative}: cannot inspect file ({exc})"]

    missing: List[str] = []
    minimum = output["min_bytes"]
    if size < minimum:
        missing.append(f"{relative}: size {size} bytes is below minimum {minimum}")

    required = output["must_contain"]
    required_any = output["must_contain_any"]
    if required or required_any:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            missing.append(f"{relative}: cannot read UTF-8 content ({exc})")
            return missing
        for needle in required:
            if needle not in text:
                missing.append(f"{relative}: missing required text {needle!r}")
        if required_any and not any(needle in text for needle in required_any):
            joined = ", ".join(repr(needle) for needle in required_any)
            missing.append(f"{relative}: requires at least one of {joined}")
    return missing


def evaluate(project_root: Path, menu: str, rules: dict) -> dict:
    """Evaluate every step live from evidence files beneath project_root."""
    menu = validate_menu(menu)
    normalized = validate_rules(rules)
    root = Path(project_root)
    steps_result: List[Dict[str, Any]] = []
    done_count = 0

    for step in normalized["steps"]:
        missing: List[str] = []
        for output in step["outputs"]:
            missing.extend(_evaluate_output(root, menu, output))
        status = "pending" if missing else "done"
        if status == "done":
            done_count += 1
        steps_result.append(
            {"id": step["id"], "title": step["title"], "status": status, "missing": missing}
        )

    current_step = next(
        (step["id"] for step in steps_result if step["status"] == "pending"), None
    )
    return {
        "menu": menu,
        "steps": steps_result,
        "current_step": current_step,
        "done_count": done_count,
        "total": len(steps_result),
    }


def can_enter(
    project_root: Path, menu: str, step_id: str, rules: dict
) -> Tuple[bool, List[str]]:
    """Allow a step only when every earlier configured step is done."""
    result = evaluate(project_root, menu, rules)
    ids = [step["id"] for step in result["steps"]]
    if step_id not in ids:
        raise ConfigError(f"unknown step id: {step_id}")
    target_index = ids.index(step_id)
    reasons: List[str] = []
    for step in result["steps"][:target_index]:
        if step["status"] == "pending":
            reasons.extend(f"{step['id']}: {reason}" for reason in step["missing"])
    return not reasons, reasons
