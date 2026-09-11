"""Route-local filters and script transforms for the webhook adapter."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT_SECONDS = 30
_MISSING = object()


def _stringify_filter_value(value: Any) -> str:
    return "" if value is _MISSING else json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def _resolve_profile_path(path_value: Any) -> Optional[Path]:
    """Resolve a user path, mapping ~/.hermes to the active profile home."""
    if not isinstance(path_value, str):
        return None
    raw = os.path.expandvars(path_value.strip())
    if not raw:
        return None
    from hermes_constants import get_hermes_home
    hermes_home = get_hermes_home()
    if raw == "~/.hermes" or raw.startswith("~/.hermes/"):
        return hermes_home / raw[len("~/.hermes/"):]
    path = Path(raw).expanduser()
    return path if path.is_absolute() else hermes_home / path


def _resolve_script_path(script_value: Any) -> tuple[Optional[Path], Optional[str]]:
    """Resolve a route script; must live under HERMES_HOME/scripts."""
    if not isinstance(script_value, str) or not script_value.strip():
        return None, "script path is empty"
    from hermes_constants import get_hermes_home
    scripts_root = (get_hermes_home() / "scripts").resolve()
    raw_text = os.path.expandvars(script_value.strip())
    if raw_text == "~/.hermes" or raw_text.startswith("~/.hermes/"):
        mapped = _resolve_profile_path(raw_text)
        candidate = mapped.resolve() if mapped is not None else scripts_root
    else:
        raw = Path(raw_text).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (scripts_root / raw).resolve()
    if not candidate.is_relative_to(scripts_root):
        return None, f"script path resolves outside {scripts_root}"
    if not candidate.exists():
        return None, f"script not found: {candidate}"
    return (candidate, None) if candidate.is_file() else (None, f"script path is not a file: {candidate}")


def _load_filter_file_values(path_value: Any) -> list[Any]:
    """Values from an ``in_file`` list: JSON list, JSON object keys, or one value per non-blank line."""
    path = _resolve_profile_path(path_value)
    if path is None:
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("[webhook] filter in_file read failed for %s: %s", path, exc)
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    return list(data.keys()) if isinstance(data, dict) else [data]


def _op_contains(value: Any, needle: Any) -> bool:
    if value is _MISSING:
        return False
    return needle in value if isinstance(value, (list, tuple, set, dict)) else str(needle) in _stringify_filter_value(value)


def _op_regex(value: Any, pattern: Any) -> bool:
    if value is _MISSING:
        return False
    try:
        return re.search(str(pattern), _stringify_filter_value(value)) is not None
    except re.error as exc:
        logger.warning("[webhook] Invalid webhook filter regex: %s", exc)
        return False


# Field operators in precedence order: (spec key, predicate(resolved value, operand)). First key present wins.
_FIELD_OPERATORS: tuple[tuple[str, Callable[[Any, Any], bool]], ...] = (
    ("exists", lambda value, arg: (value is not _MISSING) is bool(arg)),
    ("equals", lambda value, arg: value is not _MISSING and value == arg),
    ("not_equals", lambda value, arg: value is _MISSING or value != arg),
    ("contains", _op_contains),
    ("in", lambda value, arg: isinstance(arg, list) and value in arg),
    ("in_file", lambda value, arg: value in _load_filter_file_values(arg)),
    ("regex", _op_regex),
)


class WebhookRouteProcessor:
    """Evaluate declarative filters and optional script transforms."""

    def __init__(self, *, script_timeout_seconds: int = DEFAULT_SCRIPT_TIMEOUT_SECONDS) -> None:
        self.script_timeout_seconds = max(1, int(script_timeout_seconds))

    def resolve_filter_field(self, field: Any, payload: dict, event_type: str, headers: Any) -> Any:
        """Resolve a dotted filter field against payload/event/headers context; ``_MISSING`` when absent."""
        if not isinstance(field, str) or not field.strip():
            return _MISSING
        parts = [part for part in field.strip().split(".") if part]
        if not parts:
            return _MISSING
        context = {"payload": payload.get("payload", payload), "event": event_type, "event_type": event_type, "headers": dict(headers or {})}
        value: Any = context[parts.pop(0)] if parts[0] in context else payload
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, _MISSING)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if 0 <= idx < len(value) else _MISSING
            else:
                return _MISSING
            if value is _MISSING:
                return _MISSING
        return value

    def filter_matches(self, spec: Any, payload: dict, event_type: str, headers: Any) -> bool:
        """Evaluate one declarative webhook filter spec (``all``/``any``/``not`` combinators, then field ops)."""
        if not isinstance(spec, dict):
            logger.warning("[webhook] Ignoring invalid filter spec: %r", spec)
            return False

        def _sub(item) -> bool:
            return self.filter_matches(item, payload, event_type, headers)

        for key, combine in (("all", all), ("any", any)):
            if key in spec:
                items = spec.get(key)
                return isinstance(items, list) and combine(_sub(item) for item in items)
        if "not" in spec:
            return not _sub(spec.get("not"))
        value = self.resolve_filter_field(spec.get("field"), payload, event_type, headers)
        if "exists" not in spec and spec.get("missing") is True:
            return value is _MISSING
        for key, predicate in _FIELD_OPERATORS:
            if key in spec:
                return predicate(value, spec.get(key))
        logger.warning("[webhook] Filter spec has no supported operator: %r", spec)
        return False

    def route_filters_match(self, route_config: dict, payload: dict, event_type: str, headers: Any) -> bool:
        filters = route_config.get("filters") or []
        if not filters:
            return True
        if isinstance(filters, dict):
            return self.filter_matches(filters, payload, event_type, headers)
        if not isinstance(filters, list):
            logger.warning("[webhook] filters must be a list or object")
            return False
        return all(self.filter_matches(spec, payload, event_type, headers) for spec in filters)

    def run_route_script(self, script_value: Any, payload: dict) -> tuple[bool, Optional[dict]]:
        """Run a route script and return (should_continue, transformed_payload).

        Non-zero exit, empty/``[SILENT]`` stdout, or a ``[SILENT]``/``__hermes_ignore__`` flag drops the
        webhook; JSON-object stdout replaces the payload, other text is attached as ``script_output``.
        """
        path, error = _resolve_script_path(script_value)
        if error or path is None:
            logger.warning("[webhook] script ignored webhook: %s", error)
            return False, None
        is_shell = path.suffix.lower() in {".sh", ".bash"}
        interpreter = (shutil.which("bash") or ("/bin/bash" if os.path.isfile("/bin/bash") else None)) if is_shell else sys.executable
        if interpreter is None:
            logger.warning("[webhook] script ignored webhook: bash not found")
            return False, None
        try:
            from tools.environments.local import build_subprocess_env
            popen_kwargs = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
            result = subprocess.run(
                [interpreter, str(path)], input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self.script_timeout_seconds, cwd=str(path.parent), env=build_subprocess_env(), **popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            logger.warning("[webhook] script timed out: %s", path)
            return False, None
        except Exception as exc:
            logger.warning("[webhook] script execution failed: %s", exc)
            return False, None
        stdout, stderr = (result.stdout or "").strip(), (result.stderr or "").strip()
        try:
            from agent.redact import redact_sensitive_text
            stdout, stderr = redact_sensitive_text(stdout), redact_sensitive_text(stderr)
        except Exception as exc:
            logger.warning("[webhook] Failed to redact script output: %s", exc)
            stdout = stderr = "[REDACTED - redaction failed]"
        if result.returncode != 0:
            logger.info("[webhook] script ignored webhook path=%s code=%s stderr=%s", path.name, result.returncode, stderr[:200])
        if result.returncode != 0 or not stdout or stdout == "[SILENT]":
            return False, None
        try:
            transformed = json.loads(stdout)
        except json.JSONDecodeError:
            transformed = {**payload, "script_output": stdout}
        if not isinstance(transformed, dict):
            logger.warning("[webhook] script stdout must be a JSON object or text")
            return False, None
        silenced = transformed.get("[SILENT]") is True or transformed.get("__hermes_ignore__") is True
        return (False, None) if silenced else (True, transformed)
