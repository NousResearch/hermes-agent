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
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT_SECONDS = 30
_MISSING = object()


def _stringify_filter_value(value: Any) -> str:
    if value is _MISSING:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _resolve_profile_path(path_value: Any) -> Optional[Path]:
    """Resolve a user path, mapping ~/.hermes to the active profile home."""
    if not isinstance(path_value, str):
        return None
    raw = os.path.expandvars(path_value.strip())
    if not raw:
        return None
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    if raw == "~/.hermes":
        return hermes_home
    if raw.startswith("~/.hermes/"):
        return hermes_home / raw.removeprefix("~/.hermes/")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return hermes_home / path


def _resolve_script_path(script_value: Any) -> tuple[Optional[Path], Optional[str]]:
    """Resolve a route script under HERMES_HOME/scripts."""
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
    try:
        candidate.relative_to(scripts_root)
    except ValueError:
        return None, f"script path resolves outside {scripts_root}"
    if not candidate.exists():
        return None, f"script not found: {candidate}"
    if not candidate.is_file():
        return None, f"script path is not a file: {candidate}"
    return candidate, None


def _load_filter_file_values(path_value: Any) -> list[Any]:
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
    if isinstance(data, dict):
        return list(data.keys())
    return [data]


class WebhookRouteProcessor:
    """Evaluate declarative filters and optional script transforms."""

    def __init__(
        self,
        *,
        script_timeout_seconds: int = DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ) -> None:
        self.script_timeout_seconds = max(1, int(script_timeout_seconds))

    def resolve_filter_field(
        self,
        field: Any,
        payload: dict,
        event_type: str,
        headers: Any,
    ) -> Any:
        """Resolve a dotted filter field against payload/event/headers context."""
        if not isinstance(field, str) or not field.strip():
            return _MISSING
        parts = [part for part in field.strip().split(".") if part]
        if not parts:
            return _MISSING
        header_dict = dict(headers or {})
        context = {
            "payload": payload.get("payload", payload),
            "event": event_type,
            "event_type": event_type,
            "headers": header_dict,
        }
        if parts[0] in context:
            value: Any = context[parts[0]]
            parts = parts[1:]
        else:
            value = payload
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

    def filter_matches(
        self,
        spec: Any,
        payload: dict,
        event_type: str,
        headers: Any,
    ) -> bool:
        """Evaluate one declarative webhook filter spec."""
        if not isinstance(spec, dict):
            logger.warning("[webhook] Ignoring invalid filter spec: %r", spec)
            return False

        if "all" in spec:
            items = spec.get("all")
            return isinstance(items, list) and all(
                self.filter_matches(item, payload, event_type, headers)
                for item in items
            )
        if "any" in spec:
            items = spec.get("any")
            return isinstance(items, list) and any(
                self.filter_matches(item, payload, event_type, headers)
                for item in items
            )
        if "not" in spec:
            return not self.filter_matches(spec.get("not"), payload, event_type, headers)

        value = self.resolve_filter_field(
            spec.get("field"), payload, event_type, headers
        )

        if "exists" in spec:
            exists = value is not _MISSING
            return exists is bool(spec.get("exists"))
        if spec.get("missing") is True:
            return value is _MISSING
        if "equals" in spec:
            return value is not _MISSING and value == spec.get("equals")
        if "not_equals" in spec:
            return value is _MISSING or value != spec.get("not_equals")
        if "contains" in spec:
            needle = spec.get("contains")
            if value is _MISSING:
                return False
            if isinstance(value, (list, tuple, set, dict)):
                return needle in value
            return str(needle) in _stringify_filter_value(value)
        if "in" in spec:
            haystack = spec.get("in")
            return isinstance(haystack, list) and value in haystack
        if "in_file" in spec:
            return value in _load_filter_file_values(spec.get("in_file"))
        if "regex" in spec:
            if value is _MISSING:
                return False
            try:
                return (
                    re.search(str(spec.get("regex")), _stringify_filter_value(value))
                    is not None
                )
            except re.error as exc:
                logger.warning("[webhook] Invalid webhook filter regex: %s", exc)
                return False

        logger.warning("[webhook] Filter spec has no supported operator: %r", spec)
        return False

    def route_filters_match(
        self,
        route_config: dict,
        payload: dict,
        event_type: str,
        headers: Any,
    ) -> bool:
        filters = route_config.get("filters") or []
        if not filters:
            return True
        if isinstance(filters, dict):
            return self.filter_matches(filters, payload, event_type, headers)
        if not isinstance(filters, list):
            logger.warning("[webhook] filters must be a list or object")
            return False
        return all(
            self.filter_matches(spec, payload, event_type, headers)
            for spec in filters
        )

    def run_route_script(self, script_value: Any, payload: dict) -> tuple[bool, Optional[dict]]:
        """Run a route script and return (should_continue, transformed_payload)."""
        path, error = _resolve_script_path(script_value)
        if error or path is None:
            logger.warning("[webhook] script ignored webhook: %s", error)
            return False, None

        suffix = path.suffix.lower()
        if suffix in {".sh", ".bash"}:
            # Delegate to the shared bash resolver used by the native
            # terminal path (tools.environments.local._find_bash) — same as
            # cron/scheduler.py::_run_job_script.  Avoids the WSL launcher
            # stub resolution bug where ``shutil.which("bash")`` returns
            # ``C:\Windows\System32\bash.exe`` when WSL is enabled with no
            # distributions installed and Git for Windows is installed with
            # its default PATH option (adds ``Git\cmd``, not ``Git\bin``),
            # which would silently fail every ``.sh`` webhook route script.
            from tools.environments.local import _find_bash
            try:
                bash = _find_bash()
            except RuntimeError:
                logger.warning("[webhook] script ignored webhook: bash not found")
                return False, None
            argv = [bash, str(path)]
        else:
            argv = [sys.executable, str(path)]

        try:
            from tools.environments.local import _sanitize_subprocess_env

            popen_kwargs = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
            result = subprocess.run(
                argv,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.script_timeout_seconds,
                cwd=str(path.parent),
                env=_sanitize_subprocess_env(os.environ.copy()),
                **popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            logger.warning("[webhook] script timed out: %s", path)
            return False, None
        except Exception as exc:
            logger.warning("[webhook] script execution failed: %s", exc)
            return False, None

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        try:
            from agent.redact import redact_sensitive_text

            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as exc:
            logger.warning("[webhook] Failed to redact script output: %s", exc)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"
        if result.returncode != 0:
            logger.info(
                "[webhook] script ignored webhook path=%s code=%s stderr=%s",
                path.name,
                result.returncode,
                stderr[:200],
            )
            return False, None
        if not stdout or stdout == "[SILENT]":
            return False, None

        try:
            transformed = json.loads(stdout)
        except json.JSONDecodeError:
            transformed = {**payload, "script_output": stdout}
        if not isinstance(transformed, dict):
            logger.warning("[webhook] script stdout must be a JSON object or text")
            return False, None
        if (
            transformed.get("[SILENT]") is True
            or transformed.get("__hermes_ignore__") is True
        ):
            return False, None
        return True, transformed
