"""The transcript size limits callers pass to the resume and export verbs, and the payload limits they pass to the
import verbs."""

from dataclasses import fields
from pathlib import Path
from typing import Any

from persistence.housekeeping.maintenance import ImportLimits

_MAX_SAFE_MESSAGES = 20_000  # resume/export guard default


def _limit(value: Any, fallback: int) -> int:
    """A configured limit: *value* as an int when it is a non-negative number, else *fallback*; 0 disables. A YAML
    boolean is not a number, although ``bool`` subclasses ``int`` (``int(False)`` would silently disable a guard)."""
    if isinstance(value, bool):
        return fallback
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return limit if limit >= 0 else fallback


def _configured_transcript_limit(key: str, fallback: int = _MAX_SAFE_MESSAGES) -> int:
    """``sessions.<key>`` from config.yaml (lazy import: circular at load), else *fallback*; 0 disables."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("sessions") or {}).get(key)
    except Exception:  # health: allow BLE001 -- an unreadable config means the default limit
        return fallback
    return _limit(value, fallback)


def resolved_max_resume_messages() -> int:
    return _configured_transcript_limit("max_resume_messages")


def resolved_max_export_messages() -> int:
    return _configured_transcript_limit("max_export_messages")


def resolved_import_limits(home: Path) -> ImportLimits:
    """``sessions.import_max_<field>`` for each ``ImportLimits`` field from the config.yaml of profile *home* (the
    store the import writes to, never the process's own); an unset, negative or non-numeric key keeps the default and
    0 disables that guard. An unreadable or missing config means the defaults."""
    try:
        from hermes_cli.config_backend import config_exists, read_config_doc_readonly
        path = home / "config.yaml"
        doc = read_config_doc_readonly(path) if config_exists(path) else None
        sessions = doc.get("sessions") if isinstance(doc, dict) else None
    except Exception:  # health: allow BLE001 -- an unreadable config means the default limits
        sessions = None
    sessions = sessions if isinstance(sessions, dict) else {}
    return ImportLimits(**{field.name: _limit(sessions.get(f"import_{field.name}"), field.default)
                           for field in fields(ImportLimits)})
