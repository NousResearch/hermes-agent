"""What Hermes does when an existing ``config.yaml`` cannot be read or parsed.

Readers fail open (``{}`` / defaults / last-known-good) so a broken file never takes the process
down; this module makes that fallback loud (one warning per file signature, a ``corrupt`` backup),
remembers the failure for :func:`get_active_config_parse_failure`, and marks every fallback as a
:class:`FailedConfigRead` so the writers refuse to persist it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

# (config_path, mtime_ns, size) tuples already warned about, so concurrent CLI/gateway
# loads of a broken config.yaml don't spam stderr. A changed file (new mtime) warns again.
_CONFIG_PARSE_WARNED: set = set()

# path -> (mtime_ns, size, error message) of active parse failures. Written by
# _warn_config_parse_failure() (the single funnel for every load-path parse failure) and
# probed by get_active_config_parse_failure() so provider auto-resolution can refuse to
# adopt a paid provider from env keys while the user's REAL config is unreadable.
_CONFIG_PARSE_FAILURES: dict = {}

_PARSE_FAILURE_FALLBACK_MSG = {
    "last-known-good": "Hermes is running on the settings it loaded before the edit until it is fixed, so recent changes are not applied.",
    "last-known-good-backup": "Hermes is running on your last good settings until it is fixed, so recent changes are not applied.",
    "refuse-write": "Nothing was written, so the existing file is preserved."}
_PARSE_FAILURE_DEFAULTS_MSG = (
    "Hermes is running on default settings until it is fixed, so none of your saved settings are applied.")
_PARSE_FAILURE_REPAIR_MSG = "Open it with `hermes config edit`, fix {where}, then run `hermes config check`."
_FIX_PERMS = "Fix the file permissions or move it aside first."
_FIX_YAML = (
    "Fix it with `hermes config edit` and check with `hermes config check`, or copy the newest good "
    "file from {backups} over config.yaml.")


def _yaml_error_location(exc: Exception) -> str:
    """``"line 12"`` from a PyYAML problem mark (1-based), else ``""``."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = getattr(mark, "line", None)
    return f"line {line + 1}" if isinstance(line, int) else ""


def _yaml_error_details(exc: Exception) -> str:
    """Single-line ``Details:`` text: the PyYAML problem, or the exception's first line."""
    problem = getattr(exc, "problem", None)
    text = f"{problem}" if problem else str(exc).strip()
    return " ".join(text.split())


def format_config_parse_failure(config_path: Path, exc: Exception, *, fallback: str = "defaults") -> str:
    """User copy for an unparseable config.yaml: what happened, what Hermes is doing, how to fix.
    Only the problem line/column is printed; the raw PyYAML text goes to a ``Details:`` line."""
    where = _yaml_error_location(exc)
    at = f" at {where}" if where else ""
    fallback_msg = _PARSE_FAILURE_FALLBACK_MSG.get(fallback, _PARSE_FAILURE_DEFAULTS_MSG)
    if isinstance(exc, OSError):  # the file is intact; EMFILE/EIO/sharing violation, not a YAML problem
        return f"Your settings file ({config_path}) could not be read. {fallback_msg} {_read_error_fix(exc)}"
    repair = _PARSE_FAILURE_REPAIR_MSG.format(where=where or "the problem")
    return f"Your settings file ({config_path}) has a formatting error{at}. {fallback_msg} {repair}"


def _warn_config_parse_failure(
    config_path: Path, exc: Exception, *, fallback: str = "defaults") -> None:
    """Surface a config.yaml parse failure to log and stderr (once per file signature).
    Silent fallback to ``DEFAULT_CONFIG`` drops every user override, so this must be loud.

    ``fallback`` selects the message wording: ``"defaults"`` (fresh process, nothing else to serve) or
    ``"last-known-good"`` (in-process retention of the previously loaded config — see the codex#31188 port
    in ``hermes_cli.config._load_config_impl``).
    """
    try:
        from hermes_cli.config_backend import config_version
        sig = config_version(config_path)
        key = (str(config_path), *sig)
        _CONFIG_PARSE_FAILURES[str(config_path)] = (*sig, str(exc))
    except OSError:
        key = (str(config_path), 0, 0, 0, 0)
    if key in _CONFIG_PARSE_WARNED:
        return
    _CONFIG_PARSE_WARNED.add(key)
    from hermes_cli.config_backups import backup_config
    # A read error leaves an intact file behind: no "corrupt" copy of a good file.
    backup_path = None if isinstance(exc, OSError) else backup_config(config_path, "corrupt")
    msg = format_config_parse_failure(config_path, exc, fallback=fallback)
    if backup_path is not None:
        msg += f" A copy of the broken file was saved to {backup_path}."
    logger.warning("%s Details: %s", msg, _yaml_error_details(exc))
    try:
        sys.stderr.write(f"⚠️  hermes config: {msg}\n    Details: {_yaml_error_details(exc)}\n")
        sys.stderr.flush()
    except Exception:
        pass


def get_active_config_parse_failure() -> Optional[str]:
    """Return the recorded parse error while the ACTIVE config.yaml is still byte-identical
    (mtime_ns + size + ino + ctime_ns) to the file that failed to parse; else None."""
    from hermes_cli.config import get_config_path
    try:
        from hermes_cli.config_backend import config_version
        record = _CONFIG_PARSE_FAILURES[str(path := get_config_path())]
        sig = config_version(path)
        return record[len(sig)] if sig == record[:len(sig)] else None
    except Exception:
        return None


class FailedConfigRead(dict):
    """The fail-open fallback a config reader serves when an existing config.yaml could not be read
    or parsed (``{}``, defaults or last-known-good). Readers use it like any dict; ``save_config`` /
    ``atomic_config_write`` refuse to persist it. The writer merges by deletion, so saving a fallback
    after one transient EMFILE/EIO replaced the whole file with the fallback plus the caller's edit.
    A dict subclass so the refusal survives the load→mutate→save round trip at every call site."""

    def __init__(self, data: Any = (), *, error: Exception):
        super().__init__(data)
        self.read_error = error


def _read_error_fix(exc: OSError) -> str:
    return _FIX_PERMS if isinstance(exc, PermissionError) else (
        "Try again; run `hermes config check` if it keeps failing.")


def _refuse_failed_read(config_path: Path, data: Any) -> None:
    """Refuse to save a fallback; only a read error is worth retrying, bad YAML needs an edit."""
    if not isinstance(data, FailedConfigRead):
        return
    exc = data.read_error
    if isinstance(exc, OSError):
        raise _refuse_overwrite(config_path, "could not be read", exc, _read_error_fix(exc))
    raise _refuse_overwrite(
        config_path, "has a formatting error", exc, _FIX_YAML.format(backups=_backups_dir_display()))


def _refuse_overwrite(config_path: Path, reason: str, exc: Exception, fix: str) -> RuntimeError:
    """Error for a write that must not replace an existing config.yaml. Plain lead + ``Details:``."""
    where = _yaml_error_location(exc)
    at = f" ({where})" if where else ""
    return RuntimeError(
        f"Your settings file ({config_path}) {reason}{at}, so this change was not saved. {fix} "
        f"Details: {_yaml_error_details(exc)}")


def _backups_dir_display() -> str:
    from hermes_constants import display_hermes_home
    return f"{display_hermes_home()}/backups/config/"
