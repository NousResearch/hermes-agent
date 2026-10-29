"""Post-write guard for every ``config.yaml`` write: reject a file Hermes could not load.

A single duplicate YAML key (``platform_toolsets:`` written twice) made a profile's
``config.yaml`` unparseable. ruamel refuses to LOAD it, so every worker for that profile
died at startup and the multiplexer skipped the profile — one line of YAML, three
services down. The write path now proves the file it just wrote is loadable BEFORE it
counts as written.

Two defect classes are covered, by two different seams:

* **Duplicate key** — ruamel raises ``DuplicateKeyError`` while constructing a mapping.
  PyYAML (and ``yaml.safe_load``) silently keeps the LAST value, so a PyYAML-based check
  cannot see this class of defect at all. Caught at the WRITE path
  (:func:`verify_written_config`, wired into ``utils.atomic_roundtrip_yaml_save``) and at
  the PRE-COMMIT check (``scripts/check_config_yaml_duplicate_keys.py``).

* **Silent key deletion** — a regex/sed edit that dropped a top-level key and then
  re-added a second copy of the same block left the key set unchanged, so a duplicate
  check alone would have missed the precursor bug. Caught by comparing the top-level key
  set before and after the write (:func:`deleted_top_level_keys`). A write through the
  config API is caught by ``hermes_cli.config._omitted_config_paths``; an OUT-OF-BAND edit
  (regex/sed, an editor) is caught by the PRE-COMMIT check, which diffs the staged file
  against HEAD.

The guard is deliberately narrow: it validates the file that was just written, restores
the previous good bytes on failure, and raises. It never rewrites content itself.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Environment override for the guard. Set to ``0``/``false`` to disable it (an escape
#: hatch for a hand-repair of a config file that is already broken).
_ENV_DISABLE = "HERMES_CONFIG_WRITE_GUARD"


class ConfigWriteRejected(RuntimeError):
    """A write produced a config.yaml that could not be loaded, or dropped a key.

    Raised AFTER the previous good file has been restored, so the on-disk state is
    unchanged by the time the caller sees this.
    """


def guard_enabled() -> bool:
    """True unless the guard is explicitly disabled by the environment."""
    raw = os.environ.get(_ENV_DISABLE, "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _load_yaml(text: str) -> Any:
    """Parse *text* with the shared ruamel-backed reader (rejects duplicate keys)."""
    import utils

    return utils.fast_safe_load(text)


def top_level_keys(text: str) -> Optional[set[str]]:
    """The top-level key names of a config document, or None when it does not parse.

    None means "could not read the key set" — the caller must not treat that as an
    empty key set (that would read as "every key was deleted").
    """
    try:
        parsed = _load_yaml(text)
    except Exception:
        return None
    if parsed is None:
        return set()
    if not isinstance(parsed, dict):
        return None
    return {str(k) for k in parsed.keys()}


def deleted_top_level_keys(before: Optional[set[str]], after: Optional[set[str]]) -> set[str]:
    """Top-level keys present in *before* and absent from *after*.

    ``before``/``after`` of None mean "the key set could not be read" and yield an empty
    result: an unreadable key set is the duplicate-key case, not a deletion case, and the
    two are reported separately.
    """
    if before is None or after is None:
        return set()
    return before - after


def _describe_parse_error(exc: Exception) -> str:
    """One-line reason for a parse failure, with the line number when ruamel gives one."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = getattr(mark, "line", None)
    where = f" at line {line + 1}" if isinstance(line, int) else ""
    problem = getattr(exc, "problem", None)
    detail = " ".join(str(problem or exc).split())
    return f"{detail}{where}"


def _restore(previous_bytes: Optional[bytes], path: Path) -> None:
    """Put the previous good bytes back (or remove the file if it did not exist)."""
    try:
        if previous_bytes is None:
            path.unlink(missing_ok=True)
        else:
            tmp = path.with_name(path.name + ".guard-restore")
            tmp.write_bytes(previous_bytes)
            shutil.copymode(path, tmp) if path.exists() else None
            os.replace(tmp, path)
    except OSError as exc:
        logger.error("config write guard could not restore %s: %s", path, exc)


def verify_written_config(
    path: Path,
    *,
    previous_bytes: Optional[bytes] = None,
    previous_keys: Optional[set[str]] = None,
    allow_key_deletion: bool = False,
) -> None:
    """Validate the config just written to *path*; restore and raise on failure.

    *previous_bytes* is the file's content from before the write, used to restore the
    previous good file if the new content is unusable. *previous_keys* is the top-level
    key set from before the write; a write that drops one of those keys is refused
    unless *allow_key_deletion* says the drop was deliberate.

    Raises :class:`ConfigWriteRejected` after restoring the previous good file.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _restore(previous_bytes, path)
        raise ConfigWriteRejected(
            f"The settings file ({path}) could not be read back after the write, so the "
            f"change was not saved. Details: {exc}") from exc

    try:
        parsed = _load_yaml(text)
    except Exception as exc:
        _restore(previous_bytes, path)
        raise ConfigWriteRejected(
            f"The settings file ({path}) has a formatting error, so this change was not "
            f"saved and the previous file was restored. Details: {_describe_parse_error(exc)}"
        ) from exc

    if parsed is not None and not isinstance(parsed, dict):
        _restore(previous_bytes, path)
        raise ConfigWriteRejected(
            f"The settings file ({path}) must start with settings names, but its top level "
            f"is a {type(parsed).__name__}, so this change was not saved and the previous "
            f"file was restored.")

    after_keys = top_level_keys(text)
    dropped = deleted_top_level_keys(previous_keys, after_keys)
    if dropped and not allow_key_deletion:
        _restore(previous_bytes, path)
        raise ConfigWriteRejected(
            f"This change would drop {len(dropped)} top-level setting(s) from {path} "
            f"({', '.join(sorted(dropped)[:12])}), so it was not saved and the previous file "
            f"was restored. Pass the complete current config, or set the key deliberately.")


def snapshot_config(path: Path) -> tuple[Optional[bytes], Optional[set[str]]]:
    """Capture the bytes and top-level key set of *path* before a write.

    Returns ``(bytes_or_None, keys_or_None)``; both are None when the file does not
    exist yet, which the guard treats as "nothing to lose".
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None, None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, None
    return raw, top_level_keys(text)
