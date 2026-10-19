"""``.env`` key writes split out of ``hermes_cli.config``: the one locked read-modify-write
(``write_env_key``) and its profile-targeted variant.

``save_env_value_to_profile`` takes an explicit profile directory and never reads
``get_env_path()``, ``os.environ`` or the current profile scope, and never publishes the value
to the live process: a restart or the per-turn secret-scope reload picks it up.
"""

from __future__ import annotations

import stat
from pathlib import Path

# Group-write or any "other" bit: such a mode is not private enough to keep.
_NON_PRIVATE_MODE_BITS = 0o027
_FORBIDDEN_VALUE_CHARS = ("\r", "\n", "\0")


def write_env_key(env_path: Path, key: str, value: str, *, preserve_mode: bool) -> None:
    """Set ``key=value`` in ``env_path`` under the ``.env`` write lock (also matching an
    ``export KEY=`` line, so a save never appends a second line). Other lines keep their order."""
    from hermes_cli.config import (
        _env_line_defines_key, _env_write_lock, _quote_env_value, _read_env_lines, _write_env_lines)

    with _env_write_lock(env_path):
        lines = _read_env_lines(env_path) if env_path.exists() else []
        line = f"{key}={_quote_env_value(value)}\n"
        idx = next((i for i, existing in enumerate(lines) if _env_line_defines_key(existing, key)), None)
        if idx is not None:
            lines[idx] = line
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(line)
        _write_env_lines(env_path, lines, preserve_mode=preserve_mode)


def profile_env_path(profile_dir: Path) -> Path:
    """``<profile_dir>/.env``, refused when its real path leaves the real profile directory
    (a symlinked ``.env`` pointing elsewhere)."""
    root = Path(profile_dir).resolve(strict=True)
    env_path = Path(profile_dir) / ".env"
    real = env_path.resolve(strict=False)
    if real == root or not real.is_relative_to(root):
        raise ValueError("The profile .env file is outside the profile directory.")
    return env_path


def env_file_defines_key(env_path: Path, key: str) -> bool:
    """True when ``env_path`` exists and has a line that assigns ``key``."""
    from hermes_cli.config import _env_line_defines_key, _read_env_lines

    if not env_path.exists():
        return False
    return any(_env_line_defines_key(line, key) for line in _read_env_lines(env_path))


def _keeps_private_mode(env_path: Path) -> bool:
    """Keep an existing file's mode only when it has no group-write or other bits."""
    try:
        mode = stat.S_IMODE(env_path.stat().st_mode)
    except FileNotFoundError:
        return False
    return not mode & _NON_PRIVATE_MODE_BITS


def save_env_value_to_profile(profile_dir: Path, key: str, value: str) -> None:
    """Write one key to one profile's ``.env``. Raises ``ValueError`` with a fixed message (never
    the value) for a refused name, a write-locked key, a value with CR/LF/NUL, or an unsafe path."""
    from hermes_cli.config import env_write_refusal, invalidate_env_cache, validate_env_var_name_for_write

    validate_env_var_name_for_write(key)
    if env_write_refusal(key, "set"):
        raise ValueError("This key cannot be changed on this host.")
    if any(char in value for char in _FORBIDDEN_VALUE_CHARS):
        raise ValueError("The value has a line break or a NUL character.")
    env_path = profile_env_path(profile_dir)
    write_env_key(env_path, key, value, preserve_mode=_keeps_private_mode(env_path))
    invalidate_env_cache()
