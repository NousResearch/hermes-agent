"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from utils import atomic_replace


# Env var name suffixes that indicate credential values.  These are the
# only env vars whose values we sanitize on load — we must not silently
# alter arbitrary user env vars, but credentials are known to require
# pure ASCII (they become HTTP header values).
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")

# Names we've already warned about during this process, so repeated
# load_hermes_dotenv() calls (user env + project env, gateway hot-reload,
# tests) don't spam the same warning multiple times.
_WARNED_KEYS: set[str] = set()

# Per-process registry of env var names that load_hermes_dotenv itself
# placed into os.environ (i.e. they were absent before our load).  On
# reload, any name in this set that is no longer declared by any loaded
# .env file is removed from os.environ — preventing "ghost" values from
# surviving after the user deletes a key from .env.  Long-running
# processes (gateway, dashboard, CLI sessions) used to keep stale values
# forever because python-dotenv only writes keys it sees in the file.
# Names we never seeded (genuine shell exports) are never popped.
_HERMES_SEEDED_KEYS: set[str] = set()


def _format_offending_chars(value: str, limit: int = 3) -> str:
    """Return a compact 'U+XXXX ('c'), ...' summary of non-ASCII codepoints."""
    seen: list[str] = []
    for ch in value:
        if ord(ch) > 127:
            label = f"U+{ord(ch):04X}"
            if ch.isprintable():
                label += f" ({ch!r})"
            if label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
    return ", ".join(seen)


def _sanitize_loaded_credentials() -> None:
    """Strip non-ASCII characters from credential env vars in os.environ.

    Called after dotenv loads so the rest of the codebase never sees
    non-ASCII API keys.  Only touches env vars whose names end with
    known credential suffixes (``_API_KEY``, ``_TOKEN``, etc.).

    Emits a one-line warning to stderr when characters are stripped.
    Silent stripping would mask copy-paste corruption (Unicode lookalike
    glyphs from PDFs / rich-text editors, ZWSP from web pages) as opaque
    provider-side "invalid API key" errors (see #6843).
    """
    for key, value in list(os.environ.items()):
        if not any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES):
            continue
        try:
            value.encode("ascii")
            continue
        except UnicodeEncodeError:
            pass
        cleaned = value.encode("ascii", errors="ignore").decode("ascii")
        os.environ[key] = cleaned
        if key in _WARNED_KEYS:
            continue
        _WARNED_KEYS.add(key)
        stripped = len(value) - len(cleaned)
        detail = _format_offending_chars(value) or "non-printable"
        print(
            f"  Warning: {key} contained {stripped} non-ASCII character"
            f"{'s' if stripped != 1 else ''} ({detail}) — stripped so the "
            f"key can be sent as an HTTP header.",
            file=sys.stderr,
        )
        print(
            "  This usually means the key was copy-pasted from a PDF, "
            "rich-text editor, or web page that substituted lookalike\n"
            "  Unicode glyphs for ASCII letters. If authentication fails "
            "(e.g. \"API key not valid\"), re-copy the key from the\n"
            "  provider's dashboard and run `hermes setup` (or edit the "
            ".env file in a plain-text editor).",
            file=sys.stderr,
        )


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=path, override=override, encoding="latin-1")
    # Strip non-ASCII characters from credential env vars that were just
    # loaded.  API keys must be pure ASCII since they're sent as HTTP
    # header values (httpx encodes headers as ASCII).  Non-ASCII chars
    # typically come from copy-pasting keys from PDFs or rich-text editors
    # that substitute Unicode lookalike glyphs (e.g. ʋ U+028B for v).
    _sanitize_loaded_credentials()


def _sanitize_env_file_if_needed(path: Path) -> None:
    """Pre-sanitize a .env file before python-dotenv reads it.

    python-dotenv does not handle corrupted lines where multiple
    KEY=VALUE pairs are concatenated on a single line (missing newline).
    This produces mangled values — e.g. a bot token duplicated 8×
    (see #8908).

    We delegate to ``hermes_cli.config._sanitize_env_lines`` which
    already knows all valid Hermes env-var names and can split
    concatenated lines correctly.
    """
    if not path.exists():
        return
    try:
        from hermes_cli.config import _sanitize_env_lines
    except ImportError:
        return  # early bootstrap — config module not available yet

    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    try:
        with open(path, **read_kw) as f:
            original = f.readlines()
        sanitized = _sanitize_env_lines(original)
        if sanitized != original:
            import tempfile
            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".env_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(sanitized)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception:
        pass  # best-effort — don't block gateway startup


def _gather_declared_keys(*paths: Path) -> tuple[set[str], bool]:
    """Return (keys declared by these .env files, all_parsed flag).

    Uses python-dotenv's parser without mutating os.environ.  If any
    file fails to parse, all_parsed=False so callers can skip pruning
    rather than risk popping keys whose declaration we couldn't read.

    A key only counts as "declared" when its parsed value is non-None.
    A bare ``FOO`` line (no ``=``) parses as ``FOO -> None`` but is not
    actually seeded into ``os.environ`` by ``load_dotenv``, so treating
    it as declared would let a previously-seeded value linger as a
    ghost.  ``FOO=`` (empty value) parses as ``FOO -> ""`` and IS
    seeded, so it correctly counts as declared.
    """
    keys: set[str] = set()
    all_parsed = True
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            try:
                values = dotenv_values(dotenv_path=path, encoding="utf-8")
            except UnicodeDecodeError:
                values = dotenv_values(dotenv_path=path, encoding="latin-1")
        except Exception:
            all_parsed = False
            continue
        for key, value in values.items():
            if key and value is not None:
                keys.add(key)
    return keys, all_parsed


def load_hermes_dotenv(
    *,
    hermes_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
) -> list[Path]:
    """Load Hermes environment files with user config taking precedence.

    Behavior:
    - `~/.hermes/.env` overrides stale shell-exported values when present.
    - project `.env` acts as a dev fallback and only fills missing values when
      the user env exists.
    - if no user env exists, the project `.env` also overrides stale shell vars.
    - keys we previously placed into os.environ that no longer appear in any
      loaded .env file are popped.  Shell-exported keys we never seeded are
      left alone, even when they appear in .env.
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Fix corrupted .env files before python-dotenv parses them (#8908).
    if user_env.exists():
        _sanitize_env_file_if_needed(user_env)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    # Snapshot which Hermes-seeded keys are still declared by any loaded
    # .env file.  Anything we previously seeded that is now absent gets
    # pruned from os.environ so deletions in .env take effect on reload.
    declared_keys, all_parsed = _gather_declared_keys(user_env, project_env_path)
    if all_parsed:
        stale = _HERMES_SEEDED_KEYS - declared_keys
        for key in stale:
            os.environ.pop(key, None)
        _HERMES_SEEDED_KEYS.difference_update(stale)

    pre_load_keys = set(os.environ.keys())

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    # Track keys we just placed into os.environ.  A key qualifies as
    # "seeded by Hermes" only if it was absent before this load — keys
    # already present (genuine shell exports, or values an earlier load
    # already seeded) are not added a second time, and shell exports
    # remain ineligible for pruning.
    newly_seeded = (set(os.environ.keys()) - pre_load_keys) & declared_keys
    _HERMES_SEEDED_KEYS.update(newly_seeded)

    return loaded
