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


def _shared_profile_env_key(key: str) -> bool:
    """Return True for root .env keys safe to inherit into profile homes."""
    # Provider/API credentials are shared. Platform runtime identity (bot
    # tokens, home channels, allowlists) must stay profile-local; inheriting a
    # root Telegram token makes planning/worker profiles collide with Nagatha.
    return key.endswith("_API_KEY") or key in {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "FIRECRAWL_API_KEY",
        "PARALLEL_API_KEY",
    }


def _dotenv_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return {str(k) for k, v in dotenv_values(path).items() if k and v is not None}
    except Exception:
        return set()


def _discard_unshared_root_env(
    *,
    before: dict[str, str],
    root_keys: set[str],
    profile_keys: set[str],
) -> None:
    """Remove root .env values that profiles must not inherit."""
    for key in root_keys - profile_keys:
        if _shared_profile_env_key(key):
            continue
        if key in before:
            os.environ[key] = before[key]
        else:
            os.environ.pop(key, None)


def _is_placeholder_credential(value: str | None) -> bool:
    """Return True for dummy credential values that should not override real keys."""
    if value is None:
        return False
    stripped = value.strip().strip('"').strip("'")
    if not stripped:
        return True
    return stripped.lower() in {"xxx", "...", "todo", "changeme", "[redacted]"} or len(stripped) < 10


def _restore_credentials_overridden_by_placeholders(before: dict[str, str]) -> None:
    """Undo profile .env placeholder credential overrides.

    Some legacy profile .env files contain values like ``OPENROUTER_API_KEY=xxx``.
    When profile homes overlay the canonical user env, those dummy values should
    not erase a valid shared key already loaded from ~/.hermes/.env.
    """
    for key, previous in before.items():
        current = os.environ.get(key)
        if previous and not _is_placeholder_credential(previous) and _is_placeholder_credential(current):
            os.environ[key] = previous


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
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes")).expanduser()
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Profile homes live under ~/.hermes/profiles/<name>.  Load the canonical
    # user env first so shared provider credentials (OPENROUTER_API_KEY, etc.)
    # are available without copying secrets into every profile.  Then overlay
    # the profile's own .env so bot tokens / profile-specific values still win.
    root_home = Path.home() / ".hermes"
    root_env = root_home / ".env"
    env_paths: list[Path] = []
    profile_home = False
    try:
        profile_home = home_path.resolve().is_relative_to((root_home / "profiles").resolve())
        if profile_home:
            env_paths.append(root_env)
    except Exception:
        pass
    if user_env not in env_paths:
        env_paths.append(user_env)

    # Fix corrupted .env files before python-dotenv parses them (#8908).
    for env_path in env_paths:
        if env_path.exists():
            _sanitize_env_file_if_needed(env_path)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    root_keys = _dotenv_keys(root_env) if profile_home else set()
    profile_keys = _dotenv_keys(user_env) if profile_home else set()
    before_root_env = dict(os.environ) if profile_home else {}

    for i, env_path in enumerate(env_paths):
        if env_path.exists():
            previous_credentials = {
                key: value
                for key, value in os.environ.items()
                if any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES)
            }
            _load_dotenv_with_fallback(env_path, override=True)
            if i > 0:
                _restore_credentials_overridden_by_placeholders(previous_credentials)
            loaded.append(env_path)

    if profile_home:
        _discard_unshared_root_env(
            before=before_root_env,
            root_keys=root_keys,
            profile_keys=profile_keys,
        )

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    return loaded
