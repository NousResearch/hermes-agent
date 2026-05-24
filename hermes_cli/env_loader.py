"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
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

# Map of env-var name → source label ("bitwarden", etc.) for credentials
# that were injected by an external secret source during load_hermes_dotenv().
# Used by setup / `hermes model` flows to label detected credentials so
# users understand WHERE a key came from when their .env doesn't contain it
# directly (otherwise the "credentials detected ✓" line looks identical to
# the .env case and they don't know Bitwarden is wired up).
_SECRET_SOURCES: dict[str, str] = {}


def get_secret_source(env_var: str) -> str | None:
    """Return the label of the secret source that supplied ``env_var``, if any.

    Returns ``"bitwarden"`` for keys pulled from Bitwarden Secrets Manager
    during the current process's ``load_hermes_dotenv()`` call.  Returns
    ``None`` for keys that came from ``.env``, the shell environment, or
    aren't tracked.
    """
    return _SECRET_SOURCES.get(env_var)


def format_secret_source_suffix(env_var: str) -> str:
    """Return a human-readable suffix like ``" (from Bitwarden)"`` or ``""``.

    Use this when printing a detected credential so the user can see where
    it came from.  Empty string when the credential came from ``.env`` or
    the shell — those are the implicit / "default" cases users already
    understand.
    """
    source = get_secret_source(env_var)
    if not source:
        return ""
    if source == "bitwarden":
        return " (from Bitwarden)"
    # Generic fallback — future-proofing for additional secret sources
    # (e.g. 1Password, HashiCorp Vault) without having to update every
    # call site.
    return f" (from {source})"


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

    Also strips embedded null bytes which crash ``os.environ[k] = v``
    with ``ValueError: embedded null byte`` — typically introduced by
    copy-pasting API keys from terminals or rich-text editors.

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
        # Strip null bytes before _sanitize_env_lines so they never
        # reach python-dotenv (which passes them to os.environ and
        # crashes with ValueError).
        stripped = [line.replace("\x00", "") for line in original]
        sanitized = _sanitize_env_lines(stripped)
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
    - ``${HERMES_HOME}/.env`` overrides stale shell-exported values when present
      (this is the data-directory's local secrets file — installer-written on
      Windows, profile-specific in profile mode, container-volume in Docker).
    - The doc-canonical ``~/.hermes/.env`` is also loaded when it exists and
      differs from ``${HERMES_HOME}/.env``. Loaded *before* the HERMES_HOME
      file so the latter still wins on key collisions, but loaded at all so
      doc-mandated edits (``website/docs/user-guide/features/api-server.md``
      tells users to add ``API_SERVER_ENABLED=true`` to ``~/.hermes/.env``)
      actually take effect when the installer or operator has redirected
      ``HERMES_HOME`` elsewhere. Issue #31144.
    - project ``.env`` acts as a dev fallback and only fills missing values
      when one of the user envs exists.
    - if no user envs exist, the project ``.env`` also overrides stale shell vars.
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Doc-canonical secrets file. Loaded only when it exists *and* points at
    # a different inode than ``user_env`` — common only when the installer /
    # operator has redirected ``HERMES_HOME``. ``Path.samefile`` is the
    # cross-platform way to compare (handles symlinks, junctions, case
    # differences on Windows). Falls back to a plain string compare when
    # either side doesn't exist on disk yet.
    shared_user_env: Path | None = None
    default_user_env = Path.home() / ".hermes" / ".env"
    if default_user_env.exists():
        try:
            is_same_file = (
                user_env.exists() and default_user_env.samefile(user_env)
            )
        except OSError:
            is_same_file = (default_user_env == user_env)
        if not is_same_file:
            shared_user_env = default_user_env

    # Fix corrupted .env files before python-dotenv parses them (#8908).
    for env_path in (shared_user_env, user_env, project_env_path):
        if env_path and env_path.exists():
            _sanitize_env_file_if_needed(env_path)

    if shared_user_env and shared_user_env.exists():
        # Doc-canonical file: low precedence so the HERMES_HOME-resolved
        # file (loaded next) still wins on key collisions.
        _load_dotenv_with_fallback(shared_user_env, override=True)
        loaded.append(shared_user_env)

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    _apply_external_secret_sources(home_path)

    return loaded


def _apply_external_secret_sources(home_path: Path) -> None:
    """Pull secrets from external sources (currently Bitwarden) into env.

    Runs AFTER dotenv loads so .env values are visible (we use them to
    locate the access token) but BEFORE the rest of Hermes reads
    ``os.environ`` for credentials.  Any failure here is logged and
    swallowed — external secret sources must never block startup.
    """
    try:
        cfg = _load_secrets_config(home_path)
    except Exception:  # noqa: BLE001 — config errors must not block startup
        return

    bw_cfg = (cfg or {}).get("bitwarden") or {}
    if not bw_cfg.get("enabled"):
        return

    try:
        from agent.secret_sources.bitwarden import apply_bitwarden_secrets
    except ImportError:
        return

    result = apply_bitwarden_secrets(
        enabled=True,
        access_token_env=bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN"),
        project_id=bw_cfg.get("project_id", ""),
        override_existing=bool(bw_cfg.get("override_existing", False)),
        cache_ttl_seconds=float(bw_cfg.get("cache_ttl_seconds", 300)),
        auto_install=bool(bw_cfg.get("auto_install", True)),
    )

    if result.applied:
        # Re-run the ASCII sanitization pass: BSM values are user-supplied
        # and might have the same copy-paste corruption as a manually
        # edited .env (see #6843).
        _sanitize_loaded_credentials()
        # Remember where these came from so the setup / `hermes model`
        # flows can label detected credentials with "(from Bitwarden)" —
        # otherwise users see "credentials ✓" with no hint that the value
        # came from BSM rather than .env.
        for name in result.applied:
            _SECRET_SOURCES[name] = "bitwarden"
        print(
            f"  Bitwarden Secrets Manager: applied {len(result.applied)} "
            f"secret{'s' if len(result.applied) != 1 else ''} "
            f"({', '.join(sorted(result.applied))})",
            file=sys.stderr,
        )
    if result.error:
        print(
            f"  Bitwarden Secrets Manager: {result.error}",
            file=sys.stderr,
        )
    for warn in result.warnings:
        print(
            f"  Bitwarden Secrets Manager: {warn}",
            file=sys.stderr,
        )


def _load_secrets_config(home_path: Path) -> dict:
    """Read just the ``secrets:`` section out of config.yaml.

    Imported lazily and isolated from the main config loader so a
    malformed config can't take down dotenv loading entirely.
    """
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}
    return data.get("secrets") or {}
