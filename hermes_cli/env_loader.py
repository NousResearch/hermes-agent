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

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Fix corrupted .env files before python-dotenv parses them (#8908).
    if user_env.exists():
        _sanitize_env_file_if_needed(user_env)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    # config.yaml is the documented source of truth for terminal.backend, but
    # a stale TERMINAL_ENV in ~/.hermes/.env (e.g. left over from `hermes
    # setup` after the user later edits config.yaml directly) silently wins
    # because dotenv loaded with override=True above.  Re-apply config.yaml's
    # terminal.* values on top so the documented config path always wins
    # (#29186).
    _apply_config_yaml_terminal_override(home_path)

    return loaded


# Mapping of config.yaml terminal.* keys -> env vars downstream code reads.
# Kept in sync with the equivalent maps in gateway/run.py and
# hermes_cli/config.py — see tests/tools/test_terminal_config_env_sync.py.
_TERMINAL_CFG_TO_ENV = {
    "backend": "TERMINAL_ENV",
    "docker_image": "TERMINAL_DOCKER_IMAGE",
    "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
    "modal_image": "TERMINAL_MODAL_IMAGE",
    "daytona_image": "TERMINAL_DAYTONA_IMAGE",
    "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
    "timeout": "TERMINAL_TIMEOUT",
    "sandbox_dir": "TERMINAL_SANDBOX_DIR",
    "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
    "container_cpu": "TERMINAL_CONTAINER_CPU",
    "container_memory": "TERMINAL_CONTAINER_MEMORY",
    "container_disk": "TERMINAL_CONTAINER_DISK",
    "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
    "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
    "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
}


def _apply_config_yaml_terminal_override(home_path: Path) -> None:
    """Force config.yaml's terminal.* values on top of whatever .env loaded.

    Without this, a stale TERMINAL_ENV=docker in ~/.hermes/.env keeps winning
    after the user switches terminal.backend in config.yaml — the gateway
    (per-turn reload), cron scheduler, batch_runner, and doctor would all
    still run against the old backend until the user manually purged the
    .env line (#29186).
    """
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return
    try:
        import yaml
    except ImportError:
        return
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(cfg, dict):
        return
    terminal_cfg = cfg.get("terminal")
    if not isinstance(terminal_cfg, dict):
        return
    import json
    for cfg_key, env_var in _TERMINAL_CFG_TO_ENV.items():
        if cfg_key not in terminal_cfg:
            continue
        val = terminal_cfg[cfg_key]
        if val is None:
            continue
        if isinstance(val, (list, dict)):
            os.environ[env_var] = json.dumps(val)
        else:
            os.environ[env_var] = str(val)
