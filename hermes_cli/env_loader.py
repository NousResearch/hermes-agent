"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from utils import atomic_replace, fast_safe_load


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

# HERMES_HOME paths we've already pulled external secrets for during this
# process.  ``load_hermes_dotenv()`` is called at module-import time from
# several hot modules (cli.py, hermes_cli/main.py, run_agent.py,
# trajectory_compressor.py, gateway/run.py, ...), so without this guard the
# Bitwarden status line gets printed 3-5x per startup.  Bitwarden's own
# in-process cache prevents redundant network calls, but the print, the
# config re-parse, and the ASCII sanitization sweep still ran every time.
_APPLIED_HOMES: set[str] = set()


class SecretSourceRefreshError(RuntimeError):
    """Sanitized fail-closed error for no-agent secret refreshes."""

    def __init__(self, *, source: str, source_type: str, stage: str) -> None:
        self.source = source
        self.source_type = source_type
        self.stage = stage
        super().__init__(
            f"secret source refresh failed "
            f"(source={source}, type={source_type}, stage={stage})"
        )


def get_secret_source(env_var: str) -> str | None:
    """Return the label of the secret source that supplied ``env_var``, if any.

    Returns ``"bitwarden"`` for keys pulled from Bitwarden Secrets Manager
    during the current process's ``load_hermes_dotenv()`` call.  Returns
    ``None`` for keys that came from ``.env``, the shell environment, or
    aren't tracked.  The returned label is metadata only: credential-pool
    persistence may store it to explain the origin of a borrowed secret, but
    must never treat it as authorization to persist the raw value.
    """
    return _SECRET_SOURCES.get(env_var)


def reset_secret_source_cache(hermes_home: Path | None = None) -> None:
    """Forget which HERMES_HOME paths have already had external secrets applied.

    The first call to ``_apply_external_secret_sources(home_path)`` in a
    process pulls from Bitwarden (or other configured backend), records the
    applied keys in ``_SECRET_SOURCES``, and remembers ``home_path`` so
    subsequent calls in the same process are no-ops.  Call this to force the
    next call to re-pull — useful for tests, and for long-running processes
    that want to refresh after a config change.
    """
    if hermes_home is None:
        # Backward-compatible test/startup reset. Callers that need a genuine
        # backend refresh must provide the exact home so another profile's L1
        # and L2 entries are not destroyed.
        _APPLIED_HOMES.clear()
        return

    canonical_home = hermes_home.expanduser().resolve()
    _APPLIED_HOMES.discard(str(canonical_home))
    from agent.secret_sources.registry import invalidate_caches

    invalidate_caches(canonical_home)


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
    # Ask the registry for the source's human label (e.g. "1Password").
    # Fall back to the raw source name for labels the registry doesn't
    # know (stale provenance from an uninstalled plugin, tests).
    try:
        from agent.secret_sources.registry import get_source

        registered = get_source(source)
        if registered is not None and registered.label:
            return f" (from {registered.label})"
    except Exception:  # noqa: BLE001 — label lookup must never raise
        pass
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
    _apply_external: bool = True,
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

    # Load .op.env AFTER .env so that .env values win, but the bootstrap
    # token (OP_SERVICE_ACCOUNT_TOKEN) becomes available for
    # apply_onepassword_secrets() even in cron / subprocess environments
    # that inherit no shell state (no systemd EnvironmentFile, no op run).
    # .op.env is gitignored — the service-account token never enters the
    # committed .env file.
    # Users on systemd can alternatively use:
    #   EnvironmentFile=-/path/to/.hermes/.op.env
    # in their gateway unit, which takes precedence (override=False below
    # ensures .op.env never clobbers a token already in the environment).
    op_env = home_path / ".op.env"
    if op_env.exists() and not os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"):
        _load_dotenv_with_fallback(op_env, override=False)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    if _apply_external:
        _apply_external_secret_sources(home_path)
    _apply_managed_env()

    return loaded


def _apply_managed_env() -> None:
    """Apply the managed-scope .env last, with override, so it beats user/shell.

    Managed scope is machine-global (independent of HERMES_HOME / profile). v1
    enforcement is "applied last with override=True" — at the end of startup load
    ``os.environ`` holds the managed value for every managed key, beating both the
    user ``.env`` and any pre-existing shell export. This deliberately inverts the
    usual env-over-config precedence for the pinned keys (see
    ``docs/design/managed-scope.md`` §4.1).

    This does NOT prevent the agent from later mutating ``os.environ`` in-process
    or ``export``-ing in a subprocess shell; that hard boundary is a documented
    v2 item (design §8.1). v1 relies on filesystem permissions only.

    Fail-open: a missing managed dir or .env is the common case and a no-op; any
    error here is swallowed so managed scope can never block startup.
    """
    try:
        from hermes_cli import managed_scope

        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — managed scope must never block startup
        return
    if managed_dir is None:
        return
    managed_env = managed_dir / ".env"
    if not managed_env.exists():
        return
    _sanitize_env_file_if_needed(managed_env)
    _load_dotenv_with_fallback(managed_env, override=True)


def _apply_external_secret_sources(home_path: Path) -> None:
    """Pull secrets from every enabled external source into env.

    Runs AFTER dotenv loads so .env values are visible (sources use them
    to locate bootstrap tokens) but BEFORE the rest of Hermes reads
    ``os.environ`` for credentials.  Any failure here is logged and
    swallowed — external secret sources must never block startup.

    The heavy lifting (source ordering, mapped-beats-bulk precedence,
    first-claim-wins conflict handling, override semantics, provenance)
    lives in ``agent.secret_sources.registry.apply_all``; this wrapper
    owns the once-per-HERMES_HOME guard, the post-apply ASCII
    sanitization sweep, the ``_SECRET_SOURCES`` provenance map that
    UI surfaces read, and the startup status lines.

    Idempotent within a process: subsequent calls for the same
    ``home_path`` are no-ops.  ``load_hermes_dotenv()`` runs at import
    time from several hot modules (cli.py, hermes_cli/main.py,
    run_agent.py, trajectory_compressor.py, ...), so without this guard
    the status lines would print 3-5x per CLI startup.  Use
    ``reset_secret_source_cache()`` if you need to force a re-pull
    (tests, long-running processes after a config change).
    """
    home_key = str(Path(home_path).resolve())
    if home_key in _APPLIED_HOMES:
        return
    _APPLIED_HOMES.add(home_key)

    try:
        cfg = _load_secrets_config(home_path)
    except Exception:  # noqa: BLE001 — config errors must not block startup
        return
    if not cfg:
        return

    try:
        from agent.secret_sources.registry import apply_all
    except ImportError:
        return

    try:
        report = apply_all(cfg, home_path)
    except Exception:  # noqa: BLE001 — belt-and-braces; apply_all shouldn't raise
        return

    if report.applied_any:
        # Re-run the ASCII sanitization pass: vault values are
        # user-supplied and might have the same copy-paste corruption as
        # a manually edited .env (see #6843).
        _sanitize_loaded_credentials()
        # Remember where each var came from so setup / `hermes model`
        # flows can label detected credentials with "(from Bitwarden)" /
        # "(from 1Password)" — otherwise users see "credentials ✓" with
        # no hint the value came from a vault rather than .env.
        for name, applied in report.provenance.items():
            _SECRET_SOURCES[name] = applied.source

    for src in report.sources:
        if src.applied:
            print(
                f"  {src.label}: applied {len(src.applied)} "
                f"secret{'s' if len(src.applied) != 1 else ''} "
                f"({', '.join(sorted(src.applied))})",
                file=sys.stderr,
            )
        if src.result.error:
            print(f"  {src.label}: {src.result.error}", file=sys.stderr)
        for warn in src.result.warnings:
            print(f"  {src.label}: {warn}", file=sys.stderr)
    for conflict in report.conflicts:
        print(f"  Secret sources: {conflict}", file=sys.stderr)


def refresh_hermes_dotenv_strict(
    *, hermes_home: str | os.PathLike
) -> list[Path]:
    """Refresh no-agent secrets or raise a sanitized fail-closed error.

    Ordinary startup remains fail-open through ``load_hermes_dotenv``. This
    path is intentionally stricter because launching a script with a stale
    externally sourced value is less safe than refusing to launch it.
    """
    home_path = Path(hermes_home).expanduser().resolve()
    _APPLIED_HOMES.discard(str(home_path))

    # Remove every value previously attributed to an external source before
    # dotenv or vault reload. The caller holds the cron refresh lock across
    # this removal, reload, and child snapshot.
    for name in tuple(_SECRET_SOURCES):
        os.environ.pop(name, None)
        _SECRET_SOURCES.pop(name, None)

    loaded = load_hermes_dotenv(hermes_home=home_path, _apply_external=False)

    try:
        cfg = _load_secrets_config_strict(home_path)
    except Exception as exc:
        if isinstance(exc, SecretSourceRefreshError):
            raise
        raise SecretSourceRefreshError(
            source="configuration", source_type="mapping", stage="config"
        ) from None

    try:
        from agent.secret_sources.registry import apply_all, list_sources

        registered = {source.name: source for source in list_sources()}
        enabled = []
        for name, source_cfg in cfg.items():
            if name == "sources":
                continue
            source = registered.get(name)
            if source is None:
                if isinstance(source_cfg, dict) and source_cfg.get("enabled") is True:
                    raise SecretSourceRefreshError(
                        source="unregistered", source_type="unregistered", stage="config"
                    )
                continue
            if not isinstance(source_cfg, dict):
                raise SecretSourceRefreshError(
                    source=source.name,
                    source_type=type(source).__name__,
                    stage="config",
                )
            try:
                if source.is_enabled(source_cfg):
                    enabled.append(source)
            except Exception:
                raise SecretSourceRefreshError(
                    source=source.name,
                    source_type=type(source).__name__,
                    stage="config",
                ) from None
    except SecretSourceRefreshError:
        raise
    except Exception:
        raise SecretSourceRefreshError(
            source="registry", source_type="registry", stage="config"
        ) from None

    if not enabled:
        return loaded

    for source in enabled:
        try:
            source.invalidate_cache(home_path)
        except Exception:
            raise SecretSourceRefreshError(
                source=source.name,
                source_type=type(source).__name__,
                stage="invalidate",
            ) from None

    working_env = os.environ.copy()
    try:
        report = apply_all(cfg, home_path, environ=working_env)
    except Exception:
        summary = ",".join(source.name for source in enabled)
        types = ",".join(type(source).__name__ for source in enabled)
        raise SecretSourceRefreshError(
            source=summary, source_type=types, stage="apply"
        ) from None

    source_types = {source.name: type(source).__name__ for source in enabled}
    for source_report in report.sources:
        if source_report.result.error or not source_report.result.ok:
            raise SecretSourceRefreshError(
                source=source_report.name,
                source_type=source_types.get(source_report.name, "registered"),
                stage="fetch",
            )

    if report.applied_any:
        for name in report.provenance:
            os.environ[name] = working_env[name]
        _sanitize_loaded_credentials()
        for name, applied in report.provenance.items():
            _SECRET_SOURCES[name] = applied.source
    return loaded


def _load_unique_yaml_mapping(config_path: Path) -> dict:
    """Load YAML recursively rejecting duplicate mapping keys."""
    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _construct_unique_mapping(
        loader: Any, node: Any, deep: bool = False
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate mapping key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.load(config_file, Loader=_UniqueKeyLoader) or {}
    if not isinstance(data, dict):
        raise ValueError("config root is not a mapping")
    return data


def _has_unresolved_env_reference(value: Any) -> bool:
    if isinstance(value, str):
        return re.search(r"\${[^}]+}", value) is not None
    if isinstance(value, dict):
        return any(_has_unresolved_env_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_unresolved_env_reference(item) for item in value)
    return False


def _load_secrets_config_strict(home_path: Path) -> dict:
    """Parse and expand secrets config, rejecting ambiguity or unresolved refs."""
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return {}

    from hermes_cli.config import _expand_env_vars

    data = _load_unique_yaml_mapping(config_path)
    secrets = data.get("secrets", {})
    if secrets is None:
        return {}
    if not isinstance(secrets, dict):
        raise ValueError("secrets is not a mapping")
    expanded = _expand_env_vars(secrets)
    if not isinstance(expanded, dict):
        raise ValueError("expanded secrets is not a mapping")
    if _has_unresolved_env_reference(expanded):
        raise ValueError("secrets contains an unresolved environment reference")
    return expanded


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
            data = fast_safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}
    return data.get("secrets") or {}
