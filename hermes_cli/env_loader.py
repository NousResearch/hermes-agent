"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import codecs
import io
import os
import stat
import sys
from pathlib import Path

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

# Paths we've already emitted a UTF-32 refuse-to-mangle warning for.
# load_hermes_dotenv can call _sanitize_env_file_if_needed multiple times
# for the same file (user env + project env + hot-reload); once per path
# is enough.
_WARNED_UTF32_PATHS: set[str] = set()

# Project-local dotenv files are repository-controlled input. Even for a CWD
# that the user explicitly trusts in config.yaml, these names must retain
# authority from the process, ~/.hermes/.env, or config.yaml. In particular,
# blocking every HERMES_* key keeps import-time safety snapshots (approval,
# redaction, sandbox controls) outside the repository trust boundary.
_CWD_ENV_BLOCKED_PREFIXES = (
    "HERMES_",
    "_HERMES_",
    "TERMINAL_",
    "MESSAGING_",
    "SECURITY_",
    "APPROVAL_",
    "SMART_APPROVAL_",
    "AUXILIARY_APPROVAL_",
    "BROWSER_",
    "TIRITH_",
    "YOLO_",
    "REDACT_",
    "DYLD_",
    "LD_",
    "PYTHON",
    "BASH_",
    "GIT_",
    "SSH_",
    "PIP_",
    "UV_",
)
_CWD_ENV_BLOCKED_NAMES = frozenset(
    {
        "BASH_ENV",
        "BASHOPTS",
        "CDPATH",
        "CLASSPATH",
        "CURL_CA_BUNDLE",
        "DOCKER_HOST",
        "ENV",
        "GIT_SSH_COMMAND",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "IFS",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "_JAVA_OPTIONS",
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        "NODE_PATH",
        "NO_PROXY",
        "PATH",
        "PERL5OPT",
        "PERL5LIB",
        "PS4",
        "REQUESTS_CA_BUNDLE",
        "RUBYOPT",
        "RUBYLIB",
        "SHELLOPTS",
        "SSL_CERT_DIR",
        "SHELL",
        "SSL_CERT_FILE",
        "SSLKEYLOGFILE",
        "SUDO_PASSWORD",
        "TMPDIR",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WSS_PROXY",
    }
)
_WARNED_CWD_ENV_KEYS: set[tuple[str, str]] = set()
_MISSING_ENV = object()

# Map of env-var name → source label ("bitwarden", etc.) for credentials
# that were injected by an external secret source during load_hermes_dotenv().
# Used by setup / `hermes model` flows to label detected credentials so
# users understand WHERE a key came from when their .env doesn't contain it
# directly (otherwise the "credentials detected ✓" line looks identical to
# the .env case and they don't know Bitwarden is wired up).
_SECRET_SOURCES: dict[str, str] = {}
# Applied values are immutable per-home snapshots.  ``os.environ`` is shared
# across profiles and may be overwritten by a later home's source apply.
_SECRET_SOURCE_VALUES_BY_HOME: dict[str, dict[str, str]] = {}

# HERMES_HOME paths we've already pulled external secrets for during this
# process.  ``load_hermes_dotenv()`` is called at module-import time from
# several hot modules (cli.py, hermes_cli/main.py, run_agent.py,
# trajectory_compressor.py, gateway/run.py, ...), so without this guard the
# Bitwarden status line gets printed 3-5x per startup.  Bitwarden's own
# in-process cache prevents redundant network calls, but the print, the
# config re-parse, and the ASCII sanitization sweep still ran every time.
_APPLIED_HOMES: set[str] = set()


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


def get_secret_source_values(
    hermes_home: str | os.PathLike,
) -> dict[str, str]:
    """Return the external-secret value snapshot for ``hermes_home``."""
    home_key = str(Path(hermes_home).resolve())
    return dict(_SECRET_SOURCE_VALUES_BY_HOME.get(home_key, {}))


def reset_secret_source_cache() -> None:
    """Forget which HERMES_HOME paths have already had external secrets applied.

    The first call to ``_apply_external_secret_sources(home_path)`` in a
    process pulls from Bitwarden (or other configured backend), records the
    applied keys in ``_SECRET_SOURCES``, and remembers ``home_path`` so
    subsequent calls in the same process are no-ops.  Call this to force the
    next call to re-pull — useful for tests, and for long-running processes
    that want to refresh after a config change.
    """
    _APPLIED_HOMES.clear()
    _SECRET_SOURCES.clear()
    _SECRET_SOURCE_VALUES_BY_HOME.clear()


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

    Strips embedded null bytes which crash ``os.environ[k] = v``
    with ``ValueError: embedded null byte`` — typically introduced by
    copy-pasting API keys from terminals or rich-text editors.

    Encoding: sniffs a leading BOM *before* any text decode. UTF-16
    (Notepad "Unicode") is decoded correctly and rewritten as clean
    UTF-8. UTF-32 is refused (left untouched) so we never fall through
    to the errors=replace corruption path. Order of BOM checks matters:
    UTF-32-LE's BOM starts with UTF-16-LE's FF FE.

    ``hermes_cli.config._sanitize_env_lines`` normalizes line endings while
    treating content after the first ``=`` as opaque for boundary discovery.
    """
    if not path.exists():
        return
    try:
        from hermes_cli.config import _sanitize_env_lines
    except ImportError:
        return  # early bootstrap — config module not available yet

    try:
        raw = path.read_bytes()
    except Exception:
        return

    # Sniff leading BOM bytes BEFORE decoding. ORDER MATTERS:
    # codecs.BOM_UTF32_LE is FF FE 00 00, which startswith
    # codecs.BOM_UTF16_LE (FF FE). Checking UTF-16 first would
    # misdetect UTF-32-LE as UTF-16-LE and mangle the file.
    force_utf8_rewrite = False
    if raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        # Lazy import keeps the module import block identical to #65124's
        # codecs/io additions so the two PRs auto-merge either order.
        path_key = str(path.resolve())
        if path_key not in _WARNED_UTF32_PATHS:
            _WARNED_UTF32_PATHS.add(path_key)
            import logging

            logging.getLogger(__name__).warning(
                "Skipping .env sanitize for %s: UTF-32 BOM detected; "
                "leaving file untouched to avoid corruption",
                path,
            )
        return
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        # "utf-16" uses the BOM to select endianness and strips it.
        # TextIOWrapper + newline=None matches open()'s universal-newlines
        # line splitting (\\n/\\r\\n/\\r only — not splitlines()'s extra
        # Unicode boundaries like U+2028), so sanitize sees the same lines
        # as the UTF-8 path.
        try:
            with io.TextIOWrapper(
                io.BytesIO(raw), encoding="utf-16", newline=None
            ) as f:
                original = f.readlines()
        except UnicodeDecodeError:
            return
        # Source is UTF-16 on disk; always rewrite as clean UTF-8 so
        # the subsequent utf-8 dotenv load sees a canonical file.
        force_utf8_rewrite = True
    else:
        # Default path: utf-8-sig (strips UTF-8 BOM if present) with
        # errors=replace so embedded NULs can be stripped below.
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                original = f.readlines()
        except Exception:
            return
        # Defense-in-depth: errors=replace turns undecodable leading
        # bytes into U+FFFD. Persisting that glues replacement chars
        # onto the first key name and rewrites the file permanently
        # (the UTF-16-with-BOM corruption path before BOM sniffing).
        # Leave the file untouched rather than write the mangling.
        if original and original[0].startswith("\ufffd"):
            return

    try:
        # Strip null bytes before _sanitize_env_lines so they never
        # reach python-dotenv (which passes them to os.environ and
        # crashes with ValueError). Also intentionally repairs
        # BOM-less UTF-16 (NUL-padded ASCII) into clean UTF-8.
        stripped = [line.replace("\x00", "") for line in original]
        sanitized = _sanitize_env_lines(stripped)
        if sanitized != original or force_utf8_rewrite:
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


def _trusted_cwd_env_path(
    home_path: Path,
    cwd_path: Path | None = None,
    *,
    use_terminal_cwd: bool = False,
) -> Path | None:
    """Return an opted-in CWD .env path, or None outside the trust boundary.

    Trust is configured only in the user-controlled ``config.yaml``. A marker
    inside the repository cannot opt itself in. Entries are exact directories,
    resolved before comparison so a symlink cannot silently widen trust.
    """
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = fast_safe_load(f) or {}
    except Exception:
        return None
    try:
        from hermes_cli import managed_scope

        config = managed_scope.apply_managed_overlay(config)
    except Exception:
        # Managed scope is optional and must never make startup fail.
        pass
    dotenv_config = config.get("dotenv")
    if not isinstance(dotenv_config, dict):
        return None
    trusted_cwds = dotenv_config.get("trusted_cwds")
    if not isinstance(trusted_cwds, list):
        return None

    if cwd_path is not None:
        candidate_dir = cwd_path.expanduser().resolve()
    elif use_terminal_cwd:
        terminal_config = config.get("terminal")
        configured_cwd = (
            terminal_config.get("cwd")
            if isinstance(terminal_config, dict)
            else None
        )
        terminal_backend = (
            str(
                terminal_config.get("backend")
                or terminal_config.get("env_type")
                or "local"
            ).strip().lower()
            if isinstance(terminal_config, dict)
            else "local"
        )
        if (
            isinstance(configured_cwd, str)
            and configured_cwd not in ("", ".", "auto", "cwd")
        ):
            configured_path = Path(
                os.path.expandvars(configured_cwd)
            ).expanduser()
            candidate_dir = (
                configured_path.resolve()
                if configured_path.is_absolute()
                else Path.cwd().resolve()
            )
        elif terminal_backend == "local":
            candidate_dir = Path(
                os.environ.get("MESSAGING_CWD") or Path.home()
            ).expanduser().resolve()
        else:
            # A placeholder for a remote/container backend has no local host
            # directory whose .env can be read safely.
            return None
    else:
        # The local CLI/TUI contract is `cd /dir && hermes`; cli.load_cli_config
        # intentionally replaces terminal.cwd with os.getcwd() for this mode.
        candidate_dir = Path.cwd().resolve()
    for raw_path in trusted_cwds:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        # Trust entries are literal absolute paths. Environment expansion or
        # ``~`` would make trust depend on mutable process state.
        trusted_path = Path(raw_path)
        if (
            not trusted_path.is_absolute()
            or "$" in raw_path
            or "%" in raw_path
            or "~" in raw_path
        ):
            continue
        try:
            if trusted_path.resolve() == candidate_dir:
                candidate = candidate_dir / ".env"
                if candidate.is_symlink():
                    return None
                return candidate if candidate.is_file() else None
        except OSError:
            continue
    return None


def _cwd_env_key_allowed(name: str) -> bool:
    upper_name = name.upper()
    if upper_name in _CWD_ENV_BLOCKED_NAMES:
        return False
    return not upper_name.startswith(_CWD_ENV_BLOCKED_PREFIXES)


def _read_trusted_cwd_dotenv(path: Path) -> dict[str, str | None]:
    """Read a regular non-symlink dotenv file without interpolation."""
    from dotenv import dotenv_values

    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise OSError(f"not a regular file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not os.path.samestat(before, opened) or not stat.S_ISREG(opened.st_mode):
            raise OSError(f"dotenv file changed while opening: {path}")
        raw = b""
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return dict(dotenv_values(stream=io.StringIO(text), interpolate=False))


def _load_trusted_cwd_dotenv(
    path: Path,
    *,
    override: bool,
    protected_names: set[str] | None = None,
) -> None:
    """Load allowed values from an explicitly trusted project dotenv file."""
    values = _read_trusted_cwd_dotenv(path)
    protected_names = protected_names or set()

    path_key = str(path.resolve())
    for name, value in values.items():
        if value is None:
            continue
        if not _cwd_env_key_allowed(name):
            warning_key = (path_key, name.upper())
            if warning_key not in _WARNED_CWD_ENV_KEYS:
                _WARNED_CWD_ENV_KEYS.add(warning_key)
                print(
                    f"  Warning: ignored protected variable {name} from {path}",
                    file=sys.stderr,
                )
            continue
        if name.upper() in protected_names:
            continue
        if override or name not in os.environ:
            os.environ[name] = value
    _sanitize_loaded_credentials()


def _dotenv_names(path: Path) -> set[str]:
    """Return case-normalized names declared by a trusted user dotenv file."""
    from dotenv import dotenv_values

    try:
        values = dotenv_values(dotenv_path=path, encoding="utf-8")
    except UnicodeDecodeError:
        values = dotenv_values(dotenv_path=path, encoding="latin-1")
    # Windows environment-variable identity is case-insensitive. Use the same
    # conservative identity on every platform so a project cannot bypass user
    # ownership with a case variant.
    return {name.upper() for name in values}


def load_hermes_dotenv(
    *,
    hermes_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
    cwd_env: bool = False,
    cwd_path: str | os.PathLike | None = None,
    use_terminal_cwd: bool = False,
) -> list[Path]:
    """Load Hermes environment files with explicit source trust boundaries.

    Priority:
    - names declared in ``~/.hermes/.env`` have highest dotenv authority.
    - an opted-in CWD ``.env`` may override inherited shell values for other
      names; protected process and Hermes control variables are always ignored.
    - inherited values beat the source-tree ``project_env`` developer fallback
      whenever a user or CWD dotenv source exists.

    CWD loading is disabled unless ``cwd_env`` is true and the exact resolved
    directory appears under ``dotenv.trusted_cwds`` in the user's config.yaml.
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None
    requested_cwd = Path(cwd_path) if cwd_path is not None else None

    # Normalize user and developer files before parsing. Project-local CWD
    # files are parsed without rewriting repository content.
    if user_env.exists():
        _sanitize_env_file_if_needed(user_env)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    gateway_owned_env: dict[str, str | object] = {
        "_HERMES_GATEWAY": os.environ.get("_HERMES_GATEWAY", _MISSING_ENV)
    }
    if os.environ.get("_HERMES_GATEWAY") == "1":
        # TERMINAL_CWD belongs to the gateway config bridge. Preserve both its
        # value and absence across startup, credential reloads, and lazy imports
        # of cli.py/run_agent.py. The internal gateway marker itself is never
        # user-dotenv-owned, even outside a gateway process.
        gateway_owned_env["TERMINAL_CWD"] = os.environ.get(
            "TERMINAL_CWD", _MISSING_ENV
        )

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)
    for name, value in gateway_owned_env.items():
        if value is _MISSING_ENV:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    user_env_names = _dotenv_names(user_env) if user_env.exists() else set()

    # Resolve the project directory only after ~/.hermes/.env is loaded so a
    # gateway placeholder can see the existing MESSAGING_CWD compatibility
    # input. Managed terminal config is applied inside the resolver.
    cwd_env_path = (
        _trusted_cwd_env_path(
            home_path,
            requested_cwd,
            use_terminal_cwd=use_terminal_cwd,
        )
        if cwd_env
        else None
    )

    if (
        cwd_env_path is not None
        and cwd_env_path.resolve() != user_env.resolve()
        and (
            project_env_path is None
            or cwd_env_path.resolve() != project_env_path.resolve()
        )
    ):
        # The opted-in project may override inherited shell values, but never
        # names explicitly owned by the user's ~/.hermes/.env.
        _load_trusted_cwd_dotenv(
            cwd_env_path,
            override=True,
            protected_names=user_env_names,
        )
        loaded.append(cwd_env_path)

    # Load .op.env AFTER user/project dotenv sources so that their values win,
    # while the bootstrap token (OP_SERVICE_ACCOUNT_TOKEN) becomes available for
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
        # Source-tree dotenv is a fallback whenever a higher-trust dotenv
        # source exists. Preserve its legacy shell-override behavior only when
        # it is the sole dotenv source.
        _load_dotenv_with_fallback(
            project_env_path,
            override=not (user_env.exists() or cwd_env_path is not None),
        )
        loaded.append(project_env_path)

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

    try:
        cfg = _load_secrets_config(home_path)
    except Exception:  # noqa: BLE001 — config errors must not block startup
        # Deliberately NOT marked applied: a malformed config.yaml would
        # otherwise permanently disable secret loading for this process
        # even after the user fixes the file (#40597).
        return
    if not cfg:
        # No secrets section (or everything disabled at parse level).  Not
        # marked applied either — the re-parse is a cheap fast_safe_load and
        # leaving the home unmarked lets a process pick up a config change
        # on its next load_hermes_dotenv() call instead of never.
        return

    try:
        from agent.secret_sources.registry import apply_all
    except ImportError:
        return

    try:
        report = apply_all(cfg, home_path)
    except Exception:  # noqa: BLE001 — belt-and-braces; apply_all shouldn't raise
        return

    if not report.sources:
        # Config parsed but no source is enabled: keep retrying cheaply
        # (no fetch happens for disabled sources) so flipping a source on
        # mid-process takes effect on the next call.
        return

    # A real fetch attempt happened (success OR error).  Mark the home now
    # so the 3-5 import-time load_hermes_dotenv() calls per startup don't
    # re-fetch / re-print — error retries within one process are opt-in via
    # reset_secret_source_cache().  Marking AFTER the attempt (not before,
    # see #40597) is what lets the earlier failure paths stay retryable.
    _APPLIED_HOMES.add(home_key)

    if report.applied_any:
        # Re-run the ASCII sanitization pass: vault values are
        # user-supplied and might have the same copy-paste corruption as
        # a manually edited .env (see #6843).
        _sanitize_loaded_credentials()
        # Remember where each var came from so setup / `hermes model`
        # flows can label detected credentials with "(from Bitwarden)" /
        # "(from 1Password)" — otherwise users see "credentials ✓" with
        # no hint the value came from a vault rather than .env.
        values: dict[str, str] = {}
        for name, applied in report.provenance.items():
            _SECRET_SOURCES[name] = applied.source
            if name in os.environ:
                values[name] = os.environ[name]
        _SECRET_SOURCE_VALUES_BY_HOME[home_key] = values

    for src in report.sources:
        if src.applied:
            print(
                f"  {src.label}: applied {len(src.applied)} "
                f"secret{'s' if len(src.applied) != 1 else ''}",
                file=sys.stderr,
            )
        if src.result.error:
            print(f"  {src.label}: {src.result.error}", file=sys.stderr)
            hint = _remediation_hint(src.name, src.result.error_kind, cfg)
            if hint:
                print(f"  {src.label}: → {hint}", file=sys.stderr)
        for warn in src.result.warnings:
            print(f"  {src.label}: {warn}", file=sys.stderr)
    for conflict in report.conflicts:
        print(f"  Secret sources: {conflict}", file=sys.stderr)


def _remediation_hint(source_name: str, error_kind, secrets_cfg: dict) -> str:
    """Ask the failed source for its one-line fix-it hint.

    Defensive wrapper: remediation() is a pure mapping and shouldn't
    raise, but a plugin source could — and startup must never break on
    a status line.
    """
    try:
        from agent.secret_sources.registry import get_source

        source = get_source(source_name)
        if source is None:
            return ""
        src_cfg = secrets_cfg.get(source_name)
        src_cfg = src_cfg if isinstance(src_cfg, dict) else {}
        return str(source.remediation(error_kind, src_cfg) or "").strip()
    except Exception:  # noqa: BLE001 — hints must never block startup
        return ""


def _load_secrets_config(home_path: Path) -> dict:
    """Read just the ``secrets:`` section out of config.yaml.

    Imported lazily and isolated from the main config loader so a
    malformed config can't take down dotenv loading entirely.
    """
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return {}
    # Prefer the shared (mtime, size)-keyed raw-config cache — this is the
    # first config.yaml read in a normal `hermes` startup, so populating the
    # shared cache here lets main.py's early bridge and hermes_logging reuse
    # the same parse (one parse per process instead of 3-4). Falls back to a
    # direct isolated parse if the shared reader is unavailable, preserving
    # the "malformed config can't take down dotenv loading" property (the
    # shared reader also swallows parse errors and returns {}).
    if home_path == _process_hermes_home():
        try:
            from hermes_cli.config import read_raw_config

            data = read_raw_config() or {}
            return data.get("secrets") or {}
        except Exception:
            pass
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


def _process_hermes_home() -> Path:
    """The HERMES_HOME the shared config cache is keyed to."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"
