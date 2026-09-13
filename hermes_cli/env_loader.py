"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import codecs
import io
import logging
import os
import sys
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path

from dotenv.main import DotEnv
from dotenv.variables import parse_variables
from utils import atomic_replace, fast_safe_load

logger = logging.getLogger(__name__)

# The ONLY env vars sanitized on load: credentials must be pure ASCII (they become HTTP header values);
# arbitrary user env vars are never silently altered.
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")

# Once-per-process guards: load_hermes_dotenv() runs repeatedly (user + project env, gateway hot-reload,
# lazy imports mid-turn, tests) so warnings/logs fire once per key/path/home.
_WARNED_KEYS: set[str] = set()          # credential names already given the non-ASCII warning
_WARNED_UTF32_PATHS: set[str] = set()   # .env paths already given the UTF-32 refuse-to-mangle warning
_SCOPED_SKIP_LOGGED: set[str] = set()   # routed profile homes whose multiplex dotenv skip was logged

# env-var name → source label ("bitwarden", …) for externally injected credentials; setup / `hermes
# model` tell users WHERE a key came from when .env lacks it.
_SECRET_SOURCES: dict[str, str] = {}
# Immutable per-home snapshots: os.environ is shared across profiles and a later home's apply may overwrite it.
_SECRET_SOURCE_VALUES_BY_HOME: dict[str, dict[str, str]] = {}
# HERMES_HOME paths already pulled external secrets for: load_hermes_dotenv() runs at import time from
# several hot modules, so without this the Bitwarden status line prints 3-5x per startup and the config
# re-parse + ASCII sweep re-run each time (Bitwarden's own cache only saves the network call).
_APPLIED_HOMES: set[str] = set()
_SECRET_SOURCE_CACHE_LOCK = threading.RLock()

# Reload scope → (pre-dotenv baseline, values published by the preceding load). A new load peels off
# its own preceding output only when it is still present. This keeps cyclic interpolation idempotent
# without freezing changes made by the shell, an earlier dotenv layer, or the terminal config bridge.
_DOTENV_RELOAD_STATES: dict[str, tuple[dict[str, str | None], dict[str, str]]] = {}
_DOTENV_LOAD_LOCK = threading.RLock()

# Behavioral routing keys a parent Hermes process injects into child env that silently redirect a profile
# onto the wrong provider path; these — and ONLY these — are scrubbed at startup when absent from the
# profile's .env. Credentials are excluded: shell exports are a documented way to supply them, and
# read-time secret-scope checks (agent/secret_scope.py) own cross-profile credential isolation.
_PROFILE_MANAGED_ENV_KEYS: frozenset[str] = frozenset({
    "HERMES_ACP_AUTH_METHOD", "HERMES_ACP_AUTO_APPROVE", "HERMES_COPILOT_ACP_COMMAND",
    "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH", "COPILOT_ACP_BASE_URL",
})


def _env_keys_defined_in_dotenv(path: Path) -> set[str]:
    """KEY names assigned in a dotenv file (including empty ``KEY=``), via the same tokenizer that installs
    profile scopes — a key the installer sees is a key the dashboard scrub sees (BOM'd first line included)."""
    from agent.secret_scope import load_env_file

    return set(load_env_file(path))


def _clear_known_keys_missing_from_dotenv(
    path: Path, *, environ: MutableMapping[str, str] | None = None
) -> set[str]:
    """After ``.env`` loaded with override, delete inherited ``_PROFILE_MANAGED_ENV_KEYS`` it does not
    define. Deliberately NARROW: only keys that change *which provider path* is used.

    Does **not** run when the ``.env`` file does not exist (bare-profile case, which follows ``#66930`` /
    ``#67027`` semantics).
    """
    if not path.exists():
        return set()
    target = os.environ if environ is None else environ
    defined = _env_keys_defined_in_dotenv(path)
    removed: set[str] = set()
    for key in _PROFILE_MANAGED_ENV_KEYS:
        if key not in defined and key in target:
            del target[key]
            removed.add(key)
    return removed


def get_secret_source(env_var: str) -> str | None:
    """Source label that supplied ``env_var`` (``"bitwarden"`` …), None for .env/shell keys. Metadata only —
    never authorization to persist the raw value."""
    return _SECRET_SOURCES.get(env_var)


def secret_source_names() -> tuple[str, ...]:
    """Every env-var name some profile's external secret source supplied (names only — the map is
    process-wide, so a value must be resolved through the active profile's secret scope)."""
    return tuple(_SECRET_SOURCES)


def get_secret_source_values(hermes_home: str | os.PathLike) -> dict[str, str]:
    """Return the external-secret value snapshot for ``hermes_home``."""
    return dict(_SECRET_SOURCE_VALUES_BY_HOME.get(str(Path(hermes_home).resolve()), {}))


def hydrate_profile_secret_sources(hermes_home: str | os.PathLike) -> dict[str, str]:
    """Resolve one profile's configured sources without mutating ``os.environ``: multiplex gateways route
    turns to profiles that never ran the process-global dotenv path, so resolve against a private mapping
    seeded from that ``.env`` and record the per-home snapshot for ``build_profile_secret_scope()``.
    Fail-open / once-per-home like ``_apply_external_secret_sources``; never returns plaintext .env entries."""
    with _SECRET_SOURCE_CACHE_LOCK:
        return _hydrate_profile_secret_sources(Path(hermes_home))


def _hydrate_profile_secret_sources(home: Path) -> dict[str, str]:
    """Locked implementation for :func:`hydrate_profile_secret_sources`."""
    home_key = str(home.resolve())
    if home_key in _APPLIED_HOMES:
        return get_secret_source_values(home)

    try:
        cfg = _load_secrets_config(home)
    except Exception:  # noqa: BLE001 — external sources must not block routing
        return {}
    if not cfg:
        return {}

    try:
        from agent.secret_scope import _is_global_env, load_env_file
        from agent.secret_sources.registry import apply_all

        local_env = {name: value for name, value in os.environ.items() if _is_global_env(name)}
        local_env.update(load_env_file(home / ".env"))
        # Mirror load_hermes_dotenv()'s .op.env bootstrap (1Password token lives in gitignored .op.env)
        # or cold profiles fail 1Password hydration. .env wins.
        # Without seeding it here a cold profile configured for the supported .op.env flow fails 1Password
        # hydration (sweeper review on #74549). .env values win — never override an existing key.
        op_env = home / ".op.env"
        if op_env.exists():
            for _name, _value in load_env_file(op_env).items():
                local_env.setdefault(_name, _value)
        local_env["HERMES_HOME"] = str(home)
        report = apply_all(cfg, home, environ=local_env)
    except Exception:  # noqa: BLE001 — preserve fail-open startup behavior
        return {}

    if not report.sources:
        return {}

    _APPLIED_HOMES.add(home_key)
    values: dict[str, str] = {}
    for name, applied in report.provenance.items():
        value = local_env.get(name)
        if value is None:
            continue
        _SECRET_SOURCES[name] = applied.source
        values[name] = value
    if values:
        _SECRET_SOURCE_VALUES_BY_HOME[home_key] = values
    return dict(values)


def reset_secret_source_cache(hermes_home: str | os.PathLike | None = None) -> None:
    """Forget applied homes so the next load re-pulls (tests, long-running processes after config edits).

    ``hermes_home`` limits the reset to ONE home: a multiplex gateway keeps every profile's snapshot in
    this process, and a per-fire cron re-pull or a plugin-discovery refresh for one home must not wipe
    a sibling's hydrated snapshot — the sibling's next scope build would run empty until it re-hydrated
    (#102041)."""
    if hermes_home is None:
        _APPLIED_HOMES.clear()
        _SECRET_SOURCES.clear()
        _SECRET_SOURCE_VALUES_BY_HOME.clear()
        return
    home_key = str(Path(hermes_home).resolve())
    _APPLIED_HOMES.discard(home_key)
    _SECRET_SOURCE_VALUES_BY_HOME.pop(home_key, None)


def format_secret_source_suffix(env_var: str) -> str:
    """``" (from Bitwarden)"``-style suffix; ``""`` for .env/shell keys (only external sources are named)."""
    source = get_secret_source(env_var)
    if not source:
        return ""
    if source == "bitwarden":
        return " (from Bitwarden)"
    # Registry label (e.g. "1Password"); raw name for unknown sources (uninstalled plugin, tests).
    try:
        from agent.secret_sources.registry import get_source

        registered = get_source(source)
        if registered is not None and registered.label:
            return f" (from {registered.label})"
    except Exception:  # noqa: BLE001 — label lookup must never raise
        pass
    return f" (from {source})"


def _format_offending_chars(value: str, limit: int = 3) -> str:
    """Compact ``U+XXXX ('c'), ...`` summary of non-ASCII codepoints."""
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
    """Strip non-ASCII from credential env vars (``_CREDENTIAL_SUFFIXES``) so the codebase never sees them.

    Emits a one-line warning to stderr when characters are stripped. Silent stripping would mask copy-paste
    corruption (Unicode lookalike glyphs from PDFs / rich-text editors, ZWSP from web pages) as opaque
    provider-side "invalid API key" errors (see #6843).
    """
    for key, value in list(os.environ.items()):
        if not any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES):
            continue
        if value.isascii():
            continue
        cleaned = value.encode("ascii", errors="ignore").decode("ascii")
        os.environ[key] = cleaned
        if key in _WARNED_KEYS:
            continue
        _WARNED_KEYS.add(key)
        stripped = len(value) - len(cleaned)
        detail = _format_offending_chars(value) or "non-printable"
        print(f"  Warning: {key} contained {stripped} non-ASCII character"
              f"{'s' if stripped != 1 else ''} ({detail}) — stripped so the "
              f"key can be sent as an HTTP header.", file=sys.stderr)
        print(
            "  This usually means the key was copy-pasted from a PDF, "
            "rich-text editor, or web page that substituted lookalike\n"
            "  Unicode glyphs for ASCII letters. If authentication fails "
            "(e.g. \"API key not valid\"), re-copy the key from the\n"
            "  provider's dashboard and run `hermes setup` (or edit the "
            ".env file in a plain-text editor).",
            file=sys.stderr,
        )


def _normal_env_key(name: str, *, case_insensitive: bool) -> str:
    return name.upper() if case_insensitive else name


def _normalized_environment(
    environ: Mapping[str, str | None], *, case_insensitive: bool
) -> dict[str, str | None]:
    return {
        _normal_env_key(name, case_insensitive=case_insensitive): value
        for name, value in environ.items()
    }


class _InterpolationEnvironment(dict[str, str | None]):
    """Mapping used by python-dotenv atoms with Windows-compatible key lookup."""

    def __init__(self, values: Mapping[str, str | None], *, case_insensitive: bool):
        self._case_insensitive = case_insensitive
        super().__init__(
            _normalized_environment(values, case_insensitive=case_insensitive)
        )

    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(
            _normal_env_key(key, case_insensitive=self._case_insensitive), default
        )


def _read_dotenv_assignments(path: Path) -> list[tuple[str, str | None]]:
    """Read and decode one immutable snapshot, then parse it without interpolation."""
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Strip a UTF-8 BOM by hand before the existing Latin-1 fallback.
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw[len(codecs.BOM_UTF8) :]
        decoded = raw.decode("latin-1")
    parser = DotEnv(
        dotenv_path=None,
        stream=io.StringIO(decoded),
        interpolate=False,
        override=True,
    )
    return list(parser.parse())


def _resolve_dotenv_assignments(
    assignments: Iterable[tuple[str, str | None]],
    *,
    environ: Mapping[str, str],
    override: bool,
    case_insensitive: bool,
) -> dict[str, str | None]:
    """Resolve python-dotenv atoms against a private mapping, preserving its precedence."""
    current = _normalized_environment(environ, case_insensitive=case_insensitive)
    resolved_by_key: dict[str, str | None] = {}
    resolved: dict[str, str | None] = {}

    for name, value in assignments:
        if value is None:
            result = None
        else:
            if override:
                interpolation_values = {**current, **resolved_by_key}
            else:
                interpolation_values = {**resolved_by_key, **current}
            interpolation_env = _InterpolationEnvironment(
                interpolation_values, case_insensitive=case_insensitive
            )
            result = "".join(
                atom.resolve(interpolation_env) for atom in parse_variables(value)
            )

        normalized_name = _normal_env_key(
            name, case_insensitive=case_insensitive
        )
        resolved_by_key[normalized_name] = result
        resolved[name] = result

    return resolved


def _set_environment_value(
    environ: MutableMapping[str, str],
    name: str,
    value: str,
    *,
    case_insensitive: bool,
) -> None:
    if case_insensitive:
        normalized_name = _normal_env_key(name, case_insensitive=True)
        for existing in tuple(environ):
            if _normal_env_key(existing, case_insensitive=True) == normalized_name:
                if existing != name:
                    del environ[existing]
                break
    environ[name] = value


def _load_dotenv_into(
    path: Path,
    *,
    environ: MutableMapping[str, str],
    override: bool,
    case_insensitive: bool,
) -> set[str]:
    assignments = _read_dotenv_assignments(path)
    resolved = _resolve_dotenv_assignments(
        assignments,
        environ=environ,
        override=override,
        case_insensitive=case_insensitive,
    )
    applied: set[str] = set()
    existing = _normalized_environment(environ, case_insensitive=case_insensitive)
    for name, value in resolved.items():
        normalized_name = _normal_env_key(name, case_insensitive=case_insensitive)
        if value is None or (not override and normalized_name in existing):
            continue
        _set_environment_value(
            environ, name, value, case_insensitive=case_insensitive
        )
        existing[normalized_name] = value
        applied.add(name)
    return applied


def _prepare_reload_environment(
    scope: str, *, case_insensitive: bool
) -> dict[str, str]:
    """Return a private environment with this scope's prior output peeled off."""
    prepared = dict(os.environ)
    state = _DOTENV_RELOAD_STATES.get(scope)
    if state is None:
        return prepared
    baseline, published = state
    current = _normalized_environment(prepared, case_insensitive=case_insensitive)
    for normalized_name, previous_value in published.items():
        if current.get(normalized_name) != previous_value:
            continue
        for existing in tuple(prepared):
            if _normal_env_key(existing, case_insensitive=case_insensitive) == normalized_name:
                del prepared[existing]
        baseline_value = baseline.get(normalized_name)
        if baseline_value is not None:
            prepared[normalized_name] = baseline_value
    return prepared


def _publish_environment(
    environ: Mapping[str, str], names: Iterable[str], *, case_insensitive: bool
) -> None:
    normalized = _normalized_environment(environ, case_insensitive=case_insensitive)
    for name in names:
        value = normalized.get(
            _normal_env_key(name, case_insensitive=case_insensitive)
        )
        if value is not None:
            _set_environment_value(
                os.environ, name, value, case_insensitive=case_insensitive
            )


def _record_reload_state(
    scope: str,
    *,
    baseline_environ: Mapping[str, str],
    names: Iterable[str],
    case_insensitive: bool,
) -> None:
    baseline_values = _normalized_environment(
        baseline_environ, case_insensitive=case_insensitive
    )
    current = _normalized_environment(os.environ, case_insensitive=case_insensitive)
    normalized_names = {
        _normal_env_key(name, case_insensitive=case_insensitive) for name in names
    }
    _DOTENV_RELOAD_STATES[scope] = (
        {name: baseline_values.get(name) for name in normalized_names},
        {
            name: current[name]
            for name in normalized_names
            if current.get(name) is not None
        },
    )


def _load_dotenv_with_fallback(
    path: Path,
    *,
    override: bool,
    environ: MutableMapping[str, str] | None = None,
    case_insensitive: bool | None = None,
) -> set[str]:
    """Load one snapshot privately; publish only completed values when no mapping is supplied."""
    if case_insensitive is None:
        case_insensitive = os.name == "nt"
    if environ is not None:
        return _load_dotenv_into(
            path,
            environ=environ,
            override=override,
            case_insensitive=case_insensitive,
        )

    scope = f"dotenv:{path.resolve()}"
    with _DOTENV_LOAD_LOCK:
        private_env = _prepare_reload_environment(
            scope, case_insensitive=case_insensitive
        )
        baseline_env = dict(private_env)
        applied = _load_dotenv_into(
            path,
            environ=private_env,
            override=override,
            case_insensitive=case_insensitive,
        )
        _publish_environment(
            private_env, applied, case_insensitive=case_insensitive
        )
        _record_reload_state(
            scope,
            baseline_environ=baseline_env,
            names=applied,
            case_insensitive=case_insensitive,
        )
        _sanitize_loaded_credentials()  # httpx encodes headers as ASCII
        return applied


def _sanitize_env_file_if_needed(path: Path) -> None:
    """Pre-sanitize a .env file before python-dotenv reads it. Sniffs a leading BOM *before* any text
    decode: UTF-16 (Notepad "Unicode") is rewritten as clean UTF-8; UTF-32 is refused (left untouched) so
    we never fall through to the errors=replace corruption path."""
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

    # ORDER MATTERS: BOM_UTF32_LE (FF FE 00 00) startswith BOM_UTF16_LE (FF FE); UTF-16 first would mangle it.
    force_utf8_rewrite = False
    if raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        # Lazy import keeps the module import block identical to #65124's codecs/io additions so the two PRs
        # auto-merge either order.
        path_key = str(path.resolve())
        if path_key not in _WARNED_UTF32_PATHS:
            _WARNED_UTF32_PATHS.add(path_key)
            logger.warning("Skipping .env sanitize for %s: UTF-32 BOM detected; "
                           "leaving file untouched to avoid corruption", path)
        return
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        # "utf-16" uses the BOM for endianness and strips it; newline=None matches open()'s universal
        # newlines (not splitlines()'s extra boundaries like U+2028) so sanitize sees the same lines.
        try:
            with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-16", newline=None) as f:
                original = f.readlines()
        except UnicodeDecodeError:
            return
        force_utf8_rewrite = True  # always rewrite UTF-16 as UTF-8 so the dotenv load sees a canonical file
    else:
        # utf-8-sig strips a UTF-8 BOM; errors=replace so embedded NULs can be stripped below.
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                original = f.readlines()
        except Exception:
            return
        # errors=replace turns undecodable leading bytes into U+FFFD; persisting would glue them onto
        # the first key name permanently — leave the file untouched instead.
        if original and original[0].startswith("\ufffd"):
            return

    try:
        # Strip NULs (os.environ raises ValueError on them); also repairs BOM-less UTF-16 (NUL-padded ASCII).
        stripped = [line.replace("\x00", "") for line in original]
        sanitized = _sanitize_env_lines(stripped)
        if sanitized != original or force_utf8_rewrite:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".env_")
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
    load_external_secrets: bool = True,
) -> list[Path]:
    """Load Hermes env files: ``~/.hermes/.env`` overrides stale shell exports; project ``.env`` is a dev
    fallback that only fills gaps when the user env exists (and overrides shell vars when it does not)."""
    # Process home on purpose (never the per-turn override): a startup .env load must not follow a routed
    # profile — see the multiplex guard below.
    from hermes_constants import get_process_hermes_home
    home_path = Path(hermes_home) if hermes_home else get_process_hermes_home()

    # Multiplex gateway: while a routed profile-home override is active, copying that profile's .env
    # into os.environ would expose its credentials to sibling turns and every spawned child. Unscoped
    # startup loads keep the normal path; external sources still refresh against the profile mapping.
    from agent.secret_scope import is_multiplex_active
    from hermes_constants import get_hermes_home_override

    if is_multiplex_active() and get_hermes_home_override() is not None:
        home_key = str(home_path.resolve())
        if home_key not in _SCOPED_SKIP_LOGGED:
            _SCOPED_SKIP_LOGGED.add(home_key)
            logger.debug("multiplex: skipping process-global dotenv load for routed "
                         "profile home %s (credentials resolve via the profile scope)", home_path)
        if load_external_secrets:
            from hermes_cli import _early_recovery

            if not _early_recovery._should_skip_external_secret_sources():
                hydrate_profile_secret_sources(home_path)
        return []

    loaded: list[Path] = []
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None
    op_env = home_path / ".op.env"
    case_insensitive = os.name == "nt"
    project_key = str(project_env_path.resolve()) if project_env_path else ""
    reload_scope = f"hermes:{home_path.resolve()}:{project_key}"

    # One reload scope spans every dotenv layer. Resolving each file independently would let a managed
    # self-reference mistake the preceding managed result for a newly changed lower layer. Keep the whole
    # operation serialized and private, then publish completed values only.
    with _DOTENV_LOAD_LOCK:
        private_env = _prepare_reload_environment(
            reload_scope, case_insensitive=case_insensitive
        )
        baseline_env = dict(private_env)
        dotenv_names: set[str] = set()
        removed_names: set[str] = set()

        if user_env.exists():  # normalize formatting / strip NULs before parsing
            _sanitize_env_file_if_needed(user_env)
        if project_env_path and project_env_path.exists():
            _sanitize_env_file_if_needed(project_env_path)

        if user_env.exists():
            dotenv_names.update(
                _load_dotenv_with_fallback(
                    user_env, override=True, environ=private_env
                )
            )
            loaded.append(user_env)
            removed_names.update(
                _clear_known_keys_missing_from_dotenv(
                    user_env, environ=private_env
                )
            )

        # .op.env AFTER .env so .env wins, but the bootstrap OP_SERVICE_ACCOUNT_TOKEN reaches
        # apply_onepassword_secrets() even in cron with no shell state; gitignored so the token never enters
        # the committed .env. override=False lets a systemd `EnvironmentFile=-…/.op.env` token win.
        if op_env.exists() and not private_env.get("OP_SERVICE_ACCOUNT_TOKEN"):
            dotenv_names.update(
                _load_dotenv_with_fallback(
                    op_env, override=False, environ=private_env
                )
            )

        if project_env_path and project_env_path.exists():
            dotenv_names.update(
                _load_dotenv_with_fallback(
                    project_env_path, override=not loaded, environ=private_env
                )
            )
            loaded.append(project_env_path)

        _publish_environment(
            private_env, dotenv_names, case_insensitive=case_insensitive
        )
        for name in removed_names:
            normalized_name = _normal_env_key(
                name, case_insensitive=case_insensitive
            )
            for existing in tuple(os.environ):
                if (
                    _normal_env_key(
                        existing, case_insensitive=case_insensitive
                    )
                    == normalized_name
                ):
                    del os.environ[existing]
        _sanitize_loaded_credentials()

        # External sources are skipped for the updater (dotenv + managed env still load): ``update`` must
        # not import optional secret-manager native libraries into the process replacing that environment.
        from hermes_cli import _early_recovery

        if (
            load_external_secrets
            and not _early_recovery._should_skip_external_secret_sources()
        ):
            _apply_external_secret_sources(home_path)

        managed_env = dict(os.environ)
        managed_names = _apply_managed_env(environ=managed_env)
        _publish_environment(
            managed_env, managed_names, case_insensitive=case_insensitive
        )
        _sanitize_loaded_credentials()

        # config.yaml owns terminal.*. Re-apply its explicit keys LAST, after the managed overlay, so a
        # stale dotenv value cannot flip the backend in a long-lived process (#29186, #67323).
        _reapply_terminal_config_bridge(home_path)
        _record_reload_state(
            reload_scope,
            baseline_environ=baseline_env,
            names=dotenv_names | managed_names | removed_names,
            case_insensitive=case_insensitive,
        )

    return loaded


def _reapply_terminal_config_bridge(home_path: Path) -> None:
    """Re-assert config.yaml's explicit ``terminal.*`` keys over reloaded .env via the single shared bridge
    ``apply_terminal_config_to_env`` (also used by terminal_tool and the TUI/dashboard launchers) so the
    semantics can't drift between sites."""
    try:
        if Path(home_path).resolve() != _process_hermes_home().resolve():
            return
        from hermes_cli.config import apply_terminal_config_to_env

        apply_terminal_config_to_env(env=None)
    except Exception:  # noqa: BLE001 — early bootstrap / malformed config
        pass


def _apply_managed_env(
    *, environ: MutableMapping[str, str] | None = None
) -> set[str]:
    """Apply the managed-scope .env last, with override, so it beats user/shell. Does NOT stop the agent
    from later mutating os.environ (v1 relies on filesystem permissions). Fail-open: never blocks startup."""
    try:
        from hermes_cli import managed_scope

        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — managed scope must never block startup
        return set()
    if managed_dir is None:
        return set()
    managed_env = managed_dir / ".env"
    if not managed_env.exists():
        return set()
    _sanitize_env_file_if_needed(managed_env)
    return _load_dotenv_with_fallback(
        managed_env, override=True, environ=environ
    )


def _apply_external_secret_sources(home_path: Path) -> None:
    """Pull secrets from every enabled external source into env — AFTER dotenv (sources need .env bootstrap
    tokens), BEFORE Hermes reads credentials; failures never block startup. Precedence/conflicts/provenance
    live in ``registry.apply_all``; this wrapper owns the once-per-home guard, the post-apply ASCII sweep,
    the ``_SECRET_SOURCES`` map and status lines."""
    home_key = str(Path(home_path).resolve())
    if home_key in _APPLIED_HOMES:
        return

    # Neither early return marks the home applied: a malformed config.yaml would otherwise permanently
    # disable secret loading for this process, and an unmarked home picks up a config change on the next
    # load (the re-parse is a cheap fast_safe_load).
    try:
        cfg = _load_secrets_config(home_path)
    except Exception:  # noqa: BLE001 — config errors must not block startup
        # See #40597.
        return
    if not cfg:
        return

    # Defer the registry import until a source is enabled — bitwarden eagerly loads cryptography._rust.pyd,
    # which makes the Windows updater self-lock before its preflight. Detect by *shape* (dict with enabled
    # flag), not names, so plugin/test sources pass and a plain dict entry never forces the crypto load.
    any_enabled = any(isinstance(v, dict) and v.get("enabled") is True for v in cfg.values())
    if not any_enabled:
        return

    try:
        from agent.secret_sources.registry import apply_all
    except ImportError:
        return

    try:
        report = apply_all(cfg, home_path)
    except Exception:  # noqa: BLE001 — belt-and-braces; apply_all shouldn't raise
        return

    if not report.sources:  # no source enabled: keep retrying cheaply so flipping one on takes effect
        return

    # A real fetch attempt happened (success OR error): mark the home so the 3-5 import-time calls per
    # startup don't re-fetch / re-print (error retries are opt-in via reset_secret_source_cache()).
    # Marking AFTER the attempt keeps the earlier failure paths retryable.
    _APPLIED_HOMES.add(home_key)

    if report.applied_any:
        _sanitize_loaded_credentials()  # vault values carry the same copy-paste corruption risk as .env
        for name, applied in report.provenance.items():
            _SECRET_SOURCES[name] = applied.source

    # Snapshot EVERY name a source supplied, not just the newly applied ones. A name the source supplied
    # but the pre-existing process value won (``skipped_existing``) is still this home's effective value
    # for it — and on the unscoped path os.environ IS this home's environment. Skipping those names
    # latched an EMPTY snapshot whenever the key was already in the env: after the first cron/plugin
    # re-pull (the previous apply's own write-back shadows every key) or from boot under systemd
    # ``EnvironmentFile=``. Under multiplex the scope is the only credential source, so an empty
    # snapshot failed every default-profile turn for the process lifetime (#102041).
    values: dict[str, str] = {}
    supplied = set(report.provenance)
    for src in report.sources:
        supplied.update(src.skipped_existing)
    for name in supplied:
        if name in os.environ:
            values[name] = os.environ[name]
    if values:
        _SECRET_SOURCE_VALUES_BY_HOME[home_key] = values

    for src in report.sources:
        if src.applied:
            print(f"  {src.label}: applied {len(src.applied)} "
                  f"secret{'s' if len(src.applied) != 1 else ''}", file=sys.stderr)
        if src.result.error:
            print(f"  {src.label}: {src.result.error}", file=sys.stderr)
            hint = _remediation_hint(src.name, src.result.error_kind, cfg, scope=home_key)
            if hint:
                print(f"  {src.label}: → {hint}", file=sys.stderr)
        for warn in src.result.warnings:
            print(f"  {src.label}: {warn}", file=sys.stderr)
    for conflict in report.conflicts:
        print(f"  Secret sources: {conflict}", file=sys.stderr)


def _remediation_hint(source_name: str, error_kind, secrets_cfg: dict, *, scope: str | None = None) -> str:
    """The failed source's one-line fix-it hint; a plugin remediation() could raise and startup must not."""
    try:
        from agent.secret_sources.registry import get_source

        source = get_source(source_name, scope=scope)
        if source is None:
            return ""
        src_cfg = secrets_cfg.get(source_name)
        src_cfg = src_cfg if isinstance(src_cfg, dict) else {}
        return str(source.remediation(error_kind, src_cfg) or "").strip()
    except Exception:  # noqa: BLE001 — hints must never block startup
        return ""


def _load_secrets_config(home_path: Path) -> dict:
    """Read just the ``secrets:`` section of config.yaml, isolated so a malformed config can't break dotenv."""
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return {}
    # Prefer the shared raw-config cache: this is the first config.yaml read of a normal startup, so
    # populating it lets main.py's early bridge and hermes_logging reuse one parse instead of 3-4.
    if home_path == _process_hermes_home():
        try:
            from hermes_cli.config import read_raw_config

            data = read_raw_config() or {}
            return data.get("secrets") or {}
        except Exception:
            pass
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = fast_safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}
    return data.get("secrets") or {}


def _process_hermes_home() -> Path:
    """The HERMES_HOME the running process was launched under.

    Must be the *true* process home, ignoring any context-local
    ``set_hermes_home_override`` a per-request task has installed. Both
    callers depend on that:

    * ``_reapply_terminal_config_bridge`` guards "only re-bridge config into
      the shared ``os.environ`` when THIS load is for the process's own
      profile". If this followed the task override, a per-turn handler scoped
      to a *secondary* profile (the multiplex dashboard serving every profile
      from one process) would satisfy the guard and bridge that profile's
      ``terminal.backend`` into the shared env — e.g. a ``local`` sibling
      profile clobbering the launch profile's ``TERMINAL_ENV=ssh`` while
      leaving its ``TERMINAL_SSH_*`` untouched, so the session silently runs
      commands locally (cross-profile terminal-backend leak).
    * ``_load_secrets_config`` uses it to decide whether the shared
      (mtime,size)-keyed config cache is safe to reuse; under an override it
      must fall through to an isolated parse of the scoped profile.

    ``hermes_constants.get_process_hermes_home()`` is the override-immune
    resolver built for exactly this; delegate to it.
    """
    try:
        from hermes_constants import get_process_hermes_home

        return get_process_hermes_home()
    except Exception:
        return Path.home() / ".hermes"
