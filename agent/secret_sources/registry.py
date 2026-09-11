"""Secret-source registry + apply orchestrator.

Owns everything that must be uniform across backends: registration
(name/scheme uniqueness, API-version gating), the wall-clock timeout around
``fetch()``, precedence (mapped beats bulk; within a shape ``secrets.sources``
order, else registration order; first claim wins), ``override_existing``
semantics (may beat .env/shell, never another source, never a protected var),
cross-source conflict warnings, and provenance. Startup entry point:
:func:`apply_all` via ``hermes_cli.env_loader``; plugins register through
``PluginContext.register_secret_source()`` → :func:`register_source`.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION, ErrorKind, FetchResult, SecretSource, is_valid_env_name,
    reset_source_environment, set_source_environment,
)
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

# Ordered registry: name → source. Dict insertion order doubles as the default
# apply order. Origin is recorded so consumers never infer ownership from names.
_SOURCES: Dict[str, SecretSource] = {}
_SOURCE_ORIGINS: Dict[str, str] = {}
_SCOPED_SOURCES: Dict[str, Dict[str, SecretSource]] = {}
_BUILTINS_LOADED = False
_REGISTRY_LOCK = threading.RLock()

# (module, class, label) for the bundled sources, in registration order.
_BUILTIN_SOURCES = (
    ("agent.secret_sources.bitwarden", "BitwardenSource", "Bitwarden"),
    ("agent.secret_sources.onepassword", "OnePasswordSource", "1Password"),
    ("agent.secret_sources.command", "CommandSource", "command"),
)


@dataclass
class AppliedVar:
    """Provenance record for one env var the orchestrator set."""

    name: str
    source: str          # SecretSource.name
    shape: str           # "mapped" | "bulk"
    overrode_env: bool   # replaced a pre-existing .env/shell value


@dataclass
class SourceReport:
    """One source's outcome within an :class:`ApplyReport`."""

    name: str
    label: str
    result: FetchResult
    applied: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)   # .env/shell won
    skipped_claimed: List[str] = field(default_factory=list)    # earlier source won
    skipped_protected: List[str] = field(default_factory=list)  # bootstrap-auth guard
    skipped_invalid: List[str] = field(default_factory=list)    # bad env-var name


@dataclass
class ApplyReport:
    """Merged outcome of one orchestrated apply pass."""

    sources: List[SourceReport] = field(default_factory=list)
    provenance: Dict[str, AppliedVar] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)  # human-readable warnings

    @property
    def applied_any(self) -> bool:
        return bool(self.provenance)


# --- Registration -----------------------------------------------------------


def _validate_source(source: SecretSource) -> Optional[str]:
    """Reason a source is unregistrable (already formatted for the log), or None."""
    if not isinstance(source, SecretSource):
        return f"Ignoring secret source {source!r}: does not inherit from SecretSource"
    name = source.name or ""
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        return f"Ignoring secret source with invalid name {name!r}"
    if source.api_version != SECRET_SOURCE_API_VERSION:
        return (f"Ignoring secret source '{name}': built against secret-source API "
                f"v{source.api_version}, this Hermes speaks v{SECRET_SOURCE_API_VERSION}")
    if source.shape not in ("mapped", "bulk"):
        return f"Ignoring secret source '{name}': shape must be 'mapped' or 'bulk', got {source.shape!r}"
    return None


def register_source(source: SecretSource, *, replace: bool = False, builtin: bool = False,
                    scope: Optional[str] = None) -> bool:
    """Register a secret source; True on success. Rejections are logged, never
    raised. ``replace`` allows same-name override (last-writer-wins); scheme
    collisions across *different* names are always rejected."""
    problem = _validate_source(source)
    if problem:
        logger.warning(problem)
        return False
    name = source.name
    with _REGISTRY_LOCK:
        effective = dict(_SOURCES)
        if scope is not None:
            effective.update(_SCOPED_SOURCES.get(scope, {}))
        if name in effective and not replace:
            logger.warning("Secret source '%s' already registered; ignoring duplicate", name)
            return False
        owner = next((n for n, o in effective.items()
                      if n != name and source.scheme and o.scheme == source.scheme), None)
        if owner:
            logger.warning("Ignoring secret source '%s': scheme '%s://' is already owned by source '%s'",
                           name, source.scheme, owner)
            return False
        target = _SOURCES if scope is None else _SCOPED_SOURCES.setdefault(scope, {})
        target[name] = source
        if scope is None:
            _SOURCE_ORIGINS[name] = "builtin" if builtin else "plugin"
    return True


def _merged(scope: Optional[str]) -> Dict[str, SecretSource]:
    """Global sources overlaid with the scope's (default: current home) registrations."""
    merged = dict(_SOURCES)
    merged.update(_SCOPED_SOURCES.get(scope or hermes_home_key(), {}))
    return merged


def get_source(name: str, *, scope: Optional[str] = None) -> Optional[SecretSource]:
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        return _merged(scope).get(name)


def snapshot_registration(name: str, *, scope: Optional[str] = None) -> Optional[SecretSource]:
    """Return the registration owned by exactly one registry layer."""
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        return (_SOURCES if scope is None else _SCOPED_SOURCES.get(scope, {})).get(name)


def restore_registration(name: str, current: SecretSource, previous: Optional[SecretSource], *,
                         scope: Optional[str] = None) -> bool:
    """Restore a host-owned source registration if it is still current."""
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        target = _SOURCES if scope is None else _SCOPED_SOURCES.setdefault(scope, {})
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if scope is not None and not target:
            _SCOPED_SOURCES.pop(scope, None)
    return True


def list_sources(*, scope: Optional[str] = None) -> List[SecretSource]:
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        return list(_merged(scope).values())


def list_plugin_sources() -> List[SecretSource]:
    """Sources registered outside the bundled set: global ``"plugin"`` origins
    plus every scoped registration (bundled sources register with scope=None).

    Includes both legacy global plugin registrations (``_SOURCE_ORIGINS == "plugin"``) and the current
    scope's profile-keyed registrations — every scoped entry is plugin-registered by definition, since
    bundled sources register with ``scope=None`` (#64229 profile isolation).
    """
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        merged = {n: s for n, s in _SOURCES.items() if _SOURCE_ORIGINS.get(n) == "plugin"}
        merged.update(_SCOPED_SOURCES.get(hermes_home_key(), {}))
        return list(merged.values())


def _ensure_builtin_sources() -> None:
    """Idempotently register the bundled sources (lazy so import stays cheap;
    per-source guarded so one broken source can't block the others)."""
    global _BUILTINS_LOADED
    with _REGISTRY_LOCK:
        if _BUILTINS_LOADED:
            return
        _BUILTINS_LOADED = True
        for module_name, class_name, label in _BUILTIN_SOURCES:
            try:
                module = __import__(module_name, fromlist=[class_name])
                register_source(getattr(module, class_name)(), builtin=True)
            except Exception:  # noqa: BLE001 — never block startup
                logger.warning("Failed to register bundled %s secret source", label, exc_info=True)


def _reset_registry_for_tests() -> None:
    global _BUILTINS_LOADED
    with _REGISTRY_LOCK:
        _SOURCES.clear()
        _SOURCE_ORIGINS.clear()
        _SCOPED_SOURCES.clear()
        _BUILTINS_LOADED = False


# --- Orchestrated apply -----------------------------------------------------


def _fetch_with_timeout(source: SecretSource, cfg: dict, home_path: Path,
                        environ: MutableMapping[str, str]) -> FetchResult:
    """Run source.fetch() under a wall-clock budget; never raises.

    A worker thread enforces the budget: a source that blows it is reported as
    TIMEOUT and its eventual result discarded (the thread may linger until
    process exit — acceptable for a startup-only path).
    """
    timeout = source.fetch_timeout_seconds(cfg)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"secret-src-{source.name}")
    try:
        def _fetch() -> FetchResult:
            token = set_source_environment(environ)
            try:
                return source.fetch(cfg, home_path)
            finally:
                reset_source_environment(token)

        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return FetchResult().fail(f"fetch exceeded {timeout:.0f}s budget — startup continued "
                                      "without this source (raise secrets."
                                      f"{source.name}.timeout_seconds if the backend is just slow)",
                                      ErrorKind.TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — contract violation, contain it
            return FetchResult().fail(f"fetch raised {type(exc).__name__}: {exc}", ErrorKind.INTERNAL)
    finally:
        executor.shutdown(wait=False)

    if not isinstance(result, FetchResult):
        return FetchResult().fail(f"fetch returned {type(result).__name__} instead of FetchResult",
                                  ErrorKind.INTERNAL)
    return result


def _section(secrets_cfg: dict, name: str) -> dict:
    cfg = secrets_cfg.get(name)
    return cfg if isinstance(cfg, dict) else {}


def _ordered_enabled_sources(secrets_cfg: dict, *, scope: Optional[str] = None) -> List[SecretSource]:
    """Enabled sources: ``secrets.sources`` order first, then registration order
    (mapped-vs-bulk precedence is applied on top by :func:`apply_all`)."""
    sources = {source.name: source for source in list_sources(scope=scope)}

    explicit = secrets_cfg.get("sources")
    names = [e for e in explicit if isinstance(e, str)] if isinstance(explicit, list) else []
    unknown = [n for n in names if n not in sources]
    if unknown:
        logger.warning("secrets.sources names unknown source(s): %s (known: %s)",
                       ", ".join(unknown), ", ".join(sources) or "none")
    order = dict.fromkeys([n for n in names if n in sources] + list(sources))  # insertion-ordered set

    enabled: List[SecretSource] = []
    for name in order:
        try:
            if sources[name].is_enabled(_section(secrets_cfg, name)):
                enabled.append(sources[name])
        except Exception:  # noqa: BLE001
            logger.warning("Secret source '%s' is_enabled() raised; skipping", name, exc_info=True)
    return enabled


def _active_profile_name(home_path: Optional[Path]) -> str:
    """Active profile name (``~/.hermes/profiles/<name>``); "" for the default profile."""
    if home_path is not None:
        resolved = Path(home_path)
        if resolved.parent.name == "profiles" and resolved.name:
            return resolved.name
    for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        value = os.environ.get(env_name, "").strip()
        if value and value != "default":
            return value
    return ""


# Only credential-shaped names get auto-aliased — a random profile-suffixed
# var should not silently hydrate an unsuffixed name.
_ALIAS_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY", "_PASSWORD")


def _profile_alias_target(var: str, profile: str) -> Optional[str]:
    """Map ``FOO_<PROFILE>`` to ``FOO`` for the active profile when safe."""
    suffix = "_" + profile.replace("-", "_").upper()
    if not profile or not var.endswith(suffix):
        return None
    alias = var[: -len(suffix)]
    return alias if alias and is_valid_env_name(alias) and alias.endswith(_ALIAS_SUFFIXES) else None


class _Applier:
    """Apply phase state for one orchestrated pass: sequential, first-wins, attributed."""

    def __init__(self, env: MutableMapping[str, str], report: ApplyReport,
                 protected: Dict[str, str], preserve: frozenset) -> None:
        self.env, self.report, self.protected, self.preserve = env, report, protected, preserve
        self.claimed: Dict[str, str] = {}  # var → source name that won it

    def apply_source(self, source: SecretSource, cfg: dict, result: FetchResult,
                     profile: str, supplied_directly: set) -> None:
        sr = SourceReport(name=source.name, label=source.label or source.name, result=result)
        self.report.sources.append(sr)
        if not result.ok:
            return
        try:
            override = source.override_existing(cfg)
        except Exception:  # noqa: BLE001
            override = False

        for var, value in result.secrets.items():
            if not isinstance(var, str) or not isinstance(value, str):
                continue
            if not self._try_apply(sr, source, override, var, value) or not profile:
                continue
            alias = _profile_alias_target(var, profile)
            if (alias and alias not in supplied_directly and alias not in self.claimed
                    and self._try_apply(sr, source, override, alias, value)):
                result.warnings.append(f"applied profile-scoped {var} as {alias} (active profile {profile!r})")

    def _try_apply(self, sr: SourceReport, source: SecretSource, override: bool,
                   var: str, value: str) -> bool:
        """Apply one var through the shared guard chain. True = applied."""
        if not is_valid_env_name(var):
            sr.skipped_invalid.append(var)
            return False
        if var in self.protected:
            sr.skipped_protected.append(var)
            return False
        if var in self.claimed:
            sr.skipped_claimed.append(var)
            self.report.conflicts.append(f"{var}: kept value from {self.claimed[var]}; "
                                         f"{source.name} also supplies it (first source wins — "
                                         "remove one binding or reorder secrets.sources)")
            return False
        existed = bool(self.env.get(var))
        if existed and (var in self.preserve or not override):
            sr.skipped_existing.append(var)
            return False
        self.env[var] = value
        self.claimed[var] = source.name
        sr.applied.append(var)
        self.report.provenance[var] = AppliedVar(var, source.name, source.shape, overrode_env=existed)
        return True


def apply_all(secrets_cfg: dict, home_path: Path,
              environ: Optional[MutableMapping[str, str]] = None) -> ApplyReport:
    """Fetch from every enabled source and apply the merged result to ``environ``
    (default ``os.environ``).

    Precedence per env var, most-specific intent first: (1) ``secrets.preserve_existing``
    names always keep a pre-existing value, even against ``override_existing: true``;
    (2) pre-existing .env/shell value, unless the winning source has
    ``override_existing: true``; (3) mapped sources in configured order; (4) bulk
    sources in configured order. First claim wins: a later source carrying the same
    var gets ``skipped_claimed`` plus a conflict warning — never a silent clobber,
    and ``override_existing`` never applies across sources.

    Profile aliasing: under a named profile an applied ``FOO_<PROFILE>``
    (credential-shaped suffixes only) also hydrates canonical ``FOO``, under the
    same guards; disabled with ``secrets.profile_alias: false``.

    1. 2. 3. 4. See #58073.
    See #51447.
    """
    env = environ if environ is not None else os.environ
    report = ApplyReport()
    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}
    enabled = _ordered_enabled_sources(secrets_cfg, scope=hermes_home_key(home_path))
    if not enabled:
        return report

    preserve_raw = secrets_cfg.get("preserve_existing")
    preserve = frozenset(n.strip() for n in preserve_raw if isinstance(n, str) and n.strip()
                         ) if isinstance(preserve_raw, list) else frozenset()
    profile = _active_profile_name(home_path) if secrets_cfg.get("profile_alias", True) else ""

    # Mapped outranks bulk regardless of list order.
    ordered = [s for s in enabled if s.shape == "mapped"] + [s for s in enabled if s.shape == "bulk"]

    fetches: List[tuple[SecretSource, dict, FetchResult]] = []
    protected: Dict[str, str] = {}  # var → source that protects it
    for source in ordered:
        cfg = _section(secrets_cfg, source.name)
        result = _fetch_with_timeout(source, cfg, home_path, env)
        fetches.append((source, cfg, result))
        try:
            for var in source.protected_env_vars(cfg):
                protected.setdefault(var, source.name)
        except Exception:  # noqa: BLE001
            pass

    # An alias never shadows a var some source supplies by its real name.
    supplied_directly = {v for _, _, r in fetches if r.ok for v in r.secrets if isinstance(v, str)}

    applier = _Applier(env, report, protected, preserve)
    for source, cfg, result in fetches:
        applier.apply_source(source, cfg, result, profile, supplied_directly)
    return report
