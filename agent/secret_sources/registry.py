"""Secret-source registry + apply orchestrator.

This module owns everything that must be uniform across secret backends
so no individual source can get it wrong:

* registration (name/scheme uniqueness, API-version gating)
* per-source wall-clock timeout enforcement around ``fetch()``
* precedence: mapped sources beat bulk sources; within a shape,
  ``secrets.sources`` order (or registration order) decides; first
  claim wins — later sources never silently clobber an earlier one
* ``override_existing`` semantics (may beat .env/shell, never another
  secret source, never a protected var)
* cross-source conflict warnings (shadowed claims are always surfaced)
* provenance: which source supplied every applied var

The single entry point for startup is :func:`apply_all`, called from
``hermes_cli.env_loader._apply_external_secret_sources()``.

Plugins register additional sources via
``PluginContext.register_secret_source()`` which lands in
:func:`register_source`.  In-tree sources are registered lazily by
:func:`_ensure_builtin_sources` — the set of bundled sources is
deliberately closed (Bitwarden, and 1Password once it lands); new
third-party backends ship as standalone plugin repos implementing
:class:`agent.secret_sources.base.SecretSource`.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    is_valid_env_name,
)

logger = logging.getLogger(__name__)

# Ordered registry: name → source instance.  Python dicts preserve
# insertion order, which doubles as the default apply order.
_SOURCES: Dict[str, SecretSource] = {}
_BUILTINS_LOADED = False


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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_source(source: SecretSource, *, replace: bool = False) -> bool:
    """Register a secret source.  Returns True on success.

    Rejections are logged, never raised — a bad plugin must not take
    down startup.  ``replace`` allows tests / user plugins to override
    a bundled source of the same name (last-writer-wins like model
    providers), but scheme collisions across *different* names are
    always rejected.
    """
    if not isinstance(source, SecretSource):
        logger.warning(
            "Ignoring secret source %r: does not inherit from SecretSource",
            source,
        )
        return False
    name = getattr(source, "name", "") or ""
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        logger.warning("Ignoring secret source with invalid name %r", name)
        return False
    if getattr(source, "api_version", None) != SECRET_SOURCE_API_VERSION:
        logger.warning(
            "Ignoring secret source '%s': built against secret-source API v%s, "
            "this Hermes speaks v%s",
            name, getattr(source, "api_version", "?"), SECRET_SOURCE_API_VERSION,
        )
        return False
    if getattr(source, "shape", None) not in ("mapped", "bulk"):
        logger.warning(
            "Ignoring secret source '%s': shape must be 'mapped' or 'bulk', got %r",
            name, getattr(source, "shape", None),
        )
        return False
    if name in _SOURCES and not replace:
        logger.warning("Secret source '%s' already registered; ignoring duplicate", name)
        return False
    scheme = getattr(source, "scheme", None)
    if scheme:
        for other_name, other in _SOURCES.items():
            if other_name != name and getattr(other, "scheme", None) == scheme:
                logger.warning(
                    "Ignoring secret source '%s': scheme '%s://' is already "
                    "owned by source '%s'",
                    name, scheme, other_name,
                )
                return False
    _SOURCES[name] = source
    return True


def get_source(name: str) -> Optional[SecretSource]:
    _ensure_builtin_sources()
    return _SOURCES.get(name)


def list_sources() -> List[SecretSource]:
    _ensure_builtin_sources()
    return list(_SOURCES.values())


def _ensure_builtin_sources() -> None:
    """Idempotently register the bundled sources.

    Lazy so importing this module stays cheap and so a broken bundled
    source can never break registration of the others.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from agent.secret_sources.bitwarden import BitwardenSource

        register_source(BitwardenSource())
    except Exception:  # noqa: BLE001 — never block startup
        logger.warning("Failed to register bundled Bitwarden secret source",
                       exc_info=True)
    try:
        from agent.secret_sources.onepassword import OnePasswordSource

        register_source(OnePasswordSource())
    except Exception:  # noqa: BLE001 — never block startup
        logger.warning("Failed to register bundled 1Password secret source",
                       exc_info=True)


def _reset_registry_for_tests() -> None:
    global _BUILTINS_LOADED
    _SOURCES.clear()
    _BUILTINS_LOADED = False


# ---------------------------------------------------------------------------
# Orchestrated apply
# ---------------------------------------------------------------------------


def _fetch_with_timeout(
    source: SecretSource, cfg: dict, home_path: Path
) -> FetchResult:
    """Run source.fetch() under a wall-clock budget; never raises.

    The budget is enforced with a daemon worker thread: a source that
    blows its budget is reported as ``TIMEOUT`` and its (eventual)
    result is discarded.  The thread itself may linger until process
    exit — acceptable for a startup-only path, and strictly better than
    an unbounded hang on every ``hermes`` invocation.
    """
    timeout = source.fetch_timeout_seconds(cfg)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"secret-src-{source.name}"
    )
    try:
        future = executor.submit(source.fetch, cfg, home_path)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            res = FetchResult()
            res.error = (
                f"fetch exceeded {timeout:.0f}s budget — startup continued "
                "without this source (raise secrets."
                f"{source.name}.timeout_seconds if the backend is just slow)"
            )
            res.error_kind = ErrorKind.TIMEOUT
            return res
        except Exception as exc:  # noqa: BLE001 — contract violation, contain it
            res = FetchResult()
            res.error = f"fetch raised {type(exc).__name__}: {exc}"
            res.error_kind = ErrorKind.INTERNAL
            return res
    finally:
        executor.shutdown(wait=False)

    if not isinstance(result, FetchResult):
        res = FetchResult()
        res.error = (
            f"fetch returned {type(result).__name__} instead of FetchResult"
        )
        res.error_kind = ErrorKind.INTERNAL
        return res
    return result


def _ordered_enabled_sources(secrets_cfg: dict) -> List[SecretSource]:
    """Resolve which sources run, in which order.

    Order: the optional ``secrets.sources`` list wins; sources not named
    there follow in registration order.  Enabled = the source's own
    ``is_enabled`` says so for its config section.  Mapped-vs-bulk
    precedence is applied on top of this order by :func:`apply_all`.
    """
    _ensure_builtin_sources()

    explicit = secrets_cfg.get("sources")
    order: List[str] = []
    if isinstance(explicit, list):
        for entry in explicit:
            if isinstance(entry, str) and entry in _SOURCES and entry not in order:
                order.append(entry)
        unknown = [e for e in explicit
                   if isinstance(e, str) and e not in _SOURCES]
        if unknown:
            logger.warning(
                "secrets.sources names unknown source(s): %s (known: %s)",
                ", ".join(unknown), ", ".join(_SOURCES) or "none",
            )
    for name in _SOURCES:
        if name not in order:
            order.append(name)

    enabled: List[SecretSource] = []
    for name in order:
        source = _SOURCES[name]
        cfg = secrets_cfg.get(name)
        cfg = cfg if isinstance(cfg, dict) else {}
        try:
            if source.is_enabled(cfg):
                enabled.append(source)
        except Exception:  # noqa: BLE001
            logger.warning("Secret source '%s' is_enabled() raised; skipping",
                           name, exc_info=True)
    return enabled


def apply_all(secrets_cfg: dict, home_path: Path,
              environ: Optional[Dict[str, str]] = None) -> ApplyReport:
    """Fetch from every enabled source and apply the merged result to env.

    ``environ`` defaults to ``os.environ``; injectable for tests.

    Precedence per env var (most-specific intent wins):

    1. Pre-existing env (.env / shell) — unless the winning source has
       ``override_existing: true``.
    2. Mapped sources, in configured order.
    3. Bulk sources, in configured order.

    First claim wins.  A later source that also carries the var gets a
    ``skipped_claimed`` entry and a conflict warning — never a silent
    clobber, and ``override_existing`` never applies across sources.
    """
    import os as _os

    env = environ if environ is not None else _os.environ
    report = ApplyReport()

    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}
    enabled = _ordered_enabled_sources(secrets_cfg)
    if not enabled:
        return report

    # Mapped sources outrank bulk sources regardless of list order:
    # an explicit VAR→ref binding is stronger intent than a project dump.
    ordered = ([s for s in enabled if s.shape == "mapped"]
               + [s for s in enabled if s.shape == "bulk"])

    # Fetch phase.
    fetches: List[tuple[SecretSource, dict, FetchResult]] = []
    protected: Dict[str, str] = {}  # var → source that protects it
    for source in ordered:
        cfg = secrets_cfg.get(source.name)
        cfg = cfg if isinstance(cfg, dict) else {}
        result = _fetch_with_timeout(source, cfg, home_path)
        fetches.append((source, cfg, result))
        try:
            for var in source.protected_env_vars(cfg):
                protected.setdefault(var, source.name)
        except Exception:  # noqa: BLE001
            pass

    # Apply phase — sequential, first-wins, fully attributed.
    claimed: Dict[str, str] = {}  # var → source name that won it
    for source, cfg, result in fetches:
        sr = SourceReport(name=source.name,
                          label=source.label or source.name,
                          result=result)
        report.sources.append(sr)
        if not result.ok:
            continue

        try:
            override = source.override_existing(cfg)
        except Exception:  # noqa: BLE001
            override = False

        for var, value in result.secrets.items():
            if not isinstance(var, str) or not isinstance(value, str):
                continue
            if not is_valid_env_name(var):
                sr.skipped_invalid.append(var)
                continue
            if var in protected:
                sr.skipped_protected.append(var)
                continue
            if var in claimed:
                sr.skipped_claimed.append(var)
                report.conflicts.append(
                    f"{var}: kept value from {claimed[var]}; "
                    f"{source.name} also supplies it (first source wins — "
                    "remove one binding or reorder secrets.sources)"
                )
                continue
            existed = bool(env.get(var))
            if existed and not override:
                sr.skipped_existing.append(var)
                continue
            env[var] = value
            claimed[var] = source.name
            sr.applied.append(var)
            report.provenance[var] = AppliedVar(
                name=var,
                source=source.name,
                shape=source.shape,
                overrode_env=existed,
            )

    return report
