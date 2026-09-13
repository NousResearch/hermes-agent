"""Current-profile verifier discovery for callback-first Chronos cold starts."""

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Optional

from hermes_constants import get_default_hermes_root

_log = logging.getLogger("cron.chronos")
# A failed local read gets the same backoff advertised to NAS; no history is retained.
_CATALOG_RETRY_SECONDS = 5.0
_MAX_FIRE_TOKEN_BYTES = 16 * 1024
_CATALOGS_LOCK = threading.Lock()


class CronFireCatalogPending(Exception):
    """Local verifier discovery is in progress; NAS should retry the callback."""


def cron_fire_token_selectors(token: str) -> tuple[tuple[Optional[str], str], ...]:
    """Untrusted routing selectors only; the configured verifier must still accept the JWT."""
    if len(token) > _MAX_FIRE_TOKEN_BYTES:
        return ()
    try:
        import jwt

        claims = jwt.decode(token, leeway=30, options={
            "verify_signature": False, "verify_exp": True, "verify_nbf": True,
            "require": ["exp", "aud"],
        })
        if claims.get("purpose") != "cron_fire":
            return ()
        issuer, audience = claims.get("iss"), claims.get("aud")
        if issuer is not None and not isinstance(issuer, str):
            return ()
        audiences = [audience] if isinstance(audience, str) else audience
        if not isinstance(audiences, list) or not audiences:
            return ()
        if any(not isinstance(value, str) or not value for value in audiences):
            return ()
        return tuple(dict.fromkeys((issuer, value) for value in audiences))
    except (ValueError, TypeError, RecursionError, jwt.InvalidTokenError):
        return ()


@dataclass(frozen=True)
class _CatalogSnapshot:
    selectors: dict[tuple[Optional[str], str], tuple[str, ...]]
    homes: tuple[Path, ...]
    fingerprint: tuple
    env_snapshot: tuple[tuple[str, Optional[str]], ...] = field(repr=False)


def _directory_signature(root: Path) -> tuple:
    from hermes_constants import profile_tombstone_path

    values = []
    for path in (root / "profiles", profile_tombstone_path(root / "profiles" / "unused").parent):
        try:
            stat = path.stat()
            values.append((stat.st_ino, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            values.append(None)
    return tuple(values)


def _catalog_fingerprint(root: Path, homes: tuple[Path, ...]) -> tuple:
    # The loader's own signature covers both user config and the managed overlay.
    from hermes_cli.config import _load_config_cache_sig

    return (_directory_signature(root),
            tuple(_load_config_cache_sig(home / "config.yaml") for home in homes))


def _env_snapshot_current(snapshot: tuple[tuple[str, Optional[str]], ...]) -> bool:
    from hermes_cli.config import _env_ref_lookup

    return all(_env_ref_lookup(name) == value for name, value in snapshot)


def _build_cron_fire_catalog(root: Path) -> _CatalogSnapshot:
    from fastapi import HTTPException
    from hermes_cli.config import _CONFIG_LOCK, _LOAD_CONFIG_CACHE, cfg_get
    from hermes_cli.web_server_cron import _cron_profile_dicts, _load_cron_config_for_profile

    directory_signature = _directory_signature(root)
    profiles = _cron_profile_dicts()
    homes = tuple(Path(profile["path"]) for profile in profiles)
    before = _catalog_fingerprint(root, homes)
    if before[0] != directory_signature:
        raise CronFireCatalogPending
    catalog: dict[tuple[Optional[str], str], list[str]] = {}
    env_snapshot = set()
    for profile in profiles:
        name = profile["name"]
        try:
            # Couple the loader result to its own dependency snapshot while its lock
            # is held. It already covers user ${VAR}, ${env:VAR} and managed overlays.
            with _CONFIG_LOCK:
                cfg = _load_cron_config_for_profile(name)
                cached = _LOAD_CONFIG_CACHE.get(str(Path(profile["path"]) / "config.yaml"))
                if cached is not None:
                    env_snapshot.update(cached[5].items())
        except (HTTPException, OSError, ValueError):
            continue  # A concurrently removed profile has no callback authority.
        audience = cfg_get(cfg, "cron", "chronos", "expected_audience", default="")
        issuer = cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None
        key = cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="")
        if not isinstance(audience, str) or not audience or not isinstance(key, str) or not key:
            continue
        if issuer is not None and not isinstance(issuer, str):
            continue
        catalog.setdefault((issuer, audience), []).append(name)
    env_snapshot = tuple(env_snapshot)
    if _catalog_fingerprint(root, homes) != before or not _env_snapshot_current(env_snapshot):
        raise CronFireCatalogPending  # Never label old selectors with a new config signature.
    return _CatalogSnapshot(
        {selector: tuple(dict.fromkeys(names)) for selector, names in catalog.items()},
        homes, before, env_snapshot,
    )


class CronFireCatalog:
    """Single-flight discovery with stat-only freshness checks before catalog rejections.

    Candidates revalidate live config in the caller. Rejections return 401 only after checking
    the profile directory and loader signatures; changed files start an off-thread build
    and return 503. No timer, scheduler warmup, or historical job hints are required.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._snapshot: Optional[_CatalogSnapshot] = None
        self._refreshing = False
        self._checking = False
        self._retry_after = 0.0
        self.ready = threading.Event()

    def _start_refresh(self) -> None:
        self._refreshing = True
        self.ready.clear()
        threading.Thread(target=self._refresh, daemon=True, name="cron-fire-verifiers").start()

    def candidates(self, selectors) -> tuple[_CatalogSnapshot, tuple[str, ...]]:
        with self._lock:
            if self._refreshing:
                raise CronFireCatalogPending
            snapshot = self._snapshot
            if snapshot is None:
                if monotonic() >= self._retry_after:
                    self._start_refresh()
                raise CronFireCatalogPending
            names = {}
            for issuer, audience in selectors:
                # An absent configured issuer deliberately preserves the verifier's contract.
                for selector in ((issuer, audience), (None, audience)):
                    names.update(dict.fromkeys(snapshot.selectors.get(selector, ())))
            return snapshot, tuple(names)

    def require_current(self, snapshot: _CatalogSnapshot) -> None:
        """A failed candidate may simply predate another profile with the same selector."""
        with self._lock:
            if self._snapshot is not snapshot or self._refreshing or self._checking:
                raise CronFireCatalogPending
            self._checking = True
        try:
            unchanged = (
                _catalog_fingerprint(self._root, snapshot.homes) == snapshot.fingerprint
                and _env_snapshot_current(snapshot.env_snapshot)
            )
        except OSError:
            unchanged = False
        finally:
            with self._lock:
                self._checking = False
        with self._lock:
            if self._snapshot is not snapshot or self._refreshing:
                raise CronFireCatalogPending
            if unchanged:
                return
            self._start_refresh()
        raise CronFireCatalogPending

    def _refresh(self) -> None:
        snapshot = None
        try:
            # A single concurrent edit normally needs one retry. Continuous churn must
            # yield to NAS rather than keep a discovery thread spinning indefinitely.
            for _ in range(2):
                try:
                    snapshot = _build_cron_fire_catalog(self._root)
                    break
                except CronFireCatalogPending:
                    continue
        except Exception:
            _log.exception("Could not discover Chronos callback verifiers")
        finally:
            with self._lock:
                self._snapshot = snapshot
                self._retry_after = monotonic() + _CATALOG_RETRY_SECONDS
                self._refreshing = False
                self.ready.set()


@lru_cache(maxsize=8)
def _catalog_for_root(root: Path) -> CronFireCatalog:
    return CronFireCatalog(root)


def get_cron_fire_catalog() -> CronFireCatalog:
    root = get_default_hermes_root()
    # lru_cache alone permits duplicate construction on concurrent first misses.
    with _CATALOGS_LOCK:
        return _catalog_for_root(root)
