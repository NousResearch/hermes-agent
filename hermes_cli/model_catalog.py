"""Remote model catalog fetcher.

``get_catalog()`` returns the parsed manifest: in-process cache (TTL) → disk cache at
``~/.hermes/cache/model_catalog.json`` → master URL fetch; any fetch failure keeps the stale copy
(or ``{}``, and the URL is not retried for ``FETCH_RETRY_DELAY_SECONDS``). ``get_curated_openrouter_models()`` / ``get_curated_nous_models()`` are thin accessors
whose callers fall back to the in-repo lists on ``None``.

``cached_manifest()`` is the read every other provider goes through: it never opens a socket, and
what it returns only ADDS to what the install already ships (``merge_remote_models`` for picker
rows, ``manifest_model_metadata`` and ``extend_context_lengths`` for a model no table knows,
``in_model_set`` for the id sets in ``MODEL_SETS`` that gate request handling).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Sequence, TypeVar

from hermes_cli.codex_models import model_requires_account_discovery
from hermes_cli.version_info import hermes_user_agent
from utils import atomic_json_write

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_URL = (
    "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json")
# The Docusaurus site sits behind Vercel, which occasionally 403s non-browser clients (bot
# challenge); the raw GitHub copy is the same manifest and is not bot-gated.
DEFAULT_CATALOG_FALLBACK_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/static/api/model-catalog.json",
)
DEFAULT_TTL_MINUTES = 20
# Legacy key, honoured only when the user set it explicitly; ``ttl_minutes`` is the shipped default.
DEFAULT_TTL_HOURS = DEFAULT_TTL_MINUTES / 60.0
DEFAULT_FETCH_TIMEOUT = 8.0
# After a failed fetch with no cached manifest to serve, the same URL is not retried in this process
# for this long: an offline user would otherwise wait out the timeout (primary + fallback URL, ~16 s)
# on every picker build. Same window as the models.dev registry's failure backoff.
FETCH_RETRY_DELAY_SECONDS = 300.0
SUPPORTED_SCHEMA_VERSION = 1
# Rows the per-provider blocks never extend, for the publisher and the reader alike:
#   openrouter, nous  have their own blocks, read as the whole list (and the only ``default`` labels)
#   openai-codex      only the account's live Codex catalog may advertise a model
#   xai, xai-oauth    filled from the models.dev disk cache, so not the same on every machine
#   moa               a preset name, not a model list
WHOLE_LIST_ROWS = frozenset({"openrouter", "nous"})
OWN_LIST_ROWS = WHOLE_LIST_ROWS | {"openai-codex", "xai", "xai-oauth", "moa"}

# Named sets of model ids that gate request handling. Each mirrors one built-in constant, and the
# manifest's ``model_sets`` may add ids to it. The manifest cannot add a set: a set is only as real as
# the code that reads it, so a name missing here is ignored. ``True`` marks a set a model cannot be
# driven correctly without; a picker entry in one lists it under ``requires``.
MODEL_SETS: dict[str, bool] = {
    "openai_gpt62_contract": True,     # agent.reasoning_effort.GPT62_MODEL_IDS
    "bedrock_openai_responses": True,  # agent.bedrock_adapter.BEDROCK_OPENAI_RESPONSES_MODEL_IDS
    "anthropic_fast_mode": False,      # agent.model_metadata._ANTHROPIC_FAST_MODE_MODELS
    "openai_ultrafast": False,         # hermes_cli.models_catalog_static._OPENAI_ULTRAFAST_MODELS
    "openai_fast_eu": False,           # hermes_cli.models_fast_regions._FAST_REGION_MODELS (EU host)
}

# In-process cache, invalidated against the disk file's path + mtime and TTL. The path matters:
# under a multiplexed gateway each profile has its own ``<home>/cache/model_catalog.json``, and
# mtime alone cannot tell two profiles' files apart. One ``(path, mtime, manifest)`` tuple, replaced
# whole: threads serving different profiles read it, and must never see one profile's manifest
# paired with another's path.
_catalog_slot: tuple[str, float, dict[str, Any]] | None = None
# ``monotonic()`` deadline before which a failed fetch is not retried (see FETCH_RETRY_DELAY_SECONDS),
# keyed by (profile home key, "manifest" | "override", URL): one gateway process serves many profiles,
# and a dead override must not skip the manifest's fallback chain when both share a URL. Pickers and
# the gateway's refresh thread touch it concurrently, hence the lock.
_fetch_retry_after: dict[tuple[str, str, str], float] = {}
_fetch_retry_lock = threading.Lock()


def _backoff_key(kind: str, url: str) -> tuple[str, str, str]:
    from hermes_constants import hermes_home_key
    return (hermes_home_key(), kind, url)


def _backing_off(kind: str, url: str) -> bool:
    key = _backoff_key(kind, url)
    with _fetch_retry_lock:
        return time.monotonic() < _fetch_retry_after.get(key, 0.0)


def _note_fetch_result(kind: str, url: str, ok: bool) -> None:
    key = _backoff_key(kind, url)
    with _fetch_retry_lock:
        if ok:
            _fetch_retry_after.pop(key, None)
        else:
            _fetch_retry_after[key] = time.monotonic() + FETCH_RETRY_DELAY_SECONDS


def clear_fetch_backoff() -> None:
    """Forget the active profile's fetch failures so an explicit refresh retries every URL."""
    from hermes_constants import hermes_home_key
    home = hermes_home_key()
    with _fetch_retry_lock:
        for key in [k for k in _fetch_retry_after if k[0] == home]:
            del _fetch_retry_after[key]


def _load_catalog_config() -> dict[str, Any]:
    """Load the ``model_catalog`` config block with defaults filled in."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    raw = cfg.get("model_catalog")
    if not isinstance(raw, dict):
        raw = {}

    # ``ttl_hours`` (legacy) is honoured only when ``ttl_minutes`` is still at its default —
    # load_config() deep-merges the default in, so "present" alone doesn't mean "user-set".
    ttl_minutes = raw.get("ttl_minutes")
    try:
        ttl_minutes = float(ttl_minutes) if ttl_minutes not in (None, "") else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        ttl_minutes = DEFAULT_TTL_MINUTES
    if ttl_minutes == DEFAULT_TTL_MINUTES and raw.get("ttl_hours"):
        try:
            ttl_minutes = float(raw["ttl_hours"]) * 60.0
        except (TypeError, ValueError):
            pass
    if ttl_minutes <= 0:
        ttl_minutes = DEFAULT_TTL_MINUTES

    return {
        "enabled": bool(raw.get("enabled", True)),
        "url": str(raw.get("url") or DEFAULT_CATALOG_URL),
        "ttl_hours": ttl_minutes / 60.0,
        "providers": raw.get("providers") if isinstance(raw.get("providers"), dict) else {}}


def _catalog_enabled() -> bool:
    """``model_catalog.enabled`` alone, off the uncopied config: ``cached_manifest`` runs on request paths."""
    from hermes_cli.config import load_config_readonly
    raw = load_config_readonly().get("model_catalog")
    return bool(raw.get("enabled", True)) if isinstance(raw, dict) else True


def _cache_path() -> Path:
    """Disk cache path; imported lazily so tests can monkeypatch home."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "model_catalog.json"


def _fetch_manifest(url: str, timeout: float) -> dict[str, Any] | None:
    """HTTP GET the manifest URL and return a validated dict, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": hermes_user_agent()})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("model catalog fetch failed (%s): %s", url, exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.info("model catalog fetch errored (%s): %s", url, exc)
        return None
    if not _validate_manifest(data):
        logger.info("model catalog at %s failed schema validation", url)
        return None
    return data


def _fetch_manifest_with_fallback(
    primary_url: str, timeout: float, fallback_urls: tuple[str, ...] = DEFAULT_CATALOG_FALLBACK_URLS
) -> dict[str, Any] | None:
    """First manifest that fetches and validates from ``primary_url`` then ``fallback_urls`` (skipping
    any equal to the primary so a raw-GitHub-configured operator doesn't double-fetch), or None."""
    data = _fetch_manifest(primary_url, timeout)
    if data is not None:
        return data
    for url in fallback_urls:
        if not url or url == primary_url:
            continue
        data = _fetch_manifest(url, timeout)
        if data is not None:
            logger.info("model catalog primary URL failed; using fallback %s", url)
            return data
    return None


def _validate_manifest(data: Any) -> bool:
    """Return True when ``data`` matches the minimum manifest shape."""
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    if not isinstance(version, int) or version > SUPPORTED_SCHEMA_VERSION:
        return False  # future schema we don't understand — refuse rather than guess
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    for pname, pblock in providers.items():
        if not isinstance(pname, str) or not isinstance(pblock, dict):
            return False
        models = pblock.get("models")
        if not isinstance(models, list):
            return False
        if not all(isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip() for m in models):
            return False
    return True


def _read_disk_cache() -> tuple[dict[str, Any] | None, float]:
    """Return ``(data_or_none, mtime)``. mtime is 0 if file is missing."""
    path = _cache_path()
    try:
        mtime = path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return (None, 0.0)
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):  # unreadable, not UTF-8, or not JSON: all a cache miss
        return (None, 0.0)
    return (data, mtime) if _validate_manifest(data) else (None, 0.0)


def _write_disk_cache(data: dict[str, Any]) -> None:
    try:
        atomic_json_write(_cache_path(), data)
    except OSError as exc:
        logger.info("model catalog cache write failed: %s", exc)


# Stale-while-revalidate: at most one background manifest refresh in flight per cache file (i.e.
# per profile home — profile A's refresh must not suppress profile B's). The refreshed manifest
# lands on disk; the NEXT get_catalog() call picks it up via the mtime check.
_catalog_swr_lock = threading.Lock()
_catalog_swr_inflight: set[str] = set()


def _spawn_catalog_swr_refresh(url: str) -> None:
    """Refresh the catalog manifest off-thread (fire-and-forget, deduped per cache path)."""
    inflight_key = str(_cache_path())
    with _catalog_swr_lock:
        if inflight_key in _catalog_swr_inflight:
            return
        _catalog_swr_inflight.add(inflight_key)

    def _refresh() -> None:
        try:
            fetched = _fetch_manifest_with_fallback(url, DEFAULT_FETCH_TIMEOUT)
            if fetched is not None:
                _write_disk_cache(fetched)
        except Exception:
            logger.debug("catalog SWR refresh failed", exc_info=True)
        finally:
            with _catalog_swr_lock:
                _catalog_swr_inflight.discard(inflight_key)

    # copy_context: the picker may be serving a profile scoped by the HERMES_HOME ContextVar
    # (tui_gateway ``_profile_scoped``), so the worker must write THAT profile's cache file.
    context = contextvars.copy_context()
    threading.Thread(target=lambda: context.run(_refresh), daemon=True, name="model-catalog-swr").start()


def _remember(data: dict[str, Any], mtime: float) -> dict[str, Any]:
    global _catalog_slot
    _catalog_slot = (str(_cache_path()), mtime, data)
    return data


def _in_process_catalog(mtime: float | None = None) -> dict[str, Any] | None:
    """The in-process copy when it mirrors the ACTIVE profile's cache file (and, given ``mtime``,
    that version of it), else None."""
    slot = _catalog_slot
    if slot is None or slot[0] != str(_cache_path()) or (mtime is not None and slot[1] != mtime):
        return None
    return slot[2]


def get_catalog(*, force_refresh: bool = False) -> dict[str, Any]:
    """Parsed model catalog manifest, or ``{}`` on failure — never raises, so the CLI works offline
    (callers treat a missing provider/model as "use the in-repo fallback")."""
    cfg = _load_catalog_config()
    if not cfg["enabled"]:
        return {}
    ttl_seconds = max(0.0, cfg["ttl_hours"] * 3600.0)
    disk_data, disk_mtime = _read_disk_cache()
    now = time.time()
    disk_fresh = disk_data is not None and (now - disk_mtime) < ttl_seconds

    if not force_refresh and disk_data is not None:
        cached = _in_process_catalog(disk_mtime)
        if disk_fresh and cached is not None:
            return cached
        if not disk_fresh:
            # Stale-while-revalidate: serve the expired disk copy now and refresh off-thread so the
            # /model picker (which calls this on every open) never blocks on the manifest fetch.
            # Only a cold cache (no disk copy at all) still blocks.
            _spawn_catalog_swr_refresh(cfg["url"])
        return _remember(disk_data, disk_mtime)

    if not force_refresh and disk_data is None and _backing_off("manifest", cfg["url"]):
        return {}
    fetched = _fetch_manifest_with_fallback(cfg["url"], DEFAULT_FETCH_TIMEOUT)
    _note_fetch_result("manifest", cfg["url"], fetched is not None)
    if fetched is not None:
        _write_disk_cache(fetched)
        new_disk_data, new_mtime = _read_disk_cache()
        if new_disk_data is not None:
            return _remember(new_disk_data, new_mtime)
        return _remember(fetched, now)
    if disk_data is not None:
        return _remember(disk_data, disk_mtime)
    return {}


def refresh_interval_seconds() -> float:
    """Return the configured catalog TTL in seconds (the gateway poll cadence)."""
    return max(60.0, _load_catalog_config()["ttl_hours"] * 3600.0)


def refresh_catalogs() -> bool:
    """Force-refresh every remote catalog the picker reads (manifest, OpenRouter live list, and the
    Nous Portal recommendations for a Nous account, the only reader), writing each disk cache so the
    next ``/model`` open in ANY process sees them. Blocking; run it off the event loop."""
    if not _load_catalog_config()["enabled"]:
        return False
    clear_fetch_backoff()  # provider overrides are re-asked too, not only the forced manifest
    catalog = get_catalog(force_refresh=True)
    try:
        from agent.secret_scope import get_secret
        from hermes_cli.anon_auth import current_nous_state, is_guest_state
        from hermes_cli.auth import read_credential_pool
        from hermes_cli.models import _resolve_nous_portal_url, fetch_nous_recommended_models, fetch_openrouter_models

        fetch_openrouter_models(force_refresh=True)
        logins = [current_nous_state(), *read_credential_pool("nous")]
        if any(e and not is_guest_state(e) for e in logins) or get_secret("NOUS_API_KEY"):
            fetch_nous_recommended_models(_resolve_nous_portal_url(), force_refresh=True)
    except Exception:
        logger.debug("provider catalog refresh failed", exc_info=True)
    return bool(catalog)


def _fetch_provider_override(provider: str) -> dict[str, Any] | None:
    """If ``model_catalog.providers.<name>.url`` is set, fetch that instead."""
    cfg = _load_catalog_config()
    if not cfg["enabled"]:
        return None
    provider_cfg = cfg["providers"].get(provider)
    if not isinstance(provider_cfg, dict):
        return None
    override_url = provider_cfg.get("url")
    if not isinstance(override_url, str) or not override_url.strip():
        return None
    # Overrides are usually third-party self-hosted: skip the disk cache, re-request every call
    # (except during a failure backoff; the caller then falls back to the main manifest).
    override_url = override_url.strip()
    if _backing_off("override", override_url):
        return None
    data = _fetch_manifest(override_url, DEFAULT_FETCH_TIMEOUT)
    _note_fetch_result("override", override_url, data is not None)
    return data


def _block_of(manifest: dict[str, Any] | None, provider: str) -> dict[str, Any] | None:
    block = (manifest or {}).get("providers", {}).get(provider)
    return block if isinstance(block, dict) else None


def _get_provider_block(provider: str) -> dict[str, Any] | None:
    """Return the provider's manifest block, respecting per-provider overrides."""
    return _block_of(_fetch_provider_override(provider), provider) or _block_of(get_catalog(), provider)


def _block_ids(block: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    """``(id, entry)`` for every model entry of ``block`` with a non-empty id."""
    models = (block or {}).get("models", [])
    return [(mid, m) for m in models if isinstance(m, dict) and (mid := str(m.get("id") or "").strip())]


def get_curated_openrouter_models() -> list[tuple[str, str]] | None:
    """OpenRouter's curated ``[(id, description), ...]`` from the manifest."""
    rows = _block_ids(_get_provider_block("openrouter"))
    return [(mid, str(m.get("description") or "")) for mid, m in rows] or None


def get_curated_nous_models() -> list[str] | None:
    """Nous Portal's curated model ids from the manifest."""
    return [mid for mid, _ in _block_ids(_get_provider_block("nous"))] or None


def cached_manifest() -> dict[str, Any] | None:
    """This profile's manifest without opening a socket: the in-process copy while it still mirrors
    the disk cache, else the disk cache at any age. None when the catalog is disabled or was never
    fetched. The fetch points (``get_catalog``, the gateway refresh, ``hermes update``) keep it warm."""
    try:
        mtime = _cache_path().stat().st_mtime
    except OSError:
        return None  # nothing cached: the config is not consulted for a manifest that cannot exist
    if not _catalog_enabled():
        return None
    cached = _in_process_catalog(mtime)
    if cached is not None:
        return cached
    disk_data, disk_mtime = _read_disk_cache()
    return _remember(disk_data, disk_mtime) if disk_data is not None else None


_shipped_rows_only: contextvars.ContextVar[bool] = contextvars.ContextVar("model_catalog_shipped_rows_only", default=False)


@contextlib.contextmanager
def shipped_rows_only() -> Iterator[None]:
    """Inside the block the manifest adds no rows, so a provider's list is what the install ships
    plus what the provider itself reports. Provider detection confirms a match there: a published
    row is something to offer, never evidence that this provider serves the model."""
    token = _shipped_rows_only.set(True)
    try:
        yield
    finally:
        _shipped_rows_only.reset(token)


class Listings:
    """What one catalog build saw: the ids the provider's own listings reported (lowercased, to the
    provider's spelling), the ids the manifest supplied to its rows, and those of them the row only
    holds because they were in the manifest (``gated``: a builder that keeps live ids only when the
    curated row names them, such as the official OpenAI account listing)."""

    def __init__(self) -> None:
        self.reported: dict[str, str] = {}
        self.supplied: set[str] = set()
        self.gated: set[str] = set()


_listings: contextvars.ContextVar[Listings | None] = contextvars.ContextVar("model_catalog_listings", default=None)
_T = TypeVar("_T")


@contextlib.contextmanager
def recorded_listings() -> Iterator[Listings]:
    """Record, for the catalog build inside the block, which ids came from the provider and which
    from the manifest. Nothing is fetched twice: the record is taken from the build itself."""
    listings = Listings()
    token = _listings.set(listings)
    try:
        yield listings
    finally:
        _listings.reset(token)


def provider_listing(fetch: Callable[[], _T]) -> _T:
    """``fetch()``, a live listing a catalog builder works from; its ids count as reported by the
    provider while ``recorded_listings`` is active."""
    value = fetch()
    if (listings := _listings.get()) is not None and isinstance(value, (list, tuple)):
        for mid in value:
            listings.reported.setdefault(str(mid).lower(), str(mid))
    return value


def note_supplied(ids: Sequence[str], *, gated: bool = False) -> None:
    """Record ``ids`` as supplied by the manifest to the row being built (``recorded_listings``);
    ``gated`` when the builder keeps a live id only if the curated row names it."""
    if ids and (listings := _listings.get()) is not None:
        listings.supplied.update(str(mid).lower() for mid in ids)
        if gated:
            listings.gated.update(str(mid).lower() for mid in ids)


def _drivable(entry: dict[str, Any]) -> bool:
    """Whether this build reads every model set the entry lists under ``requires``."""
    requires = entry.get("requires", [])
    return isinstance(requires, list) and all(isinstance(name, str) and name in MODEL_SETS for name in requires)


def manifest_provider_models(provider: str) -> list[str]:
    """Model ids the cached manifest lists for ``provider``, in manifest order. An entry whose
    ``requires`` names a model set this build does not read is not offered: a release that cannot
    drive a model must not list it. Neither is an id that only the account's own listing may
    advertise."""
    if _shipped_rows_only.get():
        return []
    return [mid for mid, m in _block_ids(_block_of(cached_manifest(), provider))
            if _drivable(m) and not model_requires_account_discovery(provider, mid)]


def manifest_additions(provider: str, builtin: Sequence[str]) -> list[str]:
    """The ids the cached manifest adds to ``builtin``, the row this install ships for ``provider``,
    in manifest order. Nothing for a row the install does not build in (a plugin's ``static_models``
    are the plugin's), for ``OWN_LIST_ROWS``, or for an id the row already holds in any casing."""
    # Imported here: the table module resolves the Codex row through ``agent.model_metadata`` while
    # it loads, and that module reads this one.
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    if not builtin or provider not in _PROVIDER_MODELS or provider in OWN_LIST_ROWS:
        return []
    seen = {mid.lower() for mid in builtin}
    added: list[str] = []
    for mid in manifest_provider_models(provider):
        if mid.lower() not in seen:
            seen.add(mid.lower())
            added.append(mid)
    return added


def merge_remote_models(
        provider: str, builtin: Sequence[str], *, live: Sequence[str] | None = (), gated: bool = False) -> list[str]:
    """The ids the cached manifest adds for ``provider``, then ``builtin`` exactly as the install
    ships it. The manifest only extends a list: it never removes, reorders or respells a row."""
    # An id the provider's live listing already has joins at its live position, as it would without
    # the manifest: only ids the provider did not list are put in front of the shipped row.
    listed = {str(mid).lower() for mid in live or ()}
    added = [mid for mid in manifest_additions(provider, builtin) if mid.lower() not in listed]
    note_supplied(added, gated=gated)
    return [*added, *builtin]



def manifest_table(*path: str) -> dict[str, Any]:
    """The mapping at ``path`` in the cached manifest; ``{}`` when it is absent or not a mapping."""
    node: Any = cached_manifest()
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def manifest_model_metadata(provider: str, model: str) -> dict[str, Any] | None:
    """The models.dev-shaped entry the cached manifest publishes for a models.dev provider id and a
    lowercased model id, for a model no other catalog knows. An entry without a usable ``limit`` is
    no entry: a model is known with a real window or it is unknown, never known with a default.
    ``cost`` is dropped, since this entry also reaches the pricing lookup and prices are not its job."""
    entry = manifest_table("model_metadata", provider).get(model)
    limits = entry.get("limit") if isinstance(entry, dict) else None
    if not isinstance(limits, dict) or "context" not in limits:
        return None
    if any(type(limits[key]) is not int or limits[key] <= 0 for key in ("context", "input", "output") if key in limits):
        return None
    return {key: value for key, value in entry.items() if key != "cost"}


def extend_context_lengths(name: str, builtin: dict[str, int]) -> dict[str, int]:
    """``builtin`` plus the rows the cached manifest adds to the ``name`` context-length table. A key
    the install ships keeps its value and its place under any casing. An added key is lowercased,
    because every reader matches it against a lowercased model id, and a blank one is dropped: as a
    substring it would match every model."""
    shipped = {key.lower() for key in builtin}
    added: dict[str, int] = {}
    for key, value in manifest_table("context_lengths", name).items():
        if key.strip() and key.lower() not in shipped and type(value) is int and value > 0:
            added.setdefault(key.lower(), value)
    return {**builtin, **added} if added else builtin


def in_model_set(name: str, builtin: Collection[str], model: str) -> bool:
    """Whether ``model`` is in the built-in set ``name`` or among the ids the cached manifest adds to
    it. ``model`` is already in the spelling the built-in set is compared in (lowercased, bare)."""
    if model in builtin:
        return True
    added = manifest_table("model_sets").get(name)
    return isinstance(added, list) and model in added


def _default_model_from_block(block: dict[str, Any] | None) -> str | None:
    """Id of the model entry labeled ``"default": true``, or None."""
    return next((mid for mid, m in _block_ids(block) if m.get("default")), None)


def get_default_model_from_cache(provider: str) -> str | None:
    """The manifest's labeled default for ``provider`` (the model Hermes silently lands on when the
    user never picked one) — in-process then disk cache only, never a fetch. Only the two whole-list
    blocks carry the label: a block that adds rows never chooses a model."""
    if provider not in WHOLE_LIST_ROWS:
        return None
    cached = _in_process_catalog()
    found = _default_model_from_block(_block_of(cached, provider)) if cached is not None else None
    if found:
        return found
    disk_data, _mtime = _read_disk_cache()
    return _default_model_from_block(_block_of(disk_data, provider)) if disk_data is not None else None


def seed_cache_from_checkout(project_root: Path | str) -> bool:
    """Overwrite the disk cache with the checkout's ``website/static/api/model-catalog.json``.
    After ``hermes update`` that file IS the newest catalog, so the picker stays current even when
    the remote fetch is bot-gated. Validated, then written via the same atomic writer."""
    src = Path(project_root) / "website" / "static" / "api" / "model-catalog.json"
    try:
        with open(src, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("model catalog seed from checkout skipped (%s): %s", src, exc)
        return False
    if not _validate_manifest(data):
        logger.debug("model catalog seed from checkout skipped: invalid manifest at %s", src)
        return False
    _write_disk_cache(data)
    reset_cache()  # drop the in-process copy so the next read picks up the seed
    return True


def reset_cache() -> None:
    """Clear the in-process cache and fetch-failure backoff. Used by tests and ``hermes model --refresh``."""
    global _catalog_slot
    with _fetch_retry_lock:
        _fetch_retry_after.clear()
    _catalog_slot = None
