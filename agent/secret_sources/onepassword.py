"""1Password Service Account integration (onepassword-sdk).

Hermes pulls secrets from 1Password at process startup so they don't
have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* The **``onepassword-sdk``** Python package is used instead of the ``op``
  CLI because the SDK authenticates directly via the 1Password REST API
  and does not require a background daemon.  The ``op`` CLI's
  daemon-based architecture is unreliable on macOS in headless/background
  contexts (the daemon hangs on startup).  Using the Python SDK avoids
  the daemon entirely.

* The service account token is stored in ``~/.hermes/.env`` as
  ``OP_SERVICE_ACCOUNT_TOKEN`` (or whatever name the user picked in
  ``secrets.onepassword.token_env``).  This is the one bootstrap
  secret — every other provider key can live in 1Password.

* **Two mapping modes** (configurable):

  1. **Explicit** — the user lists env-var → ``op://`` reference pairs
     in ``secrets.onepassword.env`` (like the Bitwarden pattern):

     .. code-block:: yaml

        secrets:
          onepassword:
            enabled: true
            env:
              OPENAI_API_KEY: "op://Private/OpenAI/api key"
              ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"

  2. **Auto-discovery** — when ``auto_discover: true``, Hermes scans
     all items in the configured vault and extracts credential fields.
     Field titles become env var names (uppercased, spaces→underscores).
     This requires zero per-secret config but is less auditable.

  Both modes can be used together; explicit mappings take precedence.

* Pulling secrets caches the result in TWO layers so that repeated
  ``hermes`` invocations don't hammer the 1Password API:

    1. **In-process dict** — saves repeated calls within one Hermes
       process (gateway hot-reload, multiple import paths).
    2. **Disk-persisted JSON** — saves repeated calls ACROSS processes
       (CLI invocations, cron jobs, gateway forking new agents).

  Both layers share the same TTL (default 300 s).

* Failures NEVER block Hermes startup.  Missing SDK, no token, vault
  not found, network timeout, rate limits, etc. all emit a one-line
  warning and continue with whatever credentials ``.env`` already had.

Rate-limit awareness
--------------------

The 1Password Service Account API throttles at **1,000 reads/hour**
(per-token) for non-Business accounts.  The two-layer cache keeps
the real API call count near 0 for the vast majority of Hermes
invocations — only the first invocation after the cache TTL expires
actually touches the wire.  Additionally, a **cooldown mechanism**
prevents N sibling processes (gateway + dashboard + slash workers)
from retrying in lockstep when the rate limit is hit: each process
backs off for one hour, limiting total hourly calls to ``N × 3``
(one attempt with three retries each).

See: https://www.1password.dev/service-accounts/rate-limits

Transport choice: SDK vs CLI
-----------------------------

This backend uses the ``onepassword-sdk`` Python package instead of
the ``op`` CLI for two reasons:

1. **No daemon dependency.**  The ``op`` CLI spawns a background daemon
   (``op daemon --background``) that can hang indefinitely on macOS in
   headless/background contexts — exactly the environment Hermes runs
   in.  The SDK authenticates directly via HTTPS, avoiding the daemon
   entirely.

2. **Enables in-session tools.**  The SDK's Python API allows Hermes
   to expose tools (``onepassword_list_vaults``, etc.) that can be
   used mid-conversation without shelling out.  See
   ``tools/onepassword_tool.py``.

The trade-off is an extra pip dependency (``onepassword-sdk``) and
async-only API.  For users who prefer the ``op`` CLI, the companion
PR #36896 provides a CLI-based backend; the two implementations share
the same ``_cache.py`` substrate and can coexist.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import (
    CachedFetch,
    FetchResult,
    TwoLayerCache,
    is_valid_env_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Default token env var (matches the official 1Password CLI convention).
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# Disk cache basename (shared with tools/onepassword_tool.py).
_DISK_CACHE_BASENAME = "onepassword_cache.json"

# Rate-limit retry settings
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 1.0  # seconds, doubled each attempt + jitter
_RETRY_JITTER = 0.5       # ±50 % jitter on the delay

# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _token_fingerprint(token: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed.

    Uses SHA-256 instead of the raw token prefix to avoid leaking even
    the first 16 chars of the service account token through logs/cache
    files.  This matches Bitwarden's approach in the same codebase.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _cache_key(token: str, vault: str, refs_fingerprint: str = "") -> str:
    """Build a stable cache key from auth + vault + refs config.

    Changing any of: token, vault name, or the explicit ``env:`` mapping
    produces a different key so cache entries don't leak between configs.
    """
    fp = _token_fingerprint(token)
    if refs_fingerprint:
        return f"{fp}|{vault}|{refs_fingerprint}"
    return f"{fp}|{vault}"


def _refs_fingerprint(refs: Dict[str, str]) -> str:
    """Hash the configured env→reference mapping so config changes bust cache."""
    if not refs:
        return ""
    canonical = json_dumps_stable(refs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def json_dumps_stable(obj: object) -> str:
    """json.dumps with sorted keys for stable fingerprinting."""
    import json as _json
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Shared cache instance
# ---------------------------------------------------------------------------

_cache = TwoLayerCache[str](
    basename=_DISK_CACHE_BASENAME,
    cooldown_enabled=True,
    cooldown_seconds=3600.0,  # match the 1-hour rate-limit window
)


# ---------------------------------------------------------------------------
# SDK availability
# ---------------------------------------------------------------------------


def _sdk_available() -> bool:
    """Check if the onepassword-sdk is installed without importing it."""
    try:
        import onepassword  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Rate-limit retry helper with jitter
# ---------------------------------------------------------------------------


async def _retry_on_rate_limit(coro_factory, *, label: str = ""):
    """Await *coro_factory()*, retrying up to _MAX_RETRIES times on
    rate-limit errors.

    Uses a callable (``lambda: client.vaults.list()``) instead of a bare
    coroutine so each retry creates a fresh awaitable — ``await`` consumes
    a coroutine object and raises ``RuntimeError: cannot reuse already
    awaited coroutine`` on re-use.

    Uses exponential backoff WITH jitter so concurrent processes that
    all hit a rate limit at once don't pile on at the same retry tick.
    """
    from onepassword.errors import RateLimitExceededException

    for attempt in range(_MAX_RETRIES):
        try:
            return await coro_factory()
        except RateLimitExceededException:
            if attempt == _MAX_RETRIES - 1:
                raise
            base_delay = _RETRY_DELAY_BASE ** (attempt + 1)
            jitter = 1.0 + random.uniform(-_RETRY_JITTER, _RETRY_JITTER)
            delay = base_delay * jitter
            logger.warning(
                "1Password rate limit hit (%s), retrying in %.1fs (attempt %d/%d)",
                label, delay, attempt + 1, _MAX_RETRIES,
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Env var name sanitation
# ---------------------------------------------------------------------------


def _sanitise_env_name(raw: Optional[str]) -> str:
    """Convert a raw name to a valid env var name: uppercase, underscores."""
    if not raw:
        return ""
    name = raw.strip().upper().replace(" ", "_").replace("-", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    if name and is_valid_env_name(name):
        return name
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_onepassword_secrets(
    *,
    enabled: bool,
    token_env: str = _DEFAULT_TOKEN_ENV,
    vault: str = "",
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    auto_discover: bool = False,
    env_refs: Optional[Dict[str, str]] = None,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Pull secrets from 1Password and set them on ``os.environ``.

    This is the function ``load_hermes_dotenv()`` calls after the .env
    files have loaded.  It is intentionally defensive — any failure
    returns a :class:`FetchResult` with ``error`` set; it never raises.

    Parameters match the ``secrets.onepassword.*`` config keys.

    Args:
        enabled: Whether this source is active.
        token_env: Env var holding the service account token.
        vault: Vault name or ID to read from (required).
        override_existing: If True, overwrite already-set env vars.
        cache_ttl_seconds: Cache freshness window (0 = no caching).
        auto_discover: When True, scan all items in *vault* for
            credential fields and map field titles to env vars.
            Ignored when *env_refs* is the only source.
        env_refs: Explicit ``{ENV_VAR: "op://vault/item/field"}``
            mappings.  These take precedence over auto-discovered
            secrets when both are used.
        home_path: Forwarded to disk cache lookups for tests /
            non-standard installs.
    """
    result = FetchResult()

    if not enabled:
        return result

    # --- Gate checks -------------------------------------------------------

    access_token = os.environ.get(token_env, "").strip()
    if not access_token:
        result.error = (
            f"secrets.onepassword.enabled is true but {token_env} is "
            "not set in the environment."
        )
        return result

    if not vault and not env_refs:
        result.error = (
            "secrets.onepassword: at least one of 'vault' or 'env' "
            "must be configured."
        )
        return result

    refs = env_refs or {}
    if not auto_discover and not refs:
        result.error = (
            "secrets.onepassword: neither 'auto_discover' nor 'env' "
            "mapping is configured — nothing to fetch."
        )
        return result

    if not _sdk_available():
        result.error = (
            "onepassword-sdk is not installed. "
            "Run: uv pip install --python <hermes-python> onepassword-sdk"
        )
        return result

    # --- Cache key ---------------------------------------------------------

    key = _cache_key(access_token, vault, _refs_fingerprint(refs))

    # --- Two-layer cache ---------------------------------------------------

    cached = _cache.read(key, cache_ttl_seconds, home_path)
    if cached is not None:
        result.secrets = cached.secrets
        result.cache_hit = True
        _apply_secrets_to_env(result, access_token, token_env, override_existing)
        return result

    # --- Cooldown gate -----------------------------------------------------

    if _cache.is_cooldown_active(key):
        remaining = int(_cache.cooldown_remaining(key))
        result.error = (
            f"1Password rate-limit cooldown active "
            f"(retry after {remaining}s); "
            f"using previously-loaded credentials from .env"
        )
        return result

    # --- Real API call -----------------------------------------------------

    try:
        secrets, warnings_list = _fetch_onepassword_secrets(
            access_token=access_token,
            vault=vault,
            auto_discover=auto_discover,
            env_refs=refs,
        )
    except Exception as exc:
        # Record cooldown for rate-limit so sibling processes back off.
        # Other failures (auth, network) are NOT cooled down — those are
        # deterministic and need to surface every time.
        from onepassword.errors import RateLimitExceededException

        if isinstance(exc, RateLimitExceededException):
            _cache.record_cooldown(key)
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings_list)

    # Populate both cache layers
    entry = CachedFetch(secrets=secrets, fetched_at=time.time())
    _cache.write(key, entry, home_path)

    _apply_secrets_to_env(result, access_token, token_env, override_existing)

    return result


def _apply_secrets_to_env(
    result: FetchResult,
    access_token: str,
    token_env: str,
    override_existing: bool,
) -> None:
    """Apply secrets from result.secrets into os.environ."""
    for env_name, value in result.secrets.items():
        if env_name == token_env:
            result.skipped.append(env_name)
            continue
        if not override_existing and os.environ.get(env_name):
            result.skipped.append(env_name)
            continue
        os.environ[env_name] = value
        result.applied.append(env_name)


# ---------------------------------------------------------------------------
# Internal fetch
# ---------------------------------------------------------------------------


def _fetch_onepassword_secrets(
    access_token: str,
    vault: str,
    auto_discover: bool,
    env_refs: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Synchronous bridge — runs the async SDK in a fresh event loop."""
    try:
        return asyncio.run(_fetch_async(access_token, vault, auto_discover, env_refs))
    except RuntimeError:
        # If there's already a running event loop (e.g. in gateway),
        # spawn a thread with its own loop instead.
        import threading
        result: List[Tuple[Dict[str, str], List[str]]] = []

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result.append(loop.run_until_complete(
                    _fetch_async(access_token, vault, auto_discover, env_refs)
                ))
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=60)
        if not result:
            raise RuntimeError("1Password fetch timed out or failed")
        return result[0]


async def _fetch_async(
    access_token: str,
    vault: str,
    auto_discover: bool,
    env_refs: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Authenticate to 1Password and extract env-var-shaped secrets.

    Two-phase extraction:

    1. **Explicit mappings** — resolve each ``op://vault/item/field``
       reference from *env_refs*.  These are direct SDK calls
       (``items.get`` per ref) — efficient for small configs.

    2. **Auto-discovery** (optional) — scan all items in *vault* and
       extract credential fields whose titles become env var names.
       Field titles take priority; item titles are used as fallback
       for fields with generic labels (``credential``, ``password``).

    In both phases, env-var-name collisions are resolved: explicit
    mappings always win over auto-discovered ones.
    """
    from onepassword import Client

    warnings_list: List[str] = []
    secrets: Dict[str, str] = {}

    client = await Client.authenticate(
        auth=access_token,
        integration_name="Hermes Agent 1Password Secret Source",
        integration_version="1.0.0",
    )

    # --- Phase 0: resolve vault (needed for any per-item operations) --------

    target_vault = None
    if vault:
        vaults = await _retry_on_rate_limit(
            lambda: client.vaults.list(), label="vaults.list"
        )
        for v in vaults:
            if v.title == vault or v.id == vault:
                target_vault = v
                break
        if target_vault is None:
            available = ", ".join(f"{v.title} ({v.id})" for v in vaults)
            raise RuntimeError(
                f"Vault '{vault}' not found. Available: {available}"
            )

    # --- Phase 1: explicit env_refs mappings --------------------------------

    for env_name, ref in env_refs.items():
        if not isinstance(ref, str) or not ref.startswith("op://"):
            warnings_list.append(f"Skipping '{env_name}': not an op:// reference")
            continue
        if not is_valid_env_name(env_name):
            warnings_list.append(f"Skipping '{env_name}': invalid env-var name")
            continue

        parts = ref[5:].split("/")
        if len(parts) < 3:
            warnings_list.append(f"Skipping '{env_name}': ref has < 3 parts")
            continue

        ref_vault, ref_item, ref_field = parts[0], parts[1], parts[2]

        # Resolve the vault for this ref (may differ from the default vault)
        if target_vault is None or ref_vault not in (target_vault.title, target_vault.id):
            ref_target = None
            for v in vaults:
                if v.title == ref_vault or v.id == ref_vault:
                    ref_target = v
                    break
            if ref_target is None:
                warnings_list.append(
                    f"Vault '{ref_vault}' not found (ref: {ref})"
                )
                continue
        else:
            ref_target = target_vault

        try:
            value = await _resolve_ref(client, ref_target.id, ref_item, ref_field)
            if value:
                secrets[env_name] = value
            else:
                warnings_list.append(
                    f"'{env_name}': resolved to empty value from {ref}"
                )
        except Exception as exc:
            warnings_list.append(f"'{env_name}': {exc}")

    # --- Phase 2: auto-discovery (optional) ---------------------------------

    if auto_discover and target_vault is not None:
        items = await _retry_on_rate_limit(
            lambda: client.items.list(vault_id=target_vault.id),
            label="items.list",
        )

        # Generic field titles we skip for env-var naming (use item title instead)
        _GENERIC = frozenset({
            "credential", "password", "token", "secret",
            "apikey", "api_key", "api key", "",
        })

        for item_overview in items:
            try:
                item = await _retry_on_rate_limit(
                    lambda iid=item_overview.id: client.items.get(
                        vault_id=target_vault.id, item_id=iid,
                    ),
                    label=f"items.get({item_overview.title})",
                )
            except Exception as exc:
                warnings_list.append(
                    f"Could not read item '{item_overview.title}': {exc}"
                )
                continue

            found = False

            # Phase 2a — field-title-as-env-var:
            # If a CONCEALED/PASSWORD/API field has a non-generic title,
            # use the field title as the env var name.
            for field in (item.fields or []):
                ftitle = (field.title or "").strip().lower()
                ftype = str(field.field_type) if field.field_type else ""
                has_secret = (
                    "CONCEALED" in ftype
                    or "PASSWORD" in ftype
                    or "API" in ftype
                )
                if has_secret and field.value and ftitle not in _GENERIC:
                    env_name = _sanitise_env_name(field.title or "")
                    if env_name and env_name not in secrets:
                        secrets[env_name] = field.value
                        found = True
                        break

            # Phase 2b — item-title fallback:
            # Use the item title as env var name, first credential field as value.
            if not found:
                for field in (item.fields or []):
                    ftype = str(field.field_type) if field.field_type else ""
                    has_secret = (
                        "CONCEALED" in ftype
                        or "PASSWORD" in ftype
                        or "API" in ftype
                    )
                    if has_secret and field.value:
                        env_name = _sanitise_env_name(item.title or "")
                        if env_name and env_name not in secrets:
                            secrets[env_name] = field.value
                            found = True
                            break

            # Phase 2c — last resort: any non-empty field, item-title as name
            if not found:
                for field in (item.fields or []):
                    if field.value and field.value.strip():
                        env_name = _sanitise_env_name(item.title or "")
                        if env_name and env_name not in secrets:
                            secrets[env_name] = field.value
                            found = True
                            break

    return secrets, warnings_list


async def _resolve_ref(
    client,
    vault_id: str,
    item_name: str,
    field_label: str,
) -> str:
    """Resolve an ``op://vault/item/field`` reference to its value.

    Finds the item by title (case-insensitive) in *vault_id*, then
    matches the field by label (case-insensitive).
    """
    items = await _retry_on_rate_limit(
        lambda: client.items.list(vault_id=vault_id),
        label=f"items.list(resolve:{item_name})",
    )

    target_item = None
    for it in items:
        if it.title.lower() == item_name.lower():
            target_item = it
            break

    if target_item is None:
        available = ", ".join(it.title for it in items[:10])
        raise RuntimeError(
            f"Item '{item_name}' not found in vault. "
            f"Available (first 10): {available}"
        )

    item = await _retry_on_rate_limit(
        lambda: client.items.get(vault_id=vault_id, item_id=target_item.id),
        label=f"items.get({item_name})",
    )

    for field in (item.fields or []):
        label = (field.title or field.id or "").lower()
        if label == field_label.lower():
            return field.value or ""

    field_names = ", ".join(
        (f.title or f.id or "") for f in (item.fields or [])
    )
    raise RuntimeError(
        f"Field '{field_label}' not found in item '{item_name}'. "
        f"Available: {field_names}"
    )


# ---------------------------------------------------------------------------
# Test hook
# ---------------------------------------------------------------------------


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.  For hermetic tests."""
    _cache.clear(home_path)
