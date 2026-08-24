"""Per-identity Honcho client cache: cache keys, slots, and OAuth refresh hooks.

One SingletonSlot per client identity, so multi-profile processes don't pin the first
profile's workspace and bearer for every later profile. Origin-module symbols are
imported lazily so tests that monkeypatch ``client.resolve_config_path`` etc. keep
intercepting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import TYPE_CHECKING

from plugins.plugin_utils import SingletonSlot

if TYPE_CHECKING:
    from honcho import Honcho

    from plugins.memory.honcho.client import HonchoClientConfig

logger = logging.getLogger("plugins.memory.honcho.client")

# Applied when no timeout is configured anywhere: Honcho calls run on the
# post-response path, and an uncapped call can block response delivery forever.
_DEFAULT_HTTP_TIMEOUT = 30.0

_client_slots: dict[tuple, SingletonSlot] = {}
_client_slots_lock = threading.Lock()

# honcho.json-derived timeout, keyed PER CONFIG PATH on mtime_ns (-1 = absent) so
# the per-call staleness check costs one stat(). config.yaml needs no memo:
# load_config_readonly() is already cached on its files' signatures.
_honcho_json_timeout_memo: dict[str, tuple[int, float | None]] = {}


def _fingerprint_basis(block: dict, key_fn) -> str:
    """OAuth grants hash the REFRESH token (stable across access-token rotation,
    changes on re-auth/account switch); static keys hash the key itself."""
    oauth_block = block.get("oauth")
    if isinstance(oauth_block, dict) and oauth_block.get("refreshToken"):
        return f"oauth:{oauth_block['refreshToken']}"
    key = key_fn()
    return f"key:{key}" if key else ""


def _credential_fingerprint(config: HonchoClientConfig | None) -> str:
    """Stable identity for the credential a client will be built with, or ''. Must NOT change
    on in-place access-token rotation, but must change on account switch so
    'hermes honcho setup' yields a NEW cache identity."""
    from plugins.memory.honcho.client import _host_block

    try:
        if config is not None:
            basis = _fingerprint_basis(_host_block(config.raw or {}, config.host), lambda: config.api_key)
        else:
            # Ambient: correct on main threads; bound configs are the supported path for background threads.
            raw, block = _ambient_host_block()
            if raw is None:
                return ""
            from agent.secret_scope import get_secret
            basis = _fingerprint_basis(block, lambda: block.get("apiKey") or raw.get("apiKey") or get_secret("HONCHO_API_KEY") or "")
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16] if basis else ""
    except Exception:
        return ""


def _ambient_host_block() -> tuple[dict | None, dict]:
    """(raw honcho.json, active host block) for the ambient profile; (None, {}) when absent."""
    from plugins.memory.honcho.client import _host_block, resolve_active_host, resolve_config_path

    path = resolve_config_path()
    if not path.exists():
        return None, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, _host_block(raw, resolve_active_host())


def _client_cache_key(config: HonchoClientConfig | None) -> tuple:
    """Cache identity for a Honcho client build. Explicit configs key on connection identity,
    provenance paths, effective timeout, and the credential fingerprint (the access token itself
    is NOT in the key — in-place rotation must stay within one slot). Ambient callers
    (config=None) key on what from_global_config() would resolve."""
    from plugins.memory.honcho.client import resolve_active_host, resolve_config_path

    if config is not None:
        return ("explicit", config.host, config.workspace_id, config.base_url or "", config.environment,
                str(config.config_path) if config.config_path is not None else "",
                str(config.hermes_home) if config.hermes_home is not None else "",
                _resolve_timeout_from_sources(config), _credential_fingerprint(config))
    return ("ambient", str(resolve_config_path()), resolve_active_host(),
            _resolve_timeout_from_sources(None), _credential_fingerprint(None))


def _slot_identity(key: tuple) -> tuple:
    """(kind, host, paths) — the part of a cache key that survives credential/timeout churn."""
    return key[:3] if key[0] == "ambient" else (key[0], key[1], key[5], key[6])


def _slot_for(key: tuple) -> SingletonSlot:
    """Slot for ``key``, evicting stale same-identity slots: a same (kind, host, paths) identity
    with a different credential/timeout drops the old slot so the replaced client stops being
    served; otherwise credential churn leaks one pinned client per change.

    Without eviction, credential churn leaks one pinned client per change — the gap that made #81401's
    retirement machinery inert.
    """
    identity = _slot_identity(key)
    with _client_slots_lock:
        slot = _client_slots.get(key)
        if slot is None:
            for k in [k for k in _client_slots if k != key and _slot_identity(k) == identity]:
                _client_slots.pop(k, None)
            slot = SingletonSlot()
            _client_slots[key] = slot
        return slot


def _config_yaml_timeout() -> float | None:
    """Read honcho.timeout / honcho.request_timeout via the cached config loader."""
    from plugins.memory.honcho.client import _resolve_optional_float

    try:
        from hermes_cli.config import load_config_readonly
        honcho_cfg = load_config_readonly().get("honcho", {})
        if isinstance(honcho_cfg, dict):
            return _resolve_optional_float(honcho_cfg.get("timeout"), honcho_cfg.get("request_timeout"))
    except Exception:
        pass
    return None


def _honcho_json_timeout() -> float | None:
    """Read timeout/requestTimeout from honcho.json (host block wins), memoized on mtime."""
    from plugins.memory.honcho.client import _HostLookup, _resolve_optional_float, resolve_config_path

    try:
        path = resolve_config_path()
        path_key = str(path)
        try:
            mtime_ns: int = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        memo = _honcho_json_timeout_memo.get(path_key)
        if memo is not None and memo[0] == mtime_ns:
            return memo[1]
        timeout = None
        if mtime_ns != -1:
            raw, host_block = _ambient_host_block()
            timeout = _resolve_optional_float(*_HostLookup(host_block, raw).vals("timeout", "requestTimeout"))
        _honcho_json_timeout_memo[path_key] = (mtime_ns, timeout)
        return timeout
    except Exception:
        return None


def _resolve_timeout_from_sources(config: HonchoClientConfig | None) -> float:
    """Mirror the build path's timeout resolution exactly: any skew makes the staleness check
    disagree with the built client forever and rebuild it on every call."""
    from plugins.memory.honcho.client import _resolve_optional_float

    if config is not None:
        timeout = config.timeout
    else:
        timeout = _honcho_json_timeout()
        if timeout is None:
            timeout = _resolve_optional_float(os.environ.get("HONCHO_TIMEOUT"))
    if timeout is None:
        timeout = _config_yaml_timeout()
    return timeout if timeout is not None else _DEFAULT_HTTP_TIMEOUT


def _refresh_oauth(config: HonchoClientConfig | None, client: Honcho | None = None, slot: SingletonSlot | None = None) -> None:
    """Refresh a near-expiry OAuth grant. Pre-build (``client=None``): point ``config.api_key`` at the
    fresh token so a new client doesn't 401 an hour in. Cached (``client`` given): rotate its Bearer in
    place; if the in-place rotation can't apply (SDK shape change) reset ``slot`` so the next acquisition
    rebuilds. No-op for static keys or on failure (the first 401 triggers session.py's forced rotation).
    Refreshes against the config's BOUND path: the ambient resolver on daemon threads lands on the
    default profile."""
    from plugins.memory.honcho.client import resolve_active_host, resolve_config_path

    try:
        from plugins.memory.honcho import oauth
        if config is not None:
            host, path = config.host, config.bound_config_path()
        else:
            host, path = resolve_active_host(), resolve_config_path()
        token, refreshed = oauth.ensure_fresh_token(path, host)
        if client is None:
            if token:
                config.api_key = token
        elif refreshed and token and not oauth.apply_token_to_client(client, token) and slot is not None:
            slot.reset()
    except Exception:
        logger.warning("Honcho OAuth %s refresh failed", "pre-build" if client is None else "cached", exc_info=True)
