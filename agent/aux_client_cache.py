"""Auxiliary client cache + shutdown helpers (extracted from auxiliary_client.py).

Slice R4-C2 of the auxiliary_client.py god-file kill (#78635 / #78647): the
client cache, cache-key derivation, close/shutdown helpers, and
``_get_cached_client`` (source window lines 6933-7306, sha256
5e139636618beb09558c120877da698fe7fde1b545cd6b00e1de6504e64d2be2).

``agent.auxiliary_client`` re-exports every name below through an import seam,
so all existing import paths and ``unittest.mock.patch`` targets
(``agent.auxiliary_client._get_cached_client``, etc.) keep working unchanged.

Stay-behind godfile dependencies (``resolve_provider_client``,
``_normalize_aux_provider``, ``_normalize_main_runtime``, ``_peek_pool_entry``,
``_pool_runtime_api_key``, ``_pool_cache_hint``, ``_MAIN_RUNTIME_FIELDS``,
``_resolve_nous_runtime_api``, ``_create_openai_client``, ``_to_async_client``)
and ``utils.base_url_host_matches`` are lazy-imported inside the function
bodies that use them, matching the godfile's cycle-avoidance idiom.
"""

import hashlib
import inspect
import threading
from typing import Any, Dict, Optional, Tuple


# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# NOTE: loop identity is NOT part of the key.  On async cache hits we check
# whether the cached loop is the *current* loop; if not, the stale entry is
# replaced in-place.  This bounds cache growth to one entry per unique
# provider config rather than one per (config × event-loop), which previously
# caused unbounded fd accumulation in long-running gateway processes (#10200).
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


class _CallableCacheDiscriminator:
    """Hash a credential callback by identity without exposing its state."""

    __slots__ = ("_callback",)

    def __init__(self, callback: Any) -> None:
        # Retain the callback so its id cannot be reused while cached.
        self._callback = callback

    def __hash__(self) -> int:
        return id(self._callback)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _CallableCacheDiscriminator)
            and self._callback is other._callback
        )

    def __repr__(self) -> str:
        return "<callable-api-key>"


def _runtime_cache_discriminator(field: str, value: Any) -> Any:
    """Return a hashable, secret-safe runtime cache-key component."""
    if field == "api_key" and callable(value):
        return _CallableCacheDiscriminator(value)
    if field == "api_key" and isinstance(value, str) and value:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
        return ("api-key-digest", digest)
    return value


def _client_cache_key(
    provider: str,
    *,
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple:
    from agent.auxiliary_client import _MAIN_RUNTIME_FIELDS, _normalize_main_runtime, _pool_cache_hint
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple(
        _runtime_cache_discriminator(field, runtime.get(field, ""))
        for field in _MAIN_RUNTIME_FIELDS
    ) if provider == "auto" else ()
    # `auto` can now resolve through task-specific or main fallback policy,
    # so the task participates in the cache key. Non-auto providers keep the
    # old cache shape because the explicit provider/model tuple is sufficient.
    task_key = (task or "") if provider == "auto" else ""
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    # The model MUST participate in the key. Two concurrent auxiliary calls to
    # the SAME provider/base_url/key but DIFFERENT models (e.g. a MoA reference
    # fan-out running opus + gpt-5.5 in parallel threads) would otherwise share
    # one cache entry. On a cache MISS both build a client for the same key; the
    # second's _store_cached_client sees the first as the "old" entry and CLOSES
    # it — while the first call is still mid-request on it — yielding a spurious
    # APIConnectionError that fails the sibling advisor (root cause of the run2
    # double-advisor "Connection error" collapse). Keying on model gives each
    # model its own client, so concurrent fan-out calls never cross-close.
    model_key = model or runtime.get("model", "")
    api_key_key = _runtime_cache_discriminator("api_key", api_key or "")
    return (provider, async_mode, base_url or "", api_key_key, api_mode or "", runtime_key, is_vision, task_key, pool_hint, model_key)


def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any = None) -> None:
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _close_cached_client(old_entry[0])
        _client_cache[cache_key] = (client, default_model, bound_loop)


def _refresh_nous_auxiliary_client(
    *,
    cache_provider: str,
    model: Optional[str],
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Refresh Nous runtime creds, rebuild the client, and replace the cache entry."""
    from agent.auxiliary_client import _create_openai_client, _resolve_nous_runtime_api, _to_async_client
    runtime = _resolve_nous_runtime_api(force_refresh=True)
    if runtime is None:
        return None, model

    fresh_key, fresh_base_url = runtime
    sync_client = _create_openai_client(api_key=fresh_key, base_url=fresh_base_url)
    final_model = model

    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
        client, final_model = _to_async_client(sync_client, final_model or "", is_vision=is_vision)
    else:
        client = sync_client

    cache_key = _client_cache_key(
        cache_provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        model=final_model,
    )
    _store_cached_client(cache_key, client, final_model, bound_loop=current_loop)
    return client, final_model


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def _close_cached_client(client: Any) -> None:
    """Apply the canonical best-effort close policy to one cached client."""
    if client is None:
        return
    _force_close_async_httpx(client)
    try:
        close_fn = getattr(client, "close", None)
        if callable(close_fn) and not inspect.iscoroutinefunction(close_fn):
            close_fn()
    except Exception:
        pass


def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.
    """
    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            client = entry[0]
            if client is None:
                continue
            _close_cached_client(client)
        _client_cache.clear()


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                _force_close_async_httpx(client)
                stale_keys.append(key)
        for key in stale_keys:
            del _client_cache[key]


def _is_openrouter_client(client: Any) -> bool:
    from utils import base_url_host_matches
    for obj in (client, getattr(client, "_client", None), getattr(client, "client", None)):
        if obj and base_url_host_matches(str(getattr(obj, "base_url", "") or ""), "openrouter.ai"):
            return True
    return False


def _cached_client_accepts_slash_models(client: Any, cached_default: Optional[str]) -> bool:
    """Best-effort check for cached clients that accept ``vendor/model`` IDs."""
    if _is_openrouter_client(client):
        return True
    return bool(cached_default and "/" in cached_default)


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that support them.

    Mirrors the guard in resolve_provider_client() which is skipped on cache hits.
    """
    if model and "/" in model and not _cached_client_accepts_slash_models(client, cached_default):
        return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues, the cache validates on
    every async hit that the cached loop is the *current, open* loop.
    If the loop changed (e.g. a new gateway worker-thread loop), the stale
    entry is replaced in-place rather than creating an additional entry.

    This keeps cache size bounded to one entry per unique provider config,
    preventing the fd-exhaustion that previously occurred in long-running
    gateways where recycled worker threads created unbounded entries (#10200).
    """
    from agent.auxiliary_client import _normalize_aux_provider, _normalize_main_runtime, _peek_pool_entry, _pool_runtime_api_key, resolve_provider_client
    # Resolve the current event loop for async clients so we can validate
    # cached entries.  Loop identity is NOT in the cache key — instead we
    # check at hit time whether the cached loop is still current and open.
    # This prevents unbounded cache growth from recycled worker-thread loops
    # while still guaranteeing we never reuse a client on the wrong loop
    # (which causes deadlocks, see #2681).
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(
        provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        task=task,
        model=model,
    )
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # Validate: the cached client must be bound to the CURRENT,
                # OPEN loop.  If the loop changed or was closed, the httpx
                # transport inside is dead — force-close and replace.
                loop_ok = (
                    cached_loop is not None
                    and cached_loop is current_loop
                    and not cached_loop.is_closed()
                )
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return cached_client, effective
                # Stale — evict and fall through to create a new client.
                _force_close_async_httpx(cached_client)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return cached_client, effective
    # Build outside the lock.
    # For pool-backed api_key providers, derive the active API key from the
    # pool entry rather than from env vars.  resolve_api_key_provider_credentials
    # always prefers env vars (first-entry bias), which bypasses pool rotation:
    # after key #1 is marked exhausted the retry would still get key #1 from
    # the env var and fail again, causing the retry2_err handler to mark key #2.
    effective_api_key = api_key
    if not effective_api_key:
        _pe = _peek_pool_entry(_normalize_aux_provider(provider))
        if _pe is not None:
            _pk = _pool_runtime_api_key(_pe)
            if _pk:
                effective_api_key = _pk
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=effective_api_key,
        api_mode=api_mode,
        main_runtime=runtime,
        is_vision=is_vision,
        task=task,
    )
    if client is not None:
        # For async clients, remember which loop they were created on so we
        # can detect stale entries later.
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # Safety belt: if the cache has grown beyond the max, evict
                # the oldest entries (FIFO — dict preserves insertion order).
                # Do not close an evicted client here: another caller may be
                # mid-request with the object it obtained from this cache.
                # Dropping the cache reference lets normal refcount/GC cleanup
                # happen after in-flight users release it.
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key = next(iter(_client_cache))
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                built_client = client
                client, default_model, _ = _client_cache[cache_key]
                # This concurrently built loser was never exposed to a caller,
                # so it is safe to close immediately.
                _close_cached_client(built_client)
    return client, model or default_model
