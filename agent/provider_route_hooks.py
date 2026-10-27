"""Ask provider profiles about an endpoint: the named provider first, then the profile serving the host.

Endpoint-keyed ProviderProfile hooks (``client_headers``, ``request_headers``, ``max_tokens_param``,
``refresh_runtime_credentials``) belong to the provider by name AND to the backend by host: a
``custom`` route pointed at a plugin provider's host still gets that provider's quirks. The host
side is ``providers.profile_for_host`` (the profile's declared ``hostnames``), core's one
endpoint->profile lookup. Only profiles that override the hook are consulted: the base
implementations are no-ops, and hosted (``plugins.isolation: host``) proxies define only the hooks
their plugin overrides.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

# Per-request hooks documented pure in providers/base.py: one answer per argument set, so a hosted
# plugin pays one IPC round trip per distinct request shape instead of one per API call.
_PURE_HOOKS = frozenset({"request_headers", "max_tokens_param", "reasoning_payload",
                         "replays_function_thought_signature"})
_PURE_MEMO: dict[tuple, Any] = {}
_PURE_MEMO_MAX = 512
_MISSING = object()
# (profile name, hook) pairs whose failure a scopeless (one-shot) caller already logged: one warning per
# pair per process there; a scoped caller (an agent) warns once per pair per session.
_WARNED: set[tuple] = set()


def _overrides(profile: Any, hook: str) -> bool:
    return getattr(type(profile), hook, None) is not getattr(ProviderProfile, hook)


def route_profiles(hook: str, provider: str = "", base_url: str = "") -> list[Any]:
    """Profiles overriding *hook*: the one registered for *provider*, then the one serving *base_url*'s host."""
    from providers import get_provider_profile, profile_for_host

    named = get_provider_profile(provider) if provider else None
    ordered = [named] if named is not None and _overrides(named, hook) else []
    hosted = profile_for_host(base_url) if base_url else None
    if hosted is not None and hosted is not named and _overrides(hosted, hook):
        ordered.append(hosted)
    return ordered


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _ask(profile: Any, hook: str, args: tuple, kwargs: dict[str, Any]) -> Any:
    if hook not in _PURE_HOOKS:
        return getattr(profile, hook)(*args, **kwargs)
    from providers import registry_generation

    key = (registry_generation(), id(profile), type(profile), hook, _freeze(args), _freeze(kwargs))
    try:
        hash(key)
    except TypeError:  # an argument with no stable value (e.g. a set): ask every time
        return getattr(profile, hook)(*args, **kwargs)
    # One read into a local: another thread may clear() the memo between a check and a re-read.
    answer = _PURE_MEMO.get(key, _MISSING)
    if answer is _MISSING:
        answer = getattr(profile, hook)(*args, **kwargs)
        if len(_PURE_MEMO) >= _PURE_MEMO_MAX:
            _PURE_MEMO.clear()
        _PURE_MEMO[key] = answer
    return copy.deepcopy(answer) if isinstance(answer, (dict, list)) else answer


def _failures(scope: Any) -> set[tuple]:
    if scope is None:  # one-shot caller (auxiliary request): no later turn to keep stable
        return set()
    failed = getattr(scope, "_provider_hook_failures", None)
    if failed is None:
        failed = set()
        scope._provider_hook_failures = failed
    return failed


def _sticky_key(profile: Any, hook: str) -> tuple:
    return (id(profile), type(profile), getattr(profile, "name", "?"), hook)


def hook_failed(profile: Any, hook: str, scope: Any = None) -> bool:
    """True once *profile*'s *hook* failed in *scope* (``ask_profile`` then answers the fallback)."""
    return _sticky_key(profile, hook) in _failures(scope)


def ask_profile(profile: Any, hook: str, fallback: Any, *args: Any, scope: Any = None, **kwargs: Any) -> Any:
    """*profile*'s answer to the per-request *hook* (documented-pure hooks are memoized).

    ONE failure policy for every per-request profile hook: a failure answers *fallback* (the generic
    answer) and *scope* (the agent) keeps answering it without asking again until its session resets, so
    a session's request bytes never flip between turns after a plugin bug or a lost plugin host. It is
    logged once per (profile, hook) per session, so each served profile's agent reports its own
    degradation. ``scope=None`` (one-shot auxiliary calls) asks again next call, warning once per process.
    """
    failed = _failures(scope)
    sticky = _sticky_key(profile, hook)
    if sticky in failed:
        return fallback
    try:
        return _ask(profile, hook, args, kwargs)
    except Exception as exc:  # plugin code: a broken hook degrades to generic behaviour, never fails the request
        failed.add(sticky)
        if scope is not None or sticky[2:] not in _WARNED:
            _WARNED.add(sticky[2:])
            logger.warning("provider %s: %s hook failed (%s: %s); using the generic answer for the rest of "
                           "the session", sticky[2], hook, type(exc).__name__, exc, exc_info=True)
        return fallback


def ask_on_error_path(profile: Any, hook: str, fallback: Any, **kwargs: Any) -> Any:
    """*profile*'s answer to an API-error-path *hook* (credential refresh, stale-credential test, refresh
    notice), asked every time: a lost plugin host may be back for the next error. A raise answers
    *fallback* ("not refreshed"), logged with its traceback once per (provider, hook) per process, so the
    original API error still reaches the fallback walk instead of escaping its handler."""
    try:
        return getattr(profile, hook)(**kwargs)
    except Exception:  # health: allow BLE001 -- plugin hook boundary; logged with traceback once below
        key = (getattr(profile, "name", "?"), hook)
        if key not in _ERROR_PATH_WARNED:
            _ERROR_PATH_WARNED.add(key)
            logger.warning("provider %s: %s hook failed; handling the API error without it", key[0], hook,
                           exc_info=True)
        return fallback


_ERROR_PATH_WARNED: set[tuple] = set()


def model_api_mode(profile: Any, model: str, *, api_key: str | None = None) -> str | None:
    """*profile*'s ``api_mode_for_model`` answer for *model* (None = no per-model wire). The ONE guard for
    resolution, ``/model``, resume and the agent: a raising hook (a live catalog fetch, a lost plugin host)
    answers the profile's declared ``api_mode``, logged with its traceback once per provider per process,
    so a model pick never fails where runtime resolution succeeded."""
    try:
        return profile.api_mode_for_model(model, api_key=api_key)
    except Exception:  # health: allow BLE001 -- plugin hook boundary; logged with traceback once below
        name = getattr(profile, "name", "?")
        if name not in _API_MODE_WARNED:
            _API_MODE_WARNED.add(name)
            logger.warning("provider %s: api_mode_for_model failed; using its declared api_mode", name,
                           exc_info=True)
        return profile.api_mode or "chat_completions"


_API_MODE_WARNED: set[str] = set()


def route_answer(hook: str, *, provider: str = "", scope: Any = None, **kwargs: Any) -> Any:
    """First truthy answer of *hook* across ``route_profiles``; None when no profile claims the endpoint.
    Failures follow ``ask_profile``'s policy (*scope* = the agent, when there is one)."""
    for profile in route_profiles(hook, provider, str(kwargs.get("base_url") or "")):
        answer = ask_profile(profile, hook, None, scope=scope, **kwargs)
        if answer:
            return answer
    return None
