"""Route-agnostic non-interactive (bearer-token) auth seam for the dashboard.

This is the generic API-token capability (decisions.md Q-C): a reusable seam
that ANY service-to-service / machine-credential provider plugs into, NOT a
drain-specific hook. The drain bearer-secret plugin is merely the first
consumer.

How it fits the existing auth framework:

  * The interactive gate (``gated_auth_middleware``) authenticates a human
    via a session cookie on every non-public route. A service caller has no
    cookie — it presents a bearer token in the ``Authorization`` header on a
    single request. That is what this seam verifies.

  * A route opts in by registering its path via :func:`register_token_route`
    — exact match by default, or ``match="prefix"`` for routes with a dynamic
    segment (e.g. ``/api/sessions/{id}``). Only registered paths are
    token-authable; everything else is untouched, so this can never
    accidentally widen the auth surface of an existing route. Exact
    registrations always take precedence over prefix ones.

  * By default (``required=True``) a registered route is token-EXCLUSIVE:
    once registered, this seam is the only accepted auth scheme for it, and
    a request with no (or an invalid) bearer token is rejected outright —
    this is drain's original contract and remains unchanged. Pass
    ``required=False`` for a route that is ALSO reachable by the existing
    cookie/session gate (e.g. a dashboard page the desktop UI already
    renders): with no ``Authorization`` header at all, the seam is a no-op
    and the request falls through untouched to the downstream cookie gate.
    A *presented* bearer token (even an invalid one) is still fully decided
    by this seam alone — 401/503 as usual — so a genuine token-bearing
    caller gets the same contract either way. If the same path is
    registered more than once with different ``required`` values (exact,
    prefix, or regex), the stricter (``required=True``) always wins.

  * :func:`token_auth_middleware` runs OUTERMOST (installed last in
    ``web_server.py``). For a token route it fully owns the auth decision
    for any request that presents a bearer token, or for any request at all
    on a ``required=True`` route: authenticate via the stacked token
    providers, attach the verified
    :class:`~hermes_cli.dashboard_auth.base.TokenPrincipal` to
    ``request.state.token_principal`` + set ``request.state.token_authenticated``,
    and pass through; otherwise reject (401 unauthenticated, or 503 when a
    provider's backing store was unreachable). The downstream cookie/session
    gates honour ``token_authenticated`` and skip enforcement, so a
    token-authed service request is never bounced to ``/login``. A
    ``required=False`` route with no bearer token presented skips this
    seam entirely and is decided by the downstream cookie/session gates —
    which apply their own, separate trust model (see the module docstring
    caveat in ``web_server.py`` about not conflating a cookie-authenticated
    operator with a DM-scoped token principal).

  * Fails closed: a token route with no registered token provider, no token,
    or an unrecognised token gets 401 — never an open pass-through.

Provider stacking mirrors ``verify_session``: each ``supports_token`` provider
is consulted in registration order until one returns a principal. A provider
that doesn't recognise the token returns ``None`` and the seam moves on; a
provider whose backing store is unreachable raises ``ProviderError``, which the
seam remembers and surfaces as 503 only if NO provider accepts the token.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Awaitable, Callable, Iterable, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from hermes_cli.dashboard_auth import list_token_providers
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import ProviderError, TokenPrincipal

_log = logging.getLogger(__name__)

# Exact paths that accept non-interactive bearer-token auth, mapped to
# whether token auth is exclusive for that path (True) or additive/optional
# (False — see register_token_route's `required` parameter). A route
# registers itself here at import/startup; the seam only acts on registered
# paths.
_token_routes: dict[str, bool] = {}
# Prefix registrations (opt-in, see register_token_route(match="prefix")) — for
# routes with a dynamic path segment (e.g. /api/sessions/{id}) where an exact
# string can't be registered ahead of time. Checked only if no exact match
# hits, so an exact registration always takes precedence over a prefix one.
# Mapped to `required` the same way as _token_routes.
_token_route_prefixes: dict[str, bool] = {}
# Regex registrations (opt-in, see register_token_route(match="regex")) — for a
# dynamic path segment that must be constrained to a known shape (e.g. a
# session id's timestamp+hex format) so that sibling literal routes sharing
# the same prefix (e.g. /api/sessions/search) can never satisfy the pattern.
# Keyed by the raw pattern string for idempotency (compiled re.Pattern objects
# hash by identity, not by pattern text, so a plain dict would re-add on every
# call); value is (compiled_pattern, required). Checked only if no exact
# match hits — same precedence as prefix.
_token_route_regexes: dict[str, "Tuple[re.Pattern[str], bool]"] = {}

# Allowed HTTP methods per registration, keyed identically to the *required*
# registries above (exact path / prefix string / raw regex string). The value
# is a frozenset of upper-case method names, or ``None`` meaning "every method"
# — the default, and the drain provider's contract. A registration with a
# restricted method set makes the seam authenticate ONLY those methods on that
# path; a request using any other method is treated as "not a token route for
# this method" and falls through to the downstream cookie/session gate.
#
# This is the security boundary that keeps a read-only token registration from
# authenticating a mutating request that merely shares a path with a registered
# read route. ``is_token_route`` matches by path only (it cannot see the HTTP
# method), and several dashboard paths mount both a GET handler and a
# POST/PUT/PATCH/DELETE handler on the SAME path (e.g. GET vs POST
# ``/api/skills``, GET vs DELETE/PATCH ``/api/sessions/{id}``). Registering
# ``/api/skills`` token-authable for its GET reader would otherwise also make
# ``POST /api/skills`` (create-skill — agent-executed, i.e. code execution)
# token-authable for the same principal. Restricting the registration to
# ``{"GET", "HEAD"}`` confines the token to the read verb only.
_token_route_methods: dict[str, Optional[frozenset]] = {}
_token_route_prefix_methods: dict[str, Optional[frozenset]] = {}
_token_route_regex_methods: dict[str, Optional[frozenset]] = {}
_lock = threading.Lock()


def _normalize_methods(methods: Optional[Iterable[str]]) -> Optional[frozenset]:
    """``None`` → None ("all methods"); otherwise an upper-cased frozenset."""
    if methods is None:
        return None
    normalized = frozenset(m.strip().upper() for m in methods if m and m.strip())
    # An empty set would deny every method, which is never a useful
    # registration and is almost certainly a caller bug — treat it as "all".
    return normalized or None


def _merge_methods(
    registry: "dict[str, Optional[frozenset]]", key: str, new: Optional[frozenset]
) -> None:
    """Union-merge ``new`` into ``registry[key]``.

    ``None`` is the universal set: merging anything with ``None`` yields
    ``None`` ("all methods"). This mirrors the OR semantics used for repeat
    route registration elsewhere in this module — a method is token-authable
    on a path if ANY registration of that path allows it.
    """
    if key not in registry:
        registry[key] = new
        return
    existing = registry[key]
    if existing is None or new is None:
        registry[key] = None
    else:
        registry[key] = existing | new


def register_token_route(
    path: str,
    *,
    match: str = "exact",
    required: bool = True,
    methods: Optional[Iterable[str]] = None,
) -> None:
    """Mark ``path`` as token-authable.

    Idempotent. Call at module import / app setup so the seam knows which
    routes to guard. Registering a route does NOT make it public — it makes
    it authenticate by token instead of by session cookie.

    ``match`` is one of:

      * ``"exact"`` (default, unchanged behaviour) — matches only the literal
        path string.
      * ``"prefix"`` — matches any path starting with ``path``. Opt-in per
        call; existing exact registrations (e.g. the drain plugin's
        ``/api/gateway/drain``) are never reinterpreted as prefixes.
      * ``"regex"`` — ``path`` is a regex pattern matched with
        :func:`re.fullmatch` against ``request.url.path``. Use this instead
        of ``"prefix"`` when a dynamic path segment must be constrained to a
        known shape so sibling literal routes under the same parent path
        (e.g. ``/api/sessions/search`` next to ``/api/sessions/{id}``)
        cannot be swept in by accident — ``is_token_route`` has no HTTP-method
        awareness, so an over-broad prefix would make those siblings
        token-authable too even though nothing consulted their scope.

    Exact registrations always take precedence over regex/prefix ones.

    ``required`` (default ``True``) controls whether an absent bearer token
    falls through to the downstream cookie/session gate:

      * ``True`` (default) — token auth is exclusive for this path; a
        request with no bearer token is rejected 401 by this seam and never
        reaches the cookie gate. This is drain's original, unchanged
        contract.
      * ``False`` — a request with NO ``Authorization`` header at all is a
        no-op for this seam and falls through to whatever gate would have
        handled it otherwise (e.g. the existing cookie-authenticated
        desktop dashboard). A *presented* bearer token is still fully
        decided by this seam alone, same as ``required=True``.

    If the same path is registered multiple times (possibly with different
    ``required`` values), the stricter value (``True``) always wins — this
    keeps the fail-closed default from being weakened by a later, looser
    registration of the same path.

    ``methods`` restricts which HTTP methods this seam authenticates on
    ``path``. ``None`` (the default) means every method — drain's unchanged
    contract. Pass an explicit set (e.g. ``("GET", "HEAD")``) for a read-only
    registration: the seam then acts only on those methods, and a request
    using any other verb falls through to the cookie/session gate as if the
    path were not token-registered at all. This is required whenever a path
    serves both a read (GET) handler and a mutating (POST/PUT/PATCH/DELETE)
    handler — registering the reader as token-authable would otherwise expose
    the mutator to the same token principal, since route matching is
    path-based and cannot distinguish methods on its own. Method sets union
    across repeat registrations of the same path (a method is authable if ANY
    registration allows it; ``None`` unions to "all").
    """
    if match not in ("exact", "prefix", "regex"):
        raise ValueError(f"register_token_route: invalid match {match!r}")
    normalized_methods = _normalize_methods(methods)
    with _lock:
        if match == "prefix":
            existing = _token_route_prefixes.get(path)
            _token_route_prefixes[path] = required if existing is None else (existing or required)
            _merge_methods(_token_route_prefix_methods, path, normalized_methods)
        elif match == "regex":
            if path not in _token_route_regexes:
                _token_route_regexes[path] = (re.compile(path), required)
            else:
                pattern, existing = _token_route_regexes[path]
                _token_route_regexes[path] = (pattern, existing or required)
            _merge_methods(_token_route_regex_methods, path, normalized_methods)
        else:
            existing = _token_routes.get(path)
            _token_routes[path] = required if existing is None else (existing or required)
            _merge_methods(_token_route_methods, path, normalized_methods)


def is_token_route(path: str) -> bool:
    """True if ``path`` was registered as token-authable.

    An exact registration is checked first; regex and prefix registrations
    only apply if no exact registration already claimed ``path``.
    """
    with _lock:
        if path in _token_routes:
            return True
        if any(pattern.fullmatch(path) for pattern, _required in _token_route_regexes.values()):
            return True
        return any(path.startswith(prefix) for prefix in _token_route_prefixes)


def is_token_route_required(path: str) -> bool:
    """True if token auth is exclusive (no-fallthrough) for ``path``.

    Only meaningful when :func:`is_token_route` is already True for this
    path. An exact registration's ``required`` flag wins outright; otherwise
    this is the OR of every matching regex/prefix registration's flag, so a
    single ``required=True`` registration of an overlapping pattern can never
    be silently loosened by another, looser one (fail closed). Defaults to
    True (exclusive) for a path with no matching registration at all — this
    should not occur in practice since callers only reach here after
    confirming ``is_token_route(path)``.
    """
    with _lock:
        if path in _token_routes:
            return _token_routes[path]
        matched = [
            required
            for pattern, required in _token_route_regexes.values()
            if pattern.fullmatch(path)
        ]
        matched.extend(
            required
            for prefix, required in _token_route_prefixes.items()
            if path.startswith(prefix)
        )
        if not matched:
            return True
        return any(matched)


def is_token_route_method_allowed(path: str, method: str) -> bool:
    """True if the seam should authenticate ``method`` requests to ``path``.

    Only meaningful when :func:`is_token_route` is already True for ``path``.
    An exact registration's method set is authoritative for that path
    (matching the exact-takes-precedence rule of :func:`is_token_route`);
    otherwise a method is allowed if ANY matching regex/prefix registration
    permits it. A registration's ``None`` method set means "all methods".

    Returns False only when ``path`` is registered but every matching
    registration restricts token auth to methods OTHER than ``method`` — the
    security-relevant case (e.g. a GET-only Mini App route hit with DELETE).
    Defaults True when nothing matches at all (defensive; the seam only calls
    this after :func:`is_token_route` already returned True).
    """
    m = method.upper()
    with _lock:
        if path in _token_routes:
            allowed = _token_route_methods.get(path)
            return allowed is None or m in allowed
        any_match = False
        for pattern_str, (pattern, _required) in _token_route_regexes.items():
            if pattern.fullmatch(path):
                any_match = True
                allowed = _token_route_regex_methods.get(pattern_str)
                if allowed is None or m in allowed:
                    return True
        for prefix in _token_route_prefixes:
            if path.startswith(prefix):
                any_match = True
                allowed = _token_route_prefix_methods.get(prefix)
                if allowed is None or m in allowed:
                    return True
        # Matches existed but none permitted this method → deny (fall through).
        # No match at all → default allow (unreachable via the seam's guard).
        return not any_match


def clear_token_routes() -> None:
    """Test-only: drop all registered token routes (exact, prefix, and regex)."""
    with _lock:
        _token_routes.clear()
        _token_route_prefixes.clear()
        _token_route_regexes.clear()
        _token_route_methods.clear()
        _token_route_prefix_methods.clear()
        _token_route_regex_methods.clear()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def extract_bearer_token(request: Request) -> str:
    """Return the bearer token from the ``Authorization`` header, or "".

    Accepts ``<scheme> <token>`` where scheme is "bearer" (case-insensitive).
    Returns an empty string for a missing/malformed header or a non-bearer
    scheme — the caller treats "" as "no token presented".
    """
    auth = request.headers.get("authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) == 2 and parts[0].strip().lower() == "bearer":
        return parts[1].strip()
    return ""


def authenticate_token(
    request: Request,
) -> Tuple[Optional[TokenPrincipal], Optional[str]]:
    """Try every token provider against the request's bearer token.

    Returns ``(principal, unreachable_provider_name)``:
      * ``(TokenPrincipal, None)`` — a provider recognised and accepted the token.
      * ``(None, None)`` — no token, or no provider recognised it (reject 401).
      * ``(None, name)`` — no provider accepted it AND at least one provider's
        backing store was unreachable (the caller surfaces 503, not 401, so a
        transient outage doesn't read as "bad credentials").

    Never raises: a provider ``ProviderError`` is caught and remembered.
    """
    token = extract_bearer_token(request)
    if not token:
        return None, None
    unreachable: Optional[str] = None
    for provider in list_token_providers():
        try:
            principal = provider.verify_token(token=token)
        except ProviderError as e:
            _log.warning(
                "dashboard-auth: token provider %r unreachable during verify: %s",
                provider.name, e,
            )
            if unreachable is None:
                unreachable = provider.name
            continue
        except Exception as e:  # noqa: BLE001 — a buggy provider must not 500 the gate
            _log.warning(
                "dashboard-auth: token provider %r raised during verify: %s",
                provider.name, e,
            )
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


async def token_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Outermost auth seam for token-authable routes.

    No-op pass-through for any path not registered via
    :func:`register_token_route`, and for any request whose HTTP method the
    route's registration does not permit (see the ``methods`` parameter) —
    that also falls through to the cookie/session gate, so a read-only
    registration cannot authenticate a mutating verb sharing the same path.
    For a registered path+method with ``required=True`` (the default), token
    auth is the only accepted scheme:

      * valid token  → attach principal + ``token_authenticated`` flag, pass through.
      * unreachable  → 503 (provider backing store down; not "bad credentials").
      * otherwise    → 401 unauthenticated.

    For a ``required=False`` route, a request with NO bearer token at all is
    also a no-op pass-through — it falls to the downstream cookie/session
    gate untouched, so a route reachable by both the desktop dashboard
    (cookie) and a token principal (e.g. a Telegram Mini App) doesn't lock
    out the cookie-authenticated caller. A *presented* token on a
    ``required=False`` route is still decided exclusively by this seam
    (valid/unreachable/401 exactly as above) — only the "no token at all"
    case falls through.

    Runs before the cookie/session gates (installed last in ``web_server.py``).
    The cookie gates honour ``request.state.token_authenticated`` and skip
    enforcement, so a token-authed request is never redirected to ``/login``.
    """
    path = request.url.path
    if not is_token_route(path):
        return await call_next(request)

    # Method gate (security boundary): a route registered token-authable only
    # for certain methods (e.g. a read-only Mini App GET route) must NOT let
    # the seam authenticate a different verb that shares the same path (e.g.
    # POST /api/skills, DELETE /api/sessions/{id}). For a disallowed method
    # the seam does not act at all — the request falls through to the
    # cookie/session gate, which rejects a caller with no session cookie.
    if not is_token_route_method_allowed(path, request.method):
        return await call_next(request)

    if not extract_bearer_token(request) and not is_token_route_required(path):
        return await call_next(request)

    principal, unreachable = authenticate_token(request)
    if principal is not None:
        request.state.token_principal = principal
        request.state.token_authenticated = True
        return await call_next(request)

    if unreachable:
        audit_log(
            AuditEvent.TOKEN_AUTH_FAILURE,
            provider=unreachable,
            reason="provider_unreachable",
            path=path,
            ip=_client_ip(request),
        )
        return JSONResponse(
            {"detail": f"Auth provider {unreachable!r} unreachable"},
            status_code=503,
        )

    audit_log(
        AuditEvent.TOKEN_AUTH_FAILURE,
        reason="no_provider_recognises_token",
        path=path,
        ip=_client_ip(request),
    )
    return JSONResponse(
        {"error": "unauthenticated", "detail": "Unauthorized"},
        status_code=401,
    )
