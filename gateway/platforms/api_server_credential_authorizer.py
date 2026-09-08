"""Credential-authorizer ownership for the aiohttp API-server facade.

The adapter remains responsible for ordinary API routing, static operator authentication,
and profile scopes.  This module owns the bounded plugin-authorizer runner, request authority
context, credential middleware, continuing SSE reauthorization, and authorizer lifecycle.
"""

import asyncio
import hashlib
import inspect
import logging
import threading
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

from gateway.api_credentials import (
    APIServerOperation,
    AuthorizedAPICredential,
    CredentialAuthorizationRequest,
)

try:
    from aiohttp import web
    from aiohttp.web_request import RequestKey
except ImportError:
    web = None  # type: ignore[assignment]
    RequestKey = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Distinct from ``None``: the URL explicitly names a profile this gateway does not serve.
PROFILE_REJECTED = object()


@dataclass(frozen=True, slots=True)
class _CredentialAuthContext:
    principal: Optional[AuthorizedAPICredential]
    owner_key: Optional[str]
    static_admin: bool = False
    authorization_request: Optional[CredentialAuthorizationRequest] = None
    authorizer: Any = None
    authority_generation: Optional[int] = None


_api_request_auth_context: ContextVar[Optional[_CredentialAuthContext]] = ContextVar(
    "api_server_request_auth_context", default=None
)
_API_CREDENTIAL_AUTH_REQUEST_KEY = (
    RequestKey("hermes.api_credential_auth", _CredentialAuthContext)
    if RequestKey is not None
    else "hermes.api_credential_auth"
)


class _CredentialAuthorizerSaturated(RuntimeError):
    """The bounded authorizer facility has no free execution slot."""


class _CredentialAuthorizerRunner:
    """Deadline-observing async-only authorizer runner with bounded in-flight capacity.

    Capacity remains reserved until a cancelled child actually exits.  The process-level
    plugin manager owns and closes this runner; individual adapters must not close it.
    """

    def __init__(self, capacity: int):
        self._capacity = max(1, int(capacity))
        self._active = 0
        self._lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def _acquire(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("API credential authorizer runner is shut down")
            if self._active >= self._capacity:
                raise _CredentialAuthorizerSaturated
            self._active += 1

    def _release(self, task: asyncio.Task) -> None:
        with self._lock:
            self._tasks.discard(task)
            self._active -= 1
        with suppress(asyncio.CancelledError, Exception):
            task.exception()

    async def run(self, authorize, request, *, timeout: float):
        self._acquire()
        loop = asyncio.get_running_loop()
        try:
            task = loop.create_task(authorize(request))
        except BaseException:
            with self._lock:
                self._active -= 1
            raise
        with self._lock:
            if self._closed:
                self._active -= 1
                task.cancel()
                raise RuntimeError("API credential authorizer runner is shut down")
            self._tasks.add(task)
        task.add_done_callback(self._release)
        timeout_seconds = max(0.0, float(timeout))
        started_at = loop.time()
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            task.cancel()
            remaining = max(0.0, timeout_seconds - (loop.time() - started_at))
            deadline_handle = loop.call_later(remaining, task.cancel)
            task.add_done_callback(lambda _task: deadline_handle.cancel())
            raise
        if task in done:
            return task.result()
        task.cancel()
        raise asyncio.TimeoutError

    def close(self) -> None:
        with self._lock:
            self._closed = True
            tasks = tuple(self._tasks)
        for task in tasks:
            with suppress(RuntimeError):
                task.get_loop().call_soon_threadsafe(task.cancel)


def _prefix_names_served_profile(profile: str) -> bool:
    try:
        from hermes_cli.profiles import profile_matches_home

        return profile_matches_home(profile)
    except Exception:
        return False


class CredentialAuthorizerMixin:
    """Topical credential-authorizer behavior composed by ``APIServerAdapter``."""

    _API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 5.0
    _API_CREDENTIAL_AUTH_MAX_INFLIGHT = 4

    def _current_api_credential_authorizer(self):
        manager = self._api_credential_authorizer_manager
        if manager is None:
            return None, self._api_credential_authorizer
        generation, authorizer = manager.get_api_server_credential_authorizer_snapshot()
        self._api_credential_authorizer_generation = generation
        return generation, authorizer

    def _credential_authorizer_runner(self) -> _CredentialAuthorizerRunner:
        if self._api_credential_authorizer_runner is not None:
            return self._api_credential_authorizer_runner
        from hermes_cli.plugins import get_plugin_manager

        runner = get_plugin_manager().get_api_server_credential_authorizer_runner(
            capacity=self._API_CREDENTIAL_AUTH_MAX_INFLIGHT,
            factory=_CredentialAuthorizerRunner,
        )
        if not isinstance(runner, _CredentialAuthorizerRunner):
            raise RuntimeError(
                f"Expected _CredentialAuthorizerRunner from plugin manager, got {type(runner)!r}"
            )
        self._api_credential_authorizer_runner = runner
        return runner

    def _make_profile_prefix_middleware(self):
        """Resolve URL/static/plugin authority, then enter the server-derived profile scope."""

        @web.middleware
        async def profile_prefix_middleware(request: "web.Request", handler):
            url_profile = self._resolve_request_profile(request)
            if url_profile is PROFILE_REJECTED:
                return web.json_response(
                    {"error": "Unknown or unconfigured profile"}, status=404
                )

            auth_header = request.headers.get("Authorization", "")
            bearer = (
                auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
            )
            with self._profile_scope(url_profile):
                expected_key = self._expected_api_key_for_profile(url_profile)
            if bearer and expected_key and self._tokens_match(bearer, expected_key):
                context = _CredentialAuthContext(None, None, static_admin=True)
                return await self._run_scoped_request(
                    request, handler, url_profile, context
                )

            operation, canonical_route = self._credential_route_metadata(request)
            try:
                authority_generation, authorizer = (
                    self._current_api_credential_authorizer()
                )
            except Exception:
                logger.warning(
                    "[%s] API credential authority is unavailable or ambiguous for %s %s",
                    self.name,
                    request.method.upper(),
                    canonical_route,
                )
                return self._auth_failed_response()
            if not bearer or operation is None or authorizer is None:
                return await self._run_scoped_request(
                    request, handler, url_profile, None
                )
            try:
                auth_request = CredentialAuthorizationRequest(
                    bearer=bearer,
                    method=request.method.upper(),
                    canonical_route=canonical_route,
                    operation=operation,
                )
                result = await self._credential_authorizer_runner().run(
                    authorizer.authorize,
                    auth_request,
                    timeout=self._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS,
                )
                if self._api_credential_authorizer_manager is not None:
                    current_generation, current_authorizer = (
                        self._api_credential_authorizer_manager.get_api_server_credential_authorizer_snapshot()
                    )
                    if (
                        current_generation != authority_generation
                        or current_authorizer is not authorizer
                    ):
                        return self._auth_failed_response()
            except Exception:
                logger.warning(
                    "[%s] API credential authorizer rejected after an internal failure for %s %s",
                    self.name,
                    request.method.upper(),
                    canonical_route,
                )
                return self._auth_failed_response()
            if type(result) is not AuthorizedAPICredential:
                return self._auth_failed_response()
            if operation not in result.allowed_operations:
                return self._credential_forbidden_response()
            if not self._credential_profile_is_served(result.runtime_profile):
                return self._credential_forbidden_response()
            if url_profile and url_profile != result.runtime_profile:
                return self._credential_forbidden_response()
            manager = self._api_credential_authorizer_manager
            if (
                manager is not None
                and not manager.admit_api_server_credential_authorizer(
                    authority_generation, authorizer
                )
            ):
                return self._auth_failed_response()
            context = _CredentialAuthContext(
                result,
                self._credential_owner_key(result),
                authorization_request=auth_request,
                authorizer=authorizer,
                authority_generation=authority_generation,
            )
            return await self._run_scoped_request(
                request, handler, result.runtime_profile, context
            )

        return profile_prefix_middleware

    @staticmethod
    def _credential_forbidden_response() -> "web.Response":
        return web.json_response(
            {
                "error": {
                    "message": "Credential is not authorized for this operation",
                    "type": "gateway_auth_error",
                    "code": "gateway_auth_forbidden",
                }
            },
            status=403,
        )

    def _credential_route_metadata(
        self, request: "web.Request"
    ) -> tuple[Optional[APIServerOperation], str]:
        canonical = ""
        with suppress(Exception):
            canonical = str(request.match_info.route.resource.canonical)
        prefix = "/p/{profile}"
        if canonical.startswith(prefix):
            canonical = canonical[len(prefix) :] or "/"
        method = request.method.upper()
        for route in self._http_route_table():
            if route.method == method and route.path == canonical:
                return route.credential_operation, canonical
        return None, canonical

    def _credential_profile_is_served(self, profile: str) -> bool:
        cfg = getattr(self.gateway_runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            return _prefix_names_served_profile(profile)
        try:
            from hermes_cli.profiles import profiles_to_serve

            return profile in {
                name
                for name, _home in profiles_to_serve(
                    multiplex=True,
                    profile_allowlist=getattr(cfg, "multiplex_profile_allowlist", None),
                )
            }
        except Exception:
            return False

    @staticmethod
    def _credential_owner_key(principal: AuthorizedAPICredential) -> str:
        parts = (
            principal.runtime_profile,
            principal.principal_id,
            principal.agent_profile_id.value,
            principal.credential_scope_id.value,
        )
        return "api-credential:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()

    @staticmethod
    def _credential_owner() -> Optional[str]:
        context = _api_request_auth_context.get()
        return (
            context.owner_key if isinstance(context, _CredentialAuthContext) else None
        )

    async def _credential_context_is_current(
        self, request: "web.Request", operation: APIServerOperation
    ) -> bool:
        """Revalidate continuing authority before an admitted streaming disclosure."""
        try:
            context = request.get(_API_CREDENTIAL_AUTH_REQUEST_KEY)
        except (AttributeError, TypeError):
            return False
        if not isinstance(context, _CredentialAuthContext):
            return self._check_auth(request) is None
        if context.static_admin:
            return True
        auth_request = context.authorization_request
        authorizer = context.authorizer
        principal = context.principal
        if (
            not isinstance(auth_request, CredentialAuthorizationRequest)
            or authorizer is None
            or type(principal) is not AuthorizedAPICredential
            or auth_request.operation is not operation
        ):
            return False
        manager = self._api_credential_authorizer_manager
        if manager is not None and not manager.admit_api_server_credential_authorizer(
            context.authority_generation, authorizer
        ):
            return False
        try:
            current = await self._credential_authorizer_runner().run(
                authorizer.authorize,
                auth_request,
                timeout=self._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS,
            )
        except Exception:
            return False
        if manager is not None and not manager.admit_api_server_credential_authorizer(
            context.authority_generation, authorizer
        ):
            return False
        return (
            type(current) is AuthorizedAPICredential
            and current == principal
            and operation in current.allowed_operations
        )

    def _load_api_credential_authorizer(self) -> bool:
        """Resolve the listener owner's optional authorizer, rejecting ambiguity."""
        try:
            from hermes_cli.plugins import get_plugin_manager

            manager = get_plugin_manager()
        except Exception:
            logger.error(
                "[%s] Refusing to start: API credential authorizer discovery failed",
                self.name,
            )
            self._set_fatal_error(
                "api_credential_authorizer_discovery_failed",
                "API credential authorizer discovery failed.",
                retryable=False,
            )
            return False
        try:
            generation, authorizer = (
                manager.get_api_server_credential_authorizer_snapshot()
            )
        except RuntimeError:
            logger.error(
                "[%s] Refusing to start: multiple API credential authorizers are registered",
                self.name,
            )
            self._set_fatal_error(
                "api_credential_authorizer_ambiguous",
                "Multiple API credential authorizers are registered; enable exactly one.",
                retryable=False,
            )
            return False
        self._api_credential_authorizer_manager = manager
        self._api_credential_authorizer_generation = generation
        self._api_credential_authorizer = authorizer
        if authorizer is None:
            return True
        if not inspect.iscoroutinefunction(getattr(authorizer, "authorize", None)):
            logger.error(
                "[%s] Refusing to start: API credential authorizer must define async authorize()",
                self.name,
            )
            self._set_fatal_error(
                "api_credential_authorizer_not_async",
                "API credential authorizer must define async authorize(request).",
                retryable=False,
            )
            return False
        return True
