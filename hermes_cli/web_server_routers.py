"""Mount APIRouters on the dashboard app without FastAPI's per-route copy.

``FastAPI.include_router`` rebuilds every route of the included router (a fresh ``APIRoute`` with its
own dependant graph, response field and pydantic validators) and the module-level router keeps the
originals alive, so each dashboard route existed twice: ~4 MB of an idle gateway's Python heap.
For a router that adds nothing on top of its routes (no prefix, dependencies, tags, responses,
callbacks, custom response class, id generator or content-type strictness, lifespan or event
handlers), mounted on an app router with defaults (and no prefix, tags, responses or callbacks of
its own, which ``add_api_route`` merges into every route it builds), the rebuilt route is equivalent
to the original one, so the original is appended instead. The included router's own prefix, tags,
responses and callbacks were already baked into its routes when they were declared, so refusing
them is conservative rather than required; it keeps the equivalence argument trivially true.
Anything else, or a FastAPI whose defaults this module cannot recognise, goes through
``include_router`` unchanged. Only public attributes are read, so this works across the supported
FastAPI range; tests/hermes_cli/test_web_server_routers.py builds the dashboard both ways and
compares the route tables, their attribute sets and the OpenAPI schema, so FastAPI drift is caught.

Caveat: an adopted route keeps ``dependency_overrides_provider=None``, so ``app.dependency_overrides``
does not reach it (Hermes uses none; a test that needs overrides includes the router into its own app).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.datastructures import DefaultPlaceholder
from fastapi.routing import APIRoute, APIRouter, APIWebSocketRoute
from starlette.routing import Route, WebSocketRoute


def _default_lifespan_type() -> type | None:
    """The type of a lifespan-less router's ``lifespan_context``, learnt from FastAPI itself (the
    class moved from Starlette into FastAPI across versions, so it is not imported). None when that
    default is a plain function or bound method, which a custom lifespan could also be."""
    kind = type(APIRouter().lifespan_context)
    return None if kind.__module__ == "builtins" else kind


_DEFAULT_LIFESPAN = _default_lifespan_type()


def _strict_by_default(router: APIRouter) -> bool:
    """True when *router* leaves request Content-Type checking at FastAPI's default (strict), which
    is what an adopted route that never set it applies. ``FastAPI()`` stores a plain ``True``, an
    ``APIRouter`` a placeholder; FastAPI before 0.130 has no such setting."""
    value = getattr(router, "strict_content_type", True)
    return (value.value if isinstance(value, DefaultPlaceholder) else value) is True


def _adds_nothing(router: APIRouter) -> bool:
    """True when *router*'s own settings change nothing in the routes ``include_router`` builds (the
    including side and the included side share these: dependencies, deprecation, schema visibility,
    response class, operation-id generator and content-type strictness)."""
    return (not router.dependencies and not router.deprecated and router.include_in_schema
            and isinstance(router.default_response_class, DefaultPlaceholder)
            and isinstance(router.generate_unique_id_function, DefaultPlaceholder)
            and _strict_by_default(router))


def _plain(router: APIRouter) -> bool:
    """An included router with nothing of its own beyond its routes."""
    return (_DEFAULT_LIFESPAN is not None and _adds_nothing(router) and not router.prefix
            and not router.tags and not router.responses and not router.callbacks
            and not router.on_startup and not router.on_shutdown
            and type(router.lifespan_context) is _DEFAULT_LIFESPAN)


def can_adopt(app: FastAPI, router: APIRouter) -> bool:
    """True when appending *router*'s own routes to *app* is equivalent to including it."""
    target = app.router
    return _plain(router) and _adds_nothing(target) and not target.prefix and not target.tags \
        and not target.responses and not target.callbacks and all(
        isinstance(r, (APIRoute, APIWebSocketRoute, Route, WebSocketRoute)) and r.path
        for r in router.routes)


def adopt_router(app: FastAPI, router: APIRouter) -> None:
    """``app.include_router(router)`` that shares the router's own route objects when that is
    equivalent (see module docstring)."""
    if can_adopt(app, router):
        app.router.routes.extend(router.routes)
    else:
        app.include_router(router)
