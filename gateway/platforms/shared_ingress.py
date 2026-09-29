"""Shared-listener ingress for inbound-port platforms under ``gateway.multiplex_profiles``.

The default profile owns the ONE HTTP listener (api_server, and the webhook adapter's port). A
secondary profile's port-binding adapter (Twilio SMS, LINE, Teams, BlueBubbles, Microsoft Graph,
WhatsApp Cloud, WeCom callback, Feishu webhook mode) therefore cannot bind its own port; instead the
runner constructs it in *shared-listener mode*: ``bind_listener`` publishes the adapter's fully wired
``web.Application`` instead of starting a ``TCPSite``, and the default listener forwards
``/p/<profile>/<path>`` to it through ``dispatch_profile_ingress``.

Invariants: the forwarded request runs under the NAMED profile's runtime scope and is verified by
that profile's adapter with that profile's secret; the un-prefixed path keeps serving the default
profile untouched; a profile with no adapter for the path gets a 404, never the default's adapter.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # aiohttp is an optional dependency of every adapter using this module
    from aiohttp import web

logger = logging.getLogger(__name__)

_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "*"})


def shared_ingress_profile(adapter: Any) -> Optional[str]:
    """Profile name when *adapter* was constructed in shared-listener mode, else None."""
    return getattr(adapter, "_shared_listener_profile", None) or None


def listener_base_url(host: Any, port: Any) -> str:
    """``http://host:port`` clients use to reach a listener bound on ``host`` (wildcards → loopback)."""
    host = "127.0.0.1" if host is None or str(host).strip() in _WILDCARD_HOSTS else str(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port or 0}"


def shared_listener_base(runner: Any) -> Optional[str]:
    """``http://host:port`` of the default profile's live listener (api_server first, then webhook)."""
    from gateway.config import Platform
    adapters = getattr(runner, "adapters", None) or {}
    for platform in (Platform.API_SERVER, Platform.WEBHOOK):
        adapter = adapters.get(platform)
        if adapter is None:
            continue
        return listener_base_url(getattr(adapter, "_host", None), getattr(adapter, "_port", 0))
    return None


async def bind_listener(
    adapter: Any, app: "web.Application", host: Optional[str], port: int, ingress_path: str, *,
    reuse_address: Optional[bool] = None, access_log: Any = ...,
) -> Optional["web.AppRunner"]:
    """Start *app* on ``host:port`` and return its ``AppRunner`` — or, in shared-listener mode,
    publish *app* for ``/p/<profile>/`` forwarding and return None (nothing bound). ``ingress_path``
    is the adapter's primary callback path, used for the log line and runtime status."""
    from aiohttp import web
    profile = shared_ingress_profile(adapter)
    if profile:
        publish_shared_ingress(adapter, app, ingress_path)
        return None
    runner = web.AppRunner(app) if access_log is ... else web.AppRunner(app, access_log=access_log)
    await runner.setup()
    site = web.TCPSite(runner, host, port, reuse_address=reuse_address)
    try:
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    return runner


def publish_shared_ingress(adapter: Any, app: "web.Application", ingress_path: str) -> None:
    """Freeze *app* and expose it to the default listener; records the ``/p/<profile>/`` URL."""
    profile = shared_ingress_profile(adapter)
    app.freeze()
    adapter._shared_ingress_app = app
    base = shared_listener_base(getattr(adapter, "gateway_runner", None))
    prefix = f"/p/{profile}"
    adapter._shared_ingress_base = f"{base}{prefix}" if base else prefix
    adapter._shared_ingress_url = f"{adapter._shared_ingress_base}{ingress_path}"
    platform = getattr(getattr(adapter, "platform", None), "value", "adapter")
    if base:
        logger.info(
            "[%s] profile '%s' is served on the default profile's shared listener: %s "
            "(point the vendor's callback URL at this path behind your public host)",
            platform, profile, adapter._shared_ingress_url,
        )
    else:
        logger.warning(
            "[%s] profile '%s' is in shared-listener mode but the default profile has no live api_server "
            "or webhook listener yet; it will be reachable at <listener>%s%s once one is up",
            platform, profile, prefix, ingress_path,
        )
    write = getattr(adapter, "_write_runtime_status_safe", None)
    if callable(write):
        write("shared_ingress", ingress_url=adapter._shared_ingress_url)


def shared_ingress_apps(runner: Any, profile: Optional[str]) -> list[tuple[Any, "web.Application"]]:
    """``(adapter, app)`` for every shared-listener adapter of a NAMED served profile. ``default`` and
    unknown profiles yield nothing: the default's port-binders own their own ports, and a profile
    without a live adapter must never fall back to another profile's."""
    if not profile or profile == "default":
        return []
    adapters = (getattr(runner, "_profile_adapters", None) or {}).get(profile) or {}
    return [
        (adapter, app) for adapter in adapters.values()
        if (app := getattr(adapter, "_shared_ingress_app", None)) is not None
    ]


async def dispatch_profile_ingress(
    runner: Any, profile: Optional[str], tail: str, request: "web.Request", *, scoped: bool = False,
) -> "web.StreamResponse":
    """Forward ``/p/<profile>/<tail>`` to the served profile's adapter app that routes ``/<tail>``,
    under that profile's runtime scope (``scoped=True`` when the caller already entered it). 404 when
    no adapter of *profile* serves the path."""
    from aiohttp import web
    from aiohttp.web_urldispatcher import MatchInfoError
    candidates = shared_ingress_apps(runner, profile)
    if not profile or not candidates:
        raise web.HTTPNotFound(text="Unknown or unconfigured profile")
    rel_url = request.rel_url.with_path("/" + tail.lstrip("/"), keep_query=True, keep_fragment=True)
    forwarded = request.clone(rel_url=rel_url)
    chosen = None
    for _adapter, app in candidates:
        match = await app.router.resolve(forwarded)
        # A 404 means "not my path": try the profile's next adapter; 405 and real matches belong here.
        if isinstance(match, MatchInfoError) and match.http_exception.status == 404:
            continue
        chosen = app
        break
    if chosen is None:
        raise web.HTTPNotFound(text="No adapter serves this path for the profile")
    # Each adapter chose its own body cap when it built its Application; honour it for the forwarded read.
    forwarded = request.clone(rel_url=rel_url, client_max_size=chosen._client_max_size)
    if scoped:
        return await chosen._handle(forwarded)
    from gateway.run import _profile_runtime_scope
    from hermes_cli.profiles import get_profile_dir
    with _profile_runtime_scope(get_profile_dir(profile)):
        return await chosen._handle(forwarded)
