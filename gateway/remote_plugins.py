"""Remote gateway plugins — the scheme-dispatch seam for non-HTTP transports.

The server half of the remote gateway plugins extension point (the desktop
half lives in ``apps/desktop/electron/remote-plugin-registry.ts`` and shares
its contract: a remote plugin registers a URL scheme and owns turning pairing
material into a live transport; core never parses it).

Two roles, one ABC:

* ``serve()`` — the accept side: expose a local gateway (the dashboard /
  web_server URL) over the plugin's transport and hand back a pairing
  artifact (ticket, address, hostname…) that a client can dial.
* ``connect()`` — the DIAL side (optional, for headless CLI/automation):
  turn a saved ``scheme://…`` url into a live ``http://127.0.0.1:<port>``
  access point. Everything downstream consumes plain HTTP.

Registry semantics mirror ``agent.context_references``: duplicate
registrations raise ``ValueError``, the built-in ``http``/``https`` schemes
are reserved to core, and ``get*`` returns a snapshot copy.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Built-in transports stay core; a remote plugin registering one is a bug.
BUILTIN_SCHEMES = frozenset({"http", "https"})

#: RFC 3986 scheme token, lowercased at registration time.
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*$")


@dataclass
class ServeHandle:
    """Accept-side handle: what the world dials, and how to stop serving.

    ``ticket`` is the pairing artifact (plugin-shaped: an example ticket, a
    a hostname, …). Core stores/serves it verbatim — it is opaque.
    ``target_url`` is what the plugin binds its access point on (the
    gateway url it forwards to).
    """

    ticket: str = ""
    target_url: str = ""
    teardown: Callable[[], Awaitable[None]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    async def stop(self) -> None:
        """Tear down the accept side. Idempotent; errors are logged, not raised."""
        teardown, self.teardown = self.teardown, None

        if teardown is not None:
            try:
                await teardown()
            except Exception:
                logger.warning("remote plugin serve teardown failed", exc_info=True)


@dataclass
class RemoteTarget:
    """Dial-side result: a live loopback origin plus its teardown."""

    base_url: str
    teardown: Callable[[], Awaitable[None]] | None = None
    host_label: str = ""

    async def stop(self) -> None:
        teardown, self.teardown = self.teardown, None

        if teardown is not None:
            try:
                await teardown()
            except Exception:
                logger.warning("remote plugin dial teardown failed", exc_info=True)


class RemotePlugin(ABC):
    """Base class for remote gateway remote plugins.

    Registered via ``ctx.register_remote_plugin(impl())`` in a plugin's
    ``register(ctx)``; dispatched by URL scheme. ``serve()`` is the core
    obligation (a client that only ever dials can skip it by raising
    ``NotImplementedError`` from serve and implementing ``connect()``).
    """

    #: Dispatch scheme, WITHOUT '://' (e.g. 'example'). Lowercased on register.
    scheme: str = ""
    #: Owning plugin's manifest name, stamped by the plugin host at
    #: registration; the config block is looked up as
    #: ``plugins.entries.<plugin_id>`` in config.yaml.
    plugin_id: str = ""
    #: Human hint for connection-setup UIs ("paste the pairing code").
    hint: str = ""
    #: Regex SOURCE for opaque pasted pairing material without the scheme
    #: prefix (bare tickets). Core tests it generically; it never hardcodes
    #: a pairing format.
    paste_pattern: str = ""

    @abstractmethod
    async def serve(self, gateway_url: str, config: dict[str, Any] | None = None) -> ServeHandle:
        """Expose ``gateway_url`` over this transport.

        Returns a :class:`ServeHandle` whose ``ticket`` is the pairing
        artifact to publish. ``config`` is the plugin's own config block
        (``plugins.entries.<plugin_id>`` in config.yaml) — plugin-
        interpreted, core never reads inside it.
        """

    async def connect(self, saved_url: str) -> RemoteTarget:
        """Dial ``saved_url`` (``scheme://…``) to a loopback access point.

        Optional: plugins that only serve may omit it. Implementations
        must be idempotent per canonical url (reuse the live pipe).
        """
        raise NotImplementedError(f"plugin '{self.scheme}' does not implement connect()")

    def normalize(self, saved_url: str) -> str:
        """Canonical saved form for this scheme. Default: strip whitespace."""
        return saved_url.strip()

    def claims(self, raw: str) -> bool:
        """Whether a scheme-less pasted value belongs to this plugin."""
        raw = raw.strip()

        if not raw or not self.paste_pattern:
            return False

        try:
            return re.compile(self.paste_pattern, re.IGNORECASE).fullmatch(raw) is not None
        except re.error:
            return False


# ── Registry ──────────────────────────────────────────────────────────────────

_remotes: dict[str, RemotePlugin] = {}


def canonical_scheme(scheme: Any) -> str:
    """Validate + lowercase a scheme token; '' for anything invalid."""
    if not isinstance(scheme, str):
        return ""

    value = scheme.strip().lower()

    return value if _SCHEME_RE.match(value) else ""


def scheme_of(url: str) -> str:
    """``example://abc`` → ``example``; ``http://x`` → ``http``; else ``''``."""
    if not isinstance(url, str):
        return ""

    match = re.match(r"^([a-z][a-z0-9+.-]*)://", url.strip(), re.IGNORECASE)

    return canonical_scheme(match.group(1)) if match else ""


def register_remote_plugin(plugin: RemotePlugin) -> None:
    """Register a remote plugin at its scheme. Raises on anything unsafe:
    wrong type, invalid scheme, built-in scheme, duplicate claim."""
    if not isinstance(plugin, RemotePlugin):
        raise TypeError("plugin must be a RemotePlugin instance")

    scheme = canonical_scheme(plugin.scheme)

    if not scheme:
        raise ValueError(f"invalid scheme {plugin.scheme!r}")

    if scheme in BUILTIN_SCHEMES:
        raise ValueError(f"scheme '{scheme}' is a built-in transport and is not registrable")

    if scheme in _remotes:
        raise ValueError(f"scheme '{scheme}' is already registered by '{_remotes[scheme].scheme}'")

    plugin.scheme = scheme
    _remotes[scheme] = plugin


def unregister_remote_plugin(scheme: str) -> None:
    """Remove a remote plugin by scheme (idempotent)."""
    _remotes.pop(canonical_scheme(scheme), None)


def get_remote_plugin(url_or_scheme: str) -> RemotePlugin | None:
    """The plugin owning ``url_or_scheme``'s scheme, or None for
    http(s)/scheme-less values."""
    scheme = canonical_scheme(url_or_scheme) if "://" not in str(url_or_scheme) else scheme_of(url_or_scheme)

    return _remotes.get(scheme) if scheme else None


def list_remote_plugins() -> dict[str, RemotePlugin]:
    """Snapshot of registered plugins keyed by canonical scheme."""
    return dict(_remotes)
