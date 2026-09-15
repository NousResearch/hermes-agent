"""What an MCP server and a runtime plugin are, to NOVA.

Nothing in this module knows the runtime exists. The inventory arrives through
:func:`set_discovery`, which the Hermes adapter calls at import — the same injection the
channel catalogue uses, and for the same reason: ``tests/platform/test_boundaries.py``
allows runtime imports only under ``nova/runtime/<adapter>/``.

**Discovery, not invention.** Every entry here came from a manifest on disk. A server NOVA
has heard of but this deployment does not ship does not appear, because the deployment is
what decides what can actually connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: How an MCP server proves who it is. These are the runtime's own manifest values, not a
#: NOVA vocabulary — ``hermes_cli.mcp_catalog.AuthSpec.type``.
AUTH_KINDS = ("oauth", "api_key", "none")

#: Plugin kinds NOVA surfaces, and what a grant means for each. Anything else the runtime
#: discovers is carried but not offered — see :attr:`PluginEntry.grantable`.
#:
#: ``standalone`` is opt-in: it loads only when named in ``plugins.enabled``.
#: ``backend`` is bundled and auto-loads; the only honest control is turning it OFF.
#: ``platform`` is a channel. Channels have their own screen, and a second switch for the
#: same thing would eventually disagree with the first.
PLUGIN_KINDS = ("standalone", "backend", "platform")


@dataclass(frozen=True)
class CredentialVar:
    """One environment variable an extension needs, as its manifest states it."""

    name: str
    prompt: str = ""
    required: bool = True
    secret: bool = True


@dataclass(frozen=True)
class McpServer:
    """One entry of the runtime's curated MCP catalogue."""

    id: str
    description: str = ""
    source: str = ""
    transport: str = ""
    auth: str = "none"
    credentials: tuple[CredentialVar, ...] = ()
    #: The manifest's own URL for an HTTP server. Shown so an operator can see where the
    #: agent's data would go before granting it — the single most useful fact on the row.
    url: str = ""
    #: A stdio server bootstraps by cloning and building something. Worth surfacing: it is
    #: the difference between "enable this" and "this will fetch code onto the host".
    installs: bool = False

    @property
    def toolset(self) -> str:
        """The toolset the runtime registers this server's tools under.

        ``tools/mcp_tool_registration.py`` names it ``mcp-<server>``. Stated here so the
        Control Centre can say which toolset a grant creates rather than implying the
        tools appear from nowhere.
        """
        return f"mcp-{self.id}"

    @property
    def needs_interactive_auth(self) -> bool:
        """True when a browser consent NOVA cannot perform stands between grant and use."""
        return self.auth == "oauth"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "transport": self.transport,
            "auth": self.auth,
            "url": self.url,
            "installs": self.installs,
            "toolset": self.toolset,
            "needs_interactive_auth": self.needs_interactive_auth,
            # Names only. A credential's value never travels with its declaration.
            "credentials": [
                {"name": var.name, "prompt": var.prompt, "required": var.required,
                 "secret": var.secret}
                for var in self.credentials
            ],
        }


@dataclass(frozen=True)
class PluginEntry:
    """One plugin the runtime discovered, and what granting it would mean."""

    id: str
    name: str = ""
    description: str = ""
    kind: str = ""
    source: str = ""
    version: str = ""
    tools: tuple[str, ...] = ()
    credentials: tuple[CredentialVar, ...] = ()

    @property
    def auto_loads(self) -> bool:
        """True when this plugin already loads without being named in ``plugins.enabled``.

        ``gate_manifest`` auto-loads bundled backends and defers bundled platforms. For
        those, "enable" is a switch that would report a change and change nothing, so the
        Control Centre offers only the disable.
        """
        return self.source == "bundled" and self.kind in ("backend", "platform")

    @property
    def grantable(self) -> bool:
        """Whether NOVA offers this plugin at all.

        Platforms are channels and belong to the channels screen. Everything else is
        either opt-in (enable) or auto-loading (disable), both of which are real.
        """
        return self.kind != "platform"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "kind": self.kind, "source": self.source, "version": self.version,
            "tools": list(self.tools),
            "auto_loads": self.auto_loads, "grantable": self.grantable,
            "credentials": [
                {"name": var.name, "prompt": var.prompt, "required": var.required,
                 "secret": var.secret}
                for var in self.credentials
            ],
        }


@dataclass(frozen=True)
class ExtensionCatalogue:
    """Everything this deployment could grant, plus why anything is missing."""

    mcp: tuple[McpServer, ...] = ()
    plugins: tuple[PluginEntry, ...] = ()
    #: Why the inventory is empty or short. Present so "no MCP servers" is distinguishable
    #: from "the runtime could not be read", which are opposite situations.
    detail: str = ""

    def mcp_server(self, server_id: str) -> Optional[McpServer]:
        return next((entry for entry in self.mcp if entry.id == server_id), None)

    def plugin(self, plugin_id: str) -> Optional[PluginEntry]:
        return next((entry for entry in self.plugins if entry.id == plugin_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mcp": [entry.to_dict() for entry in self.mcp],
            "plugins": [entry.to_dict() for entry in self.plugins],
            "detail": self.detail,
        }


_discovery: Optional[Callable[[], dict[str, Any]]] = None
_cache: Optional[ExtensionCatalogue] = None


def set_discovery(fn: Optional[Callable[[], dict[str, Any]]]) -> None:
    """Install the runtime's extension discovery, clearing any cached inventory."""
    global _discovery, _cache
    _discovery, _cache = fn, None


def catalogue(*, refresh: bool = False) -> ExtensionCatalogue:
    """What this deployment can grant.

    Cached, because discovery walks two directory trees and parses ~120 manifests, and the
    answer only changes when the runtime is upgraded. ``refresh`` exists for tests.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache
    if _discovery is None:
        _cache = ExtensionCatalogue(
            detail="no runtime adapter is loaded, so nothing can be discovered"
        )
        return _cache
    try:
        raw = _discovery()
    except Exception as exc:  # noqa: BLE001 — a broken runtime is a report, not a crash
        _cache = ExtensionCatalogue(detail=f"the runtime could not be read: {exc}")
        return _cache
    _cache = ExtensionCatalogue(
        mcp=tuple(_mcp(row) for row in raw.get("mcp") or ()),
        plugins=tuple(_plugin(row) for row in raw.get("plugins") or ()),
        detail=str(raw.get("detail") or ""),
    )
    return _cache


def _vars(rows: Any) -> tuple[CredentialVar, ...]:
    out = []
    for row in rows or ():
        if not isinstance(row, dict) or not row.get("name"):
            continue
        out.append(CredentialVar(
            name=str(row["name"]), prompt=str(row.get("prompt") or ""),
            # A manifest that did not say is not a manifest saying "safe to show".
            required=bool(row.get("required", True)), secret=bool(row.get("secret", True)),
        ))
    return tuple(out)


def _mcp(row: dict) -> McpServer:
    return McpServer(
        id=str(row.get("id") or ""), description=str(row.get("description") or ""),
        source=str(row.get("source") or ""), transport=str(row.get("transport") or ""),
        auth=str(row.get("auth") or "none"), url=str(row.get("url") or ""),
        installs=bool(row.get("installs")), credentials=_vars(row.get("credentials")),
    )


def _plugin(row: dict) -> PluginEntry:
    return PluginEntry(
        id=str(row.get("id") or ""), name=str(row.get("name") or ""),
        description=str(row.get("description") or ""), kind=str(row.get("kind") or ""),
        source=str(row.get("source") or ""), version=str(row.get("version") or ""),
        tools=tuple(str(t) for t in (row.get("tools") or ())),
        credentials=_vars(row.get("credentials")),
    )
