"""Off-catalog connector discovery — the official MCP Registry, remotes only.

The reviewed ``optional-mcps/`` catalog stays exactly what it was: a PR-gated
trust boundary, "presence in the directory = Nous approval" (see
:mod:`hermes_cli.mcp_catalog`). This module is a *second, lower* tier beside
it, for the long tail the catalog will never enumerate — the same tier the
desktop's static ``lib/mcp-directory.ts`` already occupied, generalized from a
hardcoded list of vendor remotes into a live query.

Two rules make that tier safe enough to put behind a consent card:

**Remotes only, never packages.** A registry entry may advertise ``packages``
(``npx``/``uvx``/docker) as well as ``remotes`` (a URL). Installing a package
from an unreviewed publisher is arbitrary code execution on the user's
machine; connecting to a URL is not. Package-only entries are dropped here, in
the backend, so a renderer cannot ask for one — the filter is a trust
boundary, not a display preference.

**Publisher-owns-endpoint is the verification signal.** The registry verifies
namespace ownership: a ``com.example`` namespace requires proving control of
``example.com`` via DNS/HTTP, and ``io.github.<user>`` requires that GitHub
account. So a reverse-DNS namespace whose *endpoint host lives under the same
registrable domain* is a real statement — ``com.notion/mcp`` serving
``mcp.notion.com`` is Notion. That earns ``verified``.

Everything else is ``community``: the namespace and the endpoint disagree
(``io.github.someone`` serving ``their-app.vercel.app``), or the "domain" is a
shared-subdomain host anyone can claim (``*.trycloudflare.com``,
``*.vercel.app``). Community entries are still returned — the user asked to
see the long tail — but they are labeled, and the card shows the raw endpoint
so the choice is informed rather than implied.

Nothing here installs anything. It answers "what exists, and how much should
you trust it"; consent and the config write stay where they already are.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = "https://registry.modelcontextprotocol.io"
_SEARCH_PATH = "/v0.1/servers"

#: Reviewed catalog tier. Never produced by this module — named here so the
#: renderer and the card share one trust vocabulary across all three sources.
TRUST_CATALOG = "catalog"
#: Registry publisher proved it owns the domain serving the endpoint.
TRUST_VERIFIED = "verified"
#: Listed, but nothing ties the publisher to the endpoint it serves.
TRUST_COMMUNITY = "community"

# The registry's transport types for a hosted endpoint. Anything else (a
# package launcher) never reaches the caller.
_REMOTE_TYPES = {"streamable-http", "sse", "http"}

# Suffixes that hand out subdomains to anyone who signs up. A namespace under
# one of these verifies nothing about *who* is behind the endpoint, so it can
# never reach `verified` no matter how well the host matches.
_SHARED_SUBDOMAIN_SUFFIXES = (
    "a.run.app",
    "amplifyapp.com",
    "appspot.com",
    "azurewebsites.net",
    "cloudfunctions.net",
    "deno.dev",
    "firebaseapp.com",
    "fly.dev",
    "github.io",
    "gitpod.io",
    "glitch.me",
    "herokuapp.com",
    "koyeb.app",
    "loca.lt",
    "netlify.app",
    "ngrok-free.app",
    "ngrok.app",
    "ngrok.io",
    "onrender.com",
    "pages.dev",
    "railway.app",
    "replit.app",
    "replit.dev",
    "surge.sh",
    "trycloudflare.com",
    "vercel.app",
    "web.app",
    "workers.dev",
)

# Namespace/base labels that identify the protocol rather than the vendor —
# "com.notion/mcp" is Notion, not a server called "mcp".
_GENERIC_LABELS = {"api", "connector", "mcp", "mcp-server", "mcpserver", "remote", "server"}

_SLUG_STRIP = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True)
class RegistryEntry:
    """One hosted MCP endpoint, normalized for the connector UI."""

    #: Config-safe short name written into ``mcp_servers`` ("notion").
    name: str
    #: Fully-qualified registry identity ("com.notion/mcp"), shown as the
    #: publisher line so two servers that slug the same stay distinguishable.
    registry_name: str
    title: str
    description: str
    url: str
    #: "streamable-http" or "sse" — the endpoint's transport, as advertised.
    transport: str
    trust: str
    #: Registrable domain the namespace asserts ("notion.com"), or "".
    publisher: str
    website: str = ""
    version: str = ""
    #: Headers the endpoint documents, e.g. an API key. Names/prompts only.
    headers: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "registry_name": self.registry_name,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "transport": self.transport,
            "trust": self.trust,
            "publisher": self.publisher,
            "website": self.website,
            "version": self.version,
            "headers": list(self.headers),
        }


def _slug(value: str) -> str:
    return _SLUG_STRIP.sub("-", (value or "").strip().lower()).strip("-_")


def namespace_domain(registry_name: str) -> str:
    """Registrable domain a reverse-DNS namespace asserts ownership of.

    ``com.notion/mcp`` → ``notion.com``; ``com.paypal.mcp/mcp`` →
    ``mcp.paypal.com``. Returns "" when the namespace isn't reverse-DNS shaped
    (fewer than two labels), which classifies as community by construction.
    """
    namespace = (registry_name or "").partition("/")[0].strip().lower()
    labels = [label for label in namespace.split(".") if label]
    if len(labels) < 2:
        return ""
    return ".".join(reversed(labels))


def is_shared_subdomain(domain: str) -> bool:
    """Whether ``domain`` sits under a host that gives subdomains to anyone."""
    domain = (domain or "").lower().rstrip(".")
    return any(
        domain == suffix or domain.endswith("." + suffix)
        for suffix in _SHARED_SUBDOMAIN_SUFFIXES
    )


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        # Touching `.port` is what validates a malformed/non-numeric port —
        # `.hostname` alone happily returns a host for "example.com:notaport".
        # Same fail-closed move as urllib_security.url_origin: an authority we
        # can't fully parse must never read as verified.
        parsed.port
        return (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def classify(registry_name: str, url: str) -> Tuple[str, str]:
    """Return ``(trust, publisher_domain)`` for a registry entry's endpoint.

    ``verified`` requires all three: a reverse-DNS namespace, an endpoint host
    at or under that namespace's domain, and a domain that isn't a
    shared-subdomain host. Suffix matching is on a dot boundary, so
    ``notion.com.evil.test`` cannot pose as ``notion.com``.
    """
    domain = namespace_domain(registry_name)
    host = _host_of(url)

    if not domain or not host or is_shared_subdomain(domain):
        return TRUST_COMMUNITY, domain

    owns = host == domain or host.endswith("." + domain)
    return (TRUST_VERIFIED if owns else TRUST_COMMUNITY), domain


def _config_name(registry_name: str) -> str:
    """Short ``mcp_servers`` key for a registry identity.

    Prefers the part after the slash, falling back to the most specific
    non-generic namespace label so ``com.paypal.mcp/mcp`` lands on ``paypal``
    rather than ``mcp``.
    """
    namespace, _, base = (registry_name or "").partition("/")
    candidate = _slug(base)

    if not candidate or candidate in _GENERIC_LABELS:
        for label in reversed([label for label in namespace.split(".") if label]):
            slug = _slug(label)
            if slug and slug not in _GENERIC_LABELS:
                candidate = slug
                break

    return candidate or _slug(registry_name) or "mcp-server"


def _pick_remote(remotes: Any) -> Optional[Dict[str, Any]]:
    """First usable https remote, preferring streamable-http over sse."""
    if not isinstance(remotes, list):
        return None

    usable = [
        remote
        for remote in remotes
        if isinstance(remote, dict)
        and str(remote.get("type", "")).lower() in _REMOTE_TYPES
        # Plain http would send any header credential in clear text.
        and str(remote.get("url", "")).lower().startswith("https://")
    ]
    if not usable:
        return None

    usable.sort(key=lambda remote: 0 if str(remote.get("type")).lower() == "streamable-http" else 1)
    return usable[0]


def _is_active(payload: Dict[str, Any]) -> bool:
    """Registry status gate — deleted/deprecated entries stay out of results."""
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return True

    official = meta.get("io.modelcontextprotocol.registry/official")
    if not isinstance(official, dict):
        return True

    status = str(official.get("status", "active")).lower()
    return status == "active"


def _entry_from(payload: Dict[str, Any]) -> Optional[RegistryEntry]:
    server = payload.get("server") if isinstance(payload.get("server"), dict) else payload
    if not isinstance(server, dict):
        return None

    registry_name = str(server.get("name", "")).strip()
    if not registry_name or not _is_active(payload):
        return None

    remote = _pick_remote(server.get("remotes"))
    if remote is None:
        # Package-only entry: installing it would run someone else's code.
        return None

    url = str(remote.get("url", "")).strip()
    trust, publisher = classify(registry_name, url)

    headers = [
        {
            "name": str(header.get("name", "")),
            "description": str(header.get("description", "")),
            "required": bool(header.get("isRequired", header.get("required", False))),
            "secret": bool(header.get("isSecret", header.get("secret", False))),
        }
        for header in (remote.get("headers") or [])
        if isinstance(header, dict) and header.get("name")
    ]

    return RegistryEntry(
        name=_config_name(registry_name),
        registry_name=registry_name,
        title=str(server.get("title") or "").strip(),
        description=str(server.get("description") or "").strip(),
        url=url,
        transport=str(remote.get("type", "")).lower(),
        trust=trust,
        publisher=publisher,
        website=str(server.get("websiteUrl") or "").strip(),
        version=str(server.get("version") or "").strip(),
        headers=headers,
    )


# ─── Cache ───────────────────────────────────────────────────────────────────

# Query results, keyed by (base_url, query, limit). The registry is a
# read-mostly public index and a typing user re-queries constantly; a short
# TTL keeps the composer responsive without pinning stale results for a
# session. Guarded by a lock: the web layer calls this from a thread pool.
_cache: Dict[Tuple[str, str, int], Tuple[float, List[RegistryEntry]]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop memoized query results (tests, config change, profile switch)."""
    with _cache_lock:
        _cache.clear()


# ─── Config ──────────────────────────────────────────────────────────────────


def registry_settings() -> Dict[str, Any]:
    """``mcp.registry`` settings, defaulted for a missing/broken config."""
    defaults = {
        "enabled": True,
        "url": DEFAULT_REGISTRY_URL,
        "timeout_seconds": 8,
        "cache_ttl_minutes": 30,
        "allow_unverified": True,
    }

    try:
        from hermes_cli.config import load_config

        configured = (load_config().get("mcp") or {}).get("registry") or {}
    except Exception:
        return defaults

    if isinstance(configured, dict):
        defaults.update({k: v for k, v in configured.items() if k in defaults})

    return defaults


# ─── Search ──────────────────────────────────────────────────────────────────


def search(query: str, limit: int = 20) -> List[RegistryEntry]:
    """Search the registry for hosted connectors matching ``query``.

    Returns [] rather than raising for every failure mode — an unreachable
    registry must degrade discovery to the reviewed catalog, never break the
    card or the composer. Results are ordered verified-first, then by how
    closely the name matches, because the registry's own ranking is a plain
    substring match on name and puts real vendors below spam.
    """
    settings = registry_settings()
    if not settings.get("enabled", True):
        return []

    needle = (query or "").strip()
    if len(needle) < 2:
        return []

    base_url = str(settings.get("url") or DEFAULT_REGISTRY_URL).rstrip("/")
    # The registry caps `limit` at 100; ask for more than we return so the
    # trust sort has something to choose from after filtering.
    fetch_limit = max(1, min(100, int(limit) * 3))
    cache_key = (base_url, needle.lower(), fetch_limit)

    ttl = float(settings.get("cache_ttl_minutes", 30)) * 60
    now = time.monotonic()

    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit and now - hit[0] < ttl:
            return _rank(hit[1], needle, limit, settings)

    try:
        import httpx

        response = httpx.get(
            f"{base_url}{_SEARCH_PATH}",
            params={"search": needle, "limit": fetch_limit, "version": "latest"},
            timeout=float(settings.get("timeout_seconds", 8)),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug("MCP registry search failed for %r: %s", needle, exc)
        return []

    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        return []

    entries: List[RegistryEntry] = []
    seen: Dict[str, RegistryEntry] = {}

    for row in servers:
        if not isinstance(row, dict):
            continue

        entry = _entry_from(row)
        if entry is None:
            continue

        # Two registry identities can slug to the same config key. Keep the
        # more trustworthy one; the loser would only be installable under a
        # name the user already has.
        previous = seen.get(entry.name)
        if previous is not None:
            if previous.trust == TRUST_VERIFIED or entry.trust != TRUST_VERIFIED:
                continue
            entries.remove(previous)

        seen[entry.name] = entry
        entries.append(entry)

    with _cache_lock:
        _cache[cache_key] = (now, entries)

    return _rank(entries, needle, limit, settings)


def _rank(
    entries: List[RegistryEntry], query: str, limit: int, settings: Dict[str, Any]
) -> List[RegistryEntry]:
    """Verified first, then closest name match, then alphabetical."""
    needle = query.strip().lower()

    if not settings.get("allow_unverified", True):
        entries = [entry for entry in entries if entry.trust == TRUST_VERIFIED]

    def sort_key(entry: RegistryEntry) -> Tuple[int, int, str]:
        exact = 0 if entry.name == needle else 1 if entry.name.startswith(needle) else 2
        return (0 if entry.trust == TRUST_VERIFIED else 1, exact, entry.name)

    return sorted(entries, key=sort_key)[: max(0, int(limit))]
