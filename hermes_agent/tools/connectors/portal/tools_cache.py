from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import binascii
import hashlib
import json
import time
from typing import Callable, Literal, Protocol

from pydantic import ValidationError

from hermes_constants import get_hermes_home
from tools.connectors.gateway.errors import GatewayAuthError, GatewayUnavailable, ToolGatewayError
from tools.connectors.portal.client import NotModified, validate_slug
from tools.connectors.portal.errors import PortalToolsUnavailable
from tools.connectors.portal.wire import ConnectorTool, ConnectorToolsListing
from utils import atomic_json_write, read_json_or_empty


class ToolsClient(Protocol):
    def origin(self) -> str: ...

    def authorization_token(self) -> str | None: ...

    def tools(self, slug: str, *, if_none_match: str | None = None) -> ConnectorToolsListing | NotModified: ...


TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ToolsRead:
    connector: str
    toolkit_version: str
    etag: str
    fetched_at: float
    source: Literal["cache", "network", "revalidated"]
    stale: bool
    tools: list[ConnectorTool]


def _member_key(token: str | None) -> str | None:
    if not isinstance(token, str):
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    subject = claims.get("sub") if isinstance(claims, dict) else None
    if not isinstance(subject, str) or not subject:
        return None
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]


def _cache_path(origin: str, member_key: str, slug: str):
    origin_key = hashlib.sha256(origin.encode("utf-8")).hexdigest()[:16]
    return get_hermes_home() / "cache" / "connectors" / origin_key / member_key / f"{slug}.json"


def _cached(path) -> ToolsRead | None:
    raw = read_json_or_empty(path)
    try:
        tools = [ConnectorTool.model_validate(tool) for tool in raw["tools"]]
        return ToolsRead(
            connector=str(raw["connector"]),
            toolkit_version=str(raw["toolkit_version"]),
            etag=str(raw["etag"]),
            fetched_at=float(raw["fetched_at"]),
            source="cache",
            stale=False,
            tools=tools,
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def _store(path, entry: ToolsRead) -> ToolsRead:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        path,
        {
            "etag": entry.etag,
            "toolkit_version": entry.toolkit_version,
            "connector": entry.connector,
            "tools": [tool.model_dump(by_alias=True) for tool in entry.tools],
            "fetched_at": entry.fetched_at,
        },
    )
    return entry


def read_tools(
    slug: str,
    *,
    client: ToolsClient,
    refresh: bool = False,
    now: Callable[[], float] = time.time,
) -> ToolsRead:
    validate_slug(slug)
    member_key = _member_key(client.authorization_token())
    if member_key is None:
        listing = client.tools(slug)
        if isinstance(listing, NotModified):
            raise PortalToolsUnavailable("portal tools unavailable", code="INVALID_RESPONSE")
        return ToolsRead(
            connector=listing.connector,
            toolkit_version=listing.toolkit_version,
            etag=listing.etag,
            fetched_at=now(),
            source="network",
            stale=False,
            tools=listing.tools,
        )
    path = _cache_path(client.origin(), member_key, slug)
    cached = _cached(path)
    fetched_at = now()
    age = fetched_at - cached.fetched_at if cached is not None else None
    if cached is not None and not refresh and age is not None and 0 <= age < TTL_SECONDS:
        return cached
    try:
        listing = client.tools(slug, if_none_match=cached.etag if cached is not None else None)
    except GatewayUnavailable:
        path.unlink(missing_ok=True)
        raise
    except GatewayAuthError:
        raise
    except (PortalToolsUnavailable, ToolGatewayError):
        if cached is None:
            raise
        return replace(cached, stale=True)
    if isinstance(listing, NotModified):
        if cached is None:
            raise PortalToolsUnavailable("portal tools unavailable", code="INVALID_RESPONSE")
        return _store(path, replace(cached, fetched_at=fetched_at, source="revalidated"))
    return _store(path, ToolsRead(
        connector=listing.connector,
        toolkit_version=listing.toolkit_version,
        etag=listing.etag,
        fetched_at=fetched_at,
        source="network",
        stale=False,
        tools=listing.tools,
    ))
