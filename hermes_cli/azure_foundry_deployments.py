"""Discover deployed names, never the Azure model catalog.

Project endpoints use their deployment API. Resource endpoints use ARM with the
same explicitly selected Entra identity; API keys cannot authorize ARM reads.
Listing is advisory: unavailable permissions or endpoints leave manual entry usable.
"""

from dataclasses import replace
import re
import time
from urllib.parse import urljoin, urlsplit

from agent.azure_identity_adapter import EntraIdentityConfig, _probe_token
from hermes_cli.azure_detect import _http_get_json

_ARM = "https://management.azure.com"
_ARM_SCOPE = "https://management.azure.com/.default"


def _items(url: str, credential, deadline: float) -> list[dict] | None:
    """Read bounded pages without forwarding credentials to another origin or path."""
    first = urlsplit(url)
    items = []
    for _ in range(20):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        status, body = _http_get_json(url, credential, timeout=min(6.0, remaining))
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("value"), list):
            return None
        items.extend(item for item in body["value"] if isinstance(item, dict))
        link = body.get("nextLink")
        if not link:
            return items
        if not isinstance(link, str):
            return None
        url = urljoin(url, link)
        next_page = urlsplit(url)
        if (next_page.scheme, next_page.netloc, next_page.path) != (first.scheme, first.netloc, first.path):
            return None
    return None


def _names(items: list[dict] | None) -> list[str]:
    return sorted({item["name"] for item in items or [] if isinstance(item.get("name"), str) and item["name"].strip()})


def _project_url(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    project = re.match(r"^(/api/projects/[^/]+)(?:/|$)", parsed.path)
    if project:
        return parsed._replace(path=project[1] + "/deployments", query="api-version=v1", fragment="").geturl()
    return None


def _azure_subdomain(base_url: str) -> str | None:
    host = urlsplit(base_url).hostname or ""
    account = re.fullmatch(r"([a-zA-Z0-9-]+)\.(?:services\.ai|openai)\.azure\.com", host)
    return account[1] if account else None


def _matches_endpoint(resource: dict, base_url: str) -> bool:
    properties = resource.get("properties")
    if not isinstance(properties, dict):
        return False
    domain = properties.get("customSubDomainName")
    if isinstance(domain, str) and domain.casefold() == str(_azure_subdomain(base_url)).casefold():
        return True
    endpoints = properties.get("endpoints")
    urls = list(endpoints.values()) if isinstance(endpoints, dict) else []
    urls.append(properties.get("endpoint"))
    host = urlsplit(base_url).hostname
    return any(isinstance(url, str) and urlsplit(url).hostname == host for url in urls)


def _resource_id(base_url: str, credential, subscription: str, deadline: float) -> str | None:
    subscriptions = _items(_ARM + "/subscriptions?api-version=2022-12-01", credential, deadline)
    for row in subscriptions or []:
        sid = row.get("subscriptionId")
        if not isinstance(sid, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", sid):
            continue
        if subscription and subscription.casefold() not in (sid.casefold(), str(row.get("displayName", "")).casefold()):
            continue
        resources = _items(f"{_ARM}/subscriptions/{sid}/providers/Microsoft.CognitiveServices/accounts?api-version=2024-10-01", credential, deadline)
        for resource in resources or []:
            rid = resource.get("id")
            if (_matches_endpoint(resource, base_url) and isinstance(rid, str)
                    and re.fullmatch(rf"/subscriptions/{re.escape(sid)}/resourceGroups/[^/]+/providers/Microsoft\.CognitiveServices/accounts/[^/]+", rid, re.IGNORECASE)):
                return rid
        if time.monotonic() >= deadline:
            break
    return None


def discover_deployments(base_url: str, credential, entra: EntraIdentityConfig | None = None) -> list[str]:
    """Reuse inference credentials for project reads; acquire a separate ARM-scoped token only for Entra."""
    deadline = time.monotonic() + 18.0
    if project := _project_url(base_url):
        deployments = _items(project, credential, deadline)
        if deployments is not None:
            return _names([item for item in deployments if item.get("type") == "ModelDeployment"])
    if not _azure_subdomain(base_url) or entra is None:
        return []
    # A new scope, with the same source and selectors. No credential-chain fallback is added.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return []
    # The adapter preserves profile ContextVars and bounds even a slow SDK credential.
    probe = _probe_token(replace(entra, scope=_ARM_SCOPE), remaining)
    if not probe or not probe.get("token"):
        return []
    bearer = lambda: probe["token"].token
    resource = _resource_id(base_url, bearer, entra.subscription, deadline)
    if not resource:
        return []
    deployments = _items(f"{_ARM}{resource}/deployments?api-version=2024-10-01", bearer, deadline)
    ready = [item for item in deployments or [] if isinstance(item.get("properties"), dict)
             and item["properties"].get("provisioningState") == "Succeeded"]
    return _names(ready)
