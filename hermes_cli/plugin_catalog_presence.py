"""Whether the desktop app a catalog plugin drives is on this machine, before the plugin is installed.

A portable plugin declares its app in ``plugin.json`` (``extensions.com.nousresearch.hermes.servers``).
The catalog pins the commit, so the declaration is read once per pin from the repo at that commit and
judged by the same ``hermes_platform`` resolver the installer and the Plugins-tab pill use. Nothing is
installed, spawned or connected. Any failure to read the declaration is ``unknown``, never a guess.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_TIMEOUT = 5.0
_MAX_BYTES = 256 * 1024
# (repo, sha, subdir) -> manifest dict, or None when the pin has no readable plugin.json. A pin is
# immutable, so the process keeps the answer.
_manifests: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]] = {}


@dataclass(frozen=True)
class Presence:
    """``state`` uses the card's vocabulary (``CatalogAppState``); ``sentence`` is empty when present or unknown."""

    state: str
    sentence: str = ""


UNKNOWN = Presence("unknown")


def _raw_manifest_url(repo: str, sha: str, subdir: str) -> Optional[str]:
    match = _GITHUB.match(repo.strip())
    if not match:
        return None
    path = "/".join(p for p in (subdir.strip("/"), "plugin.json") if p)
    return f"https://raw.githubusercontent.com/{match.group(1)}/{match.group(2)}/{sha}/{path}"


def _pinned_manifest(repo: str, sha: str, subdir: str) -> Optional[Dict[str, Any]]:
    key = (repo, sha, subdir)
    if key in _manifests:
        return _manifests[key]
    url = _raw_manifest_url(repo, sha, subdir)
    manifest: Optional[Dict[str, Any]] = None
    if url:
        try:
            import httpx

            resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) <= _MAX_BYTES:
                data = json.loads(resp.content)
                manifest = data if isinstance(data, dict) else None
            elif resp.status_code != 404:
                return None  # transient: do not remember
        except Exception as exc:
            logger.debug("plugin catalog presence: %s unreadable: %s", url, exc)
            return None
    _manifests[key] = manifest
    return manifest


def _server_presence(decl: Any, liveness_raw: Any, title: str) -> Presence:
    from hermes_platform.host import facts
    from hermes_platform.resolver.app import AppResolver
    from hermes_platform.resolver.availability import availability
    from hermes_platform.resolver.base import Effort
    from tools.mcp_liveness import parse_liveness

    available = availability(decl)
    if available.state == "no_requirements":
        return UNKNOWN
    if available.state in ("missing_app", "unsupported_os"):
        return Presence("missing_app", f"needs {title}")
    if available.state == "version_too_old":
        found = f" (found {available.version})" if available.version else ""
        return Presence("missing_app", f"needs {title} {available.min_version} or newer{found}")
    try:
        live = parse_liveness(liveness_raw) if liveness_raw is not None else None
    except ValueError:
        live = None
    if live is None or live.kind == "static":
        return Presence("present")
    if live.kind == "interactive_session":
        return Presence("present") if facts.interactive_session() else Presence(
            "app_not_running", f"{title} needs an interactive desktop session")
    definition = decl.app_for(facts.os_family())
    if definition is None:
        return Presence("missing_app", f"needs {title}")
    resolver = AppResolver(live.app_definition(definition))
    probe = resolver.probe(resolver.locate(), effort=Effort.LOCAL)
    return Presence("present") if probe.running.value is True else Presence(
        "app_not_running", f"{title} is not running")


def presence(entry: Any) -> Presence:
    """The app state of one catalog entry on this host. The worst server wins (missing, then not running)."""
    from hermes_cli.agent_plugins import _server_declarations

    manifest = _pinned_manifest(entry.repo, entry.sha, entry.subdir)
    if not manifest:
        return UNKNOWN
    raw_servers = (manifest.get("extensions") or {}).get("com.nousresearch.hermes", {}).get("servers") or {}
    if not isinstance(raw_servers, dict) or not raw_servers:
        return UNKNOWN
    try:
        declarations = _server_declarations(manifest, {name: {} for name in raw_servers})
    except Exception as exc:
        logger.debug("plugin catalog presence: %s declaration invalid: %s", entry.name, exc)
        return UNKNOWN
    title = getattr(entry, "title", "") or entry.name
    rank = {"missing_app": 0, "app_not_running": 1, "present": 2, "unknown": 3}
    found = [_server_presence(d.declaration, d.liveness, title) for d in declarations.values()]
    return min(found, key=lambda p: rank[p.state]) if found else UNKNOWN


def onboarding_entries() -> list[Dict[str, Any]]:
    """Catalog entries curated for the onboarding card (``onboarding: true``) that this OS can run,
    each with its app state. Platform mismatch is the only exclusion; a missing app is reported."""
    from hermes_cli.plugin_catalog import load_catalog_live
    from hermes_cli.plugins_cmd_catalog import normalized_platforms
    from hermes_platform.host.facts import os_family

    here = os_family()
    rows = []
    for entry in load_catalog_live():
        if not entry.onboarding or (entry.platforms and here not in normalized_platforms(entry.platforms)):
            continue
        found = presence(entry)
        rows.append({
            "name": entry.name,
            "title": entry.title or entry.name,
            "description": _first_sentence(entry.description),
            "tier": entry.tier,
            "platforms": list(entry.platforms),
            "app_state": found.state,
            "sentence": found.sentence,
        })
    return rows


def _first_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    head, dot, _rest = text.partition(". ")
    return f"{head}." if dot else text
