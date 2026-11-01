"""Whether the desktop app a catalog plugin drives is on this machine, before the plugin is installed.

A portable plugin declares its app in ``plugin.json`` (``extensions.com.nousresearch.hermes.servers``).
The catalog pins the commit, so the declaration is read once per pin from the repo at that commit and
judged by the same ``hermes_platform`` resolver the installer and the Plugins-tab pill use. Nothing is
installed, spawned or connected. Any failure to read the declaration is ``unknown``, never a guess;
a pin without ``plugin.json`` (404) or without an app/GPU declaration is ``no_requirements``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_TIMEOUT = 5.0
_MAX_BYTES = 256 * 1024
_DISCLOSURE = re.compile(r"\bDisclosure\s*[:\u2014]\s*(.+)")
# (repo, sha, subdir) -> manifest dict, or None when the pin has no plugin.json (404). A pin is
# immutable, so the process keeps the answer; an unreadable or transient result is never cached.
_manifests: dict[tuple[str, str, str], Optional[dict[str, Any]]] = {}


@dataclass(frozen=True)
class Presence:
    """``state`` keeps the card vocabulary; ``reason`` records the backend resolver's distinction."""

    state: str
    sentence: str = ""
    reason: str = "uncertain"
    required_version: str | None = None
    found_version: str | None = None
    app_title: str | None = None  # the name ``sentence`` used for the app


UNKNOWN = Presence("unknown")
# Nothing to check: no portable plugin.json at the pin (the installer gate skips those too) or no server
# that declares an app or GPU. Distinct from UNKNOWN, which means the declaration could not be judged.
NO_REQUIREMENTS = Presence("unknown", reason="no_requirements")
# The one blocker an all-of plugin names, shared with the install refusal so the card, the dialog and
# the refusal agree. Structural blockers first (installing an app cannot repair them), then an app
# whose installed version cannot be read (both surfaces refuse on it), then the concrete app blockers
# the installer names. A server that merely could not be judged comes after those: it must not hide
# an actionable blocker behind "could not confirm" while the install is refused for that blocker.
BLOCKER_RANK = {"unsupported_os": 0, "unsupported_gpu": 1, "unreadable_version": 2, "missing_app": 3,
                "version_too_old": 4, "uncertain": 5, "app_not_running": 6, "available": 7}


def blocker_rank(reason: str, required_version: str | None = None, found_version: str | None = None) -> int:
    """``uncertain`` with a required but unread version is the refusable unreadable-version case."""
    key = "unreadable_version" if reason == "uncertain" and required_version and not found_version else reason
    return BLOCKER_RANK.get(key, len(BLOCKER_RANK))
_OS_LABELS = {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}

def _reviewed_app(entry: Any, actual_declaration: Any = None, *, catalog: list | None = None):
    """Only a catalog pin whose plugin.json declares exactly one app may name a vendor for it.

    The link belongs to that one server; with several app servers a single link could name the
    wrong one, so there is none. *catalog* is the caller's already-loaded live catalog; each load
    re-parses every in-tree entry."""
    if not getattr(entry, "app_download_url", None):
        return None  # the match below needs a reviewed URL on both sides
    from hermes_cli.plugin_catalog import load_catalog_live
    reviewed = next((item for item in (load_catalog_live() if catalog is None else catalog)
                     if item.name == entry.name and item.repo == entry.repo
                     and item.sha == getattr(entry, "sha", None)
                     and item.subdir == getattr(entry, "subdir", None)
                     and item.title == getattr(entry, "title", None)
                     and item.app_download_url == getattr(entry, "app_download_url", None)
                     and item.app_download_url), None)
    if reviewed is None:
        return None
    manifest = _pinned_manifest(reviewed.repo, reviewed.sha, reviewed.subdir)
    try:
        from hermes_cli.agent_plugins import _server_declarations
        servers = (manifest or {}).get("extensions", {}).get("com.nousresearch.hermes", {}).get("servers", {})
        declarations = _server_declarations(manifest, {name: {} for name in servers}) if manifest and isinstance(servers, dict) else {}
        app_servers = [name for name, candidate in declarations.items()
                       if candidate.declaration.requires_app and candidate.declaration.app is not None]
        if len(app_servers) != 1:
            return None
        declared = declarations[app_servers[0]].declaration
        return reviewed if actual_declaration is None or declared == actual_declaration else None
    except (TypeError, ValueError, AttributeError):
        return None


def app_guidance(entry: Any, reason: str, *, required_version: str | None = None,
                 found_version: str | None = None,
                 actual_declaration: Any = None, catalog: list | None = None,
                 app_title: str | None = None) -> dict[str, str | None]:
    """Backend guidance; links require catalog identity AND an app declaration at its reviewed pin.

    *app_title* is the name the caller already showed for the app; the vendor link never follows it."""
    # A link is offered only for missing_app / version_too_old, which only a required-app server
    # produces; with exactly one such server that is the server it names, so none is passed. Any
    # other reason never shows a link, so it does not pay the pinned-manifest read the review needs.
    reviewed = (_reviewed_app(entry, actual_declaration, catalog=catalog)
                if reason in ("missing_app", "version_too_old") else None)
    return {"reason": reason, "app_title": reviewed.title if reviewed else (app_title or entry.name),
            "required_version": required_version, "found_version": found_version,
            "download_url": reviewed.app_download_url if reviewed else None}


def _raw_manifest_url(repo: str, sha: str, subdir: str) -> Optional[str]:
    match = _GITHUB.match(repo.strip())
    if not match:
        return None
    path = "/".join(p for p in (subdir.strip("/"), "plugin.json") if p)
    return f"https://raw.githubusercontent.com/{match.group(1)}/{match.group(2)}/{sha}/{path}"


def _pinned_manifest(repo: str, sha: str, subdir: str) -> Optional[dict[str, Any]]:
    key = (repo, sha, subdir)
    if key in _manifests:
        return _manifests[key]
    url = _raw_manifest_url(repo, sha, subdir)
    manifest: Optional[dict[str, Any]] = None
    if url:
        try:
            import httpx

            resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) <= _MAX_BYTES:
                data = json.loads(resp.content)
                if not isinstance(data, dict):
                    return None  # unreadable, not absent: presence must not read it as "no requirements"
                manifest = data
            elif resp.status_code != 404:
                return None  # transient: do not remember
        except Exception as exc:
            logger.debug("plugin catalog presence: %s unreadable: %s", url, exc)
            return None
    _manifests[key] = manifest
    return manifest


def _server_presence(decl: Any, liveness_raw: Any, title: str) -> Presence:
    from hermes_platform.declaration import gpu_label
    from hermes_platform.host import facts
    from hermes_platform.resolver.app import AppResolver
    from hermes_platform.resolver.availability import availability
    from hermes_platform.resolver.base import Effort
    from tools.mcp_liveness import parse_liveness

    try:
        available = availability(decl)
    except Exception:
        logger.debug("plugin catalog presence: app discovery failed", exc_info=True)
        return UNKNOWN
    if available.state == "no_requirements":
        return Presence("unknown", reason="no_requirements")
    if available.state in ("missing_app", "unsupported_os"):
        return Presence("missing_app", f"needs {title}", available.state, available.min_version)
    if available.state == "unsupported_gpu":
        return Presence("missing_app", f"needs {gpu_label(decl.required_gpu)}", "unsupported_gpu", available.min_version)
    if available.state == "version_too_old":
        if not available.version:
            return Presence("unknown", f"Could not determine {title}'s version", "uncertain", available.min_version)
        return Presence("missing_app", f"needs {title} {available.min_version} or newer (found {available.version})",
                        "version_too_old", available.min_version, available.version)
    try:
        live = parse_liveness(liveness_raw) if liveness_raw is not None else None
    except ValueError:
        return UNKNOWN
    if live is None or live.kind == "static":
        return Presence("present", reason="available", required_version=available.min_version)
    if live.kind == "interactive_session":
        return Presence("present", reason="available", required_version=available.min_version) if facts.interactive_session() else Presence(
            "app_not_running", f"{title} needs an interactive desktop session", "app_not_running", available.min_version)
    definition = decl.app_for(facts.os_family())
    if definition is None:
        # Not a version question: the app was found, only its liveness cannot be resolved here.
        return Presence("unknown", f"Could not inspect {title}", "uncertain")
    resolver = AppResolver(live.app_definition(definition))
    probe = resolver.probe(resolver.locate(), effort=Effort.LOCAL)
    return Presence("present", reason="available", required_version=available.min_version) if probe.running.value is True else Presence(
        "app_not_running", f"{title} is not running", "app_not_running", available.min_version)


def presence(entry: Any) -> Presence:
    """The app state of one catalog entry on this host. The worst server wins (missing, then not running)."""
    from hermes_cli.agent_plugins import _server_declarations
    from hermes_cli.plugins_cmd_catalog import normalized_platforms
    from hermes_platform.host.facts import os_family

    here = os_family()
    platforms = getattr(entry, "platforms", ())
    if platforms and here not in normalized_platforms(platforms):
        # The catalog's platform list is the plugin's, so the row's own title names it; no manifest read.
        return Presence("missing_app", f"not supported on {_OS_LABELS.get(here, here)}", "unsupported_os",
                        app_title=getattr(entry, "title", "") or entry.name)

    manifest = _pinned_manifest(entry.repo, entry.sha, entry.subdir)
    if not manifest:
        # Only a 404 is remembered as None; a transient failure is not cached and stays unknown.
        known_absent = (_raw_manifest_url(entry.repo, entry.sha, entry.subdir) is not None
                        and (entry.repo, entry.sha, entry.subdir) in _manifests)
        return NO_REQUIREMENTS if known_absent else UNKNOWN
    raw_servers = (manifest.get("extensions") or {}).get("com.nousresearch.hermes", {}).get("servers") or {}
    if not isinstance(raw_servers, dict):
        return UNKNOWN
    if not raw_servers:
        return NO_REQUIREMENTS
    try:
        declarations = _server_declarations(manifest, {name: {} for name in raw_servers})
    except Exception as exc:
        logger.debug("plugin catalog presence: %s declaration invalid: %s", entry.name, exc)
        return UNKNOWN
    app_servers = [name for name, d in declarations.items() if getattr(d.declaration, "app", None) is not None]

    def title_for(name: str) -> str:
        # The catalog title names a lone app ("NVIDIA App", "Pay CLI"); next to another app it could
        # misname the sibling, so the plugin name stands in. Only the vendor link needs a review.
        return (getattr(entry, "title", "") or entry.name) if app_servers == [name] else entry.name

    try:
        found = []
        for name, d in declarations.items():
            title = title_for(name)
            found.append(replace(_server_presence(d.declaration, d.liveness, title), app_title=title))
    except Exception:
        logger.debug("plugin catalog presence: %s probe failed", entry.name, exc_info=True)
        return UNKNOWN
    relevant = [p for p in found if p.reason != "no_requirements"]
    return (min(relevant, key=lambda p: blocker_rank(p.reason, p.required_version, p.found_version))
            if relevant else NO_REQUIREMENTS)


def _onboarding_catalog(catalog: list | None = None) -> list:
    """Catalog entries curated for onboarding (``onboarding: true``) that this OS can run."""
    from hermes_cli.plugin_catalog import load_catalog_live
    from hermes_cli.plugins_cmd_catalog import normalized_platforms
    from hermes_platform.host.facts import os_family

    here = os_family()
    return [entry for entry in (load_catalog_live() if catalog is None else catalog)
            if entry.onboarding and (not entry.platforms or here in normalized_platforms(entry.platforms))]


def onboarding_names() -> list[str]:
    """The onboarding plugins this OS runs, by catalog name; no presence probe."""
    return [entry.name for entry in _onboarding_catalog()]


def onboarding_entries() -> list[dict[str, Any]]:
    """Catalog entries curated for the onboarding card (``onboarding: true``) that this OS can run,
    each with its app state. Platform mismatch is the only exclusion; a missing app is reported."""
    from hermes_cli.plugin_catalog import load_catalog_live

    catalog = load_catalog_live()
    rows = []
    for entry in _onboarding_catalog(catalog):
        found = presence(entry)
        rows.append({
            "name": entry.name,
            "title": entry.title or entry.name,
            "description": _first_sentence(entry.description),
            "tier": entry.tier,
            "platforms": list(entry.platforms),
            "app_state": found.state,
            "sentence": found.sentence,
            **app_guidance(entry, found.reason, required_version=found.required_version,
                           found_version=found.found_version, catalog=catalog,
                           app_title=found.app_title),
        })
    return rows


def presence_for(names: list[str]) -> list[dict[str, Any]]:
    """Named catalog plugins in request order, including those this backend OS cannot run.
    A name the live catalog does not carry (unknown or removed) is dropped."""
    from concurrent.futures import ThreadPoolExecutor

    from hermes_cli.plugin_catalog import load_catalog_live
    catalog = load_catalog_live()  # once per request: every guidance row below reuses it
    by_name = {entry.name: entry for entry in catalog}
    # An explicit named request must report unsupported_os rather than silently hiding the row.
    entries = [by_name[name] for name in dict.fromkeys(names) if name in by_name]
    if not entries:
        return []
    # Each manifest read is bounded by _TIMEOUT; in parallel the whole answer waits one timeout, not one per name.
    with ThreadPoolExecutor(max_workers=len(entries)) as pool:
        found = list(pool.map(presence, entries))
    return [{"name": entry.name, "title": entry.title or entry.name, "state": state.state,
             "sentence": state.sentence, "disclosure": _disclosure(entry.description),
             **app_guidance(entry, state.reason, required_version=state.required_version,
                            found_version=state.found_version, catalog=catalog,
                            app_title=state.app_title)}
            for entry, state in zip(entries, found)]


def _disclosure(text: str) -> str:
    """The sentence after the description's ``Disclosure:`` (or ``Disclosure —``) marker; empty when absent."""
    match = _DISCLOSURE.search(" ".join(str(text or "").split()))
    return _first_sentence(match.group(1)) if match else ""


def _first_sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    head, dot, _rest = text.partition(". ")
    return f"{head}." if dot else text
