"""``/api/i18n/languages`` and ``/api/i18n/catalog``: language packs for the web dashboard.

The HTTP twin of the ``i18n.languages`` / ``i18n.catalog`` JSON-RPC methods Desktop and the TUI use
(``tui_gateway/methods_i18n.py``): a language-pack plugin (or a user overlay under
``$HERMES_HOME/locales/``) that ships ``<lang>.web.yaml`` reaches the dashboard without rebuilding it.
The renderer merges ``messages`` over its bundled catalog, exactly like Desktop (``web/src/i18n``).

Unlike the JSON-RPC pair these take no ``profile``: the dashboard's language is a per-browser choice,
not a profile setting, and scoping a request to another profile's home switches the whole process to
multi-profile hosting. They answer for the dashboard's own home (its ``locales/`` overlay); plugin packs
are process-wide anyway. Like every ``/api/`` route these sit behind the session-token / sign-in gate.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

_log = logging.getLogger(__name__)
router = APIRouter()

WEB_SURFACE = "web"


def _load_language_packs() -> None:
    """Plugin packs register during plugin discovery, which the dashboard process may not have run
    yet. Idempotent; a failed discovery still answers with the bundled and overlay layers."""
    from hermes_cli.plugins import discover_plugins
    try:
        discover_plugins()
    except Exception:  # health: allow BLE001 -- serve bundled + overlay layers rather than fail the UI
        _log.warning("i18n: plugin discovery failed; serving languages without plugin packs", exc_info=True)


@router.get("/api/i18n/languages")
def get_languages() -> dict:
    """Every language some layer supplies (bundled ∪ user overlay ∪ plugin packs), with the surfaces
    each one has strings for — the picker offers only languages it can actually show."""
    from agent.i18n import language_options
    _load_language_packs()
    return {"languages": language_options()}


@router.get("/api/i18n/catalog")
def get_catalog(lang: str = "", surface: str = WEB_SURFACE) -> dict:
    """Pack + overlay messages (flat dotted keys) for one language and surface. ``lang`` is resolved to
    its canonical id (``pt-BR`` → ``pt-br``); an unknown id resolves to ``en`` with an empty layer."""
    from agent.i18n import resolve_language_id, surface_catalog
    from agent.i18n_layers import SURFACES
    if surface not in SURFACES:
        raise HTTPException(status_code=400, detail=f"surface must be one of {', '.join(SURFACES)}")
    _load_language_packs()
    resolved = resolve_language_id(lang) or "en"
    return {"lang": resolved, "surface": surface, "messages": surface_catalog(resolved, surface)}
