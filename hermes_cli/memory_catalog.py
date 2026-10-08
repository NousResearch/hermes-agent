"""Catalog curation shared by the memory setup picker and Desktop discovery."""
from __future__ import annotations

import logging

from hermes_cli.plugin_catalog import PluginCatalogEntry

logger = logging.getLogger(__name__)
MARKETPLACE_URL = "https://hermes-agent.nousresearch.com/docs/plugins?kind=memory"


def featured_memory_entries(*, network: bool = True) -> list[PluginCatalogEntry]:
    """Eligible curated providers, without installing or loading their code.

    ``featured`` + ``category: memory`` is a maintainer designation of a selectable
    provider, not every memory-related tool on the marketplace. Catalog pins,
    offline fallback and removals remain owned by the normal catalog resolver.
    ``network=False`` (HTTP request paths) reads the cached and in-tree catalogs only.
    """
    from hermes_cli.plugin_catalog import (
        cached_removed_entries, load_catalog_live, match_removed, resolved_removed_entries,
    )
    from hermes_cli.plugins_cmd_catalog import normalized_platforms
    from hermes_cli.plugins_manifest import requires_hermes_error
    from hermes_platform.host.facts import os_family

    try:
        entries = load_catalog_live(network=network)
        removed = resolved_removed_entries() if network else cached_removed_entries()
        host = os_family()
        return sorted((entry for entry in entries
            if entry.featured and entry.category == "memory"
            and not match_removed(entry.name, removed)
            and not match_removed(entry.repo, removed)
            and not requires_hermes_error({"requires_hermes": entry.requires_hermes})
            and (not entry.platforms or host in normalized_platforms(entry.platforms))),
            key=lambda entry: (entry.title or entry.name).casefold())
    except Exception:
        # Catalog availability must never hide installed providers or built-in memory.
        logger.warning("Featured memory catalog unavailable", exc_info=True)
        return []
