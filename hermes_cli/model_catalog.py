"""Compatibility hook for shipped updaters; live catalogue ownership is in models."""

from __future__ import annotations

def seed_cache_from_checkout(project_root) -> bool:
    from hermes_cli.catalog_context import catalog_cache_path
    from models.catalog_seed import seed_cache_from_checkout as seed_lower
    return seed_lower(project_root, catalog_cache_path())
