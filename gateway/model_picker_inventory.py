"""Gateway-owned projection of application inventory into /model provider rows."""

from __future__ import annotations

from typing import Any


def model_provider_rows(
    listing: dict[str, Any],
    *,
    max_models: int,
    interactive: bool,
    refresh: bool = False,
) -> list[dict]:
    """Build gateway picker/list rows from application facts.

    Provider/model candidate semantics remain in models/providers. Credential and
    config acquisition remain application-owned pending Phase 6.
    """
    from hermes_cli.inventory import ConfigContext, build_models_payload

    ctx = ConfigContext(
        current_provider=str(listing.get("current_provider") or ""),
        current_model=str(listing.get("current_model") or ""),
        current_base_url=str(listing.get("current_base_url") or ""),
        user_providers=listing.get("user_providers") or {},
        custom_providers=listing.get("custom_providers") or [],
        excluded_providers=listing.get("excluded_providers") or [],
    )
    payload = build_models_payload(
        ctx,
        max_models=max_models,
        for_picker=interactive,
        refresh=refresh,
        probe_custom_providers=refresh,
        probe_current_custom_provider=not refresh,
        non_blocking_catalogs=not refresh,
    )
    rows = list(payload.get("providers") or [])
    if not interactive:
        rows = [row for row in rows if str(row.get("slug") or "").lower() != "moa"]
    return rows


__all__ = ["model_provider_rows"]
