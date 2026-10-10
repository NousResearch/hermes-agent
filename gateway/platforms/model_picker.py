"""Shared entry-stage decision for interactive model pickers."""


def single_provider_for_picker(providers: list, *, grouped: bool = False):
    """Return the sole selectable provider, never a collapsed family or empty catalog.

    Inspect the complete rendered row set, not its first page. Adapters with a
    flat provider menu leave ``grouped`` false; Telegram uses its existing fold.
    """
    if grouped:
        try:
            from hermes_cli.models_catalog_static import group_providers
        except Exception:
            # Match the provider keyboard's flat-menu fallback.
            return single_provider_for_picker(providers)

        rows = group_providers([p.get("slug") for p in providers])
        if len(rows) != 1 or rows[0]["kind"] != "single":
            return None
        provider = next((p for p in providers if p.get("slug") == rows[0]["slug"]), None)
    else:
        provider = providers[0] if len(providers) == 1 else None
    return provider if provider and provider.get("models") else None
