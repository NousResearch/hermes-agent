"""Profile-scoped OpenRouter picker preference; explicit model routing stays unrestricted."""

from decimal import Decimal, InvalidOperation


def openrouter_free_only(config: dict | None = None) -> bool:
    """Read an actual YAML boolean, refusing ambiguous settings instead of widening choices."""
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    section = config
    for key in ("models", "openrouter"):
        if not isinstance(section, dict):
            raise ValueError("models.openrouter.free_only must be a boolean in nested mappings")
        section = section.get(key, {})
    if not isinstance(section, dict):
        raise ValueError("models.openrouter.free_only must be a boolean in nested mappings")
    value = section.get("free_only", False)
    if type(value) is not bool:
        raise ValueError("models.openrouter.free_only must be a YAML boolean (true or false)")
    return value


def explicitly_zero_priced(pricing) -> bool:
    """Require finite, explicit zero prompt AND completion prices; tiny decimals remain paid."""
    if not isinstance(pricing, dict):
        return False
    for key in ("prompt", "completion"):
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            return False
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return False
        if not amount.is_finite() or amount != 0:
            return False
    return True


def openrouter_picker_models(*, config: dict | None = None, force_refresh: bool = False):
    from hermes_cli.models import fetch_openrouter_models

    return fetch_openrouter_models(
        free_only=openrouter_free_only(config), force_refresh=force_refresh)


def apply_openrouter_picker_policy(rows: list[dict], *, max_models: int | None = None,
                                   force_refresh: bool = False) -> None:
    """Finalize choices after cached/configured/current-model rows have been assembled."""
    matching = [row for row in rows if str(row.get("slug", "")).lower() == "openrouter"]
    if not matching:
        return
    free_only = openrouter_free_only()
    matching = [row for row in matching if not (
        row.get("catalog_authoritative") and row.get("free_only") is free_only)]
    if not matching:
        return
    from hermes_cli.models import fetch_openrouter_models

    ids = [mid for mid, _ in fetch_openrouter_models(
        free_only=free_only, force_refresh=force_refresh)]
    for row in matching:
        # Current selection is carried separately by the inventory/session payload. Do not
        # reinsert a paid or unknown selection into a free-only list.
        row["models"] = ids[:max_models] if max_models is not None else list(ids)
        row["total_models"] = len(ids)
        row["catalog_authoritative"] = True
        row["free_only"] = free_only
