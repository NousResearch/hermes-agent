"""Canonical interpretation of model price and billing facts."""

from __future__ import annotations

from typing import Any, Optional

def _zero_priced(pricing: Any, keys: tuple[str, str], default: str) -> bool:
    """True when both pricing fields parse to 0 (missing fields read as ``default``)."""
    if not isinstance(pricing, dict):
        return False
    try:
        return all(float(pricing.get(k, default)) == 0 for k in keys)
    except (TypeError, ValueError):
        return False

def _is_subscription_billed(entry: Any) -> bool:
    """The gateway bills this catalog row to a subscription the account holds, not to credits."""
    return isinstance(entry, dict) and entry.get("billing_mode") == "subscription"

def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* costs no credits: zero-cost prompt AND completion pricing, or a row
    the gateway bills to a subscription."""
    entry = pricing.get(model_id)
    return bool(entry) and (_is_subscription_billed(entry) or _zero_priced(entry, ("prompt", "completion"), "1"))

def partition_nous_models_by_tier(
    model_ids: list[str], pricing: dict[str, dict[str, str]], free_tier: bool
) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable): free-tier users may only select free models
    (paid ones are returned as unavailable, shown grayed out)."""
    if not free_tier or not pricing:  # no pricing → can't determine, show everything
        return (model_ids, [])
    selectable = [mid for mid in model_ids if _is_model_free(mid, pricing)]
    return (selectable, [mid for mid in model_ids if mid not in selectable])

def _price_float(raw: Any, *, positive: bool) -> float | None:
    """*raw* as a finite float (> 0, or >= 0 when not *positive*); None when unset/invalid/NaN."""
    if raw in (None, ""):
        return None
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return None
    if n != n or (n <= 0 if positive else n < 0):
        return None
    return n

def _sale_pct(current: Any, original: Any) -> int | None:
    """Percent discount when *current* is strictly below *original* (both positive finite)."""
    cur, orig = _price_float(current, positive=True), _price_float(original, positive=True)
    if cur is None or orig is None or cur >= orig:
        return None
    return int(round((1.0 - (cur / orig)) * 100))

def compute_sale_discount(prompt: str, completion: str, original: Any) -> tuple[int, str, str] | None:
    """Sale chrome from gateway ``pricing.original`` (Nous Portal only; callers gate on the provider
    and opted in via ``include_sale_original=True``): ``(discount_percent, was_prompt_raw,
    was_completion_raw)`` when ``original`` is a dict and the current prompt (fallback: completion)
    rate is strictly below the original. Free / $0 models get a flat 100% off, with "was" prices
    only when the gateway served an original (a natively-free stealth model gets bare "-100%")."""
    orig_dict = original if isinstance(original, dict) else {}
    was_prompt = orig_dict.get("prompt")
    was_completion = orig_dict.get("completion")
    was_prompt_str = str(was_prompt) if was_prompt not in (None, "") else ""
    was_completion_str = str(was_completion) if was_completion not in (None, "") else ""

    if _price_float(prompt, positive=False) == 0 and _price_float(completion, positive=False) in (0, None):
        return (100, was_prompt_str, was_completion_str)

    if not isinstance(original, dict) or (not was_prompt_str and not was_completion_str):
        return None

    pct = _sale_pct(prompt, was_prompt)
    if pct is not None:
        return (pct, was_prompt_str, was_completion_str) if pct >= 1 else None
    pct = _sale_pct(completion, was_completion)
    if pct is not None:
        return (pct, was_prompt_str, was_completion_str) if pct >= 1 else None
    return None

def _pricing_entry(pricing: dict, prompt_key: str = "prompt", completion_key: str = "completion") -> dict[str, Any]:
    """Picker-shape ``{prompt, completion[, input_cache_read, input_cache_write]}`` from a catalog
    ``pricing`` block whose cache fields already use the hermes names."""
    entry: dict[str, Any] = {
        "prompt": str(pricing.get(prompt_key, "")),
        "completion": str(pricing.get(completion_key, "")),
    }
    for key in ("input_cache_read", "input_cache_write"):
        if pricing.get(key):
            entry[key] = str(pricing[key])
    return entry

def _per_token(per_mtok: Any) -> str:
    """$/MTok → the per-token price string the picker expects."""
    return str(float(per_mtok) / 1_000_000)
