"""Flatten a dict-valued model id without dropping the rest of that dict.

Readers (and the next save, which persists the canonicalizer's output) must see a
string id. ``provider`` still promotes onto ``model``. Every other nested key is
returned so the caller can lift it onto ``model`` only when that key is unset
(#153296). ``name`` is not lifted: once ``default`` is set the canonicalizer
deletes ``model.name``, so putting it back would either steal the id or lose it
on the following save.
"""

from __future__ import annotations

from typing import Any

# Consumed here, or deleted later as the model-id alias. Not data to lift.
_CONSUMED_NESTED_KEYS = frozenset({"model", "default", "provider", "name"})


def flatten_dict_valued_model_ids(model: dict[str, Any]) -> dict[str, Any]:
    """Flatten dict-valued ``default`` / ``model`` / ``name`` in place.

    Nested ``provider`` replaces an empty or ``"auto"`` outer provider and never
    a configured one. Returns the other nested keys, first one wins, for the
    caller to copy onto ``model`` after root-key migration.
    """
    lifted: dict[str, Any] = {}
    for key in ("default", "model", "name"):
        value = model.get(key)
        if not isinstance(value, dict):
            continue
        nested_model = value.get("model") or value.get("default")
        nested_provider = str(value.get("provider") or "").strip()
        model[key] = str(nested_model or "").strip()
        if nested_provider:
            outer_provider = str(model.get("provider") or "").strip()
            if not outer_provider or outer_provider == "auto":
                model["provider"] = nested_provider
        for extra_key, extra_val in value.items():
            if extra_key in _CONSUMED_NESTED_KEYS:
                continue
            if extra_key in model or extra_key in lifted:
                continue
            lifted[extra_key] = extra_val
    return lifted
