"""Desktop config surface for memory providers.

A single module that turns a provider's declarative ``get_config_schema()`` into
the field vocabulary the Desktop settings panel renders from, and assembles the
GET payload. Providers already declare their fields (for ``hermes memory setup``)
and persist them (``save_config``) and read them back (``read_config``); this
layer is purely how Desktop *presents* that config — labels, field kinds, tiers,
options, and conditional (``when``) visibility.

It lives in ``hermes_cli/`` (next to its only consumer, ``web_server``) so the
presentation vocabulary stays out of the core runtime in ``agent/``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Field kinds understood by the generic Desktop renderer.
KIND_TEXT = "text"
KIND_SECRET = "secret"
KIND_SELECT = "select"
KIND_BOOL = "bool"
KIND_NUMBER = "number"

_VALID_KINDS = {KIND_TEXT, KIND_SECRET, KIND_SELECT, KIND_BOOL, KIND_NUMBER}

# Field tiers. Tier is a GROUPING, not a lock — ``advanced`` fields are still
# editable in Desktop (desktop users may not know the CLI wizard exists);
# they just render under an "Advanced" disclosure with confirm-on-change.
TIER_SAFE = "safe"
TIER_ADVANCED = "advanced"

_VALID_TIERS = {TIER_SAFE, TIER_ADVANCED}


def _prettify(key: str) -> str:
    """``api_key`` / ``baseUrl`` -> ``Api Key`` / ``Base Url`` for a label."""
    spaced: List[str] = []
    prev_lower = False
    for ch in key.replace("_", " ").replace("-", " "):
        if ch.isupper() and prev_lower:
            spaced.append(" ")
        spaced.append(ch)
        prev_lower = ch.islower()
    return " ".join(w.capitalize() for w in "".join(spaced).split())


def _derive_kind(field: Dict[str, Any]) -> str:
    """secret:true -> secret, choices -> select, explicit kind honored, else text."""
    explicit = field.get("kind")
    if isinstance(explicit, str) and explicit in _VALID_KINDS:
        return explicit
    if field.get("secret"):
        return KIND_SECRET
    if field.get("choices"):
        return KIND_SELECT
    return KIND_TEXT


def _derive_tier(field: Dict[str, Any]) -> str:
    tier = field.get("tier")
    return tier if isinstance(tier, str) and tier in _VALID_TIERS else TIER_SAFE


def field_visible(field: Dict[str, Any], values: Dict[str, str]) -> bool:
    """True if a field's ``when`` clause matches the given values.

    No ``when`` -> always visible. Otherwise every key/value pair in ``when``
    must equal the corresponding submitted value. Mirrors the CLI wizard's
    gating so conditional fields (e.g. mode-gated Hindsight fields) behave the
    same on Desktop. Accepts raw or enriched fields — both carry ``when``.
    """
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    return all(str(values.get(k, "")) == str(v) for k, v in when.items())


def normalize_field(field: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich one raw schema field into the Desktop field shape.

    Returns ``key, label, kind, tier, description, placeholder, options,
    required, env_key`` (+ ``when``/``default`` when present). ``options`` come
    from an explicit ``options`` list or legacy ``choices``. Never a secret value.
    """
    key = str(field.get("key", ""))
    kind = _derive_kind(field)

    options: List[Dict[str, str]] = []
    raw_options = field.get("options")
    if isinstance(raw_options, list) and raw_options:
        for opt in raw_options:
            if isinstance(opt, dict):
                options.append({
                    "value": str(opt.get("value", "")),
                    "label": str(opt.get("label", opt.get("value", ""))),
                    "description": str(opt.get("description", "")),
                })
    else:
        for choice in field.get("choices") or []:
            options.append({"value": str(choice), "label": str(choice), "description": ""})

    enriched: Dict[str, Any] = {
        "key": key,
        "label": str(field.get("label") or _prettify(key)),
        "kind": kind,
        "tier": _derive_tier(field),
        "description": str(field.get("description", "")),
        "placeholder": str(field.get("placeholder", "")),
        "options": options,
        "required": bool(field.get("required", False)),
        # Where a secret lands on write; None for non-secret fields. Not a value.
        "env_key": field.get("env_var") if kind == KIND_SECRET else None,
    }

    # Conditional visibility carried through verbatim; the renderer evaluates it
    # live and the write path skips fields whose ``when`` doesn't match.
    when = field.get("when")
    if isinstance(when, dict) and when:
        enriched["when"] = {str(k): str(v) for k, v in when.items()}
    if "default" in field:
        enriched["default"] = str(field.get("default", ""))
    return enriched


def enrich_schema(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a full ``get_config_schema()`` result for the Desktop renderer."""
    return [normalize_field(f) for f in (schema or []) if f.get("key")]


def build_surface(provider, hermes_home: str) -> Dict[str, Any]:
    """Assemble the Desktop config payload: ``{name, label, fields}``.

    Driven by the provider's ``get_config_schema()`` + ``read_config()``. Each
    field carries display metadata and current state (value/is_set, secrets
    masked). A provider with no config surface yields an empty ``fields`` list
    and the panel renders nothing.
    """
    fields = enrich_schema(provider.get_config_schema() or [])
    if fields:
        state = provider.read_config(hermes_home)
        for field in fields:
            fs = state.get(field["key"], {})
            field["value"] = "" if field["kind"] == KIND_SECRET else str(fs.get("value", ""))
            field["is_set"] = bool(fs.get("is_set", False))
    return {
        "name": provider.name,
        "label": getattr(provider, "display_label", None) or provider.name.capitalize(),
        "fields": fields,
    }
