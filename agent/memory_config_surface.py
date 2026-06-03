"""Desktop config-surface helpers for memory providers.

Memory providers already declare their configurable fields via
``MemoryProvider.get_config_schema()`` (consumed by the ``hermes memory setup``
CLI wizard) and persist them via ``MemoryProvider.save_config()``. This module
adds the thin layer the Desktop app needs on top of that *existing* contract —
nothing parallel, nothing provider-specific:

* :func:`normalize_field` — enrich one raw schema field with display metadata
  (``label``, ``kind``, ``tier``, ``placeholder``) derived from what the field
  already declares, so legacy schemas work untouched.
* :func:`enrich_schema` — normalize a whole schema.
* :func:`default_read_config` — read current values for the conventional
  ``<hermes_home>/<provider>/config.json`` + env-secret layout. Providers with
  non-conventional storage (e.g. Honcho's host-keyed ``honcho.json``) override
  ``MemoryProvider.read_config()`` instead.

Secrets are write-only: their value is NEVER returned over the read path, only
an ``is_set`` flag.

This module is pure data/logic. It imports nothing from the web/HTTP layer.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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
    """Infer a renderer ``kind`` from a legacy schema field.

    Honors an explicit ``kind`` if valid, otherwise derives:
    ``secret:true`` -> secret, ``choices`` present -> select, else text.
    """
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
    if isinstance(tier, str) and tier in _VALID_TIERS:
        return tier
    return TIER_SAFE


def normalize_field(field: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich one raw schema field into the Desktop field shape.

    Always returns a dict with: ``key, label, kind, tier, description,
    placeholder, options, required, env_key``. ``options`` is a list of
    ``{value, label}`` derived from legacy ``choices`` (or an explicit
    ``options`` list of dicts). Never includes a secret's value.
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

    enriched = {
        "key": key,
        "label": str(field.get("label") or _prettify(key)),
        "kind": kind,
        "tier": _derive_tier(field),
        "description": str(field.get("description", "")),
        "placeholder": str(field.get("placeholder", "")),
        "options": options,
        "required": bool(field.get("required", False)),
        # Surfaced so the write path knows where a secret lands; harmless for
        # non-secret fields. Not a value, just the env var name.
        "env_key": field.get("env_var") if kind == KIND_SECRET else None,
    }

    # Conditional visibility: a field may be gated on the current value of
    # other fields (e.g. Hindsight's api_url is shown only when mode==cloud).
    # The clause is carried through verbatim; the renderer evaluates it live
    # against the in-progress form values, and the write path skips fields
    # whose ``when`` doesn't match the submitted values. ``default`` is carried
    # so a select that resets can fall back sensibly.
    when = field.get("when")
    if isinstance(when, dict) and when:
        enriched["when"] = {str(k): str(v) for k, v in when.items()}
    if "default" in field:
        enriched["default"] = str(field.get("default", ""))
    return enriched


def field_visible(field: Dict[str, Any], values: Dict[str, str]) -> bool:
    """True if a field's ``when`` clause matches the given form values.

    A field with no ``when`` is always visible. Otherwise every key/value pair
    in ``when`` must equal the corresponding submitted value. This is the same
    gating the CLI wizard applies, made available to the Desktop write path and
    renderer so conditional fields (e.g. mode-gated Hindsight fields) behave
    identically across surfaces.
    """
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    return all(str(values.get(k, "")) == str(v) for k, v in when.items())


def enrich_schema(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a full ``get_config_schema()`` result for the Desktop renderer."""
    return [normalize_field(f) for f in (schema or []) if f.get("key")]


def _provider_config_path(hermes_home: str, provider_name: str) -> Path:
    """Conventional per-provider config location used by the default reader."""
    return Path(hermes_home) / provider_name / "config.json"


def _read_conventional_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read provider config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def default_read_config(
    schema: List[Dict[str, Any]],
    provider_name: str,
    hermes_home: str,
) -> Dict[str, Dict[str, Any]]:
    """Read current field state for the conventional storage layout.

    Returns ``{key: {"value": str, "is_set": bool}}``. Secret fields always
    have ``value == ""`` (write-only); ``is_set`` reflects whether the env var
    is populated. Non-secret fields fall back to the schema ``default``.

    Providers whose storage differs override ``MemoryProvider.read_config()``.
    """
    data = _read_conventional_file(_provider_config_path(hermes_home, provider_name))
    state: Dict[str, Dict[str, Any]] = {}

    for raw in schema or []:
        key = raw.get("key")
        if not key:
            continue
        field = normalize_field(raw)

        if field["kind"] == KIND_SECRET:
            env_key = field["env_key"]
            is_set = bool(env_key and os.environ.get(env_key))
            state[key] = {"value": "", "is_set": is_set}
            continue

        value = data.get(key)
        if value in (None, ""):
            value = raw.get("default", "")
        state[key] = {"value": str(value), "is_set": value not in (None, "")}

    return state
