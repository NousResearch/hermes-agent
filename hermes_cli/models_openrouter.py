"""OpenRouter live-catalog helpers that the curated manifest cannot answer.

The curated manifest (``hermes_cli.model_catalog``) is a hand-picked subset, and a whole
generation of ``:free`` SKUs or OpenRouter's own router family (``openrouter/free``) can
reach the live ``/v1/models`` catalog weeks before any manifest update lands. The picker
tails its curated list with these live-only entries so the menu keeps up without
curation; the tool-support and pricing predicates are injected by the caller.
"""

from __future__ import annotations

from typing import Any, Callable


def _openrouter_model_supports_tools(item: Any) -> bool:
    """True when ``supported_parameters`` advertises ``tools`` (hermes-agent is tool-calling-first).

    Permissive when the field is absent/malformed: some OpenRouter-compatible gateways (Nous Portal,
    private mirrors) don't populate it, and the picker must not silently empty for them.

    Ported from Kilo-Org/kilocode#9068.
    """
    params = item.get("supported_parameters") if isinstance(item, dict) else None
    return "tools" in params if isinstance(params, list) else True



def append_live_free_routers(
    curated: list[tuple[str, str]],
    live_items: list[Any],
    supports_tools: Callable[[Any], bool],
    is_free: Callable[[Any], bool],
) -> list[tuple[str, str]]:
    """Append live-only entries to *curated* and return the extended list.

    Two groups qualify and must already exist in *live_items*:

    - ids ending with ``:free`` (any vendor's zero-priced SKU),
    - ids under the ``openrouter/`` namespace, i.e. OpenRouter's own router family
      (``openrouter/free`` is the case that motivated the tail: $0, tool-capable,
      absent from the manifest).

    Everything else stays out: paid vendor models belong to curation, and a hidden
    gate is intentional hermes-agent behavior (tool-calling-first — surfacing a model
    without ``tools`` fails at the first tool call). Curated ids are never duplicated
    and keep their order; tailed entries follow in live-catalog order.
    """
    seen = {mid for mid, _ in curated}
    extras: list[tuple[str, str]] = []
    for item in live_items:
        if not isinstance(item, dict):
            continue
        live_id = str(item.get("id") or "").strip()
        if not live_id or live_id in seen:
            continue
        if not (live_id.endswith(":free") or live_id.startswith("openrouter/")):
            continue
        if not supports_tools(item):
            continue
        if is_free(item.get("pricing")):
            extras.append((live_id, "free"))
        elif live_id.startswith("openrouter/"):
            extras.append((live_id, "auto-router"))
    curated.extend(extras)
    return curated
