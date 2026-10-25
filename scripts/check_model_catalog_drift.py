#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Report the drift between the curated model lists and the live provider catalogs.

The OpenRouter picker is a curated allowlist: ``OPENROUTER_MODELS`` names the model
ids the picker offers, and the live ``/api/v1/models`` catalog decides which of them
are still served and tool-capable. A curated id the live catalog no longer serves is
inert (the picker drops it) but still sits in the published manifest, which is what
an offline install falls back to. Meanwhile a live, free, tool-capable model the
curated list does not name is invisible in the picker â€” the "the free ones never show
up" symptom (#17923, #89150, #131372).

The free tier churns fast: OpenRouter retires or renames a free SKU within days, so
this report is how the curated list is kept honest. Run it before a curation PR.

Usage:
    python scripts/check_model_catalog_drift.py [--json] [--provider openrouter]

Exit code is 0 when there is no drift and 1 when there is, so it can gate CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# The curated tables import modules that read config; give them a HOME.
os.environ.setdefault("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))

DEFAULT_TIMEOUT = 20.0
USER_AGENT = "hermes-model-catalog-drift"

OPENROUTER_LIVE_URL = "https://openrouter.ai/api/v1/models"


def _fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def _supports_tools(item: dict[str, Any]) -> bool:
    """``supported_parameters`` advertises ``tools``; absent or malformed means yes."""
    params = item.get("supported_parameters")
    return "tools" in params if isinstance(params, list) else True


def _is_free(pricing: Any) -> bool:
    """Free means BOTH price fields are present and zero.

    A missing field is unknown pricing, not free pricing: reading it as 0 would
    mark a paid model free.
    """
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing["prompt"]) == 0 and float(pricing["completion"]) == 0
    except (KeyError, TypeError, ValueError):
        return False


def _live_free_ids(live_items: list[Any], *, exclude: set[str] | None = None) -> list[str]:
    """The free, tool-capable model ids in a live catalog, in catalog order.

    Free is decided by the price, not by the ``:free`` suffix: a suffix is a naming
    convention, and reading it as free would put a paid model in the free tier.
    """
    excluded = exclude or set()
    ids: list[str] = []
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid or mid in excluded:
            continue
        if not _is_free(item.get("pricing")):
            continue
        if not _supports_tools(item):
            continue
        ids.append(mid)
    return ids


def openrouter_drift() -> dict[str, Any]:
    """The curated list vs the live OpenRouter catalog."""
    from hermes_cli.models_catalog_static import OPENROUTER_MODELS

    curated = [mid for mid, _desc in OPENROUTER_MODELS]
    curated_set = set(curated)
    payload = _fetch_json(OPENROUTER_LIVE_URL)
    live_items = payload.get("data", []) if isinstance(payload, dict) else []
    live_ids = {
        str(item.get("id"))
        for item in live_items
        if isinstance(item, dict) and item.get("id")
    }
    live_free = _live_free_ids(live_items, exclude={"openrouter/free"})

    return {
        "provider": "openrouter",
        "curated_count": len(curated),
        "live_count": len(live_ids),
        # Curated ids the live catalog no longer serves: inert in the picker, still
        # published in the manifest an offline install falls back to.
        "retired": sorted(mid for mid in curated if mid not in live_ids),
        # Live free models the curated list does not name: invisible in the picker.
        "uncurated_free": sorted(mid for mid in live_free if mid not in curated_set),
        "live_free_count": len(live_free),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="openrouter", choices=["openrouter"])
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 when there is drift (for CI). The default is a report that always exits 0.")
    args = parser.parse_args()

    report = openrouter_drift()
    retired = report.pop("retired")
    uncurated = report.pop("uncurated_free")

    if args.json:
        print(json.dumps({**report, "retired": retired, "uncurated_free": uncurated}, indent=2))
    else:
        print(f"OpenRouter: {report['curated_count']} curated ids, {report['live_count']} live models, "
              f"{report['live_free_count']} live free models")
        print(f"\nCurated ids the live catalog does not publish ({len(retired)}):")
        for mid in retired:
            print(f"  - {mid}")
        print("  (a curated id the catalog does not publish may be retired OR simply unpublished —"
              " OpenRouter does not publish every curated id, the stealth SKUs among them)")
        print(f"\nLive free models the curated list does not carry ({len(uncurated)}):")
        for mid in uncurated:
            print(f"  + {mid}")
        if retired or uncurated:
            print("\nRefresh the curated list with: python scripts/build_model_catalog.py")
            print("See website/docs/reference/model-catalog.md for the curation workflow.")

    return 1 if (args.strict and (retired or uncurated)) else 0


if __name__ == "__main__":
    sys.exit(main())

