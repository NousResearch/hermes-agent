#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Build the Hermes Model Catalog — a centralized JSON manifest of curated models.

This script reads the in-repo hardcoded curated lists (``OPENROUTER_MODELS``,
``_PROVIDER_MODELS["nous"]``) and writes them to a JSON manifest that the
Hermes CLI fetches at runtime. Publishing the catalog through the docs site
lets maintainers update model lists without shipping a Hermes release.

The runtime fetcher falls back to the same in-repo hardcoded lists if the
manifest is unreachable, so this script is a convenience for keeping the
manifest in sync — not a source of truth.

Usage::

    python scripts/build_model_catalog.py

Output: ``apps/docs/public/api/model-catalog.json`` (served at
``/docs/api/model-catalog.json``) plus a byte-identical legacy copy at
``website/static/api/model-catalog.json`` — the raw fallback URL released
Hermes clients fetch, which a web redirect cannot repair.

Live URL (after ``deploy-site.yml`` runs on merge to main):
``https://hermes-agent.nousresearch.com/docs/api/model-catalog.json``
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, UTC

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Ensure HERMES_HOME is set for imports that touch it at module level.
os.environ.setdefault("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))

from hermes_cli.models import (
    OPENROUTER_MODELS,
    PREFERRED_SILENT_DEFAULT_MODEL,
    _PROVIDER_MODELS,
)

OUTPUT_PATHS = (
    # Served by the docs app at /docs/api/model-catalog.json.
    os.path.join(REPO_ROOT, "apps", "docs", "public", "api", "model-catalog.json"),
    # Legacy raw URL fetched by released clients
    # (hermes_cli.model_catalog.DEFAULT_CATALOG_FALLBACK_URLS): a web redirect
    # cannot repair a GitHub raw path, so the generator keeps this copy
    # byte-identical to the one above until a support policy retires it.
    os.path.join(REPO_ROOT, "website", "static", "api", "model-catalog.json"),
)
CATALOG_VERSION = 1


def _openrouter_entry(mid: str, desc: str) -> dict:
    entry: dict = {"id": mid, "description": desc}
    if mid == PREFERRED_SILENT_DEFAULT_MODEL:
        entry["description"] = desc or "default"
        entry["default"] = True
    return entry


def _nous_entry(mid: str) -> dict:
    entry: dict = {"id": mid}
    if mid == PREFERRED_SILENT_DEFAULT_MODEL:
        entry["default"] = True
    return entry


def build_catalog() -> dict:
    return {
        "version": CATALOG_VERSION,
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": {
            "source": "hermes-agent repo",
            "docs": "https://hermes-agent.nousresearch.com/docs/reference/model-catalog",
        },
        "providers": {
            "openrouter": {
                "metadata": {
                    "display_name": "OpenRouter",
                    "note": (
                        "Descriptions drive picker badges. Live /api/v1/models "
                        "filters curated ids by tool-calling support and free pricing. "
                        'The entry labeled "default": true is the model Hermes '
                        "silently lands on when the user never picked one."
                    ),
                },
                "models": [
                    _openrouter_entry(mid, desc)
                    for mid, desc in OPENROUTER_MODELS
                ],
            },
            "nous": {
                "metadata": {
                    "display_name": "Nous Portal",
                    "note": (
                        'The entry labeled "default": true is the model Hermes '
                        "silently lands on when the user never picked one."
                    ),
                },
                "models": [
                    _nous_entry(mid)
                    for mid in _PROVIDER_MODELS.get("nous", [])
                ],
            },
        },
    }


def main() -> int:
    catalog = build_catalog()
    # Serialize once: every published copy must be the same bytes.
    payload = json.dumps(catalog, indent=2) + "\n"
    for output_path in OUTPUT_PATHS:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"Wrote {output_path}")
    for provider, block in catalog["providers"].items():
        print(f"  {provider}: {len(block['models'])} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
