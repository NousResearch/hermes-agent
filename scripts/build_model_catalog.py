#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Build the Hermes Model Catalog — a centralized JSON manifest of curated models.

This script reads the in-repo hardcoded curated lists (``OPENROUTER_MODELS``,
``_PROVIDER_MODELS``) and writes them to a JSON manifest that the
Hermes CLI fetches at runtime. Publishing the catalog through the docs site
lets maintainers update model lists without shipping a Hermes release.

OpenRouter and Nous Portal are read as the whole list. Every other provider's
block only adds picker rows to the list an install already ships (see
``hermes_cli.model_catalog.merge_remote_models``). ``model_metadata`` and
``context_lengths`` publish the built-in metadata and context-window tables
the same way: an install reads them only for a model it has no entry for.

The runtime fetcher falls back to the same in-repo hardcoded lists if the
manifest is unreachable, so this script is a convenience for keeping the
manifest in sync — not a source of truth.

Usage::

    python scripts/build_model_catalog.py

Output: ``website/static/api/model-catalog.json``

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

from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS
from agent.model_metadata import _CODEX_OAUTH_CONTEXT_FALLBACK, DEFAULT_CONTEXT_LENGTHS
from agent.models_dev import _BUILTIN_MODEL_METADATA
from hermes_cli.model_catalog import OWN_LIST_ROWS
from hermes_cli.models import (
    OPENROUTER_MODELS,
    PREFERRED_SILENT_DEFAULT_MODEL,
    _PROVIDER_MODELS,
)

OUTPUT_PATH = os.path.join(REPO_ROOT, "website", "static", "api", "model-catalog.json")
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


def _provider_blocks() -> dict:
    """One ``{"models": [{"id": ...}]}`` block per published built-in row, in table order."""
    return {
        provider: {"models": [{"id": mid} for mid in _PROVIDER_MODELS[provider]]}
        for provider in _PROVIDER_MODELS
        if provider not in OWN_LIST_ROWS and _PROVIDER_MODELS[provider]
    }


def _model_metadata() -> dict:
    """``_BUILTIN_MODEL_METADATA`` as ``{models.dev provider id: {model id: entry}}``."""
    tables: dict = {}
    for (provider, model), entry in _BUILTIN_MODEL_METADATA.items():
        tables.setdefault(provider, {})[model] = entry
    return tables


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
            **_provider_blocks(),
        },
        # Read only for a model the install has no entry or row for.
        "model_metadata": _model_metadata(),
        "context_lengths": {
            "default": dict(DEFAULT_CONTEXT_LENGTHS),
            "bedrock": dict(BEDROCK_CONTEXT_LENGTHS),
            "codex_oauth": dict(_CODEX_OAUTH_CONTEXT_FALLBACK),
        },
    }


def main() -> int:
    catalog = build_catalog()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {OUTPUT_PATH}")
    for provider, block in catalog["providers"].items():
        print(f"  {provider}: {len(block['models'])} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
