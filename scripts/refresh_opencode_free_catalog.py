#!/usr/bin/env python3
"""Refresh and print the live OpenCode Zen free model catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a plain model list",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the in-process OpenCode catalog cache",
    )
    args = parser.parse_args()

    from hermes_cli import models as model_catalog

    free_models = model_catalog.opencode_free_model_ids(force_refresh=args.force)
    primary = model_catalog.resolve_config_model_id("opencode-zen", "auto-free", force_refresh=args.force)
    payload = {
        "provider": "opencode-zen",
        "primary_model": primary,
        "free_models": free_models,
        "catalog_url": "https://opencode.ai/zen/v1/models",
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"provider: {payload['provider']}")
        print(f"primary: {payload['primary_model']}")
        print(f"free_models ({len(free_models)}):")
        for model_id in free_models:
            print(f"  - {model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
