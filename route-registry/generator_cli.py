#!/usr/bin/env python3
"""CLI entry point for the route-registry generator.

Usage (dry-run default — nothing modified):
    python3 generator_cli.py --target-root /tmp/hermes-copy --out plan.json

Apply (explicit, triple-guarded, non-interactive only):
    python3 generator_cli.py --target-root /tmp/hermes-copy \
        --apply --confirm YES-APPLY-ROUTES
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "route_registry"))

from generator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())