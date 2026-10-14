#!/usr/bin/env python3
"""Build a validated Monet Project Primer or Agent Preview Pack.

The skill ships its audited runtime so hosted Hermes, Windows, and macOS all
produce the same wire contract regardless of which Monet Desktop version is
installed locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from project_primer_runtime import (  # type: ignore[import-not-found]
    PrimerError,
    desktop_installation_status,
    encode_setup_url,
    load_agent_preview,
    load_project_primer,
    open_package_in_monet,
    write_project_primer_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a validated Monet Project Primer.")
    parser.add_argument("primer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional Agent Preview JSON with ordered relative PNG paths.",
    )
    parser.add_argument("--open", action="store_true", dest="open_package")
    args = parser.parse_args()

    try:
        primer = load_project_primer(args.primer)
        preview = load_agent_preview(args.preview) if args.preview is not None else None
        package = write_project_primer_package(
            primer,
            args.output,
            preview=preview,
            preview_root=args.preview.parent if args.preview is not None else None,
        )
        result = {
            "package": str(package),
            "setupURL": encode_setup_url(primer),
            "desktop": desktop_installation_status(),
            "recommendedSurface": "ipad",
            "containsSecrets": False,
            "previewPages": len(preview.pages) if preview is not None else 0,
        }
        print(json.dumps(result, indent=2))
        if args.open_package:
            open_package_in_monet(package)
        return 0
    except PrimerError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
