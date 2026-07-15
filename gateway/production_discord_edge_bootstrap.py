#!/usr/bin/env python3
"""Production-named systemd bootstrap for the privileged Discord edge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gateway.full_canary_discord_edge_bootstrap import run_edge


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="absolute root-owned Discord edge JSON config",
    )
    arguments = parser.parse_args(argv)
    run_edge(
        Path(arguments.config),
        readiness_bootstrap_path=__file__,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
