#!/usr/bin/env python3
"""Backward-compatible wrapper → recipe_run.py --name product-audit."""

from __future__ import annotations

import sys

from recipe_run import main as recipe_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # If caller only passed --out, inject default name
    if "--name" not in argv and "-h" not in argv and "--help" not in argv and "--list" not in argv:
        argv = ["--name", "product-audit", *argv]
    return recipe_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
