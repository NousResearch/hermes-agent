"""Repository-root shim for ``python -m hermes_parity``."""

from __future__ import annotations

from scripts.hermes_parity import __version__
from scripts.hermes_parity.cli import main

__all__ = ["__version__", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
