#!/usr/bin/env python3
"""Run the bundled session observer without installing a Python package."""

from tmux_supervisor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
