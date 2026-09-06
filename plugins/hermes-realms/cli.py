"""Checkout launcher; installed entrypoint uses realms.cli to avoid Hermes' cli module."""

from pathlib import Path
import runpy

main = runpy.run_path(str(Path(__file__).resolve().parent / "realms/_binding.py"))["load_runtime"]("cli").main

if __name__ == "__main__":
    raise SystemExit(main())
