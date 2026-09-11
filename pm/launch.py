"""CLI entry after the isolated interpreter has been selected."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
