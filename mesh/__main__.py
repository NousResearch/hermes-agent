"""hermes mesh — runnable as `python -m mesh <cmd> ...` during development.

In-repo this dispatches via `hermes_cli/mesh.py`; this entry point lets us
exercise the implementation against a real node before upstreaming.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `hermes_cli/mesh.py` importable when running from the scaffold dir.
SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCAFFOLD_ROOT))

from hermes_cli import mesh as cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli.main())
