"""Route-sync helper for Hermes directory-plugin installs.

When using the bundled plugin from a source checkout, the plugin directory is
not installed as a Python package. This wrapper makes route sync runnable
without modifying PYTHONPATH:

    python plugins/platforms/feishu-multitenancy/sync.py apply users.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hermes_multitenancy.sync.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
