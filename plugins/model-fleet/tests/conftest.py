"""Test bootstrap for the model-fleet plugin.

The plugin imports hermes modules (``cron.jobs``, ``hermes_cli.*``, ``utils``) at
call time. In-tree those resolve because the plugin runs inside the Hermes process;
in a standalone pytest run they need the repo root on ``sys.path``. Resolve it from
the installed package first, then walk up looking for a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path | None:
    try:
        import hermes_constants

        candidate = Path(hermes_constants.__file__).resolve().parent.parent
        if (candidate / "hermes_cli").is_dir():
            return candidate
    except Exception:
        pass
    for parent in Path(__file__).resolve().parents:
        if (parent / "hermes_cli").is_dir() and (parent / "cron").is_dir():
            return parent
    return None


_ROOT = _repo_root()
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
