"""Test bootstrap for the model-fleet plugin.

The plugin imports hermes modules (``cron.jobs``, ``hermes_cli.*``, ``utils``) at
call time. In-tree those resolve because the plugin runs inside the Hermes process;
in a standalone pytest run they need the repo root on ``sys.path``. Resolve it from
the installed package first, then walk up looking for a checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _looks_like_root(path: Path) -> bool:
    return (path / "hermes_cli").is_dir() and (path / "cron").is_dir()


def _repo_root() -> Path | None:
    """Find the checkout providing ``hermes_cli``/``cron``.

    Walk up from this file first: in-tree the plugin sits inside the repo, so a parent
    is the root and the result does not depend on what is importable. Only then fall
    back to the installed package — and check the package's own directory, not its
    parent's parent, which points outside the install for a flat layout.
    """
    for parent in Path(__file__).resolve().parents:
        if _looks_like_root(parent):
            return parent
    try:
        import hermes_constants

        for candidate in Path(hermes_constants.__file__).resolve().parents:
            if _looks_like_root(candidate):
                return candidate
    except Exception:
        pass
    return None


_ROOT = _repo_root()
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
