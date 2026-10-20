"""Prepare an interpreter that can run the icon generator without a Hermes install.

Pillow and resvg-py are core runtime dependencies, so builders that have no
Hermes runtime environment (desktop bundles, product staging) render icons on
the locked runtime dependencies alone, without installing the application.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Builders import this file from a separately prepared source tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pm


def prepare_icon_environment(source: Path, out: Path, cache: Path | None) -> Path:
    """Prepare a fresh interpreter with the locked runtime dependencies only."""
    return pm.build_environment(
        source=source.resolve(), out=out.resolve(), cache=cache.resolve() if cache else None,
        no_install_project=True, explicit=True,
    )
