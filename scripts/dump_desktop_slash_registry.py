#!/usr/bin/env python3
"""Regenerate apps/desktop/src/lib/desktop-slash-registry.json from COMMAND_REGISTRY.

Run after changing any ``desktop=`` value or alias in ``hermes_cli/commands.py``;
``tests/hermes_cli/test_commands.py`` fails until the committed copy matches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "apps" / "desktop" / "src" / "lib" / "desktop-slash-registry.json"


def render() -> str:
    sys.path.insert(0, str(ROOT))
    from hermes_cli.commands import desktop_surface_registry

    return json.dumps(desktop_surface_registry(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
