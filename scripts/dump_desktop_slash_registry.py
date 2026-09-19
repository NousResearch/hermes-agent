#!/usr/bin/env python3
"""Regenerate apps/desktop/src/lib/desktop-slash-registry.json from COMMAND_REGISTRY.

Run after changing any ``desktop=`` value or alias in ``hermes_cli/commands.py``;
``tests/hermes_cli/test_desktop_slash_registry.py`` fails until the committed
copy matches. ``--check`` writes nothing and exits 1 when the committed JSON
differs from what the registry renders (the same verdict the pytest gives,
usable from any shell or CI step without pytest).
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


def check(out: Path = OUT) -> int:
    """0 when ``out`` matches the registry byte-for-byte, else 1 with a hint on stderr."""
    committed = out.read_text(encoding="utf-8-sig") if out.exists() else ""
    if committed == render():
        return 0
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"{rel} is stale — run scripts/dump_desktop_slash_registry.py", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        return check()
    if argv:
        print(f"usage: {Path(__file__).name} [--check]", file=sys.stderr)
        return 2
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
