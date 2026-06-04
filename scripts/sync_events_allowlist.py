#!/usr/bin/env python3
"""Sync agent/events_allowlist.py from packages/hermes-events/src/allowlist.ts.

Run via:  python scripts/sync_events_allowlist.py
Or npm:   npm run sync:events-allowlist
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# hermes-agent lives at <monorepo>/.hermes/hermes-agent — walk up to monorepo root
MONOREPO_ROOT = REPO_ROOT.parent.parent
TS_SOURCE = MONOREPO_ROOT / "packages" / "hermes-events" / "src" / "allowlist.ts"
PY_TARGET = REPO_ROOT / "agent" / "events_allowlist.py"


def parse_ts_allowlist(ts_path: Path) -> list[str]:
    src = ts_path.read_text(encoding="utf-8")
    # Match the array between [ ... ] in `export const EVENT_ALLOWLIST = [ ... ]`
    m = re.search(r"export const EVENT_ALLOWLIST\s*=\s*\[(.*?)\]\s*as const", src, re.DOTALL)
    if not m:
        raise ValueError(f"EVENT_ALLOWLIST array not found in {ts_path}")
    raw = m.group(1)
    return [s.strip().strip('"').strip("'") for s in raw.split(",") if s.strip().strip('"').strip("'")]


def generate_py(events: list[str]) -> str:
    items = "\n".join(f'    "{e}",' for e in events)
    return f'''# GENERATED — do not edit
# Source: packages/hermes-events/src/allowlist.ts
# Regenerate: npm run sync:events-allowlist  (or python scripts/sync_events_allowlist.py)

EVENT_ALLOWLIST = [
{items}
]

def is_allowed_event(event_name: str) -> bool:
    return event_name in EVENT_ALLOWLIST
'''


def main() -> None:
    if not TS_SOURCE.exists():
        print(f"ERROR: TypeScript source not found: {TS_SOURCE}", file=sys.stderr)
        sys.exit(1)

    events = parse_ts_allowlist(TS_SOURCE)
    py_content = generate_py(events)

    current = PY_TARGET.read_text(encoding="utf-8") if PY_TARGET.exists() else ""
    if current == py_content:
        print(f"✅  {PY_TARGET.name} already up to date ({len(events)} events)")
        return

    PY_TARGET.write_text(py_content, encoding="utf-8")
    print(f"✅  Wrote {PY_TARGET} ({len(events)} events)")


if __name__ == "__main__":
    main()
