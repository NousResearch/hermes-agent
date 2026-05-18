#!/usr/bin/env python3
from __future__ import annotations

try:
    from scripts.hermes_pm.project_status import main
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from project_status import main  # type: ignore[no-redef]


if __name__ == "__main__":
    raise SystemExit(main())
