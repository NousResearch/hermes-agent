#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Prefer the embedded runtime installed by `go-workflow init`; fall back to the
# repo root for this source checkout and to site-packages when installed.
sys.path.insert(0, str(ROOT / ".go-workflow" / "runtime"))
sys.path.insert(0, str(ROOT))

from go_workflow.__main__ import main

raise SystemExit(main(['gate', *sys.argv[1:]]))
