from __future__ import annotations

import sys
from pathlib import Path

TRACEGUARD_DIR = Path(__file__).resolve().parents[1]
if str(TRACEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(TRACEGUARD_DIR))
