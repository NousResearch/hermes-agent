#!/usr/bin/env python3
"""Entry point for Microsoft Graph delegated OAuth setup."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from microsoft_auth import main


if __name__ == "__main__":
    main()