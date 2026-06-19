#!/usr/bin/env python3
"""Run investment-assistant Pydantic HITL flows from the repo checkout."""

from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.investment_assistant.pydantic_hitl_cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
