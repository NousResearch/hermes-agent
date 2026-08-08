#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cron_control.report import main


if __name__ == "__main__":
    raise SystemExit(main())
