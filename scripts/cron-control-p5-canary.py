#!/usr/bin/env python3
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cron_control.p5 import run_p5_canaries


if __name__ == "__main__":
    print(json.dumps(run_p5_canaries(), ensure_ascii=False, indent=2, sort_keys=True))
