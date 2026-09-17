"""A local-only script actually dispatched by Hermes' no-agent cron scheduler."""

import json
import os
from pathlib import Path
import time

home = Path.cwd()
(home / "cron-started").write_text(json.dumps({"pid": os.getpid()}))
deadline = time.monotonic() + 150
while not (home / "finish-cron").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("Native fixture did not release its cron work")
    # Atomic heartbeat: the test observes work continuing after both retirements.
    temporary = home / "cron-heartbeat.tmp"
    temporary.write_text(str(time.monotonic_ns()))
    temporary.replace(home / "cron-heartbeat")
    time.sleep(0.05)
(home / "cron-finished").write_text("completed")
print("Native retirement cron completed without a model call")
