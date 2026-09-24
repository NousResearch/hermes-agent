"""Real serve entry with observable, bounded shutdown and real no-agent cron work.

Only lifecycle pacing is fixture-owned. HTTP routes, admission, scheduler dispatch,
cron ledgers, session-token auth, and uvicorn shutdown are production code.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time

# The installed interpreter supplies dependencies, never the implementation.
sys.path.insert(0, sys.argv[1])
home = Path(os.environ["HERMES_HOME"])
assert home.is_relative_to(Path(os.environ["HOME"]))

if sys.argv[2] == "cron-busy":
    from cron.jobs import create_job
    from cron.scheduler import tick

    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("cron_work.py"), scripts / "cron_work.py")
    job = create_job(
        prompt=None, schedule=datetime.now(timezone.utc).isoformat(),
        name="Native pool retirement work", script="cron_work.py", no_agent=True,
        deliver="local", repeat=1, workdir=str(home),
    )
    tick(verbose=False, sync=False)
    deadline = time.monotonic() + 30
    while not (home / "cron-started").exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"Real cron job {job['id']} did not start")
        time.sleep(0.03)
    print("RETIREMENT_CRON_STARTED", flush=True)

from hermes_cli.web_server import app, start_server

original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def observed_lifespan(application):
    async with original_lifespan(application):
        yield
        # Hold the real process after SIGTERM, not a fake exit promise. This
        # makes premature lease release observable even on a fast workstation.
        print("RETIREMENT_SHUTDOWN_STARTED", flush=True)
        deadline = time.monotonic() + 12
        while not (home / "allow-exit").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Native parent did not release shutdown barrier")
            await asyncio.sleep(0.03)


app.router.lifespan_context = observed_lifespan
start_server(host="127.0.0.1", port=0, open_browser=False, headless=True)
