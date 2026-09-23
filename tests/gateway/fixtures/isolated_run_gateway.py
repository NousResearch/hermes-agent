"""Disposable HTTP Runs controller for process-crash recovery tests."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms import api_server_runs


async def main() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(sys.argv[1])
    fixture = Path(__file__).with_name("isolated_run_synthetic.py")
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    with patch.object(api_server_runs, "_isolated_run_command", return_value=[sys.executable, str(fixture)]):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        print(site._server.sockets[0].getsockname()[1], flush=True)
        await asyncio.Event().wait()


asyncio.run(main())
