#!/usr/bin/env python3
"""Standalone aiohttp server for runtime smoke testing.

Starts a minimal server with runtime routes, RuntimeExecutor, and
a configurable AgentFactory.

Usage:
    python3 scripts/standalone_runtime_server.py [--port PORT] [--fake]
    python3 scripts/standalone_runtime_server.py --fake
    python3 scripts/standalone_runtime_server.py --port 8642

Env vars:
    DEEPSEEK_API_KEY  — used by DefaultAgentFactory when resolving live creds
    API_SERVER_KEY    — generated automatically if not set (min 16 chars)
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("standalone_runtime_server")


def _ensure_api_server_key() -> str:
    os.environ.setdefault("API_SERVER_KEY", "standalone-runtime-smoke-key-001")
    return os.environ.get("API_SERVER_KEY", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    _ensure_api_server_key()

    from aiohttp import web
    from gateway.runtime.run_manager import RunManager
    from gateway.runtime.routes import register_runtime_routes

    app = web.Application()
    rm = RunManager()

    if args.fake:
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory
        factory = FakeAgentFactory(result={
            "final_response": "runtime executor cross repo smoke ok",
            "completed": True,
        })
        executor = RuntimeExecutor(rm, agent_factory=factory)
        logger.info("Using FakeAgentFactory (deterministic mode)")
    else:
        from gateway.runtime.agent_factory import DefaultAgentFactory
        from gateway.runtime.executor import RuntimeExecutor
        factory = DefaultAgentFactory()
        executor = RuntimeExecutor(rm, agent_factory=factory)
        logger.info("Using DefaultAgentFactory (live credential mode)")

    register_runtime_routes(
        app,
        run_manager=rm,
        executor=executor,
        register_create=True,
        register_status=True,
        register_events=True,
    )
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))

    runner = web.AppRunner(app)
    loop = asyncio.get_event_loop()

    async def start():
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.port)
        await site.start()
        logger.info("Runtime smoke server listening on http://127.0.0.1:%d (fake=%s)", args.port, args.fake)
        print("SERVER_READY", flush=True)

    async def shutdown():
        await runner.cleanup()
        logger.info("Runtime smoke server shut down")

    try:
        loop.run_until_complete(start())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if not loop.is_closed():
            loop.run_until_complete(shutdown())
            loop.close()


if __name__ == "__main__":
    main()
