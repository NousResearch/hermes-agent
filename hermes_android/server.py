from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_android.bootstrap import bootstrap_android_runtime
from hermes_android.runtime_env import AndroidRuntimeEnv


@dataclass
class AndroidServerHandle:
    runtime: AndroidRuntimeEnv
    adapter: APIServerAdapter
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread

    @property
    def base_url(self) -> str:
        return f"http://{self.runtime.api_server_host}:{self.runtime.api_server_port}"

    def stop(self, timeout: float = 20.0) -> None:
        async def _shutdown() -> None:
            await self.adapter.disconnect()

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
        future.result(timeout=timeout)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=timeout)


def _build_runtime(runtime_payload: dict[str, Any]) -> AndroidRuntimeEnv:
    return AndroidRuntimeEnv(
        files_dir=Path(runtime_payload["files_dir"]),
        hermes_home=Path(runtime_payload["hermes_home"]),
        api_server_host=str(runtime_payload["api_server_host"]),
        api_server_port=int(runtime_payload["api_server_port"]),
        api_server_key=str(runtime_payload["api_server_key"]),
        api_server_model_name=str(runtime_payload["api_server_model_name"]),
    )


def start_local_api_server(
    files_dir: str,
    *,
    api_server_port: int | None = None,
    api_server_key: str | None = None,
    connect_timeout: float = 20.0,
) -> AndroidServerHandle:
    bootstrap = bootstrap_android_runtime(
        files_dir,
        api_server_port=api_server_port,
        api_server_key=api_server_key,
    )
    runtime = _build_runtime(bootstrap["runtime"])
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": runtime.api_server_host,
                "port": runtime.api_server_port,
                "key": runtime.api_server_key,
                "model_name": runtime.api_server_model_name,
                "cors_origins": [],
            },
        )
    )

    loop = asyncio.new_event_loop()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    thread = threading.Thread(target=_run_loop, name="hermes-android-api-server", daemon=True)
    thread.start()

    future = asyncio.run_coroutine_threadsafe(adapter.connect(), loop)
    if not future.result(timeout=connect_timeout):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=connect_timeout)
        raise RuntimeError("Failed to start Android local API server")

    return AndroidServerHandle(runtime=runtime, adapter=adapter, loop=loop, thread=thread)
