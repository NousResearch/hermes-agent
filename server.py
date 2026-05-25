import asyncio
import os
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn


class GatewayManager:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.logs = deque(maxlen=500)
        self.restart_count = 0
        self.last_start_time: Optional[float] = None
        self.last_exit_code: Optional[int] = None
        self.stop_requested = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.log_thread: Optional[threading.Thread] = None

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} {message}"
        self.logs.append(line)
        print(line, flush=True)

    def _start_process(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self._log("[gateway-manager] Gateway already running")
                return

            env = os.environ.copy()
            self._log("[gateway-manager] Starting Hermes gateway subprocess...")
            self.process = subprocess.Popen(
                ["hermes", "gateway", "run"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.last_start_time = time.time()
            self.last_exit_code = None

            self.log_thread = threading.Thread(
                target=self._stream_logs,
                name="gateway-log-thread",
                daemon=True,
            )
            self.log_thread.start()

    def _stream_logs(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return

        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                self.logs.append(line)
                print(f"[gateway] {line}", flush=True)
        except Exception as exc:
            self._log(f"[gateway-manager] Log streaming error: {exc}")

    def _monitor_loop(self) -> None:
        self._log("[gateway-manager] Monitor loop started")
        while not self.stop_requested:
            with self.lock:
                proc = self.process

            if proc is None:
                self._start_process()
                time.sleep(2)
                continue

            exit_code = proc.poll()
            if exit_code is not None:
                self.last_exit_code = exit_code
                self._log(f"[gateway-manager] Gateway exited with code {exit_code}")
                if not self.stop_requested:
                    self.restart_count += 1
                    self._log("[gateway-manager] Restarting gateway in 2 seconds...")
                    time.sleep(2)
                    self._start_process()
                    continue

            time.sleep(2)

        self._log("[gateway-manager] Monitor loop stopped")

    def start(self) -> None:
        if self.monitor_thread and self.monitor_thread.is_alive():
            self._log("[gateway-manager] Monitor thread already running")
            return

        self.stop_requested = False
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="gateway-monitor-thread",
            daemon=True,
        )
        self.monitor_thread.start()

    def stop(self) -> None:
        self._log("[gateway-manager] Stop requested")
        self.stop_requested = True

        with self.lock:
            proc = self.process

        if proc is not None and proc.poll() is None:
            try:
                self._log("[gateway-manager] Sending SIGTERM to gateway")
                proc.terminate()
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._log("[gateway-manager] Gateway did not stop in time; killing")
                proc.kill()
            except Exception as exc:
                self._log(f"[gateway-manager] Error during shutdown: {exc}")

    def status(self) -> dict:
        with self.lock:
            proc = self.process

        running = proc is not None and proc.poll() is None
        pid = proc.pid if proc is not None and running else None

        uptime_seconds = None
        if running and self.last_start_time is not None:
            uptime_seconds = int(time.time() - self.last_start_time)

        return {
            "running": running,
            "pid": pid,
            "restart_count": self.restart_count,
            "last_exit_code": self.last_exit_code,
            "uptime_seconds": uptime_seconds,
        }

    def recent_logs(self, limit: int = 100) -> list[str]:
        if limit <= 0:
            return []
        return list(self.logs)[-limit:]


gateway_manager = GatewayManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] FastAPI lifespan startup", flush=True)
    gateway_manager.start()
    yield
    print("[server] FastAPI lifespan shutdown", flush=True)
    gateway_manager.stop()


app = FastAPI(title="Hermes Railway Supervisor", lifespan=lifespan)


@app.get("/health")
async def health():
    """
    Liveness endpoint for Railway.
    This should return 200 as long as the HTTP server itself is alive.
    """
    return {"status": "ok"}


@app.get("/status")
async def status():
    """
    Basic runtime status for the gateway supervisor.
    """
    return gateway_manager.status()


@app.get("/logs")
async def logs(limit: int = 100):
    """
    Return recent gateway/supervisor logs.
    """
    limit = max(1, min(limit, 500))
    return JSONResponse({"logs": gateway_manager.recent_logs(limit)})


@app.post("/restart")
async def restart():
    """
    Optional manual restart endpoint for the gateway.
    """
    gateway_manager.stop()
    await asyncio.sleep(1)
    gateway_manager.start()
    return {"status": "restarting"}


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    print(f"[server] Starting HTTP server on 0.0.0.0:{port}", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
