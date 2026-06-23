import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = APP_ROOT / "frontend"


def _wait_for_http(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - retry loop
            last_error = exc
            time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {url}: {last_error}")


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def test_virtual_office_browser_e2e(tmp_path):
    backend_port = _reserve_port()
    frontend_port = _reserve_port()
    backend_url = f"http://127.0.0.1:{backend_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"

    data_root = tmp_path / "data"
    env = {
        **os.environ,
        "VIRTUAL_OFFICE_FAKE_CODEX": "1",
        "VIRTUAL_OFFICE_DATA_ROOT": str(data_root),
        "CI": "1",
    }

    backend = subprocess.Popen(
        [
            "bash",
            "-lc",
            f"source .venv/bin/activate && python -m uvicorn backend.main:app --host 127.0.0.1 --port {backend_port}",
        ],
        cwd=APP_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    frontend = None
    try:
        _wait_for_http(f"{backend_url}/api/health")

        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(frontend_port)],
            cwd=FRONTEND_ROOT,
            env={**env, "VITE_API_BASE": backend_url},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_http(frontend_url)

        result = subprocess.run(
            ["npm", "run", "test:e2e"],
            cwd=FRONTEND_ROOT,
            env={**env, "PLAYWRIGHT_BASE_URL": frontend_url},
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )

        assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    finally:
        for proc in [frontend, backend]:
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
