import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List


class RustContextCompressorClient:
    """Thin synchronous client for Rust compression planning/application."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = self._start_process()

    def _start_process(self) -> subprocess.Popen[str]:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "rust" / "Cargo.toml"
        override_bin = os.getenv("HERMES_RUST_SIDECAR_BIN", "").strip()

        if override_bin:
            command = [override_bin, "--serve"]
        else:
            command = [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(manifest_path),
                "-p",
                "hermes-sidecar",
                "--",
                "--serve",
            ]

        return subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        payload = {
            "version": "0.1",
            "id": f"req_{uuid.uuid4().hex}",
            "method": method,
            "params": params,
        }

        with self._lock:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            raw = self._proc.stdout.readline()

        if not raw:
            stderr_text = ""
            if self._proc.stderr is not None:
                stderr_text = self._proc.stderr.read().strip()
            raise RuntimeError(f"Rust compression sidecar exited unexpectedly: {stderr_text}")

        response = json.loads(raw)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RuntimeError(error.get("message", "unknown Rust compression error"))
        return response["result"]

    def plan(self, messages: List[Dict[str, Any]], protect_first_n: int, protect_last_n: int) -> Dict[str, Any]:
        return self._request(
            "compression.plan",
            {
                "messages": messages,
                "protect_first_n": protect_first_n,
                "protect_last_n": protect_last_n,
            },
        )

    def apply(
        self,
        messages: List[Dict[str, Any]],
        compress_start: int,
        compress_end: int,
        summary: str | None,
        compression_count: int,
    ) -> List[Dict[str, Any]]:
        result = self._request(
            "compression.apply",
            {
                "messages": messages,
                "compress_start": compress_start,
                "compress_end": compress_end,
                "summary": summary,
                "compression_count": compression_count,
            },
        )
        return result["messages"]

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
