import atexit
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


class RustResponseStore:
    """Synchronous Python wrapper over the Rust sidecar response store."""

    def __init__(self, max_size: int = 100, db_path: str | None = None):
        self._lock = threading.Lock()
        self._closed = False
        self._proc = self._start_process(max_size=max_size, db_path=db_path)
        atexit.register(self.close)
        self._request("health", {})

    def _start_process(self, max_size: int, db_path: str | None) -> subprocess.Popen[str]:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "rust" / "Cargo.toml"
        compiled_bin = root / "rust" / "target" / "debug" / "hermes-sidecar"
        override_bin = os.getenv("HERMES_RUST_SIDECAR_BIN", "").strip()

        if override_bin:
            command = [override_bin, "--serve"]
        elif compiled_bin.exists():
            command = [str(compiled_bin), "--serve"]
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

        db_path = db_path or os.getenv("HERMES_RUST_RESPONSE_STORE_DB_PATH", "").strip()
        if db_path:
            command.extend(["--db-path", db_path])

        command.extend(["--max-size", str(max_size)])

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
        if self._closed:
            raise RuntimeError("RustResponseStore is already closed")

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
            raise RuntimeError(f"Rust sidecar exited unexpectedly: {stderr_text}")

        response = json.loads(raw)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RuntimeError(error.get("message", "unknown Rust sidecar error"))
        return response["result"]

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        result = self._request("store.get", {"response_id": response_id})
        return result.get("value")

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        self._request("store.put", {"response_id": response_id, "data": data})

    def delete(self, response_id: str) -> bool:
        result = self._request("store.delete", {"response_id": response_id})
        return bool(result.get("deleted"))

    def get_conversation(self, name: str) -> Optional[str]:
        result = self._request("conversation.get", {"name": name})
        return result.get("value")

    def set_conversation(self, name: str, response_id: str) -> None:
        self._request("conversation.set", {"name": name, "response_id": response_id})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)

    def __len__(self) -> int:
        result = self._request("store.len", {})
        return int(result.get("value", 0))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
