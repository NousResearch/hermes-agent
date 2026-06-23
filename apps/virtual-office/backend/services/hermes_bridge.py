import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.services.settings_store import get_settings


class HermesBridge:
    def __init__(self, adapter_script_path: str | None = None) -> None:
        settings = get_settings()
        self.adapter_script_path = adapter_script_path or str(
            settings.get("hermes_adapter_path") or "adapters/hermes-adapter/main.py"
        )
        self.project_root = Path(__file__).resolve().parents[2]

    def _adapter_path(self) -> Path:
        adapter_path = Path(self.adapter_script_path)
        if adapter_path.is_absolute():
            return adapter_path
        return self.project_root / adapter_path

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params is not None:
            request["params"] = params

        adapter_path = self._adapter_path()
        commands = [["python3", str(adapter_path)], [sys.executable, str(adapter_path)]]
        last_error: Exception | None = None

        for command in commands:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.project_root,
                )
                stdout, stderr = process.communicate(json.dumps(request) + "\n", timeout=10)
            except FileNotFoundError as exc:
                last_error = exc
                continue

            if process.returncode != 0:
                raise RuntimeError(stderr.strip() or f"Adapter exited with code {process.returncode}")

            line = stdout.strip()
            if not line:
                raise RuntimeError("Adapter returned an empty response")

            response = json.loads(line)
            if "error" in response:
                error = response["error"]
                message = error.get("message", "Unknown adapter error")
                if message == "Session not found":
                    raise FileNotFoundError(message)
                raise RuntimeError(message)

            return response.get("result")

        if last_error is not None:
            raise RuntimeError(f"Unable to start Hermes adapter: {last_error}") from last_error
        raise RuntimeError("Unable to start Hermes adapter")

    def list_sessions(self) -> list[dict]:
        result = self._call("session.list")
        return result if isinstance(result, list) else []

    def resume_session(self, session_id: str) -> dict:
        result = self._call("session.resume", {"session_id": session_id})
        return result if isinstance(result, dict) else {}

    def ping(self) -> str:
        result = self._call("ping")
        return str(result)
