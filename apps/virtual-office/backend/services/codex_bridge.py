import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.services.settings_store import get_settings


class CodexBridge:
    def __init__(self, adapter_script_path: str | None = None) -> None:
        settings = get_settings()
        self.adapter_script_path = adapter_script_path or str(
            settings.get("codex_adapter_path") or "adapters/codex-adapter/main.py"
        )
        self.project_root = Path(__file__).resolve().parents[2]

    def _adapter_path(self) -> Path:
        adapter_path = Path(self.adapter_script_path)
        if adapter_path.is_absolute():
            return adapter_path
        return self.project_root / adapter_path

    def _fake_mode_enabled(self) -> bool:
        return os.getenv("VIRTUAL_OFFICE_FAKE_CODEX", "").strip().lower() in {"1", "true", "yes", "on"}

    def _fake_result(self, prompt: str, workdir: str, timeout: int) -> dict[str, Any]:
        cleaned = prompt.strip()
        output = "FAKE_OK"
        lower_cleaned = cleaned.lower()
        marker = "exactly "
        if marker in lower_cleaned:
            start = lower_cleaned.index(marker) + len(marker)
            tail = cleaned[start:].strip()
            for stopper in [" and nothing else", ".", "\n"]:
                idx = tail.lower().find(stopper)
                if idx != -1:
                    tail = tail[:idx]
                    break
            output = tail.strip(" \"'") or output
        elif cleaned:
            output = cleaned.split()[0]

        now = datetime.now(UTC).isoformat()
        return {
            "output": output,
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
            "tokens_used": 0,
            "workdir": workdir,
            "prompt_preview": cleaned[:120],
            "session_id": f"fake-codex-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            "timeout": timeout,
            "created_at": now,
        }

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params is not None:
            request["params"] = params

        adapter_path = self._adapter_path()
        commands = [
            ["python3", str(adapter_path)],
            [sys.executable, str(adapter_path)],
        ]
        last_error: Exception | None = None

        for command in commands:
            try:
                request_timeout = 30
                if method == "exec":
                    request_timeout = int((params or {}).get("timeout") or 120) + 10
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.project_root,
                )
                stdout, stderr = process.communicate(json.dumps(request) + "\n", timeout=request_timeout)
            except FileNotFoundError as exc:
                last_error = exc
                continue

            if process.returncode != 0:
                last_error = RuntimeError(stderr.strip() or f"Adapter exited with code {process.returncode}")
                continue

            line = stdout.strip()
            if not line:
                last_error = RuntimeError("Adapter returned an empty response")
                continue

            response = json.loads(line)
            if "error" in response:
                message = response["error"].get("message", "Unknown adapter error")
                raise RuntimeError(message)

            return response.get("result")

        if last_error is not None:
            raise RuntimeError(f"Unable to start Codex adapter: {last_error}") from last_error
        raise RuntimeError("Unable to start Codex adapter")

    def ping(self) -> str:
        if self._fake_mode_enabled():
            return "pong"
        result = self._call("ping")
        return str(result)

    def status(self) -> dict[str, Any]:
        if self._fake_mode_enabled():
            return {
                "name": "codex",
                "status": "online",
                "version": "fake",
                "model": "fake-codex",
            }
        result = self._call("status")
        return result if isinstance(result, dict) else {}

    def exec(self, prompt: str, workdir: str | None = None, timeout: int = 120) -> dict[str, Any]:
        settings = get_settings()
        resolved_workdir = workdir or str(settings.get("codex_workdir") or r"D:\Codex")
        if self._fake_mode_enabled():
            return self._fake_result(prompt=prompt, workdir=resolved_workdir, timeout=timeout)
        result = self._call(
            "exec",
            {
                "prompt": prompt,
                "workdir": resolved_workdir,
                "timeout": timeout,
            },
        )
        return result if isinstance(result, dict) else {}

    def session_last(self) -> dict[str, Any] | None:
        if self._fake_mode_enabled():
            return {
                "id": "fake-codex-session",
                "created_at": datetime.now(UTC).isoformat(),
                "summary": "Fake Codex session for automated E2E verification",
            }
        result = self._call("session.last")
        return result if isinstance(result, dict) else None
