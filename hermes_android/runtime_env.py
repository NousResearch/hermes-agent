from __future__ import annotations

import os
import secrets
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AndroidRuntimeEnv:
    files_dir: Path
    hermes_home: Path
    api_server_host: str
    api_server_port: int
    api_server_key: str
    api_server_model_name: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def prepare_runtime_env(
    files_dir: str | Path,
    *,
    api_server_host: str = "127.0.0.1",
    api_server_port: int | None = None,
    api_server_key: str | None = None,
    api_server_model_name: str = "hermes-agent-android",
) -> AndroidRuntimeEnv:
    files_path = Path(files_dir).expanduser().resolve()
    hermes_home = files_path / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "sessions", "skills", "downloads", "workspace"):
        (hermes_home / child).mkdir(parents=True, exist_ok=True)

    port = api_server_port or _find_free_port(api_server_host)
    key = api_server_key or secrets.token_urlsafe(32)

    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["HERMES_ANDROID_BOOTSTRAP"] = "1"
    os.environ["API_SERVER_HOST"] = api_server_host
    os.environ["API_SERVER_PORT"] = str(port)
    os.environ["API_SERVER_KEY"] = key
    os.environ["API_SERVER_MODEL_NAME"] = api_server_model_name

    return AndroidRuntimeEnv(
        files_dir=files_path,
        hermes_home=hermes_home,
        api_server_host=api_server_host,
        api_server_port=port,
        api_server_key=key,
        api_server_model_name=api_server_model_name,
    )
