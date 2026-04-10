from __future__ import annotations

import json
from typing import Any

from hermes_android.server import AndroidServerHandle, start_local_api_server

_ACTIVE_HANDLE: AndroidServerHandle | None = None


def _status_payload(handle: AndroidServerHandle | None) -> dict[str, Any]:
    if handle is None:
        return {"started": False}
    return {
        "started": True,
        "base_url": handle.base_url,
        "api_server_host": handle.runtime.api_server_host,
        "api_server_port": handle.runtime.api_server_port,
        "api_server_key": handle.runtime.api_server_key,
        "api_server_model_name": handle.runtime.api_server_model_name,
        "hermes_home": str(handle.runtime.hermes_home),
    }


def ensure_server(files_dir: str) -> str:
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is None:
        _ACTIVE_HANDLE = start_local_api_server(files_dir)
    return json.dumps(_status_payload(_ACTIVE_HANDLE), sort_keys=True)


def current_server_status() -> str:
    return json.dumps(_status_payload(_ACTIVE_HANDLE), sort_keys=True)


def stop_server() -> str:
    global _ACTIVE_HANDLE
    if _ACTIVE_HANDLE is not None:
        _ACTIVE_HANDLE.stop()
        _ACTIVE_HANDLE = None
    return json.dumps({"started": False}, sort_keys=True)
