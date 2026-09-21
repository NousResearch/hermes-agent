"""P10's read-only ComfyUI REST preflight contract.

This module deliberately cannot submit `/prompt` jobs. It verifies the local
REST schema required by a future, separately approved generation lane without
changing provider configuration or GPU state.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REQUIRED_NODES = frozenset({
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "KSampler",
    "SaveImage",
})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
JsonReader = Callable[[str], object]


class ComfyUIContractError(RuntimeError):
    """The local ComfyUI endpoint does not meet the P10 read-only contract."""


@dataclass(frozen=True)
class ComfyUIReadOnlyStatus:
    endpoint: str
    comfyui_version: str
    device_count: int
    queue_idle: bool


class ComfyUIReadOnlyProbe:
    """Read-only verifier for the local ComfyUI REST surface.

    `reader` is injected for deterministic tests. The class exposes no write
    method and never sends a generation request.
    """

    def __init__(self, endpoint: str, reader: JsonReader) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK_HOSTS:
            raise ComfyUIContractError("P10 ComfyUI endpoint must be loopback-only")
        if parsed.params or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ComfyUIContractError("P10 ComfyUI endpoint must not include a path or query")
        self._endpoint = endpoint.rstrip("/")
        self._reader = reader

    def _read_mapping(self, path: str, label: str) -> Mapping[str, Any]:
        try:
            payload = self._reader(f"{self._endpoint}{path}")
        except Exception as exc:  # reader transports network failures as exceptions
            raise ComfyUIContractError(f"unable to read {label}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ComfyUIContractError(f"{label} response must be a JSON object")
        return payload

    def probe(self) -> ComfyUIReadOnlyStatus:
        system_stats = self._read_mapping("/system_stats", "system_stats")
        system = system_stats.get("system")
        devices = system_stats.get("devices")
        if not isinstance(system, Mapping):
            raise ComfyUIContractError("system_stats.system must be a JSON object")
        version = system.get("comfyui_version")
        if not isinstance(version, str) or not version:
            raise ComfyUIContractError("system_stats must include a ComfyUI version")
        if not isinstance(devices, list):
            raise ComfyUIContractError("system_stats.devices must be a list")

        object_info = self._read_mapping("/object_info", "object_info")
        missing = sorted(REQUIRED_NODES.difference(object_info))
        if missing:
            raise ComfyUIContractError(f"missing required ComfyUI nodes: {', '.join(missing)}")

        queue = self._read_mapping("/queue", "queue")
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        if not isinstance(running, list) or not isinstance(pending, list):
            raise ComfyUIContractError("queue must expose running and pending lists")

        return ComfyUIReadOnlyStatus(
            endpoint=self._endpoint,
            comfyui_version=version,
            device_count=len(devices),
            queue_idle=not running and not pending,
        )


def http_json_reader(timeout_seconds: float = 5.0) -> JsonReader:
    """Return a GET-only JSON reader for a loopback ComfyUI endpoint."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    def read(url: str) -> object:
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 -- endpoint is loopback-validated
            if response.status != 200:
                raise ComfyUIContractError(f"unexpected HTTP status {response.status}")
            return json.loads(response.read().decode("utf-8"))

    return read


def probe_local_comfyui(endpoint: str = "http://127.0.0.1:8188") -> ComfyUIReadOnlyStatus:
    """Run the P10 read-only contract against a local ComfyUI instance."""
    return ComfyUIReadOnlyProbe(endpoint, http_json_reader()).probe()
