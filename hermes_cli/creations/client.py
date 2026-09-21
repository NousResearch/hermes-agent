from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from urllib.parse import urlsplit, urlunsplit
from typing import Any


_PATHS = {"/api/workflow-runtime", "/api/workflows", "/api/workflow-plans", "/api/workflow-runs"}


def _loopback_url(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Anvil URL must be an HTTP loopback URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Anvil URL cannot contain a path or query")
    host = parsed.hostname
    if not host:
        raise ValueError("Anvil URL must specify a host")
    try:
        addr = ipaddress.ip_address(host)
        allowed = addr.is_loopback
    except ValueError:
        allowed = host.lower() in {"localhost"}
        if allowed:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            allowed = bool(infos) and all(ipaddress.ip_address(item[4][0]).is_loopback for item in infos)
    if not allowed:
        raise ValueError("Anvil URL must resolve to loopback")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class AnvilClient:
    def __init__(self, base_url: str):
        self.base_url = _loopback_url(base_url).rstrip("/")
        self._client = None

    async def request(self, method: str, path: str, *, json: dict | None = None) -> dict:
        path_only = path.split("?", 1)[0]
        if path_only not in _PATHS and not path_only.startswith("/api/workflow-runs/"):
            raise ValueError("unsupported Anvil endpoint")
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(15.0, connect=2.0), follow_redirects=False, limits=httpx.Limits(max_connections=8))
        response = await self._client.request(method, path, json=json)
        if response.status_code in (301, 302, 303, 307, 308):
            raise RuntimeError("Anvil redirect rejected")
        if response.status_code >= 400:
            try: detail = response.json()
            except ValueError: detail = {"message": response.text[:1000]}
            raise RuntimeError(f"anvil_http_{response.status_code}: {detail}")
        if len(response.content) > 8 * 1024 * 1024:
            raise RuntimeError("Anvil response exceeded 8 MiB")
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("Anvil response must be an object")
        return value

    async def status(self) -> dict:
        try:
            runtime = await self.request("GET", "/api/workflow-runtime")
            return {"state": "ready" if runtime.get("protocolVersion") == 1 else "incompatible", "protocolVersion": runtime.get("protocolVersion"), "capabilities": runtime.get("capabilities", {}), "imageModels": runtime.get("imageModels", []), "videoModels": runtime.get("videoModels", []), "anvilUrl": self.base_url, "message": None if runtime.get("protocolVersion") == 1 else "unsupported workflow runtime"}
        except Exception as exc:
            return {"state": "offline", "protocolVersion": None, "capabilities": {}, "imageModels": [], "videoModels": [], "anvilUrl": self.base_url, "message": str(exc)}

    async def upload_asset(self, data: bytes, content_type: str) -> str:
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(15.0, connect=2.0), follow_redirects=False)
        response = await self._client.post("/api/upload-asset", content=data, headers={"content-type": content_type})
        if response.status_code in (301, 302, 303, 307, 308):
            raise RuntimeError("Anvil redirect rejected")
        if response.status_code >= 400:
            raise RuntimeError(f"anvil_http_{response.status_code}: upload failed")
        payload = response.json()
        url = payload.get("cdnUrl")
        if not isinstance(url, str) or not url:
            raise RuntimeError("Anvil upload response did not contain cdnUrl")
        return url

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
