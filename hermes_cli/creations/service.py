from __future__ import annotations

import uuid
import io
from pathlib import Path
from typing import Any

from .client import AnvilClient
from .contracts import CreationsConfig, validate_plan_body, validate_request_id
from .store import CreationsStore
from .media import stream_copy


class CreationsService:
    def __init__(self, home: Path, config: CreationsConfig, client: Any | None = None):
        self.home, self.config = Path(home), config
        self.client = client or AnvilClient(config.anvil_url)
        self.store = CreationsStore(self.home)

    async def status(self) -> dict:
        if not self.config.enabled: return {"state": "disabled", "protocolVersion": None, "capabilities": {}, "imageModels": [], "videoModels": [], "anvilUrl": self.config.anvil_url, "message": "creations disabled"}
        return await self.client.status()

    async def workflows(self) -> dict:
        if not self.config.enabled:
            return {"spaces": [], "readiness": await self.status()}
        catalog = await self.client.request("GET", "/api/workflows")
        readiness = await self.status()
        return {"spaces": catalog.get("spaces", []), "readiness": readiness}

    async def plan(self, body: dict) -> dict:
        if not self.config.enabled:
            raise RuntimeError("creations_disabled")
        body = validate_plan_body(body)
        result = await self.client.request("POST", "/api/workflow-plans", json=body)
        self.store.save_plan(result)
        return result

    async def submit(self, body: dict) -> dict:
        if not self.config.enabled:
            raise RuntimeError("creations_disabled")
        request_id = validate_request_id(body.get("requestId"))
        plan_id = body.get("planId")
        if not self.store.owns_plan(plan_id): raise KeyError("plan_not_found")
        self.store.reserve_request(request_id, "run", body)
        try:
            result = await self.client.request("POST", "/api/workflow-runs", json=body)
        except (TimeoutError, OSError) as exc:
            return {"requestId": request_id, "state": "reconciling", "message": str(exc)}
        self.store.attach_run(request_id, result)
        return result

    async def runs(self) -> dict:
        local = self.store.list_runs()
        if self.config.enabled:
            try:
                upstream = await self.client.request("GET", "/api/workflow-runs")
                known = {r.get("runId") for r in local}
                for run in upstream.get("runs", []):
                    if run.get("runId") not in known:
                        local.append(run)
            except Exception:
                pass
        return {"runs": local, "pendingRequests": self.store.pending_requests()}

    def library(self, kind: str | None = None, cursor: str | None = None) -> dict:
        return self.store.list_library(kind, cursor)

    def library_item(self, item_id: int) -> dict | None:
        return self.store.library_item(item_id)

    async def upload(self, data: bytes, content_type: str) -> dict:
        if not data or len(data) > 100 * 1024 * 1024:
            raise ValueError("upload_too_large_or_empty")
        # Validate before forwarding to Anvil, and keep only an owned URL.
        from .media import validate_upload
        mime = validate_upload(content_type, data[:64])
        url = await self.client.upload_asset(data, mime)
        self.store.register_asset(url)
        return {"assetUrl": url}

    async def reuse(self, item_id: int) -> dict:
        item = self.store.library_item(item_id)
        if not item or item.get("storageState") != "local" or not item.get("path"):
            raise KeyError("library_not_found")
        path = Path(item["path"])
        if not path.is_file():
            raise KeyError("library_not_found")
        return await self.upload(path.read_bytes(), item["mimeType"])

    async def copy_retry(self, item_id: int) -> dict:
        item = self.store.library_item(item_id)
        if not item:
            raise KeyError("library_not_found")
        # This endpoint deliberately performs no generation. The runtime's result URL is
        # copied by the normal reconciliation worker when available.
        if not item.get("sourceUrl"):
            raise ValueError("copy_source_missing")
        return item

    async def run(self, run_id: str) -> dict | None:
        return self.store.get_run(run_id)

    async def retry(self, run_id: str, body: dict) -> dict:
        if not self.config.enabled:
            raise RuntimeError("creations_disabled")
        if not self.store.get_run(run_id): raise KeyError("run_not_found")
        request_id = validate_request_id(body.get("requestId"))
        self.store.reserve_request(request_id, "retry", body)
        try:
            result = await self.client.request("POST", f"/api/workflow-runs/{run_id}/retry", json=body)
        except (TimeoutError, OSError) as exc:
            return {"requestId": request_id, "runId": run_id, "state": "reconciling", "message": str(exc)}
        self.store.attach_run(request_id, result)
        return result

    async def reconcile(self) -> None:
        for request in self.store.pending_requests():
            try:
                found = await self.client.request("GET", f"/api/workflow-runs?requestId={request['request_id']}", json=None)
                for run in found.get("runs", []):
                    if run.get("requestId") == request["request_id"]: self.store.attach_run(request["request_id"], run)
            except Exception:
                continue

    async def close(self) -> None:
        self.store.close(); await self.client.close()
