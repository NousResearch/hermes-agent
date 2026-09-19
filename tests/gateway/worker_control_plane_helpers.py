"""Test-only in-process client for the isolated Worker Control Plane."""

from __future__ import annotations

import uuid


class MockWorkerClient:
    def __init__(self, client, bootstrap_secret: str, *, worker_id: str = "server-a-worker"):
        self.client = client
        self.bootstrap_secret = bootstrap_secret
        self.worker_id = worker_id
        self.instance_id = str(uuid.uuid4())
        self.registration_id = None
        self.access_token = None

    async def register(self, *, capabilities=None, protocol_version="1.0"):
        response = await self.client.post(
            "/worker/v1/register",
            headers={"Authorization": f"Worker-Bootstrap {self.bootstrap_secret}"},
            json={"protocol_version": protocol_version, "worker_id": self.worker_id,
                  "instance_id": self.instance_id, "worker_name": "test worker",
                  "worker_version": "0.1.0", "capabilities": capabilities or ["system.echo"]},
        )
        body = await response.json()
        if response.status in (200, 201):
            self.registration_id = body["registration_id"]
            self.access_token = body["access_token"]
        return response.status, body

    def headers(self, key=None):
        data = {"Authorization": f"Bearer {self.access_token}"}
        if key:
            data["Idempotency-Key"] = key
        return data

    def base(self):
        return {"worker_id": self.worker_id, "instance_id": self.instance_id,
                "registration_id": self.registration_id}

    async def heartbeat(self, status="idle", current_task_id=None):
        data = self.base() | {"status": status, "current_task_id": current_task_id,
                              "worker_time": "2026-01-01T00:00:00Z"}
        response = await self.client.post("/worker/v1/heartbeat", headers=self.headers(), json=data)
        return response.status, await response.json()

    async def poll(self, key="poll-1"):
        data = self.base() | {"capabilities": ["system.echo"], "max_tasks": 1, "wait_seconds": 0}
        response = await self.client.post("/worker/v1/tasks/poll", headers=self.headers(key), json=data)
        return response.status, (await response.json() if response.status != 204 else None)

    async def ack(self, task, accepted=True, reason=None, key="ack-1"):
        data = self.base() | {"delivery_id": task["delivery_id"], "accepted": accepted,
                              "reason": reason, "worker_time": "2026-01-01T00:00:00Z"}
        response = await self.client.post(f'/worker/v1/tasks/{task["task_id"]}/ack', headers=self.headers(key), json=data)
        return response.status, await response.json()

    async def result(self, task, *, stdout=None, result_key="result-1", request_key="request-result-1"):
        message = task["payload"]["message"] if stdout is None else stdout
        data = self.base() | {"task_id": task["task_id"], "delivery_id": task["delivery_id"],
            "task_type": "system.echo", "status": "completed", "stdout": message, "stderr": "",
            "exit_code": 0, "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:00Z",
            "duration_ms": 0, "result_idempotency_key": result_key, "payload_hash": task["payload_hash"],
            "trace_id": task["trace_id"]}
        response = await self.client.post(f'/worker/v1/tasks/{task["task_id"]}/result', headers=self.headers(request_key), json=data)
        return response.status, await response.json()
