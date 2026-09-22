"""Typed stdlib HTTP client for the federated fleet coordinator."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement, encode_message
from hermes_cli.fleet_store import TaskClaim, TaskRecord
from hermes_cli.urllib_security import open_credentialed_url


class FleetTransportError(RuntimeError):
    """Raised for an unavailable or rejected coordinator request."""


class FleetClient:
    def __init__(self, base_url: str, *, token: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                **({"Content-Type": "application/json"} if payload is not None else {}),
            },
        )
        try:
            with open_credentialed_url(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise FleetTransportError(f"{exc.code}: coordinator rejected request") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise FleetTransportError(f"coordinator request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise FleetTransportError("coordinator returned a non-object response")
        return decoded

    @staticmethod
    def _message(value) -> dict[str, Any]:
        return json.loads(encode_message(value).decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/fleet/health")

    def register_runner(self, capability: RunnerCapability, *, now: float | None = None, ttl: float = 30.0) -> bool:
        return bool(
            self._request(
                "POST",
                "/v1/fleet/runners/register",
                {"message": self._message(capability), "now": now, "ttl": ttl},
            ).get("ok")
        )

    def heartbeat_runner(self, capability: RunnerCapability, *, now: float | None = None, ttl: float = 30.0) -> bool:
        return bool(
            self._request(
                "POST",
                f"/v1/fleet/runners/{capability.node_id}/{capability.profile}/heartbeat",
                {"now": now, "ttl": ttl},
            ).get("ok")
        )

    def submit_task(self, task: FleetTask, *, now: float | None = None) -> TaskRecord:
        return _record_from_dict(self._request(
            "POST", "/v1/fleet/tasks", {"message": self._message(task), "now": now}
        ))

    def claim_task(
        self,
        capability: RunnerCapability,
        *,
        now: float | None = None,
        lease_seconds: float = 60.0,
    ) -> TaskClaim | None:
        payload = self._request(
            "POST",
            "/v1/fleet/tasks/claim",
            {
                "capability": capability.to_dict(),
                "now": now,
                "lease_seconds": lease_seconds,
            },
        )
        claim = payload.get("claim")
        return TaskClaim(**claim) if isinstance(claim, dict) else None

    def renew_claim(
        self, task_id: str, claim_id: str, *, now: float | None = None, lease_seconds: float = 60.0
    ) -> TaskClaim | None:
        payload = self._request(
            "POST",
            f"/v1/fleet/tasks/{task_id}/renew",
            {"claim_id": claim_id, "now": now, "lease_seconds": lease_seconds},
        )
        claim = payload.get("claim")
        return TaskClaim(**claim) if isinstance(claim, dict) else None

    def complete_task(self, task_id: str, claim_id: str, *, result: str = "", now: float | None = None) -> bool:
        return bool(
            self._request(
                "POST",
                f"/v1/fleet/tasks/{task_id}/complete",
                {"claim_id": claim_id, "result": result, "now": now},
            ).get("ok")
        )

    def fail_task(self, task_id: str, claim_id: str, *, error: str, now: float | None = None) -> bool:
        return bool(
            self._request(
                "POST",
                f"/v1/fleet/tasks/{task_id}/fail",
                {"claim_id": claim_id, "error": error, "now": now},
            ).get("ok")
        )

    def retry_task(self, task_id: str, *, now: float | None = None) -> bool:
        return bool(self._request("POST", f"/v1/fleet/tasks/{task_id}/retry", {"now": now}).get("ok"))

    def cancel_task(self, task_id: str, *, now: float | None = None) -> bool:
        return bool(self._request("POST", f"/v1/fleet/tasks/{task_id}/cancel", {"now": now}).get("ok"))

    def list_tasks(self) -> list[TaskRecord]:
        return [_record_from_dict(item) for item in self._request("GET", "/v1/fleet/tasks").get("tasks", [])]


def _record_from_dict(value: dict[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=value["task_id"],
        title=value["title"],
        body=value["body"],
        requirement=TaskRequirement.from_dict(value["requirement"]),
        idempotency_key=value["idempotency_key"],
        status=value["status"],
        claim_id=value.get("claim_id"),
        node_id=value.get("node_id"),
        runner_profile=value.get("runner_profile"),
        lease_expires_at=value.get("lease_expires_at"),
        attempt=value["attempt"],
        result=value.get("result"),
        error=value.get("error"),
        created_at=value["created_at"],
        updated_at=value["updated_at"],
    )
