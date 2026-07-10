"""Small client adapter that lets Kanban worker tools use a coordinator."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from hermes_cli import kanban_db as local


class RemoteConnection:
    def close(self) -> None:
        return None


class RemoteKanban:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"coordinator {exc.code}: {detail}") from exc

    def _detail(self, task_id: str) -> dict:
        return self._request("GET", f"/v1/tasks/{task_id}")

    def get_task(self, _conn: RemoteConnection, task_id: str) -> Optional[local.Task]:
        try:
            payload = dict(self._detail(task_id)["task"])
            payload.pop("required_capabilities", None)
            return local.Task(**payload)
        except RuntimeError as exc:
            if "coordinator 404:" in str(exc):
                return None
            raise

    def list_comments(self, _conn: RemoteConnection, task_id: str) -> list[local.Comment]:
        return [local.Comment(**item) for item in self._detail(task_id)["comments"]]

    def list_events(self, _conn: RemoteConnection, task_id: str) -> list[local.Event]:
        return [local.Event(**item) for item in self._detail(task_id)["events"]]

    def list_runs(self, _conn: RemoteConnection, task_id: str) -> list[local.Run]:
        return [local.Run(**item) for item in self._detail(task_id)["runs"]]

    def latest_run(self, conn: RemoteConnection, task_id: str) -> Optional[local.Run]:
        runs = self.list_runs(conn, task_id)
        return runs[-1] if runs else None

    def parent_ids(self, _conn: RemoteConnection, task_id: str) -> list[str]:
        return self._detail(task_id)["parents"]

    def child_ids(self, _conn: RemoteConnection, task_id: str) -> list[str]:
        return self._detail(task_id)["children"]

    def build_worker_context(self, _conn: RemoteConnection, task_id: str) -> str:
        return self._detail(task_id)["worker_context"]

    def register_machine(
        self,
        machine_id: str,
        *,
        hostname: str,
        profiles: list[str],
        capabilities: list[str],
    ) -> str:
        return str(self._request("POST", "/v1/machines/register", {
            "machine_id": machine_id,
            "hostname": hostname,
            "profiles": profiles,
            "capabilities": capabilities,
        })["machine_id"])

    def claim_next(self, machine_id: str) -> Optional[local.Task]:
        payload = self._request("POST", "/v1/tasks/claim-next", {"machine_id": machine_id})
        raw = payload.get("task")
        if raw is None:
            return None
        raw = dict(raw)
        raw.pop("required_capabilities", None)
        return local.Task(**raw)

    def list_machines(self) -> dict:
        return self._request("GET", "/v1/machines")

    def heartbeat_claim(
        self, _conn: RemoteConnection, task_id: str, *, claimer: Optional[str] = None,
    ) -> bool:
        lock = claimer or os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
        return bool(self._request("POST", f"/v1/tasks/{task_id}/renew", {"claim_lock": lock})["renewed"])

    def heartbeat_worker(self, conn: RemoteConnection, task_id: str, **_kwargs: Any) -> bool:
        return self.heartbeat_claim(conn, task_id)

    def record_worker_started(self, task_id: str, *, claim_lock: str, worker_pid: int) -> bool:
        return bool(self._request("POST", f"/v1/tasks/{task_id}/worker-started", {
            "claim_lock": claim_lock,
            "worker_pid": worker_pid,
        })["recorded"])

    def complete_task(self, _conn: RemoteConnection, task_id: str, *, result: Optional[str] = None, **kwargs: Any) -> bool:
        return bool(self._request("POST", f"/v1/tasks/{task_id}/complete", {
            "result": result,
            "claim_lock": str(
                kwargs.get("claim_lock")
                or kwargs.get("claimer")
                or os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
            ),
        })["completed"])

    def block_task(self, _conn: RemoteConnection, task_id: str, *, reason: Optional[str] = None, kind: Optional[str] = None, **kwargs: Any) -> bool:
        return bool(self._request("POST", f"/v1/tasks/{task_id}/block", {
            "reason": reason,
            "kind": kind,
            "claim_lock": str(
                kwargs.get("claim_lock")
                or kwargs.get("claimer")
                or os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
            ),
        })["blocked"])

    def add_comment(self, _conn: RemoteConnection, task_id: str, *, author: str, body: str) -> int:
        return int(self._request("POST", f"/v1/tasks/{task_id}/comments", {"author": author, "body": body})["comment_id"])


def configured_coordinator_url() -> str:
    """Return the configured coordinator endpoint, if this machine uses one.

    The endpoint is ordinary deployment configuration, so ``config.yaml`` is
    authoritative. The environment fallback is retained only so existing
    installations can upgrade without an outage; it is intentionally not
    documented as the preferred setup.
    """
    try:
        from hermes_cli.config import load_config

        kanban = (load_config() or {}).get("kanban") or {}
        if isinstance(kanban, dict):
            url = str(kanban.get("coordinator_url") or "").strip()
            if url:
                return url.rstrip("/")
    except Exception:
        # A coordinator must not make a malformed optional config prevent
        # local Kanban from starting. The caller will either use the legacy
        # value below or report that no coordinator is configured.
        pass
    return os.environ.get("HERMES_KANBAN_COORDINATOR_URL", "").strip().rstrip("/")


def connect_from_config() -> tuple[RemoteKanban, RemoteConnection]:
    url = configured_coordinator_url()
    token = os.environ.get("HERMES_KANBAN_COORDINATOR_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError(
            "kanban.coordinator_url and HERMES_KANBAN_COORDINATOR_TOKEN are required"
        )
    return RemoteKanban(url, token), RemoteConnection()


# Compatibility for private deployments that set the endpoint in an
# EnvironmentFile before coordinator_url was added to config.yaml.
connect_from_env = connect_from_config
