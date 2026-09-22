"""Small authenticated HTTP coordinator for federated Hermes runners."""

from __future__ import annotations

import hmac
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from hermes_cli.fleet_protocol import (
    MAX_MESSAGE_BYTES,
    FleetTask,
    RunnerCapability,
    decode_message,
)
from hermes_cli.fleet_store import FleetStore


def _claim_dict(claim) -> dict:
    return {
        "task_id": claim.task_id,
        "claim_id": claim.claim_id,
        "node_id": claim.node_id,
        "runner_profile": claim.runner_profile,
        "lease_expires_at": claim.lease_expires_at,
        "attempt": claim.attempt,
    }


class FleetServer:
    def __init__(
        self,
        store: FleetStore,
        *,
        token: str,
        host: str = "127.0.0.1",
        port: int = 0,
        on_task_update=None,
    ):
        if not token:
            raise ValueError("fleet coordinator token must not be empty")
        self.store = store
        self.token = token
        self.on_task_update = on_task_update
        self._http = ThreadingHTTPServer((host, port), _FleetHandler)
        self._http.daemon_threads = True
        self._http.fleet_server = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    def notify_task_update(self, task_id: str, status: str, **fields) -> None:
        if self.on_task_update is None:
            return
        try:
            self.on_task_update(task_id, status, **fields)
        except Exception:
            logging.getLogger(__name__).exception("fleet task mirror failed for %s", task_id)

    @property
    def address(self) -> tuple[str, int]:
        return self._http.server_address[:2]

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._http.serve_forever, name="hermes-fleet", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is None:
            return
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None


class _FleetHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def log_message(self, _format: str, *_args) -> None:
        return

    @property
    def fleet(self) -> FleetServer:
        return self.server.fleet_server  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.fleet.token}"
        return hmac.compare_digest(header, expected)

    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("incomplete request body")
        return payload

    def _json_body(self) -> dict:
        body = json.loads(self._read_body().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        return body

    @staticmethod
    def _protocol_body(body: dict, kind: str):
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("request message is required")
        envelope = json.dumps(message, separators=(",", ":")).encode("utf-8")
        decoded = decode_message(envelope, expected_kind=kind)
        return decoded

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/fleet/health":
            self._respond(HTTPStatus.OK, {"ok": True})
            return
        if path == "/v1/fleet/tasks":
            self._respond(
                HTTPStatus.OK,
                {"tasks": [_record_dict(record) for record in self.fleet.store.list_tasks()]},
            )
            return
        if path == "/v1/fleet/runners":
            self._respond(
                HTTPStatus.OK,
                {"runners": self.fleet.store.list_runners()},
            )
            return
        self._respond(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            self._dispatch_post(urlsplit(self.path).path.rstrip("/"))
        except (ValueError, TypeError, KeyError) as exc:
            self._respond(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - final HTTP safety boundary
            self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _dispatch_post(self, path: str) -> None:
        if path == "/v1/fleet/runners/register":
            body = self._json_body()
            capability = RunnerCapability.from_dict(self._protocol_body(body, "runner_capability"))
            self.fleet.store.register_runner(
                capability,
                now=body.get("now"),
                ttl=float(body.get("ttl", 30.0)),
            )
            self._respond(HTTPStatus.OK, {"ok": True})
            return
        if path.startswith("/v1/fleet/runners/") and path.endswith("/heartbeat"):
            parts = path.split("/")
            if len(parts) != 7:
                raise ValueError("invalid heartbeat path")
            body = self._json_body()
            updated = self.fleet.store.heartbeat_runner(
                parts[4],
                parts[5],
                now=body.get("now"),
                ttl=float(body.get("ttl", 30.0)),
            )
            self._respond(HTTPStatus.OK if updated else HTTPStatus.NOT_FOUND, {"ok": updated})
            return
        if path == "/v1/fleet/tasks":
            body = self._json_body()
            task = FleetTask.from_dict(self._protocol_body(body, "fleet_task"))
            record = self.fleet.store.submit_task(task, now=body.get("now"))
            self._respond(HTTPStatus.OK, _record_dict(record))
            return
        if path == "/v1/fleet/tasks/claim":
            body = self._json_body()
            capability = RunnerCapability.from_dict(body["capability"])
            claim = self.fleet.store.claim_task(
                capability,
                now=body.get("now"),
                lease_seconds=float(body.get("lease_seconds", 60.0)),
            )
            self._respond(HTTPStatus.OK, {"claim": _claim_dict(claim) if claim else None})
            return
        parts = path.split("/")
        if len(parts) == 6 and parts[:4] == ["", "v1", "fleet", "tasks"]:
            task_id, action = parts[4], parts[5]
            body = self._json_body()
            if action == "renew":
                claim = self.fleet.store.renew_claim(
                    task_id,
                    body["claim_id"],
                    now=body.get("now"),
                    lease_seconds=float(body.get("lease_seconds", 60.0)),
                )
                self._respond(HTTPStatus.OK, {"claim": _claim_dict(claim) if claim else None})
                return
            if action == "complete":
                ok = self.fleet.store.complete_task(
                    task_id, body["claim_id"], result=str(body.get("result", "")), now=body.get("now")
                )
                if ok:
                    self.fleet.notify_task_update(task_id, "completed", result=body.get("result"))
                self._respond(HTTPStatus.OK, {"ok": ok})
                return
            if action == "fail":
                ok = self.fleet.store.fail_task(
                    task_id, body["claim_id"], error=str(body.get("error", "")), now=body.get("now")
                )
                if ok:
                    self.fleet.notify_task_update(task_id, "failed", error=body.get("error"))
                self._respond(HTTPStatus.OK, {"ok": ok})
                return
            if action == "retry":
                ok = self.fleet.store.retry_task(task_id, now=body.get("now"))
                if ok:
                    self.fleet.notify_task_update(task_id, "pending")
                self._respond(HTTPStatus.OK, {"ok": ok})
                return
            if action == "cancel":
                ok = self.fleet.store.cancel_task(task_id, now=body.get("now"))
                if ok:
                    self.fleet.notify_task_update(task_id, "cancelled")
                self._respond(HTTPStatus.OK, {"ok": ok})
                return
        self._respond(HTTPStatus.NOT_FOUND, {"error": "not found"})


def _record_dict(record) -> dict:
    return {
        "task_id": record.task_id,
        "title": record.title,
        "body": record.body,
        "requirement": record.requirement.to_dict(),
        "idempotency_key": record.idempotency_key,
        "status": record.status,
        "claim_id": record.claim_id,
        "node_id": record.node_id,
        "runner_profile": record.runner_profile,
        "lease_expires_at": record.lease_expires_at,
        "attempt": record.attempt,
        "result": record.result,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
