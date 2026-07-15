"""In-process readiness proofs for production execution boundaries.

Systemd ordering and a listening AF_UNIX socket are not sufficient evidence
that Cloud Muncho can actually use its sealed execution edges.  Immediately
before READY the gateway uses these helpers to perform one bounded command
round-trip through the isolated worker and one bounded release-local browser
command through the browser controller.

The commands are fixed liveness probes.  They do not classify work, select a
tool, inspect user text, or make a semantic decision.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Mapping

from gateway.isolated_worker import IsolatedWorkerClient, canonical_lease_id
from tools.browser_controller_client import (
    BrowserControllerClient,
    BrowserControllerClientConfig,
)


WORKER_RECEIPT_SCHEMA = "muncho-production-isolated-worker-readiness.v1"
BROWSER_RECEIPT_SCHEMA = "muncho-production-browser-controller-readiness.v1"
_WORKER_OUTPUT = b"MUNCHO_ISOLATED_WORKER_READY\n"
_WORKER_COMMAND = "printf 'MUNCHO_ISOLATED_WORKER_READY\\n'"
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ProductionExecutionReadinessError(RuntimeError):
    """Stable, secret-free readiness failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _identity_digest(label: bytes, revision: str, config_sha256: str) -> str:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ProductionExecutionReadinessError("production_revision_invalid")
    if (
        not isinstance(config_sha256, str)
        or _SHA256.fullmatch(config_sha256) is None
    ):
        raise ProductionExecutionReadinessError("production_config_digest_invalid")
    return hashlib.sha256(
        label + b"\x00" + revision.encode("ascii") + b"\x00" + config_sha256.encode("ascii")
    ).hexdigest()


def attest_isolated_worker_execution(
    *,
    socket_path: Path,
    server_uid: int,
    server_gid: int,
    socket_uid: int,
    socket_gid: int,
    revision: str,
    config_sha256: str,
    timeout_seconds: int = 10,
) -> Mapping[str, Any]:
    """Prove peer identity plus one real bwrap/shell execution round-trip."""

    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 30:
        raise ProductionExecutionReadinessError("worker_readiness_timeout_invalid")
    identity = _identity_digest(b"muncho-worker-readiness-v1", revision, config_sha256)
    client = IsolatedWorkerClient(
        Path(socket_path),
        lease_id=canonical_lease_id(f"production-readiness:{identity}"),
        expected_server_uid=server_uid,
        expected_server_gid=server_gid,
        expected_socket_uid=socket_uid,
        expected_socket_gid=socket_gid,
    )
    session_id: str | None = None
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + timeout_seconds
    try:
        session_id = client.start(
            _WORKER_COMMAND,
            cwd=Path("/workspace"),
            timeout_seconds=timeout_seconds,
        )
        while time.monotonic() < deadline:
            result = client.poll(session_id, wait_milliseconds=250)
            try:
                stdout.extend(
                    base64.b64decode(result.get("stdout_b64", ""), validate=True)
                )
                stderr.extend(
                    base64.b64decode(result.get("stderr_b64", ""), validate=True)
                )
            except (TypeError, ValueError) as exc:
                raise ProductionExecutionReadinessError(
                    "worker_readiness_response_invalid"
                ) from exc
            if len(stdout) > len(_WORKER_OUTPUT) or len(stderr) > 0:
                raise ProductionExecutionReadinessError(
                    "worker_readiness_output_invalid"
                )
            if (
                result.get("state") == "exited"
                and result.get("returncode") == 0
                and result.get("drained") is True
                and result.get("complete") is True
            ):
                if bytes(stdout) != _WORKER_OUTPUT or stderr:
                    raise ProductionExecutionReadinessError(
                        "worker_readiness_output_invalid"
                    )
                return {
                    "schema": WORKER_RECEIPT_SCHEMA,
                    "lease_identity_sha256": identity,
                    "socket_path": str(socket_path),
                    "server_uid": server_uid,
                    "server_gid": server_gid,
                    "socket_uid": socket_uid,
                    "socket_gid": socket_gid,
                    "execution_round_trip": True,
                    "output_sha256": hashlib.sha256(_WORKER_OUTPUT).hexdigest(),
                    "secret_material_recorded": False,
                }
            if result.get("state") != "running":
                raise ProductionExecutionReadinessError(
                    "worker_readiness_execution_failed"
                )
        raise ProductionExecutionReadinessError("worker_readiness_timed_out")
    except ProductionExecutionReadinessError:
        if session_id is not None:
            try:
                client.cancel(session_id)
            except Exception:
                pass
        raise
    except Exception as exc:
        if session_id is not None:
            try:
                client.cancel(session_id)
            except Exception:
                pass
        raise ProductionExecutionReadinessError(
            "worker_readiness_transport_failed"
        ) from exc
    finally:
        client.close()


def attest_browser_controller_execution(
    *,
    client_config: BrowserControllerClientConfig,
    revision: str,
    config_sha256: str,
) -> Mapping[str, Any]:
    """Prove a real controller session and release-local agent-browser command."""

    identity = _identity_digest(
        b"muncho-browser-readiness-v1", revision, config_sha256
    )
    client = BrowserControllerClient(client_config, identity)
    try:
        result = client.command("eval", ["window.location.href"])
        if result.get("success") is not True:
            raise ProductionExecutionReadinessError(
                "browser_readiness_execution_failed"
            )
        return {
            "schema": BROWSER_RECEIPT_SCHEMA,
            "session_identity_sha256": identity,
            "socket_path": str(client_config.socket_path),
            "server_uid": client_config.server_uid,
            "command_round_trip": True,
            "secret_material_recorded": False,
        }
    except ProductionExecutionReadinessError:
        raise
    except Exception as exc:
        raise ProductionExecutionReadinessError(
            "browser_readiness_transport_failed"
        ) from exc
    finally:
        client.close()


__all__ = [
    "BROWSER_RECEIPT_SCHEMA",
    "ProductionExecutionReadinessError",
    "WORKER_RECEIPT_SCHEMA",
    "attest_browser_controller_execution",
    "attest_isolated_worker_execution",
]
