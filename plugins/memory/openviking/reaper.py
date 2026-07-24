"""Finish a managed OpenViking shutdown after the Hermes CLI exits.

The interactive CLI has a short, process-wide exit watchdog. OpenViking
session commits and queued indexing can legitimately take longer, so the
provider hands its exact child process and tracked task IDs to this small
detached helper. The helper waits for every tracked task, drains OpenViking's
remaining processing queues, and only then stops that exact process.

The JSON payload is read from stdin so credentials never appear in the process
list. This module is intentionally standalone and uses only stdlib plus
``psutil``, which is a Hermes core dependency.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

import psutil


logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TERMINAL_TASK_STATUSES = {"completed", "failed"}
_ACTIVE_TASK_STATUSES = {"pending", "running"}


def _validated_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("shutdown payload must be a JSON object")

    endpoint = str(raw.get("endpoint") or "").rstrip("/")
    parsed_endpoint = urlparse(endpoint)
    if (
        parsed_endpoint.scheme.lower() != "http"
        or (parsed_endpoint.hostname or "").lower() not in _LOCAL_HOSTS
    ):
        raise ValueError("shutdown endpoint must be local HTTP")

    pid = int(raw.get("pid") or 0)
    create_time = float(raw.get("create_time") or 0)
    if pid <= 1 or pid == os.getpid() or create_time <= 0:
        raise ValueError("shutdown process identity is invalid")

    task_ids = raw.get("task_ids")
    if not isinstance(task_ids, list):
        raise ValueError("tracked task IDs must be a list")
    normalized_task_ids = []
    for task_id in task_ids:
        value = str(task_id or "").strip()
        if not value or len(value) > 256:
            raise ValueError("tracked task ID is invalid")
        normalized_task_ids.append(value)

    headers = raw.get("headers") or {}
    if not isinstance(headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in headers.items()
    ):
        raise ValueError("request headers are invalid")

    wait_timeout = float(raw.get("wait_timeout") or 0)
    poll_interval = float(raw.get("poll_interval") or 0)
    stop_timeout = float(raw.get("stop_timeout") or 0)
    if wait_timeout <= 0 or poll_interval <= 0 or stop_timeout <= 0:
        raise ValueError("shutdown timeouts must be positive")

    return {
        "endpoint": endpoint,
        "pid": pid,
        "create_time": create_time,
        "task_ids": normalized_task_ids,
        "headers": headers,
        "wait_timeout": wait_timeout,
        "poll_interval": poll_interval,
        "stop_timeout": stop_timeout,
    }


def _task_status(endpoint: str, task_id: str, headers: dict[str, str]) -> str:
    request = Request(
        f"{endpoint}/api/v1/tasks/{quote(task_id, safe='')}",
        headers=headers,
        method="GET",
    )
    # Do not inherit proxy settings for a loopback-only control request.
    response = build_opener(ProxyHandler({})).open(request, timeout=5.0)
    with response:
        payload = json.loads(response.read())
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raise ValueError(f"task {task_id} returned an invalid response")
    status = str(result.get("status") or "").strip().lower()
    if status not in _TERMINAL_TASK_STATUSES | _ACTIVE_TASK_STATUSES:
        raise ValueError(f"task {task_id} returned unknown status {status!r}")
    return status


def _wait_for_processing(
    endpoint: str,
    headers: dict[str, str],
    timeout: float,
) -> None:
    """Wait for all queued OpenViking semantic and embedding work."""
    request = Request(
        f"{endpoint}/api/v1/system/wait",
        data=json.dumps({"timeout": timeout}).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    # Give the HTTP exchange a small allowance beyond OpenViking's own bounded
    # wait. Do not inherit proxy settings for a loopback-only control request.
    response = build_opener(ProxyHandler({})).open(request, timeout=timeout + 1.0)
    with response:
        payload = json.loads(response.read())
    result = payload.get("result") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "ok"
        or not isinstance(result, dict)
    ):
        raise ValueError("system wait returned an invalid response")

    for queue_name, queue_status in result.items():
        if not isinstance(queue_status, dict):
            raise ValueError("system wait returned an invalid queue response")
        error_count = queue_status.get("error_count", 0)
        errors = queue_status.get("errors", [])
        if (
            not isinstance(error_count, int)
            or isinstance(error_count, bool)
            or error_count < 0
            or not isinstance(errors, list)
        ):
            raise ValueError("system wait returned an invalid queue response")
        if error_count or errors:
            raise RuntimeError(
                f"OpenViking queue {queue_name!r} reported processing errors"
            )


def _same_process(pid: int, expected_create_time: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - expected_create_time) > 0.01:
            return None
        # A detached helper is not the OpenViking process's parent and cannot
        # reap it. On POSIX, a server that has already exited can therefore
        # remain visible as a zombie until Hermes itself exits. Treat that as
        # stopped instead of reporting a false timeout or attempting SIGKILL.
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _stop_exact_process(pid: int, create_time: float, timeout: float) -> bool:
    process = _same_process(pid, create_time)
    if process is None:
        return True
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=timeout)
        return True
    except psutil.TimeoutExpired:
        logger.warning(
            "Managed OpenViking PID %d did not stop within %.1f seconds; force-stopping it",
            pid,
            timeout,
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, OSError) as exc:
        logger.error("Could not stop managed OpenViking PID %d: %s", pid, exc)
        return False

    # Revalidate the creation time immediately before the destructive fallback.
    process = _same_process(pid, create_time)
    if process is None:
        return True
    try:
        process.kill()
        process.wait(timeout=timeout)
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, psutil.TimeoutExpired, OSError) as exc:
        logger.error("Could not force-stop managed OpenViking PID %d: %s", pid, exc)
        return False


def run(payload: dict[str, Any]) -> int:
    config = _validated_payload(payload)
    pending = set(config["task_ids"])
    deadline = time.monotonic() + config["wait_timeout"]

    while pending:
        if _same_process(config["pid"], config["create_time"]) is None:
            return 0
        try:
            for task_id in tuple(pending):
                status = _task_status(config["endpoint"], task_id, config["headers"])
                if status in _TERMINAL_TASK_STATUSES:
                    pending.discard(task_id)
        except Exception as exc:
            logger.error(
                "Could not verify managed OpenViking background work; leaving PID %d running: %s",
                config["pid"],
                exc,
            )
            return 2

        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error(
                "Managed OpenViking tasks did not finish within %.1f seconds; leaving PID %d running",
                config["wait_timeout"],
                config["pid"],
            )
            return 3
        time.sleep(min(config["poll_interval"], remaining))

    if _same_process(config["pid"], config["create_time"]) is None:
        return 0

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        logger.error(
            "Managed OpenViking work did not finish within %.1f seconds; leaving PID %d running",
            config["wait_timeout"],
            config["pid"],
        )
        return 3
    try:
        _wait_for_processing(config["endpoint"], config["headers"], remaining)
    except Exception as exc:
        logger.error(
            "Could not verify managed OpenViking queue drain; leaving PID %d running: %s",
            config["pid"],
            exc,
        )
        return 2

    return (
        0
        if _stop_exact_process(
            config["pid"], config["create_time"], config["stop_timeout"]
        )
        else 4
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        payload = json.load(sys.stdin)
        return run(payload)
    except Exception:
        logger.exception("Managed OpenViking shutdown handoff failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
