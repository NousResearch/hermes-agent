"""Profile-owned Desktop setup jobs, also executed inside isolated plugin hosts.

Only JSON results cross the plugin boundary. Providers report progress from the
worker through a local context; no callable is sent over RPC. Jobs are deliberately
not replayed after a host restart: the user must inspect the result and retry.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from contextvars import ContextVar
from pathlib import Path

from agent.memory_provider import (
    MemoryProviderConfigConflictError,
    spawn_context_thread,
)

_lock = threading.RLock()
_jobs: dict[tuple[str, str], dict] = {}
_current: ContextVar[dict | None] = ContextVar("memory_setup_job", default=None)


def report_progress(stage: str, message: str) -> None:
    """Report a safe, credential-free setup stage from a provider action."""
    job = _current.get()
    if job is not None:
        with _lock:
            job["progress"] = {"stage": stage, "message": message}


def _key(provider, home: str) -> tuple[str, str]:
    return str(Path(home).resolve()), provider.name


def _snapshot(job: dict) -> dict:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def status(provider, *, hermes_home: str, operation_id: str = "") -> dict:
    with _lock:
        job = _jobs.get(_key(provider, hermes_home))
        if job is None or (operation_id and job["id"] != operation_id):
            return {
                "status": "unavailable",
                "message": "Setup status is no longer available. Check the saved settings before retrying.",
            }
        return _snapshot(job)


def start(provider, action: str, payload: dict, *, hermes_home: str) -> dict:
    signature = hashlib.sha256(
        json.dumps([action, payload], sort_keys=True).encode()
    ).hexdigest()
    key = _key(provider, hermes_home)
    with _lock:
        previous = _jobs.get(key)
        if previous and previous["status"] == "running":
            if previous["_signature"] == signature:
                return _snapshot(previous)
            return {
                "status": "busy",
                "message": "Setup is already running for this provider and profile. Wait for it to finish.",
            }
        job = {
            "id": uuid.uuid4().hex,
            "action": action,
            "status": "running",
            "progress": {"stage": "starting", "message": "Starting setup..."},
            "_signature": signature,
        }
        _jobs[key] = job

    def run():
        token = _current.set(job)
        try:
            result = provider.handle_desktop_config_action(
                action, payload, hermes_home=hermes_home
            )
            # Validate the public result here, including in-process plugins.
            result = json.loads(json.dumps(result))
            outcome = {"status": "completed", "result": result}
        except MemoryProviderConfigConflictError as exc:
            outcome = {
                "status": "confirmation_required",
                "confirmation": exc.confirmation,
                "message": str(exc),
            }
        except ValueError as exc:
            outcome = {"status": "failed", "message": str(exc)}
        except BaseException as exc:  # health: allow BLE001 -- Provider failures, including SystemExit, become sanitized terminal job states.
            # A setup helper may exit; it must not leave an immortal running job.
            outcome = {
                "status": "failed",
                "message": f"Setup failed ({type(exc).__name__}). Check the provider configuration and retry.",
            }
        finally:
            _current.reset(token)
        with _lock:
            job.update(outcome)

    try:
        spawn_context_thread(run, name=f"memory-setup-{provider.name}").start()
    except Exception:
        with _lock:
            job.update(
                status="failed", message="Could not start setup. Retry the operation."
            )
        raise
    with _lock:
        return _snapshot(job)
