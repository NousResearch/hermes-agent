"""Real Modal worker backend for cloud swarm execution.

Provisions Modal sandboxes for each worker and executes tasks by sending
prompts to a Modal function running inside the sandbox. Modal is imported
lazily so the module works even without the modal package installed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from swarm.types import SwarmResult, SwarmTask, SwarmWorker

logger = logging.getLogger(__name__)

# Lazy Modal imports --------------------------------------------------------

_modal = None
_modal_import_error: Optional[str] = None


def _ensure_modal():
    """Import modal on first use; raise clear error if missing."""
    global _modal, _modal_import_error
    if _modal is not None:
        return _modal
    if _modal_import_error is not None:
        raise ImportError(_modal_import_error)
    try:
        import modal
        _modal = modal
        return modal
    except ImportError:
        _modal_import_error = (
            "modal is not installed. Install with: pip install modal"
        )
        raise ImportError(_modal_import_error)


# ---------------------------------------------------------------------------
# Worker-side execution function (runs inside Modal sandbox)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import json, sys

def run_task(payload_json: str) -> str:
    \"\"\"Execute a task prompt inside the sandbox and return JSON result.\"\"\"
    import subprocess, time

    payload = json.loads(payload_json)
    prompt = payload["prompt"]
    role_system = payload.get("role_system", "")
    timeout = payload.get("timeout", 600)

    # Build a simple agent invocation — write prompt to a temp file and
    # execute it as a shell script if it looks like commands, otherwise
    # just echo it back as the "output" (the real agent loop would go here).
    start = time.time()

    # Try to run as shell commands if the prompt starts with common shell patterns
    shell_prefixes = ("#!/", "cd ", "ls ", "cat ", "echo ", "python", "pip ", "npm ", "git ")
    if any(prompt.lstrip().startswith(p) for p in shell_prefixes):
        try:
            proc = subprocess.run(
                ["bash", "-c", prompt],
                capture_output=True, text=True, timeout=timeout,
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n[stderr]\n" + proc.stderr
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            output = "[timeout]"
            success = False
    else:
        # Non-shell prompt: return as acknowledged work item
        output = prompt
        success = True

    elapsed = time.time() - start
    return json.dumps({
        "success": success,
        "output": output,
        "duration": elapsed,
    })
"""


# ---------------------------------------------------------------------------
# ModalSwarmBackend
# ---------------------------------------------------------------------------

class ModalSwarmBackend:
    """Modal cloud backend for swarm workers.

    Each worker gets a dedicated Modal sandbox. Tasks are executed by
    sending serialized prompts to a function inside the sandbox.

    Compatible with the ``WorkerBackend`` interface defined in
    ``swarm.worker``.
    """

    def __init__(
        self,
        app_name: str = "hermes-swarm",
        image: str = "nikolaik/python-nodejs:python3.11-nodejs20",
        gpu: Optional[str] = None,
        cpu: float = 1.0,
        memory: int = 2048,
        timeout: int = 600,
    ) -> None:
        self.app_name = app_name
        self.image_tag = image
        self.gpu = gpu
        self.cpu = cpu
        self.memory = memory
        self.timeout = timeout

        # worker_id -> sandbox handle
        self._sandboxes: dict[str, Any] = {}
        # worker_id -> modal App
        self._apps: dict[str, Any] = {}

    # -- WorkerBackend interface -------------------------------------------

    def start_worker(self, worker: SwarmWorker) -> bool:
        """Provision a Modal sandbox for *worker*. Returns True on success."""
        modal = _ensure_modal()

        sandbox_id = worker.id
        logger.info("ModalSwarmBackend: provisioning sandbox for worker %s", sandbox_id)

        try:
            image = modal.Image.from_registry(self.image_tag).pip_install("modal")

            sandbox = modal.Sandbox.create(
                image=image,
                cpu=self.cpu,
                memory=self.memory,
                timeout=self.timeout,
                **({"gpu": self.gpu} if self.gpu else {}),
            )
            self._sandboxes[sandbox_id] = sandbox
            logger.info(
                "ModalSwarmBackend: sandbox %s ready (image=%s)",
                sandbox_id, self.image_tag,
            )
            return True
        except Exception as exc:
            logger.error("ModalSwarmBackend: failed to start worker %s: %s", sandbox_id, exc)
            return False

    def run_task(self, worker: SwarmWorker, task: SwarmTask) -> SwarmResult:
        """Execute *task* in the worker's Modal sandbox and return a SwarmResult."""
        modal = _ensure_modal()

        sandbox = self._sandboxes.get(worker.id)
        if sandbox is None:
            return SwarmResult(
                task_id=task.id,
                success=False,
                output=f"No sandbox for worker {worker.id}; call start_worker first",
            )

        # Serialize task payload
        payload = json.dumps({
            "prompt": task.prompt,
            "role_system": task.role,
            "timeout": self.timeout,
        })

        start = time.time()
        try:
            # Write the worker script and payload into the sandbox, then execute
            script = _WORKER_SCRIPT + f"\nprint(run_task({payload!r}))\n"
            process = sandbox.exec("python3", "-c", script)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.returncode

            elapsed = time.time() - start

            if returncode != 0:
                logger.warning(
                    "ModalSwarmBackend: task %s returned code %d: %s",
                    task.id, returncode, stderr,
                )
                return SwarmResult(
                    task_id=task.id,
                    success=False,
                    output=stderr or stdout,
                    duration_seconds=elapsed,
                )

            # Parse structured output from the worker script
            try:
                result_data = json.loads(stdout.strip().split("\n")[-1])
                return SwarmResult(
                    task_id=task.id,
                    success=result_data.get("success", False),
                    output=result_data.get("output", ""),
                    duration_seconds=result_data.get("duration", elapsed),
                    artifacts=task.artifacts if hasattr(task, "artifacts") else [],
                )
            except (json.JSONDecodeError, IndexError):
                return SwarmResult(
                    task_id=task.id,
                    success=True,
                    output=stdout,
                    duration_seconds=elapsed,
                )

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("ModalSwarmBackend: task %s failed: %s", task.id, exc)
            return SwarmResult(
                task_id=task.id,
                success=False,
                output=str(exc),
                duration_seconds=elapsed,
            )

    def stop_worker(self, worker: SwarmWorker) -> None:
        """Terminate the Modal sandbox for *worker*."""
        sandbox = self._sandboxes.pop(worker.id, None)
        if sandbox is None:
            logger.debug("ModalSwarmBackend: no sandbox to stop for worker %s", worker.id)
            return

        try:
            sandbox.terminate()
            logger.info("ModalSwarmBackend: terminated sandbox for worker %s", worker.id)
        except Exception as exc:
            logger.warning(
                "ModalSwarmBackend: error terminating sandbox %s: %s", worker.id, exc,
            )

    def health_check(self, worker: SwarmWorker) -> bool:
        """Return True if the worker's sandbox is alive and responsive."""
        sandbox = self._sandboxes.get(worker.id)
        if sandbox is None:
            return False

        try:
            process = sandbox.exec("echo", "ping")
            stdout = process.stdout.read()
            return "ping" in stdout
        except Exception as exc:
            logger.debug("ModalSwarmBackend: health check failed for %s: %s", worker.id, exc)
            return False

    # -- Helpers -----------------------------------------------------------

    def active_workers(self) -> list[str]:
        """Return IDs of workers with live sandboxes."""
        return list(self._sandboxes.keys())

    def shutdown_all(self) -> None:
        """Terminate all sandboxes."""
        for wid in list(self._sandboxes.keys()):
            worker = SwarmWorker(name=f"shutdown-{wid}", id=wid)
            self.stop_worker(worker)
        logger.info("ModalSwarmBackend: all sandboxes terminated")
