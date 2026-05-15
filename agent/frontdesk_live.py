"""Opt-in live frontdesk pre-dispatch helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from utils import is_truthy_value


_DEFAULT_WORKER_LANE = "main"
_DEFAULT_WORKER_TIMEOUT_SECONDS = 60 * 60
_FRONTDESK_NOTIFIERS: dict[tuple[int, str | None], Callable[[str], Any]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _worker_prompt(goal: str) -> str:
    return (
        "You are a background worker lane launched by Hermes frontdesk. "
        "Complete the user's task independently. Do not ask clarification unless absolutely impossible. "
        "If the task references local files, inspect and edit them directly as requested. "
        "Return a concise Korean completion report with paths/artifacts changed and any caveats.\n\n"
        f"USER TASK:\n{goal}"
    )


def _worker_artifact_paths(task_id: str | None) -> dict[str, Path]:
    from hermes_constants import get_hermes_home

    safe_task_id = task_id or "untracked"
    worker_dir = get_hermes_home() / "workers" / safe_task_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    return {
        "last_message": worker_dir / "last-message.txt",
        "summary": worker_dir / "summary.md",
        "stderr": worker_dir / "stderr.log",
    }


def _write_text_artifact(path: Path | None, text: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def _run_default_worker_subprocess(
    goal: str,
    token: Any,
    *,
    task_id: str | None = None,
    on_process_start: Callable[[int], Any] | None = None,
    last_message_path: Path | None = None,
    summary_path: Path | None = None,
    stderr_path: Path | None = None,
) -> str:
    """Run one frontdesk worker task in a detached Hermes oneshot subprocess."""
    cmd = [sys.executable, "-m", "hermes_cli.main", "-z", _worker_prompt(goal)]
    env = dict(os.environ)
    env["HERMES_FRONTDESK_WORKER"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if on_process_start is not None:
        try:
            on_process_start(proc.pid)
        except Exception:
            pass
    started = time.monotonic()
    while proc.poll() is None:
        if getattr(token, "cancelled", False):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            from agent.worker_lanes import WorkerCancelled

            raise WorkerCancelled("frontdesk worker cancelled")
        if time.monotonic() - started > _DEFAULT_WORKER_TIMEOUT_SECONDS:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            _write_text_artifact(last_message_path, (stdout or "").strip())
            _write_text_artifact(stderr_path, (stderr or "").strip())
            raise TimeoutError(
                f"frontdesk worker timed out after {_DEFAULT_WORKER_TIMEOUT_SECONDS}s\n{stderr or stdout}"
            )
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    _write_text_artifact(last_message_path, (stdout or "").strip())
    _write_text_artifact(summary_path, (stdout or "").strip())
    _write_text_artifact(stderr_path, (stderr or "").strip())
    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(f"frontdesk worker exited {proc.returncode}: {detail}")
    return (stdout or "").strip() or "worker completed with no output"


def _completion_notice(task_id: str | None, summary: str) -> str:
    del summary
    return (
        f"worker complete: {task_id or 'untracked'}\n\n"
        "Result captured. Review is pending before final presentation. Use /status for details."
    )


def _linked_worker_id(runtime: Any, task_id: str | None) -> str | None:
    if not task_id:
        return None
    for _ in range(20):
        task = runtime.task_registry.get_task(task_id)
        worker_id = getattr(task, "active_worker_id", None) if task is not None else None
        if isinstance(worker_id, str) and worker_id:
            return worker_id
        time.sleep(0.01)
    return None


def _attach_worker_result(
    runtime: Any,
    *,
    task_id: str | None,
    worker_id: str | None,
    status: str,
    summary: str,
    error: str | None = None,
    paths: dict[str, Path] | None = None,
) -> None:
    if task_id is None:
        return
    artifacts = []
    if paths:
        for key, path in paths.items():
            if key == "stderr":
                continue
            artifacts.append({"kind": key, "path": str(path)})
    payload = {
        "worker_id": worker_id,
        "task_id": task_id,
        "status": status,
        "summary": summary,
    }
    if artifacts:
        payload["artifacts"] = artifacts
    if error:
        payload["error"] = error
    try:
        runtime.attach_worker_result(
            task_id=task_id,
            worker_id=worker_id or "",
            result=payload,
        )
    except Exception:
        try:
            runtime.task_registry.attach_worker_result(task_id, payload)
        except Exception:
            pass
    try:
        runtime.task_registry.update_frontdesk_metadata(
            task_id,
            last_message_path=str(paths["last_message"]) if paths and "last_message" in paths else None,
            summary_artifact_path=str(paths["summary"]) if paths and "summary" in paths else None,
        )
    except Exception:
        pass


def ensure_default_worker_lane(
    owner: Any,
    *,
    session_key: str | None = None,
    notify_callback: Callable[[str], Any] | None = None,
) -> None:
    """Ensure every live frontdesk runtime has a default worker lane.

    The live gate must not expose a worker-capable control plane with an empty
    registry.  This registers one conservative Hermes oneshot-backed lane per
    runtime.  Results are also pushed back through a per-session notifier when a
    gateway/TUI surface supplies one.
    """
    from agent.orchestration_runtime import get_or_create_orchestration_runtime
    from agent.worker_lanes import ThreadWorkerLane, WorkerSpec, CancelToken

    runtime = get_or_create_orchestration_runtime(owner)
    if notify_callback is not None:
        _FRONTDESK_NOTIFIERS[(id(owner), session_key)] = notify_callback

    if _DEFAULT_WORKER_LANE in runtime.worker_registry.lane_names():
        return

    owner_id = id(owner)
    run_default_worker = _run_default_worker_subprocess

    def runner(spec: WorkerSpec, token: CancelToken) -> str:
        worker_id = _linked_worker_id(runtime, spec.task_id)
        paths = _worker_artifact_paths(spec.task_id)
        try:
            runtime.task_registry.update_frontdesk_metadata(
                spec.task_id,
                worker_session_id=worker_id,
                last_message_path=str(paths["last_message"]),
                summary_artifact_path=str(paths["summary"]),
            )
        except Exception:
            pass

        def _record_process(pid: int) -> None:
            try:
                runtime.task_registry.update_frontdesk_metadata(
                    spec.task_id,
                    worker_process_id=str(pid),
                )
            except Exception:
                pass

        try:
            result = run_default_worker(
                spec.goal,
                token,
                task_id=spec.task_id,
                on_process_start=_record_process,
                last_message_path=paths["last_message"],
                summary_path=paths["summary"],
                stderr_path=paths["stderr"],
            )
        except BaseException as exc:
            from agent.worker_lanes import WorkerCancelled

            status = "cancelled" if isinstance(exc, WorkerCancelled) else "failed"
            summary = "worker cancelled" if status == "cancelled" else "worker failed"
            _attach_worker_result(
                runtime,
                task_id=spec.task_id,
                worker_id=worker_id,
                status=status,
                summary=summary,
                error=str(exc) or type(exc).__name__,
                paths=paths,
            )
            raise
        _attach_worker_result(
            runtime,
            task_id=spec.task_id,
            worker_id=worker_id,
            status="succeeded",
            summary=result,
            paths=paths,
        )
        sk = None
        if isinstance(spec.metadata, dict):
            raw_sk = spec.metadata.get("session_key")
            sk = raw_sk if isinstance(raw_sk, str) else None
        notifier = _FRONTDESK_NOTIFIERS.get((owner_id, sk)) or _FRONTDESK_NOTIFIERS.get(
            (owner_id, None)
        )
        if notifier is not None:
            try:
                notifier(_completion_notice(spec.task_id, result))
            except Exception:
                pass
        return result

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name=_DEFAULT_WORKER_LANE))


def frontdesk_live_enabled(owner: Any, *, session: dict | None = None) -> bool:
    """Return whether live frontdesk interception is explicitly enabled."""
    for carrier in (session, owner):
        if not carrier:
            continue
        if isinstance(carrier, dict):
            if "frontdesk_live_enabled" in carrier:
                return bool(carrier.get("frontdesk_live_enabled"))
            cfg = carrier.get("config")
        else:
            if hasattr(carrier, "frontdesk_live_enabled"):
                return bool(getattr(carrier, "frontdesk_live_enabled"))
            if hasattr(carrier, "_frontdesk_live_enabled"):
                return bool(getattr(carrier, "_frontdesk_live_enabled"))
            cfg = getattr(carrier, "config", None)
        if isinstance(cfg, dict):
            orchestration = cfg.get("orchestration") or {}
            if isinstance(orchestration, dict):
                raw = orchestration.get("frontdesk_live_enabled")
                if raw is not None:
                    return is_truthy_value(raw, default=False)
    return False


def handle_frontdesk_live_input(
    owner: Any,
    request_text: Any,
    *,
    session: dict | None = None,
    session_key: str | None = None,
    source_surface: str,
    main_in_flight: bool = False,
    steer_callback: Callable[[str], Any] | None = None,
    cancel_callback: Callable[[str], Any] | None = None,
    notify_callback: Callable[[str], Any] | None = None,
):
    """Run the frontdesk control gate and return a consumed result or ``None``.

    ``None`` means the caller should continue its existing main-model path.
    Every non-``None`` result is a local control response and must not be
    enqueued, replayed, or sent to the main model.
    """
    if not frontdesk_live_enabled(owner, session=session):
        return None
    if not isinstance(request_text, str) or not request_text.strip():
        return None

    from agent.orchestration_runtime import (
        OrchestrationRuntime,
        get_or_create_orchestration_runtime,
    )

    if session is None:
        ensure_default_worker_lane(
            owner,
            session_key=session_key,
            notify_callback=notify_callback,
        )

    runtime_owner = session if session is not None else owner
    if isinstance(runtime_owner, dict):
        runtime = runtime_owner.get("_orchestration_runtime")
        if not isinstance(runtime, OrchestrationRuntime):
            runtime = OrchestrationRuntime.create()
            runtime_owner["_orchestration_runtime"] = runtime
    else:
        runtime = get_or_create_orchestration_runtime(runtime_owner)
    result = runtime.handle_frontdesk_input(
        request_text,
        frontdesk_mode_active=True,
        session_key=session_key,
        source_surface=source_surface,
        main_in_flight=main_in_flight,
        steer_callback=steer_callback,
    )
    if result.action == "main":
        return None
    if result.action == "stopped" and cancel_callback is not None:
        cancel_callback(request_text)
    return result
