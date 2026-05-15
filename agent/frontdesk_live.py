"""Opt-in live frontdesk pre-dispatch helpers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from utils import is_truthy_value


_LOG = logging.getLogger(__name__)
_DEFAULT_WORKER_LANE = "main"
_DEFAULT_WORKER_TIMEOUT_SECONDS = 60 * 60
_DEFAULT_DURABLE_LEASE_SECONDS = 60 * 60 * 6
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


def frontdesk_durable_store_path() -> Path:
    """Return the default durable frontdesk SQLite path under Hermes home."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "frontdesk" / "frontdesk.sqlite3"


def _cfg_bool(owner: Any, session: dict | None, key: str) -> bool | None:
    for carrier in (session, owner):
        if not carrier:
            continue
        if isinstance(carrier, dict):
            if key in carrier:
                return is_truthy_value(carrier.get(key), default=False)
            cfg = carrier.get("config")
        else:
            if hasattr(carrier, key):
                return is_truthy_value(getattr(carrier, key), default=False)
            private_key = f"_{key}"
            if hasattr(carrier, private_key):
                return is_truthy_value(getattr(carrier, private_key), default=False)
            cfg = getattr(carrier, "config", None)
        if isinstance(cfg, dict):
            orchestration = cfg.get("orchestration") or {}
            if isinstance(orchestration, dict):
                raw = orchestration.get(key)
                if raw is not None:
                    return is_truthy_value(raw, default=False)
    return None


def frontdesk_durable_store_enabled(owner: Any, *, session: dict | None = None) -> bool:
    """Return whether the durable live-worker bridge is explicitly enabled."""
    configured = _cfg_bool(owner, session, "frontdesk_durable_store_enabled")
    if configured is not None:
        return configured
    raw = os.getenv("HERMES_FRONTDESK_DURABLE_STORE", "").strip()
    if raw:
        return is_truthy_value(raw, default=False)
    return False


def _owner_durable_store_path(owner: Any, session: dict | None = None) -> Path:
    for carrier in (session, owner):
        if not carrier:
            continue
        if isinstance(carrier, dict):
            raw = carrier.get("frontdesk_durable_store_path") or carrier.get(
                "_frontdesk_durable_store_path"
            )
            cfg = carrier.get("config")
        else:
            raw = getattr(carrier, "frontdesk_durable_store_path", None) or getattr(
                carrier, "_frontdesk_durable_store_path", None
            )
            cfg = getattr(carrier, "config", None)
        if raw:
            return Path(raw)
        if isinstance(cfg, dict):
            orchestration = cfg.get("orchestration") or {}
            if isinstance(orchestration, dict):
                raw = orchestration.get("frontdesk_durable_store_path")
                if raw:
                    return Path(raw)
    return frontdesk_durable_store_path()


def _record_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


def recover_durable_frontdesk_store(
    *,
    path: str | os.PathLike[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Recover expired durable leases without launching replacement workers."""
    from agent.frontdesk_store import FrontdeskStore

    db_path = Path(path) if path is not None else frontdesk_durable_store_path()
    store = FrontdeskStore(db_path)
    try:
        recovered = store.recover_expired_leases(now=now)
        task_ids = {job.task_id for job in recovered}
        tasks = [store.get_task(task_id) for task_id in sorted(task_ids)]
        return {
            "path": str(db_path),
            "recovered_jobs": [_record_dict(job) for job in recovered],
            "tasks": [_record_dict(task) for task in tasks if task is not None],
        }
    finally:
        store.close()


def _artifact_pointer_dict(record: Any) -> dict[str, Any]:
    pointer = {
        "id": record.id,
        "job_id": record.job_id,
        "path": record.path,
        "type": record.artifact_type,
        "import_status": record.import_status,
    }
    if record.checksum is not None:
        pointer["checksum"] = record.checksum
    if record.size is not None:
        pointer["size"] = record.size
    return pointer


def _review_status_from_adapter_result(raw: Any) -> tuple[str, dict[str, Any]]:
    from agent.frontdesk_store import REVIEW_REJECTED, REVIEW_UNSAFE
    from agent.task_registry import REVIEW_BLOCKED, REVIEW_FAILED, REVIEW_NEEDS_ITERATION, REVIEW_PASSED

    valid = {
        REVIEW_PASSED,
        REVIEW_FAILED,
        REVIEW_NEEDS_ITERATION,
        REVIEW_BLOCKED,
        REVIEW_REJECTED,
        REVIEW_UNSAFE,
    }
    if isinstance(raw, str):
        payload = {"review_status": raw}
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        raise TypeError("review adapter must return a verdict string or dictionary")
    status = payload.get("review_status") or payload.get("verdict")
    if status == "reject":
        status = REVIEW_REJECTED
    if status not in valid:
        raise ValueError(f"unknown durable frontdesk review verdict {status!r}")
    payload["review_status"] = status
    payload["verdict"] = status
    return status, payload


def _json_safe_or_string(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        return str(value)


def _default_durable_frontdesk_review(context: dict[str, Any]) -> dict[str, Any]:
    from agent.frontdesk_store import JOB_SUCCEEDED, REVIEW_REJECTED
    from agent.task_registry import REVIEW_BLOCKED, REVIEW_PASSED

    worker = context.get("worker_job") or {}
    worker_result = worker.get("result") if isinstance(worker, dict) else None
    if worker.get("state") != JOB_SUCCEEDED:
        return {
            "review_status": REVIEW_BLOCKED,
            "summary": "worker result is not a successful terminal result",
        }
    if isinstance(worker_result, dict):
        forced = worker_result.get("durable_review_verdict")
        if forced in {"passed", "failed", "needs_iteration", "blocked", "rejected", "reject", "unsafe"}:
            status = REVIEW_REJECTED if forced == "reject" else forced
            return {
                "review_status": status,
                "summary": str(worker_result.get("durable_review_summary") or f"review verdict: {status}"),
            }
    return {
        "review_status": REVIEW_PASSED,
        "summary": "worker result and artifact metadata are present for later presentation/import",
    }


def run_one_durable_frontdesk_review(
    *,
    path: str | os.PathLike[str] | None = None,
    lease_owner: str | None = None,
    lease_seconds: float = _DEFAULT_DURABLE_LEASE_SECONDS,
    now: float | None = None,
    review_adapter: Callable[[dict[str, Any]], dict[str, Any] | str] | None = None,
) -> dict[str, Any] | None:
    """Claim and complete one queued durable frontdesk reviewer job.

    This helper is deliberately explicit/default-off.  It does not start the
    live gateway, import artifacts, apply changes, or mark final output as
    presented.  Callers that want a different deterministic policy can pass a
    small ``review_adapter`` returning one of: ``passed``, ``needs_iteration``,
    ``blocked``, ``rejected``, or ``unsafe``.
    """
    from agent.frontdesk_store import FrontdeskStore, JOB_REVIEWER, JOB_SUCCEEDED, JOB_WORKER
    from agent.task_registry import REVIEW_FAILED, REVIEW_PASSED

    db_path = Path(path) if path is not None else frontdesk_durable_store_path()
    owner = lease_owner or f"frontdesk-reviewer:{os.getpid()}:{uuid.uuid4().hex}"
    store = FrontdeskStore(db_path)
    try:
        claimed = store.claim_job(
            kind=JOB_REVIEWER,
            lease_owner=owner,
            lease_seconds=lease_seconds,
            now=now,
        )
        if claimed is None:
            return None
        task = store.get_task(claimed.task_id)
        if task is None:
            raise RuntimeError(f"reviewer job {claimed.id} references missing task {claimed.task_id}")
        worker_jobs = store.list_jobs(task_id=task.id, kind=JOB_WORKER)
        worker_job = next((job for job in worker_jobs if job.state == JOB_SUCCEEDED), None)
        if worker_job is None and worker_jobs:
            worker_job = worker_jobs[-1]
        if worker_job is None:
            raise RuntimeError(f"reviewer job {claimed.id} has no linked worker job")
        artifacts = store.list_artifacts(task_id=task.id)
        worker_artifacts = [
            _artifact_pointer_dict(artifact)
            for artifact in artifacts
            if artifact.job_id == worker_job.id
        ]
        context = {
            "task": _record_dict(task),
            "reviewer_job": _record_dict(claimed),
            "worker_job": _record_dict(worker_job),
            "artifacts": worker_artifacts,
        }
        adapter = review_adapter or _default_durable_frontdesk_review
        try:
            review_status, adapter_payload = _review_status_from_adapter_result(adapter(context))
            summary = str(adapter_payload.get("summary") or f"review verdict: {review_status}")
            result = {
                "review_status": review_status,
                "verdict": review_status,
                "summary": summary,
                "task_id": task.id,
                "reviewer_job_id": claimed.id,
                "worker_job_id": worker_job.id,
                "worker_result": worker_job.result or {},
                "artifact_pointers": worker_artifacts,
                "imported": False,
                "presented": False,
            }
            for key in ("risks", "tests_run", "changed_files", "recommended_next_action"):
                if key in adapter_payload:
                    result[key] = _json_safe_or_string(adapter_payload[key])
        except Exception as exc:
            review_status = REVIEW_FAILED
            result = {
                "review_status": review_status,
                "verdict": review_status,
                "summary": "review adapter failed; task remains non-presentable pending iteration",
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "task_id": task.id,
                "reviewer_job_id": claimed.id,
                "worker_job_id": worker_job.id,
                "worker_result": worker_job.result or {},
                "artifact_pointers": worker_artifacts,
                "imported": False,
                "presented": False,
            }
        completed = store.complete_reviewer_job(
            claimed.id,
            review_status=review_status,
            lease_owner=owner,
            attempt=claimed.attempt,
            exit_status="pass" if review_status == REVIEW_PASSED else review_status,
            result=result,
        )
        completed_task = store.get_task(task.id)
        return {
            "path": str(db_path),
            "task": _record_dict(completed_task) if completed_task is not None else None,
            "reviewer_job": _record_dict(completed),
            "worker_job": _record_dict(worker_job),
            "review_result": completed.result or result,
        }
    finally:
        store.close()


class _DurableWorkerBridge:
    def __init__(
        self,
        *,
        store: Any,
        task_id: str,
        job_id: str,
        lease_owner: str,
        attempt: int,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.job_id = job_id
        self.lease_owner = lease_owner
        self.attempt = attempt

    def heartbeat(
        self,
        *,
        pid: str | int | None = None,
        session_id: str | None = None,
    ) -> None:
        self.store.heartbeat_job(
            self.job_id,
            lease_owner=self.lease_owner,
            attempt=self.attempt,
            extend_seconds=_DEFAULT_DURABLE_LEASE_SECONDS,
            pid=pid,
            session_id=session_id,
        )

    def complete(
        self,
        *,
        success: bool,
        cancelled: bool = False,
        result: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        exit_status: str | int | None = None,
    ) -> None:
        self.heartbeat()
        self.store.complete_worker_job(
            self.job_id,
            success=success,
            cancelled=cancelled,
            lease_owner=self.lease_owner,
            attempt=self.attempt,
            exit_status=exit_status,
            result=result or {},
            artifacts=artifacts or [],
        )

    def close(self) -> None:
        self.store.close()


def _artifact_records(paths: dict[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind, path in paths.items():
        record: dict[str, Any] = {"path": str(path), "type": kind}
        try:
            if path.exists():
                record["size"] = path.stat().st_size
        except OSError:
            pass
        records.append(record)
    return records


def _begin_durable_worker_bridge(
    owner: Any,
    *,
    spec: Any,
    worker_id: str | None,
    paths: dict[str, Path],
    session: dict | None = None,
) -> _DurableWorkerBridge | None:
    if not frontdesk_durable_store_enabled(owner, session=session):
        return None
    from agent.frontdesk_store import FrontdeskStore, JOB_WORKER

    db_path = _owner_durable_store_path(owner, session)
    store = None
    try:
        store = FrontdeskStore(db_path)
        task_id = spec.task_id or f"task-{uuid.uuid4().hex}"
        metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
        session_key = metadata.get("session_key") if isinstance(metadata.get("session_key"), str) else None
        source_surface = (
            metadata.get("source_surface") if isinstance(metadata.get("source_surface"), str) else None
        )
        origin = {
            "platform": source_surface or "frontdesk_live",
            "session_key": session_key,
            "source_surface": source_surface,
            "in_memory_task_id": spec.task_id,
            "in_memory_worker_id": worker_id,
            "worker_lane": getattr(spec, "lane", None),
            "frontdesk_fingerprint": metadata.get("frontdesk_fingerprint"),
            "artifact_paths": {key: str(path) for key, path in paths.items()},
        }
        durable_task, durable_job = store.create_task_with_worker_job(
            spec.goal,
            session_key=session_key,
            origin=origin,
            task_id=task_id,
        )
        del durable_task
        lease_owner = f"frontdesk-live:{os.getpid()}:{id(owner)}:{worker_id or task_id}"
        claimed = store.claim_job(
            kind=JOB_WORKER,
            job_id=durable_job.id,
            lease_owner=lease_owner,
            lease_seconds=_DEFAULT_DURABLE_LEASE_SECONDS,
            session_id=worker_id,
        )
        if claimed is None:
            raise RuntimeError(f"durable worker job was not claimable: {durable_job.id}")
        bridge = _DurableWorkerBridge(
            store=store,
            task_id=task_id,
            job_id=claimed.id,
            lease_owner=lease_owner,
            attempt=claimed.attempt,
        )
        bridge.heartbeat(session_id=worker_id)
        return bridge
    except Exception:
        if store is not None:
            store.close()
        _LOG.exception("frontdesk durable bridge initialization failed; continuing without durable mirror")
        return None


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
    durable_store_enabled: bool | None = None,
    durable_store_path: str | os.PathLike[str] | None = None,
) -> None:
    """Ensure every live frontdesk runtime has a default worker lane.

    The live gate must not expose a worker-capable control plane with an empty
    registry.  This registers one conservative Hermes oneshot-backed lane per
    runtime.  Results are also pushed back through a per-session notifier when a
    gateway/TUI surface supplies one.
    """
    from agent.orchestration_runtime import get_or_create_orchestration_runtime
    from agent.worker_lanes import ThreadWorkerLane, WorkerSpec, CancelToken

    if durable_store_enabled is not None:
        try:
            setattr(owner, "frontdesk_durable_store_enabled", bool(durable_store_enabled))
        except Exception:
            pass
    if durable_store_path is not None:
        try:
            setattr(owner, "_frontdesk_durable_store_path", str(durable_store_path))
        except Exception:
            pass

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
        durable = _begin_durable_worker_bridge(
            owner,
            spec=spec,
            worker_id=worker_id,
            paths=paths,
        )
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
            if durable is not None:
                try:
                    durable.heartbeat(pid=pid, session_id=worker_id)
                except Exception:
                    _LOG.exception("frontdesk durable bridge heartbeat failed; continuing worker lane")

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
            if durable is not None:
                try:
                    try:
                        durable.complete(
                            success=False,
                            cancelled=status == "cancelled",
                            exit_status=status,
                            result={
                                "status": status,
                                "summary": summary,
                                "error": str(exc) or type(exc).__name__,
                                "in_memory_task_id": spec.task_id,
                                "in_memory_worker_id": worker_id,
                            },
                            artifacts=_artifact_records(paths),
                        )
                    except Exception:
                        _LOG.exception(
                            "frontdesk durable bridge failure completion failed; preserving original worker error"
                        )
                finally:
                    durable.close()
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
        if durable is not None:
            try:
                try:
                    durable.complete(
                        success=True,
                        exit_status=0,
                        result={
                            "status": "succeeded",
                            "summary": result,
                            "in_memory_task_id": spec.task_id,
                            "in_memory_worker_id": worker_id,
                        },
                        artifacts=_artifact_records(paths),
                    )
                except Exception:
                    _LOG.exception(
                        "frontdesk durable bridge success completion failed; continuing in-memory worker lane"
                    )
            finally:
                durable.close()
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
