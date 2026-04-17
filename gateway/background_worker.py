"""External durable background worker entrypoint for gateway jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from gateway.agent_execution_service import (
    create_gateway_agent,
    run_gateway_approved_conversation,
)
from gateway.agent_runtime import build_gateway_agent_runtime
from gateway.background_jobs import BackgroundJobStore, ExternalApprovalBridge
from gateway.session import SessionSource


logger = logging.getLogger(__name__)


def _heartbeat_interval_seconds() -> float:
    raw = os.getenv("HERMES_BACKGROUND_HEARTBEAT_SECONDS", "8")
    try:
        return max(float(raw), 1.0)
    except Exception:
        return 8.0


def _start_job_heartbeat(store: BackgroundJobStore, task_id: str) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _run() -> None:
        try:
            store.touch_job_heartbeat(task_id)
        except Exception:
            logger.debug("Initial background heartbeat failed for %s", task_id, exc_info=True)

        interval = _heartbeat_interval_seconds()
        while not stop_event.wait(interval):
            try:
                store.touch_job_heartbeat(task_id)
            except Exception:
                logger.debug("Background heartbeat failed for %s", task_id, exc_info=True)

    thread = threading.Thread(
        target=_run,
        name=f"hermes-bg-heartbeat-{task_id}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one durable Hermes background job")
    parser.add_argument("--task-id", required=True, help="Background job task ID")
    return parser.parse_args(argv)


def _build_agent_runtime(job: dict):
    source = SessionSource.from_dict(job["source"])
    runtime_spec = build_gateway_agent_runtime(
        source=source,
        user_message=str(job.get("prompt") or ""),
        context_prompt=str(job.get("context_prompt") or ""),
        preloaded_skills=list(job.get("preloaded_skills") or []),
        skill_task_id=str(job.get("task_id") or ""),
    )
    return {
        "source": source,
        "runtime_spec": runtime_spec,
        "loaded_skills": runtime_spec.loaded_skills,
        "missing_skills": runtime_spec.missing_skills,
    }


def _run_btw_job(*, job: dict, runtime_spec, source: SessionSource):
    agent = create_gateway_agent(
        runtime_spec=runtime_spec,
        session_id=job["task_id"],
        source=source,
        max_iterations=min(int(runtime_spec.max_iterations), 8),
        enabled_toolsets=[],
        quiet_mode=True,
        verbose_logging=False,
        skip_memory=True,
        skip_context_files=True,
        persist_session=False,
    )
    btw_prompt = (
        "[Ephemeral /btw side question. Answer using the conversation "
        "context. No tools available. Be direct and concise.]\n\n"
        + str(job.get("prompt") or "")
    )
    return agent.run_conversation(
        user_message=btw_prompt,
        conversation_history=list(job.get("conversation_history") or []) or None,
        task_id=job["task_id"],
    )


def run_background_job(task_id: str) -> int:
    store = BackgroundJobStore()
    job = store.get_job(task_id)
    if not job:
        logger.error("Background job %s not found", task_id)
        return 2
    if str(job.get("status") or "").strip().lower() in {"completed", "failed", "cancelled"}:
        logger.info("Background job %s already terminal (%s)", task_id, job.get("status"))
        return 0

    os.environ["HERMES_EXEC_ASK"] = "1"
    store.mark_job_running(task_id, launcher_pid=os.getpid())
    heartbeat_stop, heartbeat_thread = _start_job_heartbeat(store, task_id)

    def _handle_signal(signum, _frame):
        try:
            store.mark_job_cancelled(task_id, reason=f"terminated by signal {signum}")
        finally:
            raise SystemExit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    runtime = _build_agent_runtime(job)
    store.update_job_skills(
        task_id,
        loaded_skills=runtime["loaded_skills"],
        missing_skills=runtime["missing_skills"],
    )
    source = runtime["source"]
    runtime_spec = runtime["runtime_spec"]
    approval_bridge = ExternalApprovalBridge(
        store=store,
        task_id=task_id,
        session_key=str(job.get("session_key") or task_id),
        source=source,
    )

    try:
        if str(job.get("kind") or "").strip().lower() == "btw":
            result = _run_btw_job(job=job, runtime_spec=runtime_spec, source=source)
        else:
            agent = create_gateway_agent(
                runtime_spec=runtime_spec,
                session_id=task_id,
                source=source,
                max_iterations=runtime_spec.max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=runtime_spec.enabled_toolsets,
            )
            result = run_gateway_approved_conversation(
                agent=agent,
                message=str(job.get("prompt") or ""),
                pending_model_note=None,
                conversation_history=list(job.get("conversation_history") or []) or None,
                task_id=task_id,
                session_key=str(job.get("session_key") or task_id),
                admin_user_ids=list(job.get("admin_user_ids") or []),
                is_admin_user=job.get("is_admin_user"),
                status_adapter=None,
                status_chat_id=getattr(source, "chat_id", None),
                status_thread_metadata={"thread_id": source.thread_id} if getattr(source, "thread_id", None) else None,
                loop_for_step=None,
                logger=logger,
                admin_only_message_builder=lambda action: None,
                external_backend=approval_bridge,
            )
        response = ""
        if result:
            response = str(result.get("final_response") or "")
            if not response and result.get("error"):
                raise RuntimeError(str(result.get("error")))
        store.mark_job_completed(task_id, raw_response=response)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Background worker failed for %s", task_id)
        store.mark_job_failed(task_id, error=str(exc))
        return 1
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return run_background_job(args.task_id)


if __name__ == "__main__":
    raise SystemExit(main())
