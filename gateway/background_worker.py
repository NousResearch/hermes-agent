"""External durable background worker entrypoint for gateway jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from agent.skill_commands import build_preloaded_skills_prompt
from gateway.background_jobs import BackgroundJobStore, ExternalApprovalBridge
from gateway.session import SessionSource
from run_agent import AIAgent
from tools.approval import (
    reset_current_admin_policy,
    reset_current_session_key,
    reset_external_approval_backend,
    set_current_admin_policy,
    set_current_session_key,
    set_external_approval_backend,
)


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
    from agent.smart_model_routing import resolve_turn_route
    from gateway.run import (
        GatewayRunner,
        _load_gateway_config,
        _platform_config_key,
        _resolve_gateway_model,
        _resolve_runtime_agent_kwargs,
    )
    from hermes_cli.tools_config import _get_platform_tools

    source = SessionSource.from_dict(job["source"])
    runtime_kwargs = _resolve_runtime_agent_kwargs()
    if not runtime_kwargs.get("api_key"):
        raise RuntimeError("no provider credentials configured")

    user_config = _load_gateway_config()
    model = _resolve_gateway_model(user_config)
    platform_key = _platform_config_key(source.platform)
    enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
    provider_routing = GatewayRunner._load_provider_routing()
    fallback_model = GatewayRunner._load_fallback_model()
    reasoning_config = GatewayRunner._load_reasoning_config()
    smart_model_routing = GatewayRunner._load_smart_model_routing()

    primary = {
        "model": model,
        "api_key": runtime_kwargs.get("api_key"),
        "base_url": runtime_kwargs.get("base_url"),
        "provider": runtime_kwargs.get("provider"),
        "api_mode": runtime_kwargs.get("api_mode"),
        "command": runtime_kwargs.get("command"),
        "args": list(runtime_kwargs.get("args") or []),
        "credential_pool": runtime_kwargs.get("credential_pool"),
    }
    turn_route = resolve_turn_route(job["prompt"], smart_model_routing, primary)

    combined_ephemeral = str(job.get("context_prompt") or "").strip()
    loaded_skill_names: list[str] = []
    missing_skills: list[str] = []
    if job.get("preloaded_skills"):
        skill_prompt, loaded_skill_names, missing_skills = build_preloaded_skills_prompt(
            list(job.get("preloaded_skills") or []),
            task_id=job["task_id"],
        )
        if skill_prompt:
            combined_ephemeral = (
                f"{skill_prompt}\n\n{combined_ephemeral}".strip()
                if combined_ephemeral
                else skill_prompt
            )

    platform_system_prompt = ""
    platform_cfg = (user_config.get("gateway", {}) or {}).get("platforms", {}) if isinstance(user_config, dict) else {}
    raw_platform_cfg = platform_cfg.get(source.platform.value, {}) if isinstance(platform_cfg, dict) else {}
    extra = raw_platform_cfg.get("extra") if isinstance(raw_platform_cfg, dict) else None
    if isinstance(extra, dict):
        platform_system_prompt = str(extra.get("system_prompt") or "").strip()
    if platform_system_prompt:
        combined_ephemeral = (combined_ephemeral + "\n\n" + platform_system_prompt).strip()

    return {
        "source": source,
        "enabled_toolsets": enabled_toolsets,
        "provider_routing": provider_routing,
        "fallback_model": fallback_model,
        "reasoning_config": reasoning_config,
        "turn_route": turn_route,
        "combined_ephemeral": combined_ephemeral or None,
        "loaded_skills": loaded_skill_names,
        "missing_skills": missing_skills,
        "max_iterations": int(os.getenv("HERMES_MAX_ITERATIONS", "90")),
    }


def _run_btw_job(*, job: dict, runtime: dict, source: SessionSource):
    provider_routing = runtime["provider_routing"]
    turn_route = runtime["turn_route"]
    agent = AIAgent(
        model=turn_route["model"],
        **turn_route["runtime"],
        max_iterations=min(int(runtime["max_iterations"]), 8),
        quiet_mode=True,
        verbose_logging=False,
        enabled_toolsets=[],
        ephemeral_system_prompt=runtime["combined_ephemeral"],
        reasoning_config=runtime["reasoning_config"],
        providers_allowed=provider_routing.get("only"),
        providers_ignored=provider_routing.get("ignore"),
        providers_order=provider_routing.get("order"),
        provider_sort=provider_routing.get("sort"),
        provider_require_parameters=provider_routing.get("require_parameters", False),
        provider_data_collection=provider_routing.get("data_collection"),
        session_id=job["task_id"],
        platform=source.platform.value if source.platform else "unknown",
        user_id=source.user_id,
        fallback_model=runtime["fallback_model"],
        session_db=None,
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
    approval_bridge = ExternalApprovalBridge(
        store=store,
        task_id=task_id,
        session_key=str(job.get("session_key") or task_id),
        source=source,
    )

    approval_session_token = set_current_session_key(str(job.get("session_key") or task_id))
    approval_admin_tokens = set_current_admin_policy(
        list(job.get("admin_user_ids") or []),
        job.get("is_admin_user"),
    )
    approval_bridge_token = set_external_approval_backend(approval_bridge)
    try:
        provider_routing = runtime["provider_routing"]
        turn_route = runtime["turn_route"]
        if str(job.get("kind") or "").strip().lower() == "btw":
            result = _run_btw_job(job=job, runtime=runtime, source=source)
        else:
            agent = AIAgent(
                model=turn_route["model"],
                **turn_route["runtime"],
                max_iterations=runtime["max_iterations"],
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=runtime["enabled_toolsets"],
                ephemeral_system_prompt=runtime["combined_ephemeral"],
                reasoning_config=runtime["reasoning_config"],
                providers_allowed=provider_routing.get("only"),
                providers_ignored=provider_routing.get("ignore"),
                providers_order=provider_routing.get("order"),
                provider_sort=provider_routing.get("sort"),
                provider_require_parameters=provider_routing.get("require_parameters", False),
                provider_data_collection=provider_routing.get("data_collection"),
                session_id=task_id,
                platform=source.platform.value if source.platform else "unknown",
                user_id=source.user_id,
                fallback_model=runtime["fallback_model"],
            )
            result = agent.run_conversation(
                user_message=str(job.get("prompt") or ""),
                conversation_history=list(job.get("conversation_history") or []) or None,
                task_id=task_id,
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
        reset_external_approval_backend(approval_bridge_token)
        reset_current_admin_policy(approval_admin_tokens)
        reset_current_session_key(approval_session_token)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return run_background_job(args.task_id)


if __name__ == "__main__":
    raise SystemExit(main())
