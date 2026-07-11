"""Provider-neutral whole-agent runtime attempts used by the main loop."""

from __future__ import annotations

import os
import platform
import re
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent.claude_cli_boundary import (
    attest_claude_max_auth,
    create_exact_env_cli_wrapper,
    invalidate_claude_auth_attestation,
)
from agent.claude_agent_runtime import ClaudeProjection, RuntimeFailure
from agent.claude_sdk_session import (
    ClaudeAgentSdkSession,
    build_claude_agent_options,
    load_claude_agent_sdk,
)
from agent.error_classifier import FailoverReason, classify_api_error
from agent.claude_subscription_env import build_claude_subscription_env
from agent.claude_workspace_terminal import build_workspace_seatbelt_profile
from agent.claude_workspace_files import WorkspaceFileBroker
from hermes_constants import get_hermes_home, get_host_user_home


_runtime_events_logger = logging.getLogger("hermes.runtime_events")


def _emit_runtime_event(agent: Any, event: str, **fields: Any) -> None:
    """Log a structured external-runtime event and surface significant state."""

    payload = {
        "event": event,
        "ts": time.time(),
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "runtime": str(getattr(agent, "runtime", "hermes") or "hermes"),
        **fields,
    }
    _runtime_events_logger.info(json.dumps(payload, default=str, sort_keys=True))
    status = {
        "runtime_attempt_failure": f"Runtime attempt failed: {fields.get('reason', 'unknown')}",
        "runtime_circuit_open": "Runtime circuit opened; trying fallback.",
        "runtime_fallback_activated": "Runtime fallback activated.",
    }.get(event)
    if status:
        try:
            agent._emit_status(status)
        except Exception:
            pass


def _claude_effort(agent: Any) -> str | None:
    try:
        from agent.routing_contract import active_reasoning_effort

        effort = str(active_reasoning_effort(agent) or "").lower()
    except Exception:
        return None
    if effort in {"low", "medium", "high", "xhigh", "max"}:
        return effort
    return None


def _claude_workspace(agent: Any) -> Path:
    configured = os.getenv("HERMES_KANBAN_WORKSPACE") or getattr(
        agent, "session_cwd", None
    )
    if configured:
        return Path(configured).expanduser().resolve()
    from agent.runtime_cwd import resolve_agent_cwd

    return Path(resolve_agent_cwd()).expanduser().resolve()


def _claude_project_state_dir(host_home: Path, workspace: Path) -> Path:
    project_key = re.sub(r"[^A-Za-z0-9]", "-", str(workspace))
    path = host_home / ".claude" / "projects" / project_key
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _load_persisted_claude_session_id(agent: Any) -> str | None:
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        return None
    try:
        row = db.get_session(session_id) or {}
        raw = row.get("model_config")
        config = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        value = str(config.get("claude_session_id") or "").strip()
        return value or None
    except Exception:
        return None


def _persist_claude_session_id(agent: Any, claude_session_id: str | None) -> None:
    if not claude_session_id:
        return
    agent._claude_resume_session_id = claude_session_id
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id or not hasattr(db, "update_session_meta"):
        return
    try:
        row = db.get_session(session_id) or {}
        raw = row.get("model_config")
        config = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        config = dict(config or {})
        config["claude_session_id"] = claude_session_id
        db.update_session_meta(
            session_id,
            json.dumps(config, sort_keys=True),
            str(getattr(agent, "model", "") or "") or None,
        )
    except Exception:
        pass


def prepare_claude_agent_sdk_runtime(agent: Any) -> RuntimeFailure | None:
    """Attest and sandbox the subscription runtime before circuit lookup."""

    if getattr(agent, "_claude_runtime_context", None) is not None:
        return None
    kanban_task_id = os.getenv("HERMES_KANBAN_TASK", "").strip() or None
    if not kanban_task_id:
        return RuntimeFailure(
            FailoverReason.auth_permanent,
            "Claude Agent SDK runtime currently supports Kanban workers only",
        )
    try:
        sdk = load_claude_agent_sdk()
        host_home = get_host_user_home()
        if not host_home:
            raise RuntimeError(
                "Claude subscription runtime could not resolve its HOME boundary"
            )
        host_home = Path(host_home).expanduser().resolve()
        workspace = _claude_workspace(agent)
        if platform.system() != "Darwin":
            raise RuntimeError(
                "Claude subscription worker filesystem isolation is supported on macOS only"
            )
        sdk_root = Path(sdk.__file__).resolve().parent
        cli_name = "claude.exe" if os.name == "nt" else "claude"
        bundled_cli = sdk_root / "_bundled" / cli_name
        exact_env = build_claude_subscription_env(os.environ, host_home=host_home)
        worker_tmp = workspace / ".hermes-claude-runtime" / "tmp"
        worker_tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
        worker_tmp.chmod(0o700)
        exact_env["TMPDIR"] = str(worker_tmp)
        sandbox_profile = build_workspace_seatbelt_profile(
            workspace=workspace,
            host_home=host_home,
            allow_network=True,
            readable_roots=[sdk_root],
            restrict_reads=False,
            control_write_paths=[
                host_home / ".claude.json",
                host_home / ".claude.json.lock",
            ],
            control_write_roots=[_claude_project_state_dir(host_home, workspace)],
        )
        cli_wrapper = create_exact_env_cli_wrapper(
            bundled_cli,
            exact_env,
            get_hermes_home() / "cache" / "claude-agent-sdk" / "launchers",
            sandbox_profile=sandbox_profile,
        )
        attestation = attest_claude_max_auth(cli_wrapper)
    except Exception as exc:
        classified = classify_api_error(
            exc,
            provider=str(getattr(agent, "provider", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
        )
        reason = (
            FailoverReason.auth_permanent
            if isinstance(exc, RuntimeError) and "attestation" in str(exc).lower()
            else classified.reason
        )
        return RuntimeFailure(reason, classified.message or str(exc))
    agent._claude_max_attestation = attestation
    agent._claude_cli_wrapper = str(cli_wrapper)
    agent._claude_runtime_context = {
        "sdk": sdk,
        "host_home": host_home,
        "workspace": workspace,
        "cli_wrapper": cli_wrapper,
        "kanban_task_id": kanban_task_id,
    }
    return None


def run_claude_agent_sdk_attempt(
    agent: Any,
    *,
    user_message: str,
    effective_task_id: str,
) -> ClaudeProjection:
    """Run one resumable Claude SDK attempt using the active runtime target."""

    preflight_failure = prepare_claude_agent_sdk_runtime(agent)
    if preflight_failure is not None:
        return ClaudeProjection(failure=preflight_failure)
    context = agent._claude_runtime_context
    sdk = context["sdk"]
    host_home = context["host_home"]
    workspace = context["workspace"]
    cli_wrapper = context["cli_wrapper"]
    kanban_task_id = context["kanban_task_id"]
    key = (
        str(getattr(agent, "provider", "") or ""),
        str(getattr(agent, "model", "") or ""),
    )
    sessions = getattr(agent, "_claude_sdk_sessions", None)
    if sessions is None:
        sessions = {}
        agent._claude_sdk_sessions = sessions

    if key not in sessions:
        from model_tools import handle_function_call
        file_broker = WorkspaceFileBroker(workspace)

        def _options(resume: str | None) -> Any:
            return build_claude_agent_options(
                sdk=sdk,
                model=agent.model,
                system_prompt=str(getattr(agent, "_cached_system_prompt", "") or ""),
                workspace=workspace,
                host_home=host_home,
                profile_home=host_home,
                inherited_env=os.environ,
                tool_definitions=list(getattr(agent, "tools", None) or []),
                dispatch=handle_function_call,
                effective_task_id=effective_task_id,
                kanban_task_id=kanban_task_id,
                max_turns=getattr(agent, "max_iterations", None),
                resume=resume,
                effort=_claude_effort(agent),
                cli_path=cli_wrapper,
                file_broker=file_broker,
                capability_mode=str(
                    getattr(agent, "_claude_capability_mode", "worker") or "worker"
                ),
                auxiliary_tool_names=tuple(
                    getattr(agent, "_claude_auxiliary_tool_names", ()) or ()
                ),
            )

        sessions[key] = ClaudeAgentSdkSession(
            sdk=sdk,
            options_factory=_options,
            stream_delta_callback=getattr(agent, "stream_delta_callback", None),
            tool_progress_callback=getattr(agent, "tool_progress_callback", None),
            resources=[file_broker],
            initial_session_id=(
                getattr(agent, "_claude_resume_session_id", None)
                or _load_persisted_claude_session_id(agent)
            ),
        )

    def _clear_auth_state() -> None:
        invalidate_claude_auth_attestation(cli_wrapper)
        agent._claude_max_attestation = None
        agent._claude_runtime_context = None
        failed_session = sessions.pop(key, None)
        if failed_session is not None:
            failed_session.close()

    try:
        projection = sessions[key].run_turn(user_message)
        _persist_claude_session_id(agent, projection.session_id)
        if getattr(agent, "stream_delta_callback", None) is not None:
            agent.stream_delta_callback(None)
        if projection.failure and projection.failure.reason in {
            FailoverReason.auth,
            FailoverReason.auth_permanent,
        }:
            _clear_auth_state()
        return projection
    except Exception as exc:
        classified = classify_api_error(
            exc,
            provider=str(getattr(agent, "provider", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
        )
        if classified.reason in {FailoverReason.auth, FailoverReason.auth_permanent}:
            _clear_auth_state()
        return ClaudeProjection(
            failure=RuntimeFailure(classified.reason, classified.message or str(exc))
        )


def record_claude_subscription_usage(agent: Any, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Record subscription usage without inventing a pay-as-you-go cost."""

    attestation = getattr(agent, "_claude_max_attestation", None)
    included = bool(attestation and getattr(attestation, "included_usage", False))
    raw = dict(usage or {})
    _emit_runtime_event(
        agent, "runtime_billing_mode", billing_mode="subscription_included"
    )

    def _int(name: str) -> int:
        try:
            return max(int(raw.get(name) or 0), 0)
        except (TypeError, ValueError):
            return 0

    input_tokens = _int("input_tokens")
    output_tokens = _int("output_tokens")
    cache_read = _int("cache_read_input_tokens") or _int("cache_read_tokens")
    cache_write = _int("cache_creation_input_tokens") or _int("cache_write_tokens")
    from agent.usage_pricing import CanonicalUsage

    canonical = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw_usage=raw,
    )
    prompt_tokens = canonical.prompt_tokens
    total = canonical.total_tokens
    agent.session_api_calls += 1
    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += output_tokens
    agent.session_total_tokens += total
    agent.session_input_tokens += input_tokens
    agent.session_output_tokens += output_tokens
    agent.session_cache_read_tokens += cache_read
    agent.session_cache_write_tokens += cache_write
    cost_status = "included" if included else "unknown"
    cost_source = "claude_max_subscription" if included else "unattested"
    agent.session_cost_status = cost_status
    agent.session_cost_source = cost_source
    if getattr(agent, "_session_db", None) is not None and getattr(
        agent, "session_id", None
    ):
        try:
            if not getattr(agent, "_session_db_created", False):
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                estimated_cost_usd=None,
                cost_status=cost_status,
                cost_source=cost_source,
                billing_provider="anthropic",
                billing_mode="subscription_included" if included else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception:
            # Accounting must never turn a successful worker result into a
            # failed task; the in-memory counters still remain authoritative.
            pass
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "estimated_cost_usd": None,
        "cost_status": cost_status,
        "cost_source": cost_source,
    }


__all__ = [
    "_emit_runtime_event",
    "prepare_claude_agent_sdk_runtime",
    "record_claude_subscription_usage",
    "run_claude_agent_sdk_attempt",
]
