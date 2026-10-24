"""``hermes doctor --turn`` — opt-in end-to-end agent turn on the configured main model.

The static and ``--live`` checks prove a provider answers ``/models`` with your key; they never send
a completion. A route can pass all of them and still fail every chat: a model slug the provider does
not serve, an entitlement the account lacks, the wrong ``api_mode``, or a model that cannot call tools.
This check runs ONE real turn through the same runtime resolution ``hermes -z`` uses, with fallback
off (a healthy fallback would hide a dead primary), only the ``todo`` tool exposed, and no session
row written. One short turn is a few small model requests (about 2-4, capped at 6 iterations), so it
runs only when the user passes ``--turn``.
"""

from __future__ import annotations

import logging
import time

from hermes_cli.doctor import _section
from hermes_cli.doctor_report import check_fail, check_ok, check_warn

logger = logging.getLogger(__name__)

_SENTINEL = "HERMES_DOCTOR_OK"
_PROMPT = (
    "This is an automated Hermes health check. Call the todo_list tool exactly once to add one "
    "pending item whose content is 'doctor check'. After the tool result comes back, reply with "
    f"exactly {_SENTINEL} and nothing else."
)


def _called_todo(messages: list) -> bool:
    """True when the transcript holds an executed todo_list result. Judged on the tool RESULT row:
    when tools are deferred the assistant reaches todo_list through the ``tool_call`` dispatcher, so
    the assistant-side call name is ``tool_call`` while the result row names the real tool."""
    return any(isinstance(m, dict) and m.get("role") == "tool"
               and "todo_list" in (m.get("name"), m.get("tool_name"))
               and "doctor check" in str(m.get("content") or "") for m in messages)


def _build_agent():
    """Primary route only: same resolution as a ``hermes -z`` turn minus the fallback chain."""
    from hermes_cli.config import load_config
    from hermes_cli.oneshot import _resolve_model_and_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import resolve_reasoning_config
    from run_agent import AIAgent

    cfg = load_config()
    choice = _resolve_model_and_provider(cfg, None, None)
    runtime = resolve_runtime_provider(requested=choice.provider, target_model=choice.model or None,
                                       explicit_base_url=choice.base_url, explicit_api_key=choice.api_key)
    agent = AIAgent(
        api_key=runtime.get("api_key"), base_url=runtime.get("base_url"), provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"), api_mode=runtime.get("api_mode"),
        model=choice.model, enabled_toolsets=["todo"], quiet_mode=True, platform="cli",
        credential_pool=runtime.get("credential_pool"), request_overrides=runtime.get("request_overrides"),
        reasoning_config=resolve_reasoning_config(cfg, choice.model), max_iterations=6,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
    )
    # A health check is not a conversation: no state.db row, no MCP refresh, no streamed output.
    agent._persist_disabled = agent._skip_mcp_refresh = agent.suppress_status_output = True
    return agent


def run_turn_check(issues: list[str]) -> str:
    """Run the turn and report one row; returns ``pass`` / ``warn`` / ``fail``."""
    _section("End-to-End Turn (opt-in, a few real model requests)")
    agent = None
    started = time.monotonic()
    try:
        agent = _build_agent()
        label = f"Turn: {agent.provider or 'auto'}/{agent.model} ({agent.api_mode})"
        result = agent.run_conversation(_PROMPT)
    except Exception as exc:  # any failure to build or run the turn IS the diagnosis
        logger.debug("doctor --turn could not run", exc_info=True)
        check_fail("Turn", f"(could not run: {exc})")
        issues.append(f"End-to-end turn failed before reaching the model: {exc}")
        return "fail"
    finally:
        if agent is not None:
            agent.close()
    elapsed = time.monotonic() - started
    usage = f"{elapsed:.1f}s, {agent.session_input_tokens} in / {agent.session_output_tokens} out tokens"
    # Same outcome rules as a ``hermes -z`` exit code: a failed, partial or interrupted turn is a failure
    # even when it left explanatory text in final_response.
    from hermes_cli.oneshot import _oneshot_exit_code
    if _oneshot_exit_code(result.get("final_response"), result) != 0:
        error = str(result.get("error") or result.get("final_response") or "no response").strip()
        check_fail(label, f"({error[:300]})")
        issues.append(f"End-to-end turn on the main model failed: {error[:200]}")
        return "fail"
    if not _called_todo(result.get("messages") or []):
        check_warn(label, f"(answered in {usage}, but never called the tool — tool calling looks unreliable "
                          "on this model/route)")
        return "warn"
    if _SENTINEL not in result["final_response"]:
        check_warn(label, f"(tool round trip ok in {usage}, but the final reply ignored the instruction)")
        return "warn"
    check_ok(label, f"(tool round trip ok, {usage})")
    return "pass"


def maybe_run_turn_check(args, issues: list[str]):
    """No-op (None) unless ``--turn`` was passed (``run_doctor`` wraps the call in ``warn_on_error``)."""
    if not getattr(args, "turn", False):
        return None
    return run_turn_check(issues)
