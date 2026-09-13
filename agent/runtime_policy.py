"""Runtime stop consumption and session binding for authoritative runs."""
import logging
from typing import Any
logger = logging.getLogger(__name__)
INVALID_STOP_REASON = "invalid_stop_directive"


def is_authoritative(policy: Any) -> bool:
    """True for a policy id that actually confers authority.

    A truthiness test is not enough: callers hold anything from ``None`` to a
    test double, and ``"observer"`` is a real policy name that deliberately
    carries no authority.
    """
    return isinstance(policy, str) and bool(policy) and policy != "observer"

def _fail_closed_stop(agent, policy: Any) -> None:
    """Stop the run when a trusted directive cannot be trusted."""
    if getattr(agent, "_runtime_stop_reason", None) is not None:
        return
    agent._runtime_stop_reason = INVALID_STOP_REASON
    agent._runtime_terminal_outcome = {
        "status": "failure",
        "reason": INVALID_STOP_REASON,
        "policy": str(policy),
    }
    logger.warning(
        "unreadable trusted stop directive under policy %s — stopping the run",
        policy,
        exc_info=True,
    )

def apply_mcp_runtime_stop(agent) -> None:
    """Consume the stop directive left in the context by the last MCP call.

    Under an active runtime policy this is an admission boundary, so a
    directive that cannot be read — the import fails, it is not a mapping,
    it carries no ``reason`` — must stop the run rather than be dropped.
    Swallowing it lets the remaining calls in the assistant batch execute
    after the policy already said stop, which is the fail-open path this
    hook exists to close. With no policy in force there is no authority to
    enforce and the directive is observer data, so a failure is logged and
    the run continues.
    """
    policy = getattr(agent, "runtime_policy", None)
    try:
        from tools.mcp_tool_handlers import consume_mcp_runtime_stop

        directive = consume_mcp_runtime_stop()
        if not directive:
            return
        if getattr(agent, "_runtime_stop_reason", None) is not None:
            return
        reason = directive["reason"]
        outcome = dict(directive)
    except Exception:
        if is_authoritative(policy):
            _fail_closed_stop(agent, policy)
        else:
            logger.debug("mcp runtime stop directive dropped", exc_info=True)
        return

    agent._runtime_stop_reason = reason
    agent._runtime_terminal_outcome = outcome

def _rebind_authoritative_run(agent) -> None:
    """Re-point an authoritative run lease at the rotated transcript session.

    Best-effort by design: authority resolves primarily by the run's immutable
    fire id, which this rotation never touches. The reverse index refreshed
    here is only the fallback for callers that know a session id and nothing
    else, so a failure here cannot open the authority gate.
    """
    run_id = str(getattr(agent, "runtime_task_id", "") or "")
    if not run_id:
        return
    try:
        from hermes_cli.plugins_authority import bind_authoritative_run_session

        bind_authoritative_run_session(run_id, str(getattr(agent, "session_id", "") or ""))
    except Exception:
        logger.debug("authoritative run rebind skipped", exc_info=True)
