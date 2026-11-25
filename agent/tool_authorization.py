"""Hermes's authorization of one agent-loop tool call, separate from running it.

Policy order (unchanged): hosted refusal -> tool-scope block -> ``pre_tool_call`` plugin hooks (block / approve /
modify) -> the pruned-argument guard -> loop guardrails. ``authorize_call`` runs it once and either returns the
result the model must see (the call is blocked, its terminal ``post_tool_call`` already emitted) or ``None`` (the call
may run, with ``ref.args`` updated by any ``modify`` hook).

The agent loop runs it inside the execution callback handed to ``tool_execution`` middleware; a middleware that runs
the call somewhere else runs it through ``hermes_cli.middleware.authorize_tool_call`` instead.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from agent.compression_marker import _COMPRESSION_MARKER_PREFIX
from agent.inline_tool_executors import tool_hook_ids
from agent.tool_dispatch_helpers import _context_pruned_argument_paths

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent.tool_executor import _ConcurrentToolAuthorizationGate, _ManagedToolResult, _ToolCallRef

# A tool-body check authorize_tool_call already ran for one call ({"call_id", "terminal": (guard_key, approval)} or
# {"call_id", "edit": args}), handed to the local tool if the call is then dispatched here, so nobody is asked twice.
# Single use; bound for the life of one agent-loop tool call.
BODY_DECISION: ContextVar[dict | None] = ContextVar("hermes_authorized_tool_body", default=None)
_EDIT_TOOLS = frozenset({"write_file", "patch"})

PRUNED_TOOL_ARGUMENTS_ERROR = "suspected_pruned_tool_arguments"
PRUNED_TOOL_ARGUMENTS_MESSAGE = (
    "Tool was not executed because effect-capable arguments contain a Hermes context-compression artifact. "
    "Recover the exact content from its durable source or re-read it, then issue a complete new call; "
    "do not retry these arguments. To remove a marker that already landed in a file, match it by its "
    f"{_COMPRESSION_MARKER_PREFIX.strip('⟪:')} prefix (e.g. a terminal sed on that line) instead of quoting the full marker."
)


def blocked_tool_result(agent, ref: _ToolCallRef, *, block_body: dict[str, Any] | None, block_error_type: str,
                        guardrail_decision) -> str:
    """Synthesize the result for a call blocked by scope/plugin/pruned-args (``block_body``, the
    JSON the model sees) or by guardrail policy (``guardrail_decision``) and emit its terminal post_tool_call."""
    if block_body is not None:
        result = json.dumps(block_body, ensure_ascii=False)
        error_type, error_message = block_error_type, block_body.get("message") or block_body["error"]
    else:
        result = agent._guardrail_block_result(guardrail_decision)
        error_type = "guardrail_block"
        error_message = getattr(guardrail_decision, "message", None) or "Tool blocked by guardrail policy"
    ref.emit_post(agent, result, status="blocked", error_type=error_type, error_message=error_message)
    return result


def pre_tool_block(agent, ref: _ToolCallRef):
    """Run ``pre_tool_call`` plugin hooks; returns ``(block_message, final_args)`` with any
    hook-modified args applied. Hook failures never block."""
    try:
        from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

        block_msg, modified_args = _dispatch_pre_tool_call_hooks(
            ref.name,
            ref.args,
            **tool_hook_ids(agent, ref.task_id, ref.call_id),
            middleware_trace=list(ref.trace),
        )
        return block_msg, (ref.args if modified_args is None else modified_args)
    except Exception:
        return None, ref.args


def authorize_call(
    agent,
    state: _ManagedToolResult,
    ref: _ToolCallRef,
    *,
    scope_block: str | None,
    authorization_gate: _ConcurrentToolAuthorizationGate | None,
) -> str | None:
    """Hermes policy for one call (scope -> plugin pre-hooks -> pruned-arg check -> guardrails).

    Returns the blocked result (``state.blocked`` set, terminal ``post_tool_call`` emitted) or ``None`` when the call
    may run. Plugin ``modify`` hooks may rewrite ``ref.args`` (mirrored into ``state.args``).
    """
    # Deny-by-default for a hosted scope, before plugin hooks, guardrails or any executor: every
    # sequential, concurrent and batched call reaches its executor (inline, memory provider, context
    # engine, registry) only through here.
    from tools.hosted_tool_policy import hosted_tool_refusal
    block_message, block_error_type = hosted_tool_refusal(ref.name), "hosted_tool_refused"
    if block_message is None:
        block_message, block_error_type = scope_block, "tool_scope_block"
    if block_message is None:
        block_error_type = "plugin_block"
        resolve = lambda: pre_tool_block(agent, ref)
        block_message, ref.args = resolve() if authorization_gate is None else authorization_gate.run(resolve)
        state.args = ref.args
    block_body = None if block_message is None else {"error": block_message}

    # Checked once, after plugin modify hooks (which may replace arguments) and
    # before guardrails or real dispatch: a copied compression marker in an
    # effect-capable argument must never reach the tool.
    if block_body is None:
        pruned_paths = _context_pruned_argument_paths(ref.name, ref.args)
        if pruned_paths:
            block_body = {
                "error": PRUNED_TOOL_ARGUMENTS_ERROR,
                "message": PRUNED_TOOL_ARGUMENTS_MESSAGE,
                "argument_paths": pruned_paths,
            }
            block_error_type = PRUNED_TOOL_ARGUMENTS_ERROR

    guardrail_decision = None
    if block_body is None:
        guardrail_decision = agent._tool_guardrails.before_call(ref.name, ref.args)
        if guardrail_decision.allows_execution:
            guardrail_decision = None

    if block_body is None and guardrail_decision is None:
        return None
    state.blocked = True
    return blocked_tool_result(
        agent, ref, block_body=block_body, block_error_type=block_error_type, guardrail_decision=guardrail_decision,
    )


def refuse_call(agent, state: _ManagedToolResult, ref: _ToolCallRef, result: str, error_type: str) -> str:
    """Report a call Hermes refused (terminal ``post_tool_call``, status blocked) and mark it blocked."""
    ref.emit_post(agent, result, status="blocked", error_type=error_type, error_message=result[:200])
    state.blocked = True
    return result


def authorize_tool_body(agent, ref: _ToolCallRef) -> tuple[str, str] | None:
    """The checks that otherwise run inside the tool itself, for a call that runs elsewhere: the terminal
    dangerous-command gate and ACP edit approval, through Hermes's own code. Returns ``(result, error_type)`` when the
    call may not run, else ``None``; an approval is recorded in ``BODY_DECISION`` for a later local dispatch."""
    if ref.name == "terminal":
        return _authorize_terminal(agent, ref)
    if ref.name in _EDIT_TOOLS:
        return _authorize_edit(ref)
    return None


def _authorize_terminal(agent, ref: _ToolCallRef) -> tuple[str, str] | None:
    from tools import terminal_tool as tt
    from tools.approval_context import reset_current_observability_context, set_current_observability_context
    from tools.terminal_tool_guards import trailing_amp_background_command
    command = ref.args.get("command")
    if not isinstance(command, str):
        return None                         # the terminal tool refuses a non-string command itself
    if not ref.args.get("background"):      # a foreground `cmd &` runs as `cmd`: gate what will run
        command = trailing_amp_background_command(command) or command
    config = tt._get_env_config()
    key = (command, config["env_type"], tt._docker_has_host_access(config))
    tokens = set_current_observability_context(
        tool_call_id=ref.call_id, session_id=getattr(agent, "session_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "")
    try:
        approval = tt._check_all_guards(*key)   # a desktop batch's prepared answer is consumed here, as locally
    finally:
        reset_current_observability_context(tokens)
    rejected = tt.approval_rejection(approval, key[0])
    if rejected is not None:
        return rejected, "terminal_guard"
    BODY_DECISION.set({"call_id": ref.call_id, "terminal": (key, approval)})
    return None


def _authorize_edit(ref: _ToolCallRef) -> tuple[str, str] | None:
    try:
        from acp_adapter.edit_approval import maybe_require_edit_approval
        denied = maybe_require_edit_approval(ref.name, ref.args)
    except Exception:   # fail closed: an approval guard that cannot run denies the edit
        logger.exception("ACP edit approval guard failed for %s; the edit is denied", ref.name)
        return json.dumps({"error": "Edit approval denied: approval guard failed"}), "edit_approval_error"
    if denied is not None:
        return denied, "edit_approval_denied"
    BODY_DECISION.set({"call_id": ref.call_id, "edit": dict(ref.args)})
    return None


def consume_authorized_terminal(command: str, env_type: str, has_host_access: bool, call_id: str) -> dict | None:
    """The terminal gate's answer authorize_tool_call already got for exactly this call and command, once."""
    record = BODY_DECISION.get()
    if not record or record.get("call_id") != call_id or "terminal" not in record:
        return None
    key, approval = record["terminal"]
    if key != (command, env_type, has_host_access):
        return None
    BODY_DECISION.set(None)
    return approval


def consume_authorized_edit(name: str, args: dict, call_id: str) -> bool:
    """True once when authorize_tool_call already got ACP edit approval for exactly this call and arguments."""
    record = BODY_DECISION.get()
    if not record or record.get("call_id") != call_id or record.get("edit") != args or name not in _EDIT_TOOLS:
        return False
    BODY_DECISION.set(None)
    return True


class CallAuthorizer:
    """``hermes_cli.middleware.authorize_tool_call`` for one agent-loop call: Hermes's checks once, without running it.

    One decision per call (``decided``: ``decision`` and the ``requested`` arguments), reused by the executor's
    dispatch. ``start(ref)`` starts the call (progress, start order) once policy allowed it; ``advance()`` moves the
    start order for a refused call.
    """

    def __init__(self, agent, state: _ManagedToolResult, *, lock, make_ref, start, advance, scope_block,
                 authorization_gate) -> None:
        self.agent, self.state, self.lock = agent, state, lock
        self.make_ref, self.start, self.advance = make_ref, start, advance
        self.scope_block, self.authorization_gate = scope_block, authorization_gate
        self.decided: dict[str, Any] = {}

    def __call__(self, requested_args: dict[str, Any]):
        from hermes_cli import middleware as hmw
        with self.lock:
            if self.state.dispatched:
                raise RuntimeError("authorize_tool_call() was called after this tool call already ran")
            if "decision" in self.decided:
                first = self.decided["decision"]
                if requested_args in (first.args, self.decided["requested"]):
                    return first
                return hmw.ToolCallAuthorization(allowed=False, args=dict(requested_args), result=hmw.tool_call_refusal(
                    "this tool call was already authorized with other arguments; it was not run"))
            ref = self.make_ref(requested_args)
            reentry = hmw._TOOL_CALL_AUTHORIZER.set(_no_reentry)    # a hook calling it again must not deadlock
            try:
                decision = self._decide(ref)
            finally:
                hmw._TOOL_CALL_AUTHORIZER.reset(reentry)
            self.decided.update(decision=decision, requested=dict(requested_args))
            return decision

    def _decide(self, ref: _ToolCallRef):
        """Policy, the start of the call, then the checks that otherwise live in the tool body."""
        from hermes_cli.middleware import ToolCallAuthorization, tool_call_refusal
        agent, state = self.agent, self.state
        state.args = ref.args
        if getattr(agent, "_interrupt_requested", False):
            self.advance()
            return ToolCallAuthorization(allowed=False, args=ref.args, result=refuse_call(
                agent, state, ref, tool_call_refusal("the turn was interrupted; the tool call was not run"),
                "interrupted"))
        blocked = authorize_call(agent, state, ref, scope_block=self.scope_block,
                                 authorization_gate=self.authorization_gate)
        if blocked is not None:
            self.advance()
            return ToolCallAuthorization(allowed=False, args=ref.args, result=blocked)
        try:
            self.start(ref)                 # the local checkpoint waits for a local dispatch, if any
            body_refusal = authorize_tool_body(agent, ref)
        except Exception as exc:            # fail closed, and recorded so a fall-through never runs policy again
            logger.exception("authorizing %s failed; the call is refused", ref.name)
            return ToolCallAuthorization(allowed=False, args=ref.args, result=refuse_call(
                agent, state, ref, tool_call_refusal(
                    f"the tool call could not be authorized ({type(exc).__name__}); it was not run"),
                "authorization_error"))
        if body_refusal is not None:
            return ToolCallAuthorization(allowed=False, args=ref.args,
                                         result=refuse_call(agent, state, ref, *body_refusal))
        state.authorized = True
        return ToolCallAuthorization(allowed=True, args=ref.args)


def _no_reentry(_args=None):
    raise RuntimeError("authorize_tool_call() cannot be called from inside Hermes's own authorization of the call "
                       "(for example from a pre_tool_call hook)")
