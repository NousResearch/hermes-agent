"""The CLI's /goal semantics, shared by every goal command surface.

Adapters supply authorization and schedule the returned prompt; this module alone
parses subcommands and mutates goal state. It never changes conversation history.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from hermes_cli import goals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoalCommandResult:
    output: str
    prompt: str | None = None
    kickoff: bool = False
    clear_pending: str | None = None
    error: bool = False


def _english(key, default, **values):
    return default.format(**values)


def _status(mgr, arg, render):
    return GoalCommandResult(mgr.status_line())


def _show(mgr, arg, render):
    return GoalCommandResult(f"{mgr.status_line()}\n{mgr.render_contract()}")


def _pause(mgr, arg, render):
    state = mgr.pause(reason="user-paused")
    return GoalCommandResult(
        render("gateway.goal.paused", "⏸ Goal paused: {goal}", goal=state.goal) if state
        else render("gateway.goal.no_goal_set", "No goal set."),
        clear_pending="pause" if state else None,
    )


def _resume(mgr, arg, render):
    state = mgr.resume()
    if state is None:
        return GoalCommandResult(render("gateway.goal.no_resume", "No goal to resume."))
    return GoalCommandResult(render("gateway.goal.resumed", "▶ Goal resumed: {goal}", goal=state.goal),
                             prompt=mgr.next_continuation_prompt())


def _clear(mgr, arg, render):
    had = mgr.has_goal()
    mgr.clear()
    return GoalCommandResult(render("gateway.goal_cleared", "✓ Goal cleared.") if had else
                             render("gateway.no_active_goal", "No active goal."), clear_pending="clear")


def _unwait(mgr, arg, render):
    return GoalCommandResult("▶ Wait barrier cleared — goal loop resumes." if mgr.stop_waiting()
                             else "No wait barrier set.")


def _wait(mgr, arg):
    if not arg:
        return GoalCommandResult("Usage: /goal wait <pid> [reason]", error=True)
    tokens = arg.split(None, 1)
    try:
        pid = int(tokens[0])
    except ValueError:
        return GoalCommandResult("/goal wait: <pid> must be an integer process id.", error=True)
    reason = tokens[1].strip() if len(tokens) > 1 else ""
    mgr.wait_on(pid, reason=reason)
    suffix = f" ({reason})" if reason else ""
    return GoalCommandResult(f"⏳ Goal parked on pid {pid}{suffix}. Loop pauses until it exits.")


def _gate_add(mgr, arg):
    gate = mgr.add_gate(arg)
    return GoalCommandResult(f"⚿ Gate added: $ {gate.command} "
                             f"({gate.max_retries} retries, {gate.timeout_seconds}s timeout). "
                             "It must pass before the goal can complete.")


def _gate_remove(mgr, arg):
    return GoalCommandResult(f"✓ Gate removed: $ {mgr.remove_gate(int(arg))}")


def _gate_clear(mgr, arg):
    count = mgr.clear_gates()
    return GoalCommandResult(f"✓ Cleared {count} gate{'s' if count != 1 else ''}.")


_GATE_HANDLERS = {"add": _gate_add, "remove": _gate_remove, "rm": _gate_remove, "clear": _gate_clear}
_EXACT_HANDLERS = {
    "": _status, "status": _status, "show": _show, "pause": _pause,
    "resume": _resume, "clear": _clear, "stop": _clear, "done": _clear, "unwait": _unwait,
}


def _gate(mgr, arg, authorize_gate):
    if not arg or arg.lower() == "list":
        return GoalCommandResult(mgr.render_gates())
    tokens = arg.split(None, 1)
    verb, rest = tokens[0].lower(), tokens[1].strip() if len(tokens) > 1 else ""
    handler = _GATE_HANDLERS.get(verb)
    if handler is None or (verb == "clear" and rest) or (verb != "clear" and not rest):
        return GoalCommandResult("Usage: /goal gate [list | add <command> | remove <N> | clear]", error=True)
    # Gates run shell commands without a later approval. The adapter must explicitly
    # authorize creation; recovery commands remain available to non-admin senders.
    if verb == "add" and (denial := authorize_gate()):
        return GoalCommandResult(denial, error=True)
    try:
        return handler(mgr, rest)
    except (RuntimeError, ValueError, IndexError) as exc:
        operation = "remove" if verb == "rm" else verb
        return GoalCommandResult(f"/goal gate {operation}: {exc}", error=True)


def _set(mgr, arg, *, drafting, last_user_message, render, progress):
    if drafting:
        if not arg:
            return GoalCommandResult("Usage: /goal draft <objective in plain language>", error=True)
        if progress is not None:
            progress("Drafting completion contract…")
        try:
            contract = goals.draft_contract(arg)
        except Exception as exc:
            logger.debug("goal draft failed: %s", exc)
            contract = None
        headline = arg
    else:
        headline, contract = goals.parse_contract(arg)
        contract = contract if not contract.is_empty() else None
    state = mgr.set(headline or arg, contract=contract)
    output = render("gateway.goal.set", "⊙ Goal set ({budget}-turn budget): {goal}",
                    budget=state.max_turns, goal=state.goal)
    if state.has_contract():
        label = "Drafted completion contract:" if drafting else "Completion contract:"
        output += f"\n{label}\n{state.contract.render_block()}"
    if drafting:
        output += ("\nTighten any field by re-setting the goal with inline lines "
                   "(e.g. verify: <command>), then /goal resume. Use /goal show to review."
                   if state.has_contract() else
                   "\nCouldn't draft a contract (aux model unavailable) — running as a "
                   "free-form goal. The per-turn judge still applies.")
    else:
        against = " against the contract above" if state.has_contract() else ""
        output += (f"\nAfter each turn, a judge model checks if the goal is done{against}. "
                   "Hermes keeps working until it is, you pause/clear it, or the budget is "
                   "exhausted. Use /goal status, /goal show, /goal pause, /goal resume, /goal clear.")
    return GoalCommandResult(output, goals.goal_kick_prompt(state.goal, last_user_message), kickoff=True)


def is_goal_control(arg: str) -> bool:
    """Whether this command controls an existing goal rather than replacing it."""
    normalized = arg.strip().lower()
    return normalized in _EXACT_HANDLERS or normalized.split(None, 1)[0] in {"wait", "gate"}


def dispatch_goal_command(
    mgr: goals.GoalManager, arg: str, *, authorize_gate: Callable[[], str | None],
    last_user_message=None, render: Callable = _english,
    progress: Callable[[str], None] | None = None,
) -> GoalCommandResult:
    """Apply one command. ``authorize_gate`` returns a denial or None (explicit approval).

    Synchronous like GoalManager: async adapters run this off-loop with copied
    ContextVars so draft credentials and persisted I/O stay in the caller's profile.
    """
    arg = arg.strip()
    tokens = arg.split(None, 1)
    verb = tokens[0].lower() if tokens else ""
    rest = tokens[1].strip() if len(tokens) > 1 else ""
    prefix = "Invalid goal"
    try:
        if handler := _EXACT_HANDLERS.get(arg.lower()):
            return handler(mgr, "", render)
        if verb == "wait":
            prefix = "/goal wait"
            return _wait(mgr, rest)
        if verb == "gate":
            prefix = "/goal gate"
            return _gate(mgr, rest, authorize_gate)
        return _set(mgr, rest if verb == "draft" else arg,
                    drafting=verb == "draft", last_user_message=last_user_message,
                    render=render, progress=progress)
    except (RuntimeError, ValueError, IndexError) as exc:
        output = (render("gateway.goal.invalid", "Invalid goal: {error}", error=str(exc))
                  if prefix == "Invalid goal" else f"{prefix}: {exc}")
        return GoalCommandResult(output, error=True)
