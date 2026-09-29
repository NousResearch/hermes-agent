"""The CLI's /goal semantics, shared by every goal command surface.

Adapters supply authorization and schedule the returned prompt; this module alone
parses subcommands and mutates goal state. It never changes conversation history.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Callable, List

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


def _resume(mgr, arg, render, *, dropped: str = ""):
    """Resume semantics — never goal text.

    A resume-shaped input is control input, not a goal statement: it must not overwrite the active
    goal's text (2026-09-28: `/goal resume last goal` replaced a standing goal with that sentence).
    A paused/waited goal comes back with its text byte-identical; an already-active goal is a
    no-op that says so.
    """
    state = mgr.state
    if state is None or state.status == "cleared":
        base = render("gateway.goal.no_resume", "No goal to resume.")
        if dropped:
            return GoalCommandResult(
                base + f" Nothing was set: {dropped!r} is not goal text — start one with "
                       "/goal <goal text>.")
        return GoalCommandResult(base)
    if state.status == "active":
        parked = bool(state.waiting_on_session or state.waiting_on_pid or state.waiting_until)
        note = (" It is parked on a wait barrier — /goal unwait clears it, or the loop resumes on its "
                "own when the barrier lifts." if parked else "")
        return GoalCommandResult(
            f"⊙ Goal is already active — nothing to resume: {state.goal}{note}" + _dropped_hint(dropped))
    resumed = mgr.resume()
    if resumed is None:
        return GoalCommandResult(render("gateway.goal.no_resume", "No goal to resume.") + _dropped_hint(dropped))
    return GoalCommandResult(
        render("gateway.goal.resumed", "▶ Goal resumed: {goal}", goal=resumed.goal) + _dropped_hint(dropped),
        prompt=mgr.next_continuation_prompt())


def _dropped_hint(dropped: str) -> str:
    """Say plainly that trailing words after a control verb were not taken as goal text."""
    if not dropped:
        return ""
    return (f"\n(ignored {dropped!r}: a control command never sets goal text — use /goal <goal text> "
            "to replace the goal, /subgoal <text> to add criteria, or /goal -- <text> when the text "
            "really starts with a control word.)")


def _bare_control(mgr, verb: str, rest: str, render):
    """A bare control verb with trailing words.

    The verb reading wins (that is the bug fix): the words after it are reported as ignored rather
    than silently becoming the new goal. The terminal verbs are the exception — clearing a goal the
    user did not mean to clear is expensive, so with an existing goal they refuse and name both
    readings instead of acting. Goal text that genuinely starts with a control word is written
    verbatim with the ``--`` prefix (``/goal -- clear the cache``).
    """
    if rest and verb in _TERMINAL_VERBS and mgr.has_goal():
        return GoalCommandResult(
            f"/goal {verb} takes no argument, and {rest!r} is NOT a new goal — replacing goal text "
            f"needs its own command. Say '/goal {verb}' to {_TERMINAL_HINT[verb]}, or "
            "'/goal <new goal text>' to replace the goal (prefix it with -- if the text should "
            "start with a control word).",
            error=True,
        )
    result = _EXACT_HANDLERS[verb](mgr, "", render)
    if rest:
        result = replace(result, output=result.output + _dropped_hint(rest))
    return result


def _history(mgr, arg, render):
    return GoalCommandResult(mgr.render_history())


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
# Resume-shaped verbs: control input that resumes an existing goal and never sets goal text.
_RESUME_VERBS = frozenset({"resume", "continue", "unpause"})
# Bare control verbs (no argument of their own): with trailing words the verb reading still wins —
# the alternative is the silent-overwrite bug — except for the TERMINAL verbs, which end the goal,
# where a wrong guess is expensive and the handler refuses instead (``_bare_control``).
_TERMINAL_VERBS = frozenset({"clear", "stop", "done"})
_BARE_CONTROL_VERBS = _RESUME_VERBS | {"pause", "status", "show", "unwait", "history"} | _TERMINAL_VERBS
_TERMINAL_HINT = {"clear": "clear the goal", "stop": "clear the goal", "done": "mark it done/clear it"}
_EXACT_HANDLERS = {
    "": _status, "status": _status, "show": _show, "pause": _pause,
    "resume": _resume, "continue": _resume, "unpause": _resume,
    "clear": _clear, "stop": _clear, "done": _clear, "unwait": _unwait, "history": _history,
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
        headline, criteria = arg, []
    else:
        headline, contract, criteria = goals.parse_goal_input(arg)
        contract = contract if not contract.is_empty() else None

    # Criteria-only input (`/goal criteria: …`) extends the ACTIVE goal: the headline is left
    # byte-identical and the criteria land on the judge-rendered subgoal surface.
    if criteria and not headline:
        return _criteria_only(mgr, criteria)

    previous = (mgr.state.goal.strip() if mgr.state is not None and (mgr.state.goal or "").strip() else "")
    state = mgr.set(headline or arg, contract=contract, subgoals=criteria)
    output = render("gateway.goal.set", "⊙ Goal set ({budget}-turn budget): {goal}",
                    budget=state.max_turns, goal=state.goal)
    if previous and previous != state.goal:
        # A replace is allowed (plain goal text IS the replace path) but never silent about the loss.
        output += f"\n(replaced the previous goal — its text is kept: /goal history)\n  was: {previous}"
    if criteria:
        output += f"\nCriteria (subgoals the judge checks):\n{state.render_subgoals_block()}"
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


def _criteria_only(mgr, criteria: List[str]) -> GoalCommandResult:
    """Append criteria to the ACTIVE goal as subgoals (state only — the goal text is untouched)."""
    try:
        added = mgr.add_subgoals(criteria)
    except RuntimeError:
        return GoalCommandResult(
            "No active goal to add criteria to — set one with /goal <goal text>, or supply the "
            "criteria alongside it on their own lines (criteria: …).", error=True)
    if not added:
        return GoalCommandResult("No criteria text found to add.", error=True)
    noun = "criterion" if len(added) == 1 else "criteria"
    return GoalCommandResult(
        f"⊙ Added {len(added)} {noun} to the active goal (its text is unchanged): {mgr.state.goal}\n"
        f"{mgr.state.render_subgoals_block()}\n"
        "The per-turn judge checks every criterion above.")


def is_goal_control(arg: str) -> bool:
    """Whether this command controls an existing goal rather than replacing it."""
    normalized = arg.strip().lower()
    first = normalized.split(None, 1)[0] if normalized else ""
    if normalized in _EXACT_HANDLERS or first in _BARE_CONTROL_VERBS or first in {"wait", "gate"}:
        return True
    # Criteria-only input adds to the goal (like /subgoal) instead of replacing its text.
    headline, _contract, criteria = goals.parse_goal_input(arg)
    return bool(criteria) and not headline


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
    if arg.startswith("--"):
        # Verbatim goal text: the escape hatch for text that starts with a control word
        # (`/goal -- clear the cache`). Contract/criteria lines still parse — they are explicit.
        return _set(mgr, arg[2:].strip(), drafting=False, last_user_message=last_user_message,
                    render=render, progress=progress)
    tokens = arg.split(None, 1)
    verb = tokens[0].lower() if tokens else ""
    rest = tokens[1].strip() if len(tokens) > 1 else ""
    prefix = "Invalid goal"
    try:
        if handler := _EXACT_HANDLERS.get(arg.lower()):
            return handler(mgr, "", render)
        if verb in _RESUME_VERBS:
            # `/goal resume last goal` (or any resume verb with trailing words) resumes; it never
            # becomes the goal text.
            return _resume(mgr, arg, render, dropped=rest)
        if verb in _BARE_CONTROL_VERBS:
            return _bare_control(mgr, verb, rest, render)
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
