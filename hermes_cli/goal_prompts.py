"""Profile-scoped continuation policy; defaults preserve the shipped prompt bytes."""

from __future__ import annotations


_POLICIES = {
    "best_judgement": (
        "For reversible decisions within the goal's boundaries, use your best judgement, "
        "state the choice and reason briefly, and keep working. Ask only when an "
        "irreversible or expensive decision requires authorization, or progress genuinely "
        "requires missing credentials or information only the user can provide."
    ),
    "never_ask": (
        "Do not stop to ask the user to choose an approach. Make reasonable decisions "
        "within the goal's boundaries and keep working within the turn budget, trying "
        "authorized alternatives when blocked. This does not grant authorization, supply "
        "missing credentials, or override explicit stop conditions. If no authorized path "
        "remains, report the concrete blocker and stop."
    ),
}

# Rewrite only shipped template text, before formatting user-owned goal/contract/evidence.
_STOP_SENTENCES = {
    "If you are blocked and need input from the user, say so clearly and stop.": "",
    "If you hit the stated stop condition or are otherwise blocked and need user input, "
    "say so clearly and stop.": "If you hit the stated stop condition, say so clearly and stop. ",
    "If the gate itself is wrong or cannot pass, say so clearly and stop.": (
        "Respect the gate's acceptance criteria; do not weaken them to manufacture success. "
    ),
    "If you are blocked and need human input, call kanban_block with a reason.": (
        "If no authorized path remains, call kanban_block with the concrete reason."
    ),
    "If something still blocks completion, call kanban_block with the reason instead.": (
        "If no authorized path remains, call kanban_block with the concrete reason instead."
    ),
}


def _instructions(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ""


def _settings() -> tuple[str, str, str]:
    try:
        from hermes_cli.config_effective import load_user_config_effective

        goals = load_user_config_effective().get("goals", {})
        autonomy = goals.get("autonomy", "ask")
        if not isinstance(autonomy, str) or autonomy not in _POLICIES:
            autonomy = "ask"
        instructions = _instructions(goals.get("continuation_instructions"))
        worker = _instructions(goals.get("worker_instructions")) or instructions
        return autonomy, instructions, worker
    except Exception:
        return "ask", "", ""


def render_goal_continuation(template: str, *, worker: bool = False, **fields: object) -> str:
    autonomy, instructions, worker_instructions = _settings()
    if autonomy != "ask":
        for original, replacement in _STOP_SENTENCES.items():
            template = template.replace(original, replacement)
    prompt = template.format(**fields)
    if autonomy != "ask":
        prompt += f"\n\nGoal autonomy ({autonomy}): {_POLICIES[autonomy]}"
        if worker:
            prompt += " Report a genuine blocker with kanban_block."
    extra = worker_instructions if worker else instructions
    if extra:
        prompt += f"\n\nAdditional goal instructions:\n{extra}"
    return prompt


def goal_judge_prompts(system: str, user: str) -> tuple[str, str]:
    autonomy, _, _ = _settings()
    if autonomy == "ask":
        return system, user
    system = system.replace(
        "- The response explains progress is blocked and the next step needs user input to proceed.\n",
        "- Progress genuinely requires missing credentials, unavailable information, or "
        "authorization outside the goal's boundaries, and no authorized alternative remains.\n",
    )
    user += (
        f"\n\nGoal autonomy: {autonomy}. {_POLICIES[autonomy]} "
        "A pending reversible decision, a question about approach, or a reasoned choice "
        "within the boundaries is not by itself BLOCKED: use CONTINUE if work remains. "
        "Explicit contract stop conditions and genuinely unachievable goals still mean BLOCKED. "
        "Completion still requires the stated deliverable and verification."
    )
    return system, user
