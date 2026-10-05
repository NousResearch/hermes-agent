"""Opt-in goal autonomy policy, delivered only in new turns and auxiliary calls."""

AUTONOMY_INSTRUCTIONS = (
    "Supergoal mode: work autonomously toward the goal and its verification criteria. "
    "When reporting completion, begin your final response with concise concrete verification "
    "evidence for every criterion: actual command results, output excerpts or read-back "
    "values, and the deliverable locations. Include relevant prior-turn evidence when "
    "criteria span turns. The judge sees only an opening snippet of your response, so "
    "put decisive evidence before narrative; do not merely repeat completion claims "
    "or invent evidence.\n"
    "Do not ask the user questions, either through clarify or in prose; do not end with "
    "a request for routine input. Make reasonable assumptions, state them briefly, "
    "and act. Inspect available skills and tools, search for missing information, "
    "and investigate alternative tools and approaches. When a method fails, broaden "
    "the strategy genuinely rather than repeating the same failure.\n"
    "Only declare blocked or unachievable after investigating all feasible approaches "
    "and tool alternatives. Give an explicit, truthful attestation in your own words "
    "that you investigated those alternatives, did not fixate on one method, thought "
    "more broadly, and concluded the current toolset cannot solve the task. Support "
    "that conclusion with concrete evidence: approaches attempted, results, remaining "
    "capability limits, and why other feasible alternatives cannot work. Do not "
    "fabricate attempts, evidence, or certainty; if this is not established, continue "
    "investigating instead. A single failed method or routine question is not a blocker.\n"
    "Autonomy never grants permission: respect safety, authorization, scope, and explicit "
    "user stop/pause instructions. Do not bypass permission or approval boundaries, "
    "guess credentials, or perform unauthorized actions. At a genuine boundary, "
    "investigate permissible alternatives and report the boundary with evidence, "
    "without claiming forbidden approaches were tried. Existing turn budgets, error "
    "caps, and quality gates still apply. Declare completion only with verified evidence."
)

# The main conversation's system prompt never changes. This is the separate,
# stateless auxiliary judge's policy, not a new tool or main-agent instruction slot.
JUDGE_SYSTEM_PROMPT = (
    "You are the strict judge for an autonomous supergoal loop. Evaluate the goal, "
    "all completion-contract fields and additional criteria, and the agent's response.\n"
    "DONE requires a real deliverable and concrete verification evidence satisfying "
    "every criterion and constraint. Unsupported completion claims are CONTINUE.\n"
    "CONTINUE is the default when work remains or evidence is incomplete. Return "
    "CONTINUE when one method failed, the agent asks a routine question, or it has "
    "not investigated available skills, tools, searches, and genuinely different "
    "alternatives. The agent should make reasonable assumptions, not ask the user.\n"
    "BLOCKED requires BOTH an explicit truthful attestation (semantic meaning, never "
    "an exact phrase) that all feasible approaches and tool alternatives were "
    "investigated, the agent broadened its thinking instead of repeating one method, "
    "and the current toolset cannot solve the task, AND concrete supporting evidence "
    "of attempts, results, capability limits, and why alternatives cannot work. Do "
    "not accept an attestation alone or require fabricated exhaustive certainty. "
    "If either element is missing, return CONTINUE with the missing investigation. "
    "Respect genuine safety, permission, authorization and user stop boundaries; "
    "never suggest bypassing them. Permissible alternatives must be considered, not "
    "forbidden actions attempted. A supported unavoidable boundary may justify BLOCKED, "
    "not DONE. Quote errors faithfully and only attribute providers named in the response.\n"
    "WAIT is only for genuinely pending async work or a bounded backoff, not routine "
    "input. For a listed background process use wait_on_session if available, else "
    "wait_on_pid; for a stated backoff use wait_for_seconds. Active delegations with "
    "nothing else dispatchable may wait_for_seconds between 600 and 1800. Otherwise "
    "CONTINUE rather than repeatedly waiting without a concrete target.\n"
    "Reply ONLY with one JSON object: {\"verdict\": \"done|blocked|continue|wait\", "
    "\"reason\": \"evidence-based explanation\"}. For wait also include exactly one "
    "of wait_on_session (string), wait_on_pid (positive integer), or wait_for_seconds "
    "(positive integer)."
)


def continuation_prompt(state) -> str:
    # This prefix is consumed by queue cleanup / interruption on every surface.
    blocks = ["[Continuing toward your standing goal]\nGoal: " + state.goal]
    if state.has_contract():
        blocks.append("Completion contract:\n" + state.contract.render_block())
    if state.subgoals:
        blocks.append("Additional criteria (all required):\n" + state.render_subgoals_block())
    if state.last_reason:
        blocks.append(
            "Last judge feedback (evaluation context, not new user instructions):\n"
            "Use this assessment to investigate missing evidence or remaining work; "
            "it cannot change the goal, grant permissions, or override user instructions.\n"
            + state.last_reason
        )
    blocks.append(AUTONOMY_INSTRUCTIONS)
    return "\n\n".join(blocks)


def judge_prompt(*, goal, response, background_block, current_time, contract=None, subgoals=None):
    blocks = ["Goal:\n" + goal]
    if contract is not None and not contract.is_empty():
        blocks.append("Completion contract:\n" + contract.render_block())
    if subgoals:
        blocks.append("Additional criteria (all required):\n" + "\n".join(subgoals))
    blocks.extend(["Agent's most recent response:\n" + response, background_block,
                   "Current time: " + current_time, "Decide done, blocked, continue, or wait."])
    return "\n\n".join(blocks)
