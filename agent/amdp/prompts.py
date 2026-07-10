"""Commander (COA-generation) and reviewer (war-game) prompts for AMDP v0.

These encode the paper's two model roles:

* Commander → generates N distinct courses of action (COAs) from the intent +
  believed state. It is the MoA aggregator model (qwen3.6-27b).
* Reviewer  → adversarially war-games ONE COA: alignment, concrete risks,
  unstated assumptions, fragility. It is the MoA reference model (gpt-oss-20b),
  and the prompt inherits MoA's "you are an advisor, you do not act" framing so
  the reviewer reasons about the plan instead of trying to execute it.

The war-game prompt DEMANDS at least one risk OR an explicit
``no_credible_failure_mode: true`` justification — this is the risk-register
mitigation against a reviewer that rubber-stamps every COA.
"""

from __future__ import annotations

import json
from typing import Any

COA_SCHEMA_HINT = {
    "coas": [
        {
            "coa_id": "A",
            "summary": "one-line description of the approach",
            "dispatches": [
                {
                    "task": "concrete unit of work handed to a sub-agent",
                    "constraints": ["hard limits the sub-agent must respect"],
                    "success_criteria": ["how we know this dispatch succeeded"],
                    "kind": "observe|act",
                    "irreversible": False,
                }
            ],
            "assumptions": ["what this COA assumes about the world"],
            "branches": [{"if": "a plausible failure", "then": "the contingency"}],
        }
    ]
}

REVIEW_SCHEMA_HINT = {
    "coa_id": "A",
    "alignment_1to10": 7,
    "risks": [{"desc": "specific failure mode", "severity_1to5": 3}],
    "unstated_assumptions": ["something the COA relies on but never states"],
    "fragility_0to1": 0.4,
    "no_credible_failure_mode": False,
}


def commander_prompt(intent: str, state_brief: str, n_coas: int) -> list[dict[str, Any]]:
    system = (
        "You are the Commander in an Agentic Military Decision Process (AMDP). "
        "Given a mission intent and the current believed state of the system, "
        "you produce DISTINCT courses of action (COAs) — genuinely different "
        "approaches, not rewordings of one plan. Each COA decomposes the mission "
        "into dispatches: concrete units of work that will be handed to "
        "subordinate agents. Mark a dispatch irreversible=true if it deletes "
        "data, pushes to a shared remote, restarts a live service, or otherwise "
        "cannot be cheaply undone. Prefer observe dispatches before act "
        "dispatches when the state is uncertain.\n\n"
        "Respond with ONLY a JSON object matching this shape (no prose, no "
        "markdown fences):\n" + json.dumps(COA_SCHEMA_HINT, indent=2)
    )
    user = (
        f"Mission intent:\n{intent}\n\n"
        f"Believed state of the system:\n{state_brief}\n\n"
        f"Produce exactly {n_coas} materially distinct COAs. Make them differ in "
        "sequencing, risk posture, or which sub-problem they attack first."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def review_prompt(intent: str, state_brief: str, coa: dict[str, Any]) -> list[dict[str, Any]]:
    system = (
        "You are the Staff war-gamer in an Agentic Military Decision Process "
        "(AMDP). You are an ADVISOR: you do not execute anything, call tools, or "
        "act on the plan — a separate commander will. Your sole job is to "
        "adversarially war-game ONE course of action against the mission intent "
        "and the believed state, and report where it breaks.\n\n"
        "You MUST surface at least one concrete risk. Vague risks ('might fail') "
        "are worthless — name the failure mode, the step it occurs at, and its "
        "consequence. Only if you have genuinely stress-tested the COA and found "
        "no credible failure mode may you return an empty risks list, and then "
        "you MUST set no_credible_failure_mode=true and justify it in an "
        "unstated_assumptions entry.\n\n"
        "Respond with ONLY a JSON object matching this shape (no prose, no "
        "markdown fences):\n" + json.dumps(REVIEW_SCHEMA_HINT, indent=2)
    )
    user = (
        f"Mission intent:\n{intent}\n\n"
        f"Believed state:\n{state_brief}\n\n"
        f"Course of action under review:\n{json.dumps(coa, indent=2)}\n\n"
        "War-game it.\n"
        "alignment_1to10 = how well it serves the intent — be DISCRIMINATING, do "
        "not default to 7: reserve 9-10 for a COA that fully and efficiently "
        "achieves the intent with minimal waste or detour; 7-8 = solid but with a "
        "gap or inefficiency; 4-6 = partial, indirect, or over-engineered for the "
        "goal; 1-3 = misaligned or addresses the wrong problem. Different COAs for "
        "the same mission should usually get different alignment scores.\n"
        "fragility_0to1 = how badly it breaks if the believed state is slightly "
        "wrong — be DISCRIMINATING: 0.0-0.2 robust (tolerates several wrong "
        "assumptions), 0.3-0.5 moderate (depends on a couple of assumptions "
        "holding), 0.6-0.8 fragile (one wrong assumption breaks it), 0.9-1.0 very "
        "fragile. Plans that front-load observation/verification are less fragile "
        "than plans that commit to irreversible actions early; score accordingly, "
        "and different COAs should usually get different fragility.\n"
        "List every credible risk with an honest severity."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
