"""System prompt + tool spec for the LLM-as-skill-selector.

Keep the prompt tight. The state JSON gives the LLM the spatial info it
needs; the rendered top-down adds spatial intuition; the tool spec
constrains output to a known skill name.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Dict, List, Tuple

from .skills import SKILL_REGISTRY


SYSTEM_PROMPT = dedent(
    """\
    You are the high-level controller for a Spot quadruped robot in a
    foraging task. Your goal is to direct the robot to walk over and
    collect every yellow battery scattered on the ground.

    You DO NOT control joints directly. You pick a movement skill from
    a fixed library; a low-level locomotion policy executes that skill
    for ~1 second and then returns control to you for the next pick.

    World frame: +X east, +Z north (top-down). The robot's heading is
    the direction it is currently facing (and walking when "walk_forward").
    Batteries are collected when the robot's body passes within ~0.4m of
    them. The robot can fall — if it does, the episode ends in failure.
    Your reward is dominated by batteries collected; secondary penalties
    encourage efficiency (fewer skill picks per battery).

    Each turn you receive:
      - A JSON status block with the robot's xy, heading, battery
        positions, recent skill history, and time remaining.
      - A top-down rendered view of the scene.

    Pick the next skill by calling the `select_skill` tool. Do not write
    long explanations — the trainee model needs to learn fast skill
    selection, not verbose reasoning. One short sentence of rationale
    before the tool call is fine; multi-paragraph chain-of-thought is
    counterproductive at this control rate.
    """
).strip()


def build_tool_spec() -> List[Dict]:
    """OpenAI-format tool spec exposing only the available skills."""
    skill_names = sorted(SKILL_REGISTRY.keys())
    return [
        {
            "type": "function",
            "function": {
                "name": "select_skill",
                "description": (
                    "Choose the next movement skill for the robot. The "
                    "low-level policy will execute it for the skill's "
                    "fixed duration, then return control to you."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "enum": skill_names,
                            "description": (
                                "Skill name. Available skills (with brief "
                                "descriptions):\n"
                                + "\n".join(
                                    f"  - {s.name}: {s.description}"
                                    for s in SKILL_REGISTRY.values()
                                )
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "One short sentence: why this skill, given "
                                "the current state. Keep under 20 words."
                            ),
                        },
                    },
                    "required": ["skill"],
                },
            },
        }
    ]


def format_state_json(
    *,
    spot_xy: Tuple[float, float],
    spot_yaw_deg: float,
    batteries: List[Tuple[float, float]],
    collected: int,
    skill_history: List[str],
    steps_remaining: int,
    energy: float | None = None,
) -> str:
    """Compact JSON the LLM can parse quickly."""
    state = {
        "robot": {
            "position": [round(spot_xy[0], 2), round(spot_xy[1], 2)],
            "heading_deg": round(spot_yaw_deg, 1),
        },
        "batteries_remaining": [
            [round(x, 2), round(z, 2)] for x, z in batteries
        ],
        "batteries_collected": collected,
        "skill_history_recent": skill_history[-5:],
        "skill_picks_remaining": steps_remaining,
    }
    if energy is not None:
        state["robot"]["energy"] = round(energy, 2)
    return json.dumps(state, indent=2)


def build_user_message_content(state_json: str, image_part: Dict | None) -> List[Dict]:
    """Compose the multimodal user message for one skill-pick turn."""
    parts: List[Dict] = [
        {"type": "text", "text": f"Current state:\n```json\n{state_json}\n```"}
    ]
    if image_part is not None:
        parts.append(image_part)
    parts.append(
        {
            "type": "text",
            "text": "Call `select_skill` with the next movement skill.",
        }
    )
    return parts
