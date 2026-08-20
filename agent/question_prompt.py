"""Prompt builder for the /question slash command."""

from __future__ import annotations


def build_question_prompt(request: str = "") -> str:
    """Build the agent seed for /question.

    The slash command itself does not run a special loop. It rewrites the user
    turn into an instruction that asks the normal agent loop to use the clarify
    tool repeatedly until the task is actionable.
    """
    topic = (request or "").strip()
    if not topic:
        topic = "The user invoked /question without an initial topic. First ask what they want clarified."

    return f"""You are in /question mode.

Initial topic/request:
{topic}

Your objective is to reach actionable clarity before solving, planning, or executing anything.

Rules:
- Use the clarify tool to ask the user one focused question at a time.
- If choices would make the answer easier, provide up to 4 concise choices; otherwise ask an open-ended question.
- After each answer, reassess what is still ambiguous and ask another clarify question if needed.
- Continue until you have enough information to restate the goal, constraints, success criteria, and next action without guessing.
- Do not do the actual task yet unless the user explicitly asks you to proceed after clarity is reached.
- When clarity is reached, summarize the clarified request briefly and ask whether to proceed.
""".strip()
