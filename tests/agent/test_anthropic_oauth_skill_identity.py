"""Claude-plan identity rewriting preserves skill index identifiers.

Regression for #146711: the model must receive the real skill slug that
skill_view accepts while ordinary product-name prose is still sanitized.
"""

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.prompt_builder import _render_skills_index


def test_oauth_identity_preserves_hermes_skill_index_slug():
    skills_index = _render_skills_index(
        {
            "autonomous-ai-agents": [
                ("hermes-agent", "Core Hermes Agent workflows."),
            ],
            "software-development": [
                ("claude-code", "Claude Code delegation."),
                ("github", "GitHub integration."),
            ],
            "archived": [
                ("alpha", "Alpha skill."),
                ("hermes-agent", "Core Hermes Agent workflows."),
            ],
        },
        frozenset({"archived"}),
        None,
    )
    assert "\n  - hermes-agent: Core Hermes Agent workflows." in skills_index
    assert "archived [names only]: alpha, hermes-agent" in skills_index

    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-20250514",
        messages=[
            {
                "role": "system",
                "content": (
                    skills_index
                    + "\nHermes Agent uses hermes-agent for general work."
                ),
            },
            {"role": "user", "content": "Hi"},
        ],
        tools=None,
        max_tokens=4096,
        reasoning_config=None,
        is_oauth=True,
    )

    system_text = "\n".join(block["text"] for block in kwargs["system"])

    assert "\n  - hermes-agent: Core Claude Code workflows." in system_text
    assert "    - claude-code: Claude Code delegation." in system_text
    assert "archived [names only]: alpha, hermes-agent" in system_text
    assert "Claude Code uses claude-code for general work." in system_text
