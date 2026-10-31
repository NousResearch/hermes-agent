"""Claude-plan identity rewriting preserves skill index identifiers.

Regression for #146711: the model must receive the real skill slug that
skill_view accepts while ordinary product-name prose is still sanitized.
"""

from agent.anthropic_adapter import build_anthropic_kwargs


def test_oauth_identity_preserves_hermes_skill_index_slug():
    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-20250514",
        messages=[
            {
                "role": "system",
                "content": (
                    "## Skills\n"
                    "    - hermes-agent: Core Hermes Agent workflows.\n"
                    "    - claude-code: Claude Code delegation.\n"
                    "\nHermes Agent uses hermes-agent for general work."
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

    assert "    - hermes-agent: Core Claude Code workflows." in system_text
    assert "    - claude-code: Claude Code delegation." in system_text
    assert "Claude Code uses claude-code for general work." in system_text
