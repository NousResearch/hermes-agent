"""The review prompts must override review judgment with the user's no-save
consent request (#116788): a background reviewer once proposed adding a
sensitive fact whose own text recorded that request, and the ungated ``add``
would have applied it unattended. These assert the consent *rule* is present —
they do NOT snapshot the full prompt text (change-detector). The memory tool's
staging of self-reporting adds is the second, fail-closed line of defense
(tested in tests/tools/test_memory_tool.py).
"""

from run_agent import AIAgent


def test_memory_prompts_forbid_persisting_no_save_facts():
    """Both memory-writing prompts must override review judgment with the user's no-save
    request, including the trap of recording the request itself as a memory entry."""
    for label, prompt in (
        ("_MEMORY_REVIEW_PROMPT", AIAgent._MEMORY_REVIEW_PROMPT),
        ("_COMBINED_REVIEW_PROMPT", AIAgent._COMBINED_REVIEW_PROMPT),
    ):
        lower = prompt.lower()
        assert "consent" in lower, f"{label}: must frame no-save requests as a consent rule"
        assert "not to save" in lower or "don't save" in lower or "don't remember" in lower, (
            f"{label}: must name the no-save request shape"
        )
        assert "never" in lower, f"{label}: must be an absolute prohibition"
        assert "request itself" in lower or "record the request" in lower, (
            f"{label}: must cover recording the request as the fact (the #116788 near-miss shape)"
        )
