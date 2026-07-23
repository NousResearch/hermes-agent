"""The review prompts must describe the write guard the fork actually runs into.

The background review fork is told to improve the skill library, then meets a
guard that refuses most of it. When the prompt and the guard disagree, the fork
picks a forbidden target and loops on real refusals. Two failure modes are
pinned down here:

* Coverage — every reason the guard can refuse on must be named in the prompt,
  so a new reason cannot be added without telling the reviewer about it.
* Contradiction — the prompt used to say pinned skills "CAN be improved" while
  the guard refused every pinned write.

These assert instructions are present, not the prompt's exact wording.
"""

import pytest

from run_agent import AIAgent
from tools.skill_provenance import (
    BACKGROUND_REVIEW_BLOCK_REASONS,
    BLOCK_BUNDLED,
    BLOCK_EXTERNAL,
    BLOCK_HUB_INSTALLED,
    BLOCK_NOT_AGENT_CREATED,
    BLOCK_PINNED,
    BLOCK_PROTECTED_BUILTIN,
)


# Each guard reason and the vocabulary that tells the reviewer about it. The
# reason code itself counts — the prompt enumerates the exact strings that
# come back in ``blocked_because``, which is what the reviewer matches on.
# Keeping this keyed by the real constants means a new block reason fails
# these tests until both prompts describe it.
REASON_VOCABULARY = {
    BLOCK_PINNED: (BLOCK_PINNED,),
    BLOCK_EXTERNAL: (BLOCK_EXTERNAL, "external_dirs", "external dir"),
    BLOCK_PROTECTED_BUILTIN: (
        BLOCK_PROTECTED_BUILTIN,
        "protected built-in",
        "protected builtin",
    ),
    BLOCK_HUB_INSTALLED: (BLOCK_HUB_INSTALLED, "hub-installed", "hub installed"),
    BLOCK_BUNDLED: (BLOCK_BUNDLED,),
    BLOCK_NOT_AGENT_CREATED: (
        BLOCK_NOT_AGENT_CREATED,
        "manually authored",
        "created_by",
        "user wrote",
    ),
}

# Parametrize by attribute name, not by prompt text — the text is thousands of
# characters and pytest would put all of it in every test id.
PROMPT_NAMES = ["_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"]

prompt_cases = pytest.mark.parametrize("label", PROMPT_NAMES)


def test_vocabulary_map_covers_every_guard_reason():
    """Guard reasons and prompt vocabulary must stay in step."""
    assert set(REASON_VOCABULARY) == set(BACKGROUND_REVIEW_BLOCK_REASONS), (
        "a block reason exists with no prompt vocabulary — teach the reviewer "
        "about it and add it here"
    )


@prompt_cases
def test_prompt_names_every_reason_a_write_can_be_refused(label):
    prompt = getattr(AIAgent, label)
    lower = prompt.lower()
    for reason, keywords in REASON_VOCABULARY.items():
        assert any(k in lower for k in keywords), (
            f"{label}: nothing tells the reviewer about the {reason!r} block; "
            f"expected one of {keywords}"
        )


@prompt_cases
def test_prompt_does_not_promise_pinned_skills_are_editable(label):
    """The guard refuses every pinned write — the prompt must not invite one."""
    prompt = getattr(AIAgent, label)
    lower = prompt.lower()
    assert "can be improved" not in lower, (
        f"{label}: still claims pinned skills can be improved; the background "
        f"guard refuses pinned writes outright"
    )
    assert "pin only blocks deletion" not in lower, (
        f"{label}: still describes pin as deletion-only for this pass"
    )


@prompt_cases
def test_prompt_directs_target_selection_by_writable_metadata(label):
    """skills_list advertises writability — the prompt must say to read it."""
    prompt = getattr(AIAgent, label)
    assert "writable" in prompt, (
        f"{label}: must tell the reviewer to select targets by the writable flag"
    )
    assert "blocked_because" in prompt, (
        f"{label}: must name blocked_because as the reason field"
    )
    assert "skills_list" in prompt, (
        f"{label}: must point at the tool that carries the writability metadata"
    )


@prompt_cases
def test_prompt_applies_the_writable_check_to_already_loaded_skills(label):
    """The gap the 11:13 trace exposed.

    Session 20260722_103517_c4b416e4: the fork's very first API call was
    ``skill_manage(patch, 'fix-pr-review')`` — it never called ``skills_list``,
    because preference step 1 says to patch a skill already loaded in the
    conversation. Writability metadata the fork never fetches cannot help it,
    so the loaded-skill step has to carry the check itself.
    """
    prompt = getattr(AIAgent, label)
    lower = prompt.lower()
    sentences = [s for s in lower.replace("\n", " ").split(". ")]
    assert any("loaded" in s and "writable" in s for s in sentences), (
        f"{label}: the currently-loaded-skill step must require a writability "
        f"check too — a skill loaded in the parent conversation is the target "
        f"the fork reaches for first, without ever calling skills_list"
    )


@prompt_cases
def test_prompt_makes_the_owned_fallback_mandatory(label):
    """A blocked best-target must redirect, never end the pass empty."""
    prompt = getattr(AIAgent, label)
    lower = prompt.lower()
    assert "do not retry" in lower or "never retry" in lower, (
        f"{label}: must forbid retrying a blocked target"
    )
    # The two destinations the fork owns.
    assert "action=create" in lower or "create a new" in lower, (
        f"{label}: must name creating a new agent-owned skill as the fallback"
    )
    assert "memory tool" in lower, (
        f"{label}: must name memory as the other owned destination"
    )
    # The old escape hatch: "only protected skills need updating, so stop".
    assert "if the only skills that need updating are protected" not in lower, (
        f"{label}: must not let a protected best-target end the pass with "
        f"nothing saved — the fallback is mandatory"
    )
