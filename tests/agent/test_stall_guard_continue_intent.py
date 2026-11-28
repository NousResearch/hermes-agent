"""Said-continue-but-stopped detector: bare intent forms vs. ordinary dialog (PR #114174).

``trailing_continue_intent`` decides whether a visible reply with no tool calls is re-prompted to
act. A miss leaves a stalled run reporting "complete"; a false fire talks over the user. These
pin both sides: the bare announced-action shapes models actually write must fire, and the
conversational turns and handoffs that precede a user's approval must not.
"""

import pytest

from agent.agent_runtime_helpers_stall_tail import trailing_continue_intent

# Final turns from stalled runs: the model announced a tool action and stopped.
ANNOUNCED_ACTIONS = (
    "Let me search for the registry file.",
    "Now I need to check the Prisma schema to understand the phase fields.",
    "Let me create the executor file with the required functions.",
    "I need to read the file first to see its current content.",
    "The file is named `registry` without extension. Let me read it.",
    "Let's run the test suite",
    "I'm going to update the handler",
    "Found the bug. I'll fix it now.",
    "Okay, let me look at the config",
    "Tests are red. Next I run the linter",
    "Now I’ll also update the docs",
    "Let me open agent/turn_final_response.py.",
    "Typecheck passes. Proceeding to run the integration suite.",
    # "now"-qualified forms the guard caught before this change.
    "Found the config file. Let me now update it.",
    "The tests pass. I will now push the branch.",
    "Done with step one. Next, I check the logs",
)

# Ordinary dialog: offers, handoffs and approval gates. Re-prompting any of these overrides the user.
DIALOG_AND_HANDOFFS = (
    "I've pushed the branch. Let me know if you want the CI run too.",
    "That's the full diff — let me know how you'd like to proceed.",
    "Options ready. I will wait for your decision.",
    "Options ready — next, I will wait for your decision.",
    "Plan is above. I'll run the migration once you confirm.",
    "I'll run it if you say so",
    "Let me check with you before changing the schema.",
    "I'll update you when the build finishes.",
    "I'll be happy to help with that.",
    "I'll be here if you have questions.",
    "Thanks! I'll keep that in mind.",
    "I need to be clear about the tradeoff here",
    "I need to know which environment you want.",
    "I'm going to be honest, that approach is risky.",
    "Let's talk about it.",
    "Let's go with option B.",
    "This lets us skip the check",
    "I will happily review the PR once CI is green",
    "I can help with that tomorrow if you'd like.",
    "Should I run the tests?",
    "Done. All tests pass and the branch is pushed.",
    # An intent followed by the work itself: nothing is dangling.
    "Let me explain the design. The registry is keyed by entity name, and each entry carries the model.",
    "Let me check. The answer is 42.",
)


@pytest.mark.parametrize("reply", ANNOUNCED_ACTIONS)
def test_announced_tool_action_at_the_tail_fires(reply):
    assert trailing_continue_intent(reply)


@pytest.mark.parametrize("reply", DIALOG_AND_HANDOFFS)
def test_dialog_and_user_handoffs_do_not_fire(reply):
    assert not trailing_continue_intent(reply)
