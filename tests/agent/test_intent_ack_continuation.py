"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881): the historical ``codex_responses``-only path is byte-stable
under the default ``"auto"`` mode, while an explicit ``true``/model-list opt-in
extends the "you announced an action but called no tool — keep going" nudge to
every api_mode and relaxes the codebase/workspace requirement so general
autonomous workflows ("I'll run a health check on the server") are caught.

These are invariant assertions about how the mode string and the detector
gates relate, not snapshots of the marker lists.
"""

from types import SimpleNamespace
from typing import Union

from agent.agent_runtime_helpers import (
    intent_ack_continuation_mode,
    looks_like_codex_intermediate_ack,
)


def _agent(
    mode: Union[str, bool, list] = "auto",
    api_mode="chat_completions",
    model="anthropic/claude-sonnet-4",
):
    # _strip_think_blocks is a no-op for these plain-text fixtures.
    return SimpleNamespace(
        _intent_ack_continuation=mode,
        api_mode=api_mode,
        model=model,
        _strip_think_blocks=lambda c: c,
    )


# The reporter's exact repro (#27881): server-ops task, no filesystem reference.
REPRO_USER = (
    "check the current status of the server, grab the latest error logs, "
    "and let me know if there's anything critical"
)
REPRO_ACK = "I will start by running a health check command on the server to see its current status."

# The codex-coding case the detector was originally built for.
CODE_USER = "review the codebase in /app"
CODE_ACK = "Let me inspect the repository files first."


# ── mode resolution ────────────────────────────────────────────────────────




def test_true_is_all_api_modes():
    for am in ("chat_completions", "anthropic", "codex_responses"):
        assert intent_ack_continuation_mode(_agent(True, am)) == "all"
    for s in ("true", "always", "yes", "on", "ON"):
        assert intent_ack_continuation_mode(_agent(s, "chat_completions")) == "all"








def test_missing_attr_defaults_to_auto():
    bare = SimpleNamespace(api_mode="chat_completions", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare) == "off"
    bare_codex = SimpleNamespace(api_mode="codex_responses", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare_codex) == "codex_only"

# ── detector: workspace requirement ─────────────────────────────────────────




def test_codex_only_path_accepts_pronounless_workspace_announcement():
    """The default codex path keeps ACP-style action narration moving."""
    a = _agent("auto", "codex_responses")
    user = "add a migration"
    msgs = [{"role": "user", "content": user}]
    assert looks_like_codex_intermediate_ack(
        a,
        user,
        "Creating the migration file in the repo now.",
        msgs,
        require_workspace=True,
    )
    assert not looks_like_codex_intermediate_ack(
        a,
        user,
        "Testing complete. The repo is clean.",
        msgs,
        require_workspace=True,
    )


def test_multipart_user_message_does_not_crash_on_workspace_path():
    """#9562: vision requests forward ``user_message`` as a multi-part list.

    The OpenAI-compat API server passes the raw ``content`` field straight
    through for vision turns, so ``user_message`` reaches the detector as
    ``[{type:"text",...}, {type:"image_url",...}]``. The ``require_workspace``
    path flattened it with ``(user_message or "").strip()`` — a truthy list
    survived and ``.strip()`` raised ``AttributeError``, killing the turn.
    The text part still has to drive workspace detection.
    """
    a = _agent("auto", "codex_responses")
    multipart = [
        {"type": "text", "text": CODE_USER},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    msgs = [{"role": "user", "content": multipart}]
    # No crash, and the text part ("review the codebase in /app") still
    # satisfies the workspace requirement so the ack fires.
    assert looks_like_codex_intermediate_ack(
        a, multipart, CODE_ACK, msgs, require_workspace=True
    )


def test_all_path_drops_workspace_requirement():
    """The #27881 fix: opted-in turns catch non-codebase intent acks."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


def test_pronounless_action_announcements_continue_when_opted_in():
    """#72692: Copilot ACP uses terse log-style action narration."""
    a = _agent(True, "chat_completions")
    user = "write the brief and launch the session"
    msgs = [{"role": "user", "content": user}]
    announcements = (
        "Launching it now.",
        "Launching it on Copilot via acpx.",
        "Writing the task brief, then launching it on Copilot via acpx.",
        "Brief written. Creating the session now.",
        "Brief written. Creating the session on Copilot.",
        "EXIT=1 — checking the actual log.",
        "Exited — checking the actual log.",
        "Relaunching with corrected arguments.",
        "Relaunching with globals before the agent name.",
        "Running now (pid 19441, no early exit). Checking the log…",
        "Brief written\nCreating the session now",
        "Checking whether the service is healthy.",
        "Launching now — see https://ci.example.com/run?id=5 for progress.",
        "Creating the session now. Then launching the worker.",
        "Checking the log now. Will report back.",
        "Exit code 1. Checking the actual log.",
        "Step 2. Creating the session now.",
        "Attempt 2. Relaunching with corrected arguments.",
        "Exit code 1. Attempt 2. Relaunching with corrected arguments.",
        "Exit code 1. Step 2. Checking the actual log.",
        "Attempt 1. Attempt 2. Relaunching with corrected arguments.",
    )
    for announcement in announcements:
        assert looks_like_codex_intermediate_ack(
            a, user, announcement, msgs, require_workspace=False
        ), announcement


def test_pronounless_action_guardrails_reject_questions_and_finals():
    a = _agent(True, "chat_completions")
    user = "launch the session"
    msgs = [{"role": "user", "content": user}]
    final_answers = (
        "Should I launch it now?",
        "Nothing to do here.",
        "Interesting result; no action is needed.",
        "Done. The deployment succeeded.",
        "Testing completed successfully.",
        "Checking finished with no issues.",
        "Testing has completed successfully.",
        "Checking is complete.",
        "Running was successful.",
        "Testing complete.",
        "Checking done.",
        "Running successful.",
        "Testing the parser completed successfully.",
        "Checking the logs finished with no errors.",
        "Testing completed in 3.2 seconds.",
        "Checking finished at 10:42.",
        "Running the suite passed 42 of 42 assertions.",
        "Testing completed and everything looks good.",
        "Running the suite passed. 42 tests, 0 failures.",
        "Running the suite in parallel with pytest-xdist is the fastest win.",
        "Checking the CI logs would be the first step.",
        "Reading the traceback tells you which parser rule failed.",
        "Reading the traceback gives you the failing rule.",
        "Running the suite locally reproduces the failure.",
        "Checking the CI logs seems like the first step.",
        "Running the suite locally reproduced the failure.",
        "Checking the CI logs revealed a stale cache.",
        "Testing the parser found three bugs.",
        "Reviewing the diff surfaced two issues.",
        "Running the tests fails intermittently.",
        "Running the suite that ships with the repo reproduces the failure.",
        "Reading the traceback that pytest prints gives the failing rule.",
        "Testing the branch that CI builds found three bugs.",
        "Reading the traceback explains the failure.",
        "Checking the logs confirms the theory.",
        "Running the migration breaks the schema.",
        "Testing the parser produces a stack trace.",
        "Creating the migration that adds the users table.",
        "Checking the job that failed in CI.",
        "Running the migration via script improves speed.",
        "Running now — this improves reliability.",
        "Checking the actual log reveals the cause.",
        "Relaunching with corrected arguments improves reliability.",
        "Checking whether the service is healthy improves reliability.",
        "Then launching the worker improves reliability.",
        "Running the suite now. 42 passed, 0 failed.",
        "Checking the log now. Testing completed successfully.",
        "Checking the actual log. The job failed because the token expired.",
        "Running the suite locally works now.",
        "Creating the index speeds queries now.",
        "Running the suite works on Copilot.",
        "Testing happens via acpx.",
        "Checking the actual log. The error was a missing token.",
        "Checking the log now. The build is green.",
        "Launching it now. The deployment is live.",
        "Reviewing the diff. Everything looks fine.",
        "Testing the parser.\n\nAll 42 tests pass.",
        "Running the numbers. The total is 42.",
        "A few options:\n- Running the suite with a fixed seed\n- Pinning the dependency.",
        "Two ideas:\n* Checking the CI cache\n* Bumping the timeout.",
        "Options:\n1. Running the suite locally.",
        "Two options. 1. Running the suite locally. 2. Pinning the dependency.",
        "Two options. 1. Pinning the dependency. 2. Running the suite locally.",
        "Two options. 1. Checking the actual log. 2. Bumping the timeout.",
        "Two options. 1. Relaunching with corrected arguments. 2. Pinning the dep.",
        "Two approaches. 1. Checking the actual log. 2. Bumping the timeout.",
        "Two approaches. 1. Pinning the dependency. 2. Checking the actual log.",
        "Two ways. 1. Creating the session now. 2. Pinning the dep.",
        "Two options. Step 1. Pinning the dependency. Step 2. Relaunching with corrected arguments.",
        "Step 1. Pinning the dependency. Step 2. Relaunching with corrected arguments.",
        "Step 1. Pinning the dependency. Step 3. Relaunching with corrected arguments.",
        "Two options. 2. Pinning the dependency. 3. Checking the actual log.",
        "Possible fixes. 1. Checking the actual log. 2. Bumping the timeout.",
        "Two options: pinning the dep, checking the actual log.",
        "Two fixes: bumping the timeout, relaunching with corrected arguments.",
        "Option: 1. Launching it now.",
        "Options: 1. Creating the session now.",
    )
    for final in final_answers:
        assert not looks_like_codex_intermediate_ack(
            a, user, final, msgs, require_workspace=False
        ), final


# ── detector: guardrails that hold regardless of workspace ───────────────────







