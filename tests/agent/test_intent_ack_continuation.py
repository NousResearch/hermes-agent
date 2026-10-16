"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881). Under the default ``"auto"`` mode the detector is invoked only
for the ``codex_responses`` api_mode (``codex_only``, ``require_workspace=True``);
an explicit ``true``/model-list opt-in extends the "you announced an action but
called no tool — keep going" nudge to every api_mode and relaxes the
codebase/workspace requirement so general autonomous workflows ("I'll run a
health check on the server") are caught. The detector's shared internals (the
future-ack, action-verb, and sign-off patterns) were tightened for both modes,
so ``codex_only`` is not byte-stable — it inherits the same false-positive
guards; only the workspace requirement and the current-turn after-tool gate are
codex_only-specific.

These are invariant assertions about how the mode string and the detector
gates relate, plus false-positive guards on realistic prose — not snapshots of
the marker lists.
"""

from types import SimpleNamespace
from typing import Union

from agent.agent_runtime_helpers import (
    intent_ack_continuation_mode,
    looks_like_codex_intermediate_ack,
)


def _agent(
    mode: str | bool | list = "auto",
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


# ── build/ops action detection (announce-then-stop on local models) ──────────

# Real announce-then-stop shapes local/general models produce. The original
# inspection-oriented marker list missed these; they carry a future-ack + a
# build/ops verb + no prior tool call, so opted-in "all" mode must catch them.
BUILD_ACKS = [
    "I'll delegate the FastAPI app creation to a sub-agent, then bring it up.",
    "Sure — I'll set up the docker compose stack and deploy it.",
    "Let me install the dependencies and launch the service.",
    "Now I'll scaffold the project and generate the config.",
    "I'll verify the endpoints and curl them.",
    "Great! Now let me verify what was created and bring it all up properly.",
]


def test_build_ops_acks_caught_in_all_mode():
    ag = _agent(True, "chat_completions", "ornith-35b")
    for ack in BUILD_ACKS:
        assert looks_like_codex_intermediate_ack(
            ag, "build me a service", ack, [], require_workspace=False
        ), ack


def test_after_tool_ack_continues_in_all_mode():
    # Opted-in "all" mode now CONTINUES even when a tool ran this turn — the
    # dominant local-model failure is "do work, announce the next step, stop"
    # ("files are in place; let me launch the stack:"), which lands after a tool.
    # Bounded by the loop's max-consecutive-continuation cap.
    ag = _agent(True, "chat_completions", "ornith-35b")
    msgs = [
        {"role": "user", "content": "build me a service"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "wrote files"},
    ]
    assert looks_like_codex_intermediate_ack(
        ag, "build me a service",
        "Now let me verify what was created and bring it all up.",
        msgs, require_workspace=False,
    )


def test_historical_tool_does_not_block():
    # The store keeps a role=='tool' row for every past tool call; an ack at the
    # START of a new turn must still be caught even in a session that used tools
    # earlier. (Previously the full-history scan short-circuited here.)
    ag = _agent(True, "chat_completions", "ornith-35b")
    hist = [
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "old result"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "build me a service"},   # current turn's user msg
    ]
    assert looks_like_codex_intermediate_ack(
        ag, "build me a service",
        "I'll delegate this to a sub-agent, then bring it up.",
        hist, require_workspace=False,
    )


def test_after_tool_still_gated_in_codex_only():
    # codex_only mode keeps the original "a tool ran this turn -> don't continue"
    # behavior (scoped to the current turn, never full history).
    ag = _agent("auto", "codex_responses")
    msgs = [
        {"role": "user", "content": "review the codebase in /app"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "listed files"},
    ]
    assert not looks_like_codex_intermediate_ack(
        ag, "review the codebase in /app",
        "Let me inspect the repository files next.",
        msgs, require_workspace=True,
    )


def test_truncated_colon_ack_caught_in_all_mode():
    # Robust to phrasing that dodges the verb lists: a short reply ending
    # mid-thought on a colon is a classic announced-then-cut-off stall.
    ag = _agent(True, "chat_completions", "ornith-35b")
    for text in ("Going straight to execution now:", "Let me check:", "I'll set it up:"):
        assert looks_like_codex_intermediate_ack(ag, "do the thing", text, [], require_workspace=False), text
    # Colon endings without a first-person announcement are NOT stalls: a
    # question to the user or a list header must not trigger the nudge.
    for text in ("Sure, happy to help!", "The answer is 42.",
                 "Please paste the failing test output here:", "The three checks to run:"):
        assert not looks_like_codex_intermediate_ack(ag, "do the thing", text, [], require_workspace=False), text


def test_no_false_positive_on_benign_prose_in_all_mode():
    """Opt-in mode must not nudge on conversational replies, sign-offs, or
    questions to the user. Guards against the false-positive class that
    keyword-matching invites: substring stems (creat/creative, kill/skill,
    boot/booties), phantom future-acks ("suite on it"), colon endings with no
    action, and first-person idioms governing a verb only across a clause."""
    ag = _agent(True, "chat_completions", "ornith-35b")
    benign = [
        "I need your sudo password to continue - please type it here:",
        "Here are the three options:",
        "All three files are written and the tests pass. Let me know if you'd like me to create more examples.",
        "The deploy finished cleanly and the health checks pass. Let me know if you want me to run it again later.",
        "Let me know if you want anything else.",
        "Would you like me to deploy it now?",
        "I fixed the bug and ran the full test suite on it. Everything is green.",
        "Let's talk about your creative writing project first.",
        "I'll admit, building rapport with a new team takes time.",
        "Let's talk about the skill tree in your game design.",
        "I'll say it: dark mode is the killer feature here.",
        "Let's plan for the next installment of the series.",
        "Let's go with the curly brace style you prefer.",
        "Let's get your dog some booties for winter.",
        "Let's think about generational wealth strategies.",
        "I'll be honest, in the long run this pays off.",
        "I'll check in with you next week.",
        "Done. The tests pass.",
        # "ill" is not the lead-in "I'll"; a bare/2nd-person "going to" is not a
        # first-person announcement.
        "This error message is ill-formed when running the old parser.",
        "The parser is now ill-suited to run large files.",  # "now ill" != "now I'll"
        "If you're going to run a marathon, start with base mileage.",
        "The team is going to review the proposal on Friday.",
        "Right now I think we should run the tests together.",  # proposal, not announce
        "Now I am done reviewing the code and everything runs.",  # completion, not "now I'm <verb>ing"
        "Now I am unable to run the tests because docker is down.",  # failure report
        # a refusal must not get an "execute the tool calls" nudge
        "I'll never run that command on prod.",
        "I will never delete your data without asking.",
        # the governed window must not cross a newline into the next clause
        "I'll be brief\nkill the process when you are done.",
        # colon ending without a first-person announcement (question / header).
        "Please paste the output of the failing test here:",
        "The three checks I'd run:",
        # phrasal verbs whose meaning is conversational, not a tool action.
        "I'll run through my reasoning first.",
        "I'll open with a quick summary.",
        "Let me build on your earlier point.",
        "Let me read you the key line.",
        "Now I'll walk you through the tradeoffs.",
    ]
    for text in benign:
        assert not looks_like_codex_intermediate_ack(
            ag, "help me plan", text, [], require_workspace=False
        ), text
    # KNOWN RESIDUAL (bounded, opt-in, max-2 nudges): a first-person lead
    # governing an everyday verb whose object is conversational, not a task
    # ("I'll test that assumption", "I'll review your essay"), still fires — the
    # verb match cannot see the object. "all" mode is intended for autonomous
    # task sessions, where relaxing the workspace requirement is the point;
    # codex_only keeps the workspace requirement that prevents this.


def test_governed_announcement_fires_across_variants_in_all_mode():
    """A first-person lead-in governing an action verb in the same clause is the
    core positive signal (announce-then-stop, including mid-task)."""
    ag = _agent(True, "chat_completions", "ornith-35b")
    fires = [
        "I'll delegate the FastAPI app creation, then bring it up.",
        "Now let me verify what was created and bring it all up properly.",
        "I'll set up the service and deploy it now.",
        "I'll run the tests and fix any failures.",
        "let me verify + finish directly:",
        "Let me kick off the migration.",
        "Now I'll bring the stack up.",
        "I'm going to run the full test suite.",
        "Let me delete the stale containers and redeploy.",
        "I'll scaffold the project and install deps.",
        "Now I'm deploying the fix.",  # "now I'm <verb>ing" progressive
        "Now I am going to run the migration.",
    ]
    for text in fires:
        assert looks_like_codex_intermediate_ack(
            ag, "bring up the stack", text, [], require_workspace=False
        ), text


def test_offer_guard_is_load_bearing_over_a_governed_announcement():
    """A governed announcement that also defers to the user ("...but let me know
    if you'd prefer...") is waiting on input, not stalling. The offer guard must
    suppress it even though the governed-announcement pattern matches — this is
    the case the guard exists for (the lead-in lookahead alone does not catch)."""
    ag = _agent(True, "chat_completions", "ornith-35b")
    assert not looks_like_codex_intermediate_ack(
        ag, "set it up",
        "I'll deploy it, but let me know if you'd prefer a different approach.",
        [], require_workspace=False,
    )


def test_offer_guard_suppresses_in_codex_only_too():
    """The sign-off/offer guard is shared: a conditional offer is not a stall in
    codex_only either, even with a workspace reference."""
    ag = _agent("auto", "codex_responses", "gpt-5-codex")
    assert not looks_like_codex_intermediate_ack(
        ag, "review the repo",
        "Let me know if you'd like me to inspect the repository files.",
        [], require_workspace=True,
    )


def test_colon_ack_after_tool_fires_in_all_mode():
    # The "let me launch it:" mid-task stall lands after a tool call — opted-in
    # mode must catch it so the task actually reaches completion.
    ag = _agent(True, "chat_completions", "ornith-35b")
    msgs = [{"role": "user", "content": "bring up the stack"},
            {"role": "tool", "content": "wrote compose"}]
    assert looks_like_codex_intermediate_ack(
        ag, "bring up the stack",
        "The files are in place; let me launch the stack:",
        msgs, require_workspace=False,
    )


def test_conversational_future_not_caught():
    # A future-ack with no build/ops or inspection verb must not fire.
    ag = _agent(True, "chat_completions", "ornith-35b")
    for text in ("I'll help you brainstorm some names.", "Let me know what you think."):
        assert not looks_like_codex_intermediate_ack(
            ag, "hi", text, [], require_workspace=False
        )


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
    # #69778 / ClaySecAI #69779: old tool work must not disable a new turn.
    history = [{"role": "tool", "content": "old result"},
               {"role": "assistant", "content": "Earlier task finished."}, *msgs]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, history, require_workspace=False
    )
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, "Let me know if you want me to run it again.",
        history, require_workspace=False,
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






def test_all_mode_continues_after_a_tool_already_ran():
    # Opted-in mode intentionally continues after a tool ran this turn (the
    # after-tool relaxation). codex_only still suppresses — covered by
    # test_after_tool_still_gated_in_codex_only.
    a = _agent(True, "chat_completions")
    msgs = [
        {"role": "user", "content": REPRO_USER},
        {"role": "tool", "content": "health check result"},
    ]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )



def test_explicit_user_decline_suppresses_on_both_paths():
    """An explicit refusal governs the turn: never continue against it.

    Regression test for #69779 (reported by @cygnostik). ``_DECLINED_RE`` was
    applied only inside ``intent_ack.clarification_ack``, which runs on an
    answered-clarification turn. On the general path the detector never consulted
    it, so a declined request still produced a continuation signal. Both
    ``require_workspace`` values are asserted because the opted-in path returns
    before the workspace check that is the only other place user text is read.

    Verbatim repro from the report: False at 14b330c3, True at 40193d60.
    """
    ag = _agent(True, "chat_completions", "ornith-35b")
    user = "Do not inspect the project yet."
    msgs = [{"role": "user", "content": user}]
    for require_workspace in (True, False):
        assert not looks_like_codex_intermediate_ack(
            ag, user, "I'll inspect the files.", msgs,
            require_workspace=require_workspace,
        ), f"continued against an explicit decline (require_workspace={require_workspace})"


def test_scoped_decline_suppresses():
    """A refusal scoped to part of the request still suppresses: "explain X; do
    not edit anything" must not license an edit announcement. Also from the
    14b330c3 negative table."""
    ag = _agent(True, "chat_completions", "ornith-35b")
    user = "Explain the project options; do not edit anything."
    assert not looks_like_codex_intermediate_ack(
        ag, user, "I'll update the project files.",
        [{"role": "user", "content": user}], require_workspace=False,
    )


def test_decline_guard_does_not_suppress_an_ordinary_request():
    """The inverse the original table did not assert: a plain request with no
    refusal wording still fires, so restoring the guard cannot silently disable
    continuation altogether."""
    ag = _agent(True, "chat_completions", "ornith-35b")
    user = "Please inspect the project and report what you find."
    assert looks_like_codex_intermediate_ack(
        ag, user, "I'll inspect the files.",
        [{"role": "user", "content": user}], require_workspace=False,
    )
