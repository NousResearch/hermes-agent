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
    intent_ack_continuation_enabled,
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


def test_auto_is_codex_only():
    assert intent_ack_continuation_mode(_agent("auto", "codex_responses")) == "codex_only"
    assert intent_ack_continuation_mode(_agent("auto", "chat_completions")) == "off"
    assert intent_ack_continuation_mode(_agent("auto", "anthropic")) == "off"


def test_true_is_all_api_modes():
    for am in ("chat_completions", "anthropic", "codex_responses"):
        assert intent_ack_continuation_mode(_agent(True, am)) == "all"
    for s in ("true", "always", "yes", "on", "ON"):
        assert intent_ack_continuation_mode(_agent(s, "chat_completions")) == "all"


def test_false_is_off_even_for_codex():
    assert intent_ack_continuation_mode(_agent(False, "codex_responses")) == "off"
    for s in ("false", "never", "no", "off"):
        assert intent_ack_continuation_mode(_agent(s, "codex_responses")) == "off"


def test_list_matches_model_substring():
    assert intent_ack_continuation_mode(
        _agent(["gemini", "qwen"], "chat_completions", "google/gemini-3-pro")
    ) == "all"
    assert intent_ack_continuation_mode(
        _agent(["gemini", "qwen"], "chat_completions", "anthropic/claude-sonnet-4")
    ) == "off"


def test_unrecognised_value_falls_back_to_auto():
    assert intent_ack_continuation_mode(_agent("garbage", "codex_responses")) == "codex_only"
    assert intent_ack_continuation_mode(_agent("garbage", "chat_completions")) == "off"


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


def test_enabled_is_mode_not_off():
    assert intent_ack_continuation_enabled(_agent(True, "chat_completions")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "codex_responses")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "chat_completions")) is False
    assert intent_ack_continuation_enabled(_agent(False, "codex_responses")) is False


# ── detector: workspace requirement ─────────────────────────────────────────


def test_codex_only_path_requires_workspace():
    a = _agent("auto", "codex_responses")
    msgs = [{"role": "user", "content": CODE_USER}]
    # codebase ack matches workspace markers → fires
    assert looks_like_codex_intermediate_ack(a, CODE_USER, CODE_ACK, msgs, require_workspace=True)
    # server-ops ack has no filesystem reference → does NOT fire (historical scope)
    repro_msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, repro_msgs, require_workspace=True
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


# ── detector: guardrails that hold regardless of workspace ───────────────────


def test_real_final_answer_does_not_fire():
    a = _agent(True, "chat_completions")
    final = "Done. The server is healthy and there are no critical errors in the logs."
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(a, REPRO_USER, final, msgs, require_workspace=False)


def test_conversational_reply_without_action_verb_does_not_fire():
    a = _agent(True, "chat_completions")
    brainstorm = "I'll help you think through the tradeoffs here."
    msgs = [{"role": "user", "content": "help me decide"}]
    assert not looks_like_codex_intermediate_ack(
        a, "help me decide", brainstorm, msgs, require_workspace=False
    )


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


def test_long_response_is_not_treated_as_an_ack():
    a = _agent(True, "chat_completions")
    long_ack = "I will run the check. " + ("x" * 1300)
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_USER, long_ack, msgs, require_workspace=False
    )
