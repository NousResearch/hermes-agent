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


def test_enabled_is_mode_not_off():
    assert intent_ack_continuation_enabled(_agent(True, "chat_completions")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "codex_responses")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "chat_completions")) is False
    assert intent_ack_continuation_enabled(_agent(False, "codex_responses")) is False


# ── detector: workspace requirement ─────────────────────────────────────────




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


# The #101868 repro: a complete gateway answer whose product list contains
# "bread" (a substring collision with the "read" action marker) plus a closing
# future-tense offer. The reply is substantive and must NOT be treated as an
# intermediate ack, or the gateway delivers only the synthetic continuation.
REPRO_101868_USER = "List four products."
REPRO_101868_REPLY = (
    "Here are four options:\n"
    "1. Bread maker.\n"
    "2. Coffee grinder.\n"
    "3. Electric kettle.\n"
    "4. Stand mixer.\n"
    "\n"
    "These are the available options. If useful, I'll provide a comparison table."
)


def test_substantive_reply_with_marker_substring_is_not_an_ack():
    """#101868: ``"read" in "bread"`` must not make a complete reply an ack.

    On the opted-in gateway path (``require_workspace=False``) the substring
    collision combined with the closing ``I'll`` offer misclassified the whole
    answer as an action announcement, so the first, substantive response was
    never delivered.
    """
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": REPRO_101868_USER}]
    assert not looks_like_codex_intermediate_ack(
        a, REPRO_101868_USER, REPRO_101868_REPLY, msgs, require_workspace=False
    )


def test_substantive_reply_with_thread_substring_is_not_an_ack():
    """Sibling collision from the same marker tuple: ``"read" in "thread"``."""
    a = _agent(True, "chat_completions")
    reply = (
        "Here is the summary you asked for.\n"
        "The discussion thread converged on two designs.\n"
        "If you want, I'll provide the full transcript."
    )
    msgs = [{"role": "user", "content": "Summarize the discussion."}]
    assert not looks_like_codex_intermediate_ack(
        a, "Summarize the discussion.", reply, msgs, require_workspace=False
    )


def test_workspace_substring_collisions_do_not_satisfy_codex_scope():
    """#101868 bug class, workspace markers: ``"repo" in "report"``,
    ``"path" in "empathy"``, ``"files" in "profiles"`` must not target a
    workspace on the codex_only path."""
    a = _agent("auto", "codex_responses")
    user = "Give me a status report with empathy."
    reply = "I'll draft the report now."
    msgs = [{"role": "user", "content": user}]
    assert not looks_like_codex_intermediate_ack(
        a, user, reply, msgs, require_workspace=True
    )

    # Assistant-side only: the ack mentions "profiles"/"report" but no real
    # workspace word, and the user targets none either.
    reply2 = "I'll review the profiles section now."
    assert not looks_like_codex_intermediate_ack(
        a, "Describe the layout.", reply2, msgs, require_workspace=True
    )


def test_word_bounded_actions_still_fire():
    """Positive side of #101868: inflected action mentions governed by a
    future-ack keep triggering continuation in ``all`` mode."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": "Get the log summary."}]
    for ack in (
        "I'll read the file now.",
        "I'll start reading the file.",
        "Let me check the logs.",
        "I'll begin analyzing the results.",
        "I'll keep scanning the records.",
        "Let me look into it.",
        "I'll report back shortly.",
    ):
        assert looks_like_codex_intermediate_ack(
            a, "Get the log summary.", ack, msgs, require_workspace=False
        ), ack


def test_word_bounded_workspace_markers_still_fire():
    """Positive side of the workspace-marker fix: genuine repo/path/file-tree
    references still satisfy the codex workspace scope."""
    a = _agent("auto", "codex_responses")
    msgs = [{"role": "user", "content": "check the failing build"}]
    for ack in (
        "Let me inspect the repository files first.",
        "I'll scan the codebase next.",
        "I'll check the current directory.",
        "Let me look at the file tree.",
    ):
        assert looks_like_codex_intermediate_ack(
            a, "check the failing build", ack, msgs, require_workspace=True
        ), ack


def test_plain_offers_of_optional_future_action_are_not_acks():
    """Completed conversational answers that merely offer an optional next
    step must be delivered as-is (#101868 "negative/substantive" cases)."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": "What are my options?"}]
    for reply in (
        "You can pick A or B. I can do that for you if you want.",
        "The answer is 42. Let me know if you'd like details.",
    ):
        assert not looks_like_codex_intermediate_ack(
            a, "What are my options?", reply, msgs, require_workspace=False
        ), reply








