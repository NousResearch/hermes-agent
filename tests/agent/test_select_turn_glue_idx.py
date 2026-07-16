"""Regression tests for the ephemeral-context *glue target* selection.

Context: ``current_turn_user_idx`` is snapshotted once at turn-start
(``agent/turn_context.py``), right after the user message is appended. Before
the glue site in ``agent/conversation_loop.py`` injects ephemeral plugin/memory
context onto the current turn's user message, ``messages`` can be mutated by:

* preflight context compression — which does **not** keep the same list *or*
  the same dict objects: it ``.copy()``s the protected tail into fresh dicts
  (``context_compressor.compress`` / ``_fresh_compaction_message_copy``), so a
  turn-start index goes stale *and* object identity is lost; and
* ``repair_message_sequence_with_cursor`` — in-place alternation repair that
  reorders/relocates messages.

An earlier fix recomputed the *globally-last* user message at glue time. That
over-corrects: the loop appends synthetic continuation/recovery user nudges
(length-continue, codex-ack, verify-stop) which become the last user message,
so glue would drift off the real request and onto a "[System: continue]"
nudge (hermes-sweeper review on #63348).

The current fix stamps the current turn's user message with a per-turn marker
(``TURN_USER_MARKER_KEY``) that rides the compaction ``.copy()`` through and is
absent on the synthetic nudges. ``select_turn_glue_idx`` re-identifies that
exact message at its post-mutation position — never the global last user.

These tests cover both the pure selector and a faithful mirror of the
``conversation_loop`` request-assembly glue (same pattern as
``tests/run_agent/test_steer.py``, whose helper tests mirror the run_conversation
injection): relocated-after-compaction, a following synthetic nudge, and the
compacted-away drop.
"""

from agent.agent_runtime_helpers import TURN_USER_MARKER_KEY, select_turn_glue_idx


# ── Faithful mirror of the conversation_loop glue step ────────────────────────
# Mirrors ``agent/conversation_loop.py`` request assembly: pick the glue index
# with the REAL selector, then append plugin/memory context onto that message's
# api copy (the original ``messages`` entry is never mutated). Kept minimal and
# byte-for-byte with the loop's ``_base + "\n\n" + ...`` concatenation.
def _assemble_api_messages(messages, marker, plugin_user_context):
    glue_idx = select_turn_glue_idx(messages, marker)
    api_messages = []
    for idx, msg in enumerate(messages):
        api_msg = msg.copy()
        if idx == glue_idx:
            _base = api_msg.get("content", "")
            if isinstance(_base, str) and plugin_user_context:
                api_msg["content"] = _base + "\n\n" + plugin_user_context
        api_msg.pop(TURN_USER_MARKER_KEY, None)  # never sent to the provider
        api_messages.append(api_msg)
    return glue_idx, api_messages


def _marked(content, marker="M1"):
    return {"role": "user", "content": content, TURN_USER_MARKER_KEY: marker}


# ── Pure selector ─────────────────────────────────────────────────────────────

def test_selects_marked_current_turn_message():
    messages = [
        _marked("current question"),
        {"role": "assistant", "content": "hello"},
    ]
    assert select_turn_glue_idx(messages, "M1") == 0


def test_relocated_and_reobjected_after_compaction_is_still_found():
    """Compression drops old turns and rebuilds the list from *copies* — the
    current turn's user message survives as a NEW dict object at a new index.
    Object identity would fail here; the marker re-identifies it."""
    pre = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        _marked("turn 2 (current)"),
    ]
    assert select_turn_glue_idx(pre, "M1") == 2

    # ``[m.copy() for m in tail]`` — fresh objects, marker preserved.
    post = [{"role": "user", "content": "[compressed summary]"}] + [
        m.copy() for m in pre[2:]
    ]
    assert post[1] is not pre[2]                 # genuinely re-objected
    assert select_turn_glue_idx(post, "M1") == 1


def test_none_marker_returns_none():
    assert select_turn_glue_idx([_marked("q")], None) is None


def test_absent_marker_returns_none():
    messages = [
        {"role": "user", "content": "some other turn"},
        {"role": "assistant", "content": "hi"},
    ]
    assert select_turn_glue_idx(messages, "M1") is None


def test_stale_previous_turn_marker_is_not_matched():
    """Each turn mints a fresh marker; a previous turn's user message lingering
    in history must not be re-selected for the current turn's glue."""
    messages = [
        _marked("previous turn", marker="OLD"),
        {"role": "assistant", "content": "reply"},
        _marked("current turn", marker="NEW"),
    ]
    assert select_turn_glue_idx(messages, "NEW") == 2


def test_non_dict_elements_do_not_crash():
    messages = [None, "garbage", 42, _marked("ok")]
    assert select_turn_glue_idx(messages, "M1") == 3


# ── Request-level regressions (reviewer-requested) ────────────────────────────

def test_glue_lands_on_relocated_current_turn_after_compaction():
    """Request-level: after compaction relocates + re-objects the current-turn
    user message (a summary sits ahead of it), the ephemeral context is glued
    onto that message, not the global-last user (which is the summary here)."""
    post = [
        {"role": "user", "content": "[compressed summary of earlier turns]"},
        _marked("the real current question"),
    ]
    glue_idx, api = _assemble_api_messages(post, "M1", "[Mem] recall block")

    assert glue_idx == 1
    assert api[1]["content"] == "the real current question\n\n[Mem] recall block"
    assert "[Mem] recall block" not in api[0]["content"]   # summary untouched
    assert TURN_USER_MARKER_KEY not in api[1]              # stripped for provider


def test_following_synthetic_continuation_nudge_does_not_steal_glue():
    """Request-level: an UNFLAGGED length-continue / codex-ack nudge appended
    after the marked message is the global-last user, but must NOT receive the
    glue — it has no marker."""
    messages = [
        _marked("real question"),
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "[System: Continue now. Execute the tool calls.]"},
    ]
    glue_idx, api = _assemble_api_messages(messages, "M1", "[Mem] recall block")

    assert glue_idx == 0
    assert api[0]["content"] == "real question\n\n[Mem] recall block"
    assert "[Mem] recall block" not in api[3]["content"]   # nudge untouched


def test_following_flagged_verify_nudge_does_not_steal_glue():
    """Same, for a FLAGGED verify-stop nudge (``_verification_stop_synthetic``);
    marker absence — not flag inspection — is what protects the target."""
    messages = [
        _marked("real question"),
        {"role": "assistant", "content": "premature answer",
         "_verification_stop_synthetic": True},
        {"role": "user", "content": "[verify] run your checks before answering",
         "_verification_stop_synthetic": True},
    ]
    glue_idx, api = _assemble_api_messages(messages, "M1", "[Mem] recall block")

    assert glue_idx == 0
    assert api[0]["content"] == "real question\n\n[Mem] recall block"
    assert "[Mem] recall block" not in api[2]["content"]


def test_current_turn_message_compacted_away_drops_context_not_misattaches():
    """Request-level: if the current-turn user message was compacted away
    entirely (marker gone), context is dropped from the request — NOT glued
    onto a surviving synthetic nudge. (conversation_loop warns in this case.)"""
    messages = [
        {"role": "user", "content": "[compressed summary]"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "[System: Continue now.]"},   # nudge, no marker
    ]
    glue_idx, api = _assemble_api_messages(messages, "M1", "[Mem] recall block")

    assert glue_idx is None
    assert all("[Mem] recall block" not in (m.get("content") or "") for m in api)
