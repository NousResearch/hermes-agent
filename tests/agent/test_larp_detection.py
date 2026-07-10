"""Unit tests for the post-turn LARP guard (agent/larp_detection.py).

Tier-1 (deterministic) only — no LLM judge. Asserts the three-way contract:
claim+no-tool -> re-prompt; claim+failed-tool -> pass (escalate); claim+success -> pass.
"""

from __future__ import annotations

from agent.larp_detection import build_larp_nudge, larp_detection_enabled

CFG = {"larp_detection": {"max_reprompts": 2, "exempt_toolsets": []}}


def _u(c):
    return {"role": "user", "content": c}


def _a(name):
    return {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": name, "arguments": "{}"}}]}


def _t(name, content):
    return {"role": "tool", "tool_call_id": "1", "name": name, "content": content}


def test_claim_with_no_tool_call_is_larp():
    assert build_larp_nudge(messages=[_u("do x")], final_response="I have updated the file.", config=CFG) is not None


def test_claim_with_successful_tool_passes():
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "ok wrote 10 lines")]
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=CFG) is None


def test_claim_with_failed_tool_passes_not_reprompt():
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "Error executing tool 'write_file': denied")]
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=CFG) is None


def test_no_claim_passes():
    assert build_larp_nudge(messages=[_u("hi")], final_response="Here is a summary of the weather.", config=CFG) is None


def test_modal_is_not_a_claim():
    fr = "I should update the file but need confirmation first."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None


def test_narrate_then_stop_is_larp():
    fr = "Sounds good. I'll now run the ingestion script."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_any_tool_call_passes_by_default():
    # default exempt is empty -> a memory call counts as real work.
    msgs = [_u("x"), _a("memory"), _t("memory", "saved")]
    assert build_larp_nudge(messages=msgs, final_response="I have saved that.", config=CFG) is None


def test_strict_exempt_flags_housekeeping_only_turn():
    msgs = [_u("x"), _a("memory"), _t("memory", "saved")]
    cfg = {"larp_detection": {"exempt_toolsets": ["memory"]}}
    assert build_larp_nudge(messages=msgs, final_response="I have updated the database.", config=cfg) is not None


def test_reprompt_cap():
    assert build_larp_nudge(messages=[_u("x")], final_response="I have updated the file.", config=CFG, attempts=2) is None


def test_disabled_by_default():
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}) is False


def test_env_override_enables(monkeypatch):
    monkeypatch.setenv("HERMES_LARP_DETECTION", "1")
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}) is True


# ---- tuning from real session strings (narrate-then-stop / terminal action) ----

def test_present_progressive_narrate_is_larp():
    # "I am dispatching ..." — the dominant form the old future-only regex MISSED.
    fr = "The next 5 products to research (Phase 3): ...\n\nI am dispatching the sub-agents now."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_bare_terminal_action_is_larp():
    for fr in ("Executing now.", "Starting Batch 1 now.", "Proceeding with dispatch...",
               "Correcting the script creation now..."):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None, fr


def test_proceeding_with_item_now_is_larp():
    fr = "### Project State\n- Total Completed: 36\n\nI am proceeding with PGPx9944 now."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_permission_question_now_is_not_larp():
    # trailing "?" => asking permission, not claiming -> must NOT flag.
    for fr in ("Want me to execute this now?", "Want me to snip Figures 3 and 4 now?",
               "Should I proceed now?"):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_waiting_status_is_not_larp():
    fr = "**Current KG state:** 103 products.\n\nWaiting for batch 3 result..."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None


def test_i_am_state_is_not_larp():
    # "I am ready/unable" are states, not gerund actions -> must NOT match narrate.
    for fr in ("I am ready to help with the next step.", "I am unable to do that right now."):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_present_progressive_with_tool_passes():
    # (c) a real tool call this turn backs the announcement -> pass.
    msgs = [_u("x"), _a("delegate_task"), _t("delegate_task", "spawned 5 subagents")]
    fr = "I am dispatching the sub-agents now."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=CFG) is None


def test_post_compaction_window_enables_when_disabled():
    cfg = {"larp_detection": {"enabled": False, "post_compaction_window": 3}}

    class _InWindow:
        _turns_since_compaction = 1

    class _PastWindow:
        _turns_since_compaction = 5

    class _NeverCompacted:
        _turns_since_compaction = None

    assert larp_detection_enabled(cfg) is False                          # no agent -> off
    assert larp_detection_enabled(cfg, agent=_InWindow()) is True        # within window -> on
    assert larp_detection_enabled(cfg, agent=_PastWindow()) is False     # past window -> off
    assert larp_detection_enabled(cfg, agent=_NeverCompacted()) is False # never compacted -> off


def test_post_compaction_window_default_off():
    class _JustCompacted:
        _turns_since_compaction = 0

    # default window 0 -> disabled stays disabled even right after compaction.
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}, agent=_JustCompacted()) is False
