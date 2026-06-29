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
