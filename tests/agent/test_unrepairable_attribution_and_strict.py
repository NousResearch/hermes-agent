"""A1/A2/A3 polarity tests (RCA 2026-09-17, follow-up to 3d5161618c).

A3: Unrepairable census lines carry model/session attribution.
A1: HERMES_STRICT_REPAIRED_ARGS refuses repaired write-class args (returns "{}"
    so the sentinel path drops them); default OFF keeps repairs flowing.
A2: blank args + finish_reason length/max_tokens never execute silently.
"""
import json
import logging
from types import SimpleNamespace

from agent.message_sanitization import _repair_tool_call_arguments


def _unrepairable_payload() -> str:
    # Real 02:54 17/9 truncation subclass — missing closing quote: brace-closing
    # cannot fix this (verified: passes 1-4 all fail)
    return '{"action": "list}'


def _repairable_payload() -> str:
    # Real repairable subclass (pass 1-3 fixable): trailing comma — brace-closing
    # yields balanced, loads-valid JSON
    return '{"operations": [{"action": "list"},]}'


def test_a3_attribution_on_unrepairable_line(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
        out = _repair_tool_call_arguments(_unrepairable_payload(), "delegate_task",
                                          model="z-ai/glm-5.3", session="abc123")
    assert out == "{}"
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "Unrepairable tool_call arguments for delegate_task" in joined
    assert "[model=z-ai/glm-5.3 session=abc123]" in joined


def test_a3_no_attribution_keeps_legacy_format(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
        _repair_tool_call_arguments(_unrepairable_payload(), "delegate_task")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "[model=" not in joined  # kwargs absent → census line unchanged


def test_a1_default_off_repaired_write_class_still_repaired(monkeypatch):
    monkeypatch.delenv("HERMES_STRICT_REPAIRED_ARGS", raising=False)
    out = _repair_tool_call_arguments(_repairable_payload(), "memory")
    assert out != "{}"
    assert json.loads(out) == {"operations": [{"action": "list"}]}


def test_a1_strict_on_refuses_repaired_write_class(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", "1")
    with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
        out = _repair_tool_call_arguments(_repairable_payload(), "memory",
                                          model="z-ai/glm-5.3", session="s1")
    assert out == "{}"  # call-site converts to sentinel → dropped + re-issue error
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "Strict repaired-args mode: refusing repaired write-class args for memory" in joined


def test_a1_strict_on_non_write_class_still_repaired(monkeypatch):
    monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", "1")
    out = _repair_tool_call_arguments(_repairable_payload(), "terminal")
    assert out != "{}"


def test_a1_strict_truthy_variants(monkeypatch):
    for val in ("true", "YES", "1"):
        monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", val)
        assert _repair_tool_call_arguments(_repairable_payload(), "patch") == "{}"
    monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", "0")
    assert _repair_tool_call_arguments(_repairable_payload(), "patch") != "{}"


def test_a2_blank_args_with_length_finish_reason_refused():
    """End-to-end through the streaming assembler: blank args + length must not
    produce an executable call."""
    from agent.chat_completion_helpers import _StreamingCall
    acc = {0: {"id": "call_0", "type": "function", "extra_content": None,
               "function": {"name": "delegate_task", "arguments": ""}}}
    calls, flagged = _StreamingCall._assemble_tool_calls(acc, "length")
    # blank args + finish_reason present: not flagged by assembler (existing
    # behavior), but validation below must refuse — asserted in test_a2_validation.
    assert flagged is False


def test_a2_validation_blank_args_length_never_executes():
    """BEHAVIOUR test (CodeRabbit PR #3, Minor): call validate_tool_calls with
    blank args + finish_reason='length' and assert the verdict refuses
    execution (action 'return' or 'continue' — never 'ok'), instead of
    grepping module source."""
    from types import SimpleNamespace
    from agent import turn_tool_validation as ttv

    agent = SimpleNamespace(
        valid_tool_names={"delegate_task"},
        _uniquify_tool_call_ids=lambda tcs: None,
        _repair_tool_call=lambda name: None,
        _buffer_vprint=lambda *a, **k: None,
        _vprint=lambda *a, **k: None,
        log_prefix="[test] ",
        _invalid_tool_retries=0,
        _invalid_json_args_strikes=0,
        _invalid_json_retries=0,
        _cleanup_task_resources=lambda task_id: None,
        _persist_session=lambda msgs, hist: None,
    )
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(
        id="call_0", type="function",
        function=SimpleNamespace(name="delegate_task", arguments=""))])

    verdict = ttv.validate_tool_calls(
        agent, msg, "length",
        messages=[{"role": "assistant", "tool_calls": []}],
        conversation_history=[], api_call_count=1, effective_task_id=None,
    )
    assert verdict.action in ("return", "continue"), (
        f"blank-args+length must not dispatch, got action={verdict.action!r}"
    )
    assert verdict.action != "ok"


def test_repair_unchanged_for_clean_json():
    good = json.dumps({"a": 1})
    assert json.loads(_repair_tool_call_arguments(good, "memory")) == {"a": 1}


def test_pass0_strict_mode_gates_control_char_repair(monkeypatch):
    """CodeRabbit PR #3 (Major): a write-class arg with literal control chars
    is pass-0-repaired (strict=False + reserialise) BEFORE the strict guards
    used to run — strict mode must refuse it there too."""
    raw = '{"text": "line1\nline2"}'  # literal \n inside string
    monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", "1")
    out = _repair_tool_call_arguments(raw, "memory")
    parsed = json.loads(out)
    assert parsed.get("__hermes_malformed_tool_arguments__") is True
    monkeypatch.setenv("HERMES_STRICT_REPAIRED_ARGS", "0")
    out_off = _repair_tool_call_arguments(raw, "memory")
    assert json.loads(out_off) == {"text": "line1\nline2"}


def test_unrepairable_still_sentinel_at_assemble():
    from agent.chat_completion_helpers import _StreamingCall
    acc = {0: {"id": "call_0", "type": "function", "extra_content": None,
               "function": {"name": "memory", "arguments": _unrepairable_payload()}}}
    calls, flagged = _StreamingCall._assemble_tool_calls(acc, "tool_calls",
                                                          model_name="z-ai/glm-5.3",
                                                          session_id="s9")
    parsed = json.loads(calls[0].function.arguments)
    assert parsed.get("__hermes_malformed_tool_arguments__") is True
    assert flagged is True
