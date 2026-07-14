from types import SimpleNamespace

from agent.conversation_loop import (
    _tool_calls_include_verification_evidence,
    _verification_gate_has_evidence_tools,
    _verification_gate_required,
)


def test_verification_gate_required_for_explicit_file_check():
    assert _verification_gate_required(
        "Check /tmp/project/app.py and do not answer from memory."
    )


def test_verification_gate_required_for_chinese_check_prompt():
    assert _verification_gate_required(
        "检查一下 /Users/vivien/.hermes/config.yaml 里有没有这个字段，不要凭记忆回答。"
    )


def test_verification_gate_does_not_trigger_for_plain_explanation():
    assert not _verification_gate_required("Explain what tool_use_enforcement means.")


def test_verification_evidence_ignores_housekeeping_tools():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="memory", arguments="{}"),
    )

    assert not _tool_calls_include_verification_evidence([tool_call])


def test_verification_evidence_accepts_substantive_tools():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )

    assert _tool_calls_include_verification_evidence([tool_call])


def test_verification_gate_requires_substantive_available_tool():
    assert not _verification_gate_has_evidence_tools([])
    assert not _verification_gate_has_evidence_tools(["memory", "todo"])
    assert _verification_gate_has_evidence_tools(["memory", "terminal"])
