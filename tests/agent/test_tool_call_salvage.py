"""Regression tests for text-form tool-call salvage (IQ3 reasoning-content artifact)."""

import json

from agent.tool_call_salvage import salvage_tool_calls_from_text

VALID = {"terminal", "execute_code", "read_file"}

SAMPLE_PLAIN = 'I should check the repo state first.\n\n<function=terminal>\n<parameter=command>\ncd ~/.hermes/hermes-agent && git status --short | head\n</parameter>\n</function>\n'
SAMPLE_TWO_CALLS = 'Lass mich die Datei lesen.\n\n<function=read_file>\n<parameter=path>/home/n0g00d/.hermes/hermes-agent/agent/turn_response_intake.py\n</parameter>\n<parameter=offset>120\n</parameter>\n</function>\n<function=terminal>\n<parameter=command>echo hi\n</parameter>\n</function>'
SAMPLE_BAD_NAME = '<function=not_a_real_tool>\n<parameter=x>1\n</parameter>\n</function>'
SAMPLE_NO_PARAMS = '<function=terminal>\n</function>'
SAMPLE_MULTILINE = "<function=execute_code>\n<parameter=code>from hermes_tools import terminal\nr = terminal('ls')\nprint(r['output'])\n</parameter>\n</function>"


def test_single_call_salvaged():
    calls = salvage_tool_calls_from_text(SAMPLE_PLAIN, VALID)
    assert len(calls) == 1
    tc = calls[0]
    assert tc.function.name == "terminal"
    args = json.loads(tc.function.arguments)
    assert args["command"].startswith("cd ~/.hermes/hermes-agent")
    assert tc.type == "function" and tc.id


def test_multiple_calls_and_params():
    calls = salvage_tool_calls_from_text(SAMPLE_TWO_CALLS, VALID)
    assert len(calls) == 2
    a0 = json.loads(calls[0].function.arguments)
    assert a0["path"].endswith("turn_response_intake.py") and a0["offset"] == "120"
    assert json.loads(calls[1].function.arguments)["command"] == "echo hi"


def test_invalid_tool_name_rejected():
    assert salvage_tool_calls_from_text(SAMPLE_BAD_NAME, VALID) == []


def test_no_params_skipped():
    assert salvage_tool_calls_from_text(SAMPLE_NO_PARAMS, VALID) == []


def test_multiline_code_value_preserved():
    calls = salvage_tool_calls_from_text(SAMPLE_MULTILINE, VALID)
    assert len(calls) == 1
    code = json.loads(calls[0].function.arguments)["code"]
    assert code.startswith("from hermes_tools import terminal")
    assert "print(r['output'])" in code


def test_plain_text_no_salvage():
    assert salvage_tool_calls_from_text("Die Analyse ist abgeschlossen.", VALID) == []
    assert salvage_tool_calls_from_text("", VALID) == []
    assert salvage_tool_calls_from_text(None, VALID) == []


def test_attr_style_name():
    text = '<function name="terminal"><parameter=command>ls\n</parameter></function>'
    calls = salvage_tool_calls_from_text(text, VALID)
    assert len(calls) == 1 and calls[0].function.name == "terminal"
