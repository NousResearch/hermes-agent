from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_single_action_json_silence_envelopes_are_intentional_silence():
    for payload in (
        '{"action":"NO_REPLY"}',
        '{\n  "action": "NO_REPLY"\n}',
    ):
        assert is_intentional_silence_response(payload)
        assert is_autonomous_silence_response(payload)


def test_json_silence_envelope_requires_one_exact_action_key():
    for payload in (
        '{"action":"NO_REPLY","reason":"duplicate"}',
        '{"Action":"NO_REPLY"}',
        '{"action":"SILENT"}',
        '{"action":" NO_REPLY "}',
        '{"action":"NO_REPLY","action":"NO_REPLY"}',
        '["NO_REPLY"]',
        '{"action":',
        '\u00a0{"action":"NO_REPLY"}',
        '{"action":"NO_REPLY"}\u2028',
    ):
        assert not is_intentional_silence_response(payload)
        assert not is_autonomous_silence_response(payload)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


