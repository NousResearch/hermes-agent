from gateway.response_filters import (
    HANDOFF_GUARD_REPLACEMENT,
    guard_handoff_response,
    is_handoff_leak_response,
    is_partial_handoff_leak_candidate,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_edge_punctuation_silence_tokens_are_intentional_silence():
    for token in (".NO_REPLY", "*NO_REPLY*", " .NO_REPLY ", "*[SILENT]*", "NO_REPLY."):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")
    assert not is_intentional_silence_response("😄 NO_REPLY")
    assert not is_intentional_silence_response("[SILENT")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def test_long_internal_handoff_is_replaced():
    raw = "<analysis>" + ("internal state " * 50)
    assert is_handoff_leak_response(raw)
    assert guard_handoff_response(raw) == HANDOFF_GUARD_REPLACEMENT


def test_handoff_guard_keeps_short_or_ordinary_text():
    short = "<summary>brief user-facing summary</summary>"
    ordinary = "Here is how the <analysis> tag works. " + ("x" * 700)
    assert guard_handoff_response(short) == short
    assert guard_handoff_response(ordinary) == ordinary


def test_handoff_stream_prefix_is_held_until_decided():
    assert is_partial_handoff_leak_candidate("<ana")
    assert is_partial_handoff_leak_candidate("<analysis>" + ("x" * 500))
    assert not is_partial_handoff_leak_candidate("<analysis>" + ("x" * 601))


def test_handoff_guard_escape_hatch(monkeypatch):
    raw = "<summary>" + ("internal state " * 50)
    monkeypatch.setenv("HERMES_HANDOFF_GUARD", "0")
    assert guard_handoff_response(raw) == raw
