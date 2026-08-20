from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


def test_autonomous_silence_tolerates_dropped_closing_bracket():
    """Models intermittently emit "[SILENT" — the marker must still suppress."""
    assert is_autonomous_silence_response("[SILENT")
    assert is_autonomous_silence_response("[SILENT\n\nNothing new this tick.")
    assert is_autonomous_silence_response("Nothing new this tick.\n\n[SILENT")
    assert is_autonomous_silence_response("[SILENT No changes detected")
    # Genuine prose that merely starts similarly must still be delivered.
    assert not is_autonomous_silence_response("[SILENT-mode] rollout finished")
    assert not is_autonomous_silence_response("Silent retry succeeded")


