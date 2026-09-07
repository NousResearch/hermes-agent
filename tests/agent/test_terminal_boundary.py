from agent.terminal_boundary import enforce_public_terminal_response


def test_plain_provider_response_is_preserved_and_gets_external_disposition():
    result = enforce_public_terminal_response(
        {"final_response": "4", "completed": True, "failed": False},
        platform="webui",
    )
    assert result["final_response"] == "4"
    assert result["core_control"]["outcome"] == "DONE"
    assert result["core_control"]["response_preserved"] is True


def test_incomplete_response_is_not_rewritten_by_core_control():
    result = enforce_public_terminal_response(
        {"final_response": "The work is not finished yet.", "completed": False},
        platform="webui",
    )
    assert result["final_response"] == "The work is not finished yet."
    assert result["core_control"]["outcome"] == "ARGH"


def test_explicit_meeting_is_metadata_only():
    response = "MEETING\nShould we approve the next step?"
    result = enforce_public_terminal_response({"final_response": response}, platform="webui")
    assert result["final_response"] == response
    assert result["core_control"]["outcome"] == "MEETING"


def test_internal_child_response_is_not_rewritten_or_invoked():
    original = {"final_response": "The child should continue."}
    assert enforce_public_terminal_response(original, platform="subagent") is original
    assert "core_control" not in original
