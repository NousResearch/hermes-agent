from agent.terminal_boundary import enforce_public_terminal_response


def test_owner_response_without_terminal_outcome_fails_closed_to_argh():
    result = enforce_public_terminal_response(
        {"final_response": "The work is not finished yet.", "completed": False},
        platform="webui",
    )
    assert result["public_terminal_outcome"] == "ARGH"
    assert result["final_response"].startswith("ARGH\n")
    assert result["failed"] is True
    assert result["completed"] is False


def test_owner_response_with_one_terminal_outcome_passes_through():
    result = enforce_public_terminal_response(
        {"final_response": "DONE\nThe verified artifact is ready."},
        platform="webui",
    )
    assert result["public_terminal_outcome"] == "DONE"
    assert result["terminal_boundary_enforced"] is True


def test_internal_child_response_is_not_rewritten():
    original = {"final_response": "The child should continue."}
    assert enforce_public_terminal_response(original, platform="subagent") is original
    assert "public_terminal_outcome" not in original
