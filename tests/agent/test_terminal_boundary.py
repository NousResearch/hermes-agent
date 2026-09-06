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


def test_done_without_directed_completion_receipt_fails_closed_to_argh():
    result = enforce_public_terminal_response(
        {"final_response": "DONE\nThe verified artifact is ready."},
        platform="webui",
    )
    assert result["public_terminal_outcome"] == "ARGH"
    assert result["final_response"].startswith("ARGH\n")
    assert "what was done" in result["done_contract_missing"]
    assert "exactly one recommended next objective" in result["done_contract_missing"]


def test_done_with_directed_completion_receipt_passes_through():
    response = """DONE
What was done: implemented the public response boundary.
Evidence: focused boundary tests and a fresh-process invocation passed.
How Hermes verified it: independently reran the tests and read back the public result.
How completion advanced the project: ordinary turns can no longer bypass the governed stop contract.
Recommended next objective: approve the next boundary hardening slice.
Evidence: the remaining public entry points are enumerated in the acceptance record.
Acceptance-state criteria: each entry point emits exactly one canonical outcome.
Verification method: fresh-process tests through the actual user-facing route.
"""
    result = enforce_public_terminal_response(
        {"final_response": response},
        platform="webui",
    )
    assert result["public_terminal_outcome"] == "DONE"
    assert result["terminal_boundary_enforced"] is True


def test_owner_response_with_one_terminal_outcome_passes_through():
    result = enforce_public_terminal_response(
        {"final_response": "MEETING\nShould we approve the next step?"},
        platform="webui",
    )
    assert result["public_terminal_outcome"] == "MEETING"
    assert result["terminal_boundary_enforced"] is True


def test_internal_child_response_is_not_rewritten():
    original = {"final_response": "The child should continue."}
    assert enforce_public_terminal_response(original, platform="subagent") is original
    assert "public_terminal_outcome" not in original
