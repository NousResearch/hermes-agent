"""A hidden verification candidate must not be mistaken for a delivered answer."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from tests.agent.test_verification_continuation_budget import _response

pytest_plugins = ("tests.agent.test_verification_continuation_budget",)


@pytest.mark.parametrize("gate", ["verify_on_stop", "pre_verify"])
@pytest.mark.parametrize("delivery", ["hidden", "interim", "streamed", "partial"])
def test_verification_continuation_knows_whether_user_received_candidate(
    agent, monkeypatch, gate, delivery,
):
    agent.max_iterations = 3
    agent.iteration_budget.max_total = 3
    candidate = "The app has 100 downloads. All checks pass."
    corrected = "The app has 46 downloads; the check found duplicate exports."
    requests = []
    delivered = []
    agent.interim_assistant_callback = (
        (lambda text, **kwargs: delivered.append(text)) if delivery == "interim" else None
    )

    def model_call(kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        agent._turn_file_mutation_paths = {"changed.py"}
        if len(requests) == 1 and delivery in {"streamed", "partial"}:
            agent._current_streamed_assistant_text = (
                candidate if delivery == "streamed" else "The app"
            )
        return _response(candidate if len(requests) == 1 else corrected)

    agent._interruptible_api_call = model_call
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1" if gate == "verify_on_stop" else "0")
    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", side_effect=["Run checks.", None]),
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: gate == "pre_verify" and name == "pre_verify"),
        patch("hermes_cli.plugins.get_pre_verify_continue_message", side_effect=["Run checks.", None]),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("How is the app doing? Analyze the reports.")

    assert len(requests) == 2
    # The next model request retains its prefix and original question unchanged.
    assert requests[1][:len(requests[0])] == requests[0]
    continuation = requests[1][-1]
    assert continuation["role"] == "user"
    assert delivered == ([candidate] if delivery == "interim" else [])
    if delivery in {"interim", "streamed"}:
        assert "not been delivered" not in continuation["content"]
    else:
        assert "not been delivered" in continuation["content"]
        assert "complete, self-contained answer" in continuation["content"]
        assert "correct any claims" in continuation["content"]
    # Verification can invalidate the first report. Never resurrect it by concatenation.
    assert result["final_response"] == corrected
    assert result["completed"] is True
    assert not result["response_previewed"]
