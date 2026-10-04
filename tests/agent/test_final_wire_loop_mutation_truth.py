"""Detached mutation truth at later-loop compression, separate from prologue."""
from copy import deepcopy
import pytest
from tests.run_agent.test_413_compression import agent  # noqa: F401
from tests.agent.test_provider_bound_dispatch_guard import history, run_local, assert_unsafe_stopped


def test_local_admission_diagnostic_does_not_invent_timeout():
    from agent.conversation_compression import over_limit_local_stop_status
    text = over_limit_local_stop_status(transcript_rewritten=False)
    assert "No messages were dropped" in text
    assert "timed out" not in text
    assert "refused locally" in text


@pytest.mark.parametrize("mode", ["inplace", "equalcopy", "unknown"])
def test_later_loop_mutation_truth_uses_detached_snapshot(agent, mode):
    def compact(rows, *args, **kwargs):
        if mode == "equalcopy":
            return deepcopy(rows), agent._cached_system_prompt
        if mode == "unknown":
            rows[0]["opaque_test_state"] = object()
        else:
            rows[0]["content"] += "changed"
        return rows, agent._cached_system_prompt
    observation = run_local(agent, messages=history(), compress=compact)
    assert_unsafe_stopped(observation)
    text = " ".join(observation[3])
    if mode == "equalcopy":
        assert "No messages were dropped" in text
    else:
        assert "No messages were dropped" not in text
    if mode == "unknown":
        assert "mutation outcome is unknown" in text
