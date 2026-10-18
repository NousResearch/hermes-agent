"""A post-renewal read race cannot downgrade exact source into ordinary text."""

import pytest

from tests.agent.test_run_agent import agent as _agent_fixture

agent = _agent_fixture


@pytest.mark.parametrize("change", ["replace", "delete"])
def test_source_changed_after_renewal_fails_before_sdk(agent, tmp_path, monkeypatch, change):
    from agent import llm_egress_runtime as runtime
    from agent.source_provenance import SourceProvenanceError
    from tests.agent.test_context_source_lifecycle import _dispatch, _prepare, _sdk
    source, message, turn = _prepare(agent, tmp_path)
    read = runtime._read_grant_text
    def changed_at_final_read(grant):
        if change == "delete":
            source.unlink()
        else:
            source.write_text("my_value = changed_value + 456\n")
        return read(grant)
    monkeypatch.setattr(runtime, "_read_grant_text", changed_at_final_read)
    captured = []
    with _sdk(agent, captured) as sdk:
        with pytest.raises(SourceProvenanceError):
            _dispatch(agent, sdk, message, turn, 1)
    assert captured == []
