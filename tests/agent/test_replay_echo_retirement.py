"""Replay-echo retirement for interrupt placeholders (#132949, alongside #81841).

A hidden assistant row still reaches the provider as assistant content on replay.
Before the wording change it carried ``[response interrupted]`` — a short
natural-language phrase in the model's own prior turn, which the model reproduces
verbatim (clean ``finish_reason=stop`` answering an ordinary instruction with just
the placeholder). The prep filter neutralises (never drops) hidden rows carrying a
pre-change spelling on a copy: removal could form ``tool -> user`` (#48879) or a
``user -> user`` pair that repair merges.
"""

from typing import Any

LEGACY = "[response interrupted]"
NEW_PLACEHOLDER = "[interrupt: no assistant output for this turn]"


def _agent(tmp_path, monkeypatch) -> Any:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _prepare(tmp_path, monkeypatch, messages):
    """Run prepare_iteration and return the messages it produced."""
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        user_message = messages[-1].get("content") if messages else None
        prep = prepare_iteration(
            agent, messages=messages, api_call_count=1,
            user_message=user_message, current_turn_user_idx=max(len(messages) - 1, 0),
        )
        return list(prep.messages)
    finally:
        agent._session_db.close()


def _hidden_row(text):
    return {"role": "assistant", "content": "", "display_kind": "hidden", "api_content": text}


def test_tool_tail_row_is_kept_and_neutralised(tmp_path, monkeypatch):
    """assistant(tool_calls) -> tool -> hidden(legacy) -> user: dropping would
    recreate tool -> user (#48879). The row stays, but its echoable text is
    replaced with the post-fix placeholder. The persisted redirect shape
    user -> hidden(legacy) -> user(checkpoint) must not collapse either: a
    dropped row leaves user -> user, which repair merges, losing the
    correction's checkpoint sidecar and rewriting the first row in place."""
    import copy
    checkpoint = "[Context from the interrupted assistant response]\nhalf a poem\n\nmake it short"
    redirect = [
        {"role": "user", "content": "write a poem"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "make it short", "api_content": checkpoint},
    ]
    stored = copy.deepcopy(redirect)
    (tmp_path / "redirect").mkdir()
    out = _prepare(tmp_path / "redirect", monkeypatch, redirect)
    assert redirect == stored, "stored rows were rewritten in place"
    assert [m["role"] for m in out] == ["user", "assistant", "user"], out
    assert out[1]["api_content"] == NEW_PLACEHOLDER
    assert out[2].get("api_content") == checkpoint, f"checkpoint sidecar lost: {out}"

    messages = [
        {"role": "user", "content": "edit the file"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "patch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "they do! increase the timing"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    assert messages[3]["api_content"] == LEGACY  # durable row dict is never rewritten in place

    hidden = [m for m in out
              if m.get("role") == "assistant" and m.get("display_kind") == "hidden"]
    assert len(hidden) == 1, f"tool-tail row was dropped (tool -> user would revive): {out}"
    row = hidden[0]
    assert row.get("content") == ""
    assert row.get("api_content") == NEW_PLACEHOLDER, (
        f"legacy text survived neutralisation: {row.get('api_content')!r}"
    )
    # The sequence still has no tool -> user adjacency.
    for i in range(len(out) - 1):
        if out[i].get("role") == "tool":
            assert out[i + 1].get("role") != "user", (
                f"role-alternation violation: tool -> user at index {i}: {out}"
            )


def test_visible_echoed_placeholder_rows_are_neutralised_to_hidden(tmp_path, monkeypatch):
    """#150052: When an earlier turn echoed ``_INTERRUPTED_PLACEHOLDER`` (or a legacy
    placeholder) and persisted it as a visible assistant message (``display_kind=None``,
    ``content=NEW_PLACEHOLDER``), ``prepare_iteration`` neutralises a copy into the
    canonical hidden row (``display_kind="hidden"``, ``content=""``,
    ``api_content=NEW_PLACEHOLDER``) without mutating stored dicts or re-triggering on
    already-hidden canonical rows."""
    import copy
    from agent.turn_iteration_prep import _neutralise_replay_echo_ghosts

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": NEW_PLACEHOLDER, "finish_reason": "stop"},
        {"role": "user", "content": "why did you stop?"},
        {"role": "assistant", "content": LEGACY, "finish_reason": "stop"},
        {"role": "user", "content": "still there?"},
    ]
    stored = copy.deepcopy(messages)
    out = _prepare(tmp_path, monkeypatch, messages)
    assert messages == stored, "stored rows were mutated in place"
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    for idx in (1, 3):
        assert out[idx].get("display_kind") == "hidden", f"row {idx} stayed visible: {out[idx]}"
        assert out[idx].get("content") == "", f"row {idx} kept visible content: {out[idx]}"
        assert out[idx].get("api_content") == NEW_PLACEHOLDER

    # Idempotence: feeding the neutralised list back must not re-trigger neutralisation.
    again, count = _neutralise_replay_echo_ghosts(out)
    assert count == 0
    assert again is out


def test_echoed_placeholder_response_is_retried_and_never_persisted(tmp_path, monkeypatch):
    """#150052: When the provider echoes ``_INTERRUPTED_PLACEHOLDER`` as a clean
    ``finish_reason=stop`` response (with no tool calls in the last 5 messages),
    the loop must treat it as an empty/echoed turn, nudge once, return the
    recovered answer, and never persist the placeholder as a visible assistant turn."""
    from unittest.mock import MagicMock, patch
    from tests.agent.test_run_agent import _mock_response

    agent = _agent(tmp_path, monkeypatch)
    try:
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

        def _create_side_effect(*_a, **_kw):
            idx = agent.client.chat.completions.create.call_count - 1
            if idx == 0:
                agent._current_streamed_assistant_text = NEW_PLACEHOLDER
                return _mock_response(content=NEW_PLACEHOLDER, finish_reason="stop")
            assert agent._current_streamed_assistant_text == "", "echoed placeholder survived in stream buffer"
            return _mock_response(content="Real answer after recovery.", finish_reason="stop")

        agent.client.chat.completions.create.side_effect = _create_side_effect
        history = [
            {"role": "user", "content": "earlier task"},
            _hidden_row(NEW_PLACEHOLDER),
        ]
        with patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation("how are you reading this?", conversation_history=history)

        assert result["completed"] is True
        assert result["final_response"] == "Real answer after recovery."
        assert agent.client.chat.completions.create.call_count == 2
        retry_wire = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        from agent.agent_runtime_helpers_placeholders import _ECHOED_PLACEHOLDER_NUDGE
        assert retry_wire[-1]["content"] == _ECHOED_PLACEHOLDER_NUDGE

        visible_assistants = [
            m for m in result["messages"]
            if m.get("role") == "assistant" and m.get("display_kind") != "hidden"
        ]
        assert [m.get("content") for m in visible_assistants] == ["Real answer after recovery."]
        assert not any(m.get("_empty_recovery_synthetic") for m in result["messages"])
    finally:
        agent._session_db.close()




