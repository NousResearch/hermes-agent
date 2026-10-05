"""Replay-echo retirement for interrupt placeholders (#132949, alongside #81841).

A hidden assistant row still reaches the provider as assistant content on replay.
Before the wording change it carried ``[response interrupted]`` — a short
natural-language phrase in the model's own prior turn, which the model reproduces
verbatim (clean ``finish_reason=stop`` answering an ordinary instruction with just
the placeholder). The prep filter must retire rows carrying a pre-change spelling,
while keeping the row whenever removing it would form ``tool -> user`` — the strict
provider failure ``close_interrupted_tool_sequence`` exists to prevent (#48879).
"""

import pytest


LEGACY = "[response interrupted]"
SCAFFOLD = "[This response was interrupted by a user correction.]"
NEW_PLACEHOLDER = "[interrupt: no assistant output for this turn]"


def _agent(tmp_path, monkeypatch):
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


def _hidden_row(text, *, content="", api_content=None):
    row = {"role": "assistant", "content": content, "display_kind": "hidden"}
    row["api_content"] = api_content if api_content is not None else text
    return row


# ---------------------------------------------------------------------------
# 1. Role-safe removal: the row is dropped outright.
# ---------------------------------------------------------------------------

def test_role_safe_legacy_row_is_dropped(tmp_path, monkeypatch):
    """assistant -> hidden(legacy) -> user: removal leaves a valid alternation,
    so the poisoned row must leave the replay entirely."""
    messages = [
        {"role": "user", "content": "do the thing"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "continue"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    assert not any(
        m.get("role") == "assistant" and m.get("display_kind") == "hidden"
        for m in out
    ), f"legacy hidden row survived: {out}"


def test_role_safe_scaffold_row_is_dropped(tmp_path, monkeypatch):
    """The pre-existing #81841 path keeps working under the widened filter."""
    messages = [
        {"role": "user", "content": "hi"},
        _hidden_row(SCAFFOLD),
        {"role": "user", "content": "continue"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    assert not any(
        m.get("role") == "assistant" and m.get("display_kind") == "hidden"
        for m in out
    ), f"scaffold ghost survived: {out}"


# ---------------------------------------------------------------------------
# 2. tool -> user protection: keep the row, neutralise the text.
# ---------------------------------------------------------------------------

def test_tool_tail_row_is_kept_and_neutralised(tmp_path, monkeypatch):
    """assistant(tool_calls) -> tool -> hidden(legacy) -> user: dropping would
    recreate tool -> user (#48879). The row stays, but its echoable text is
    replaced with the post-fix placeholder."""
    messages = [
        {"role": "user", "content": "edit the file"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "patch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "they do! increase the timing"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)

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


# ---------------------------------------------------------------------------
# 3. Rows already carrying the post-fix placeholder are untouched.
# ---------------------------------------------------------------------------

def test_post_fix_placeholder_row_is_untouched(tmp_path, monkeypatch):
    """The current placeholder is not a hazard: a freshly written hidden row must
    not be retired or rewritten on every request."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "display_kind": "hidden",
         "api_content": NEW_PLACEHOLDER},
        {"role": "user", "content": "continue"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    hidden = [m for m in out
              if m.get("role") == "assistant" and m.get("display_kind") == "hidden"]
    assert len(hidden) == 1, f"post-fix placeholder row was retired: {out}"
    assert hidden[0].get("api_content") == NEW_PLACEHOLDER


def test_visible_interrupt_banner_is_kept(tmp_path, monkeypatch):
    """Caller-supplied banners (real diagnostic text) are not hidden and never
    match the filter — they stay exactly as written."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant",
         "content": "Operation interrupted: waiting for model response (4.9s elapsed)."},
        {"role": "user", "content": "continue"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    assert any(
        m.get("content", "").startswith("Operation interrupted: waiting")
        for m in out
    ), f"visible banner lost: {out}"
