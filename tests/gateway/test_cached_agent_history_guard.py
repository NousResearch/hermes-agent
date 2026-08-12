"""Regression tests for cached gateway transcript selection."""

from gateway.run import _build_gateway_agent_history, _select_cached_agent_history


def test_select_cached_history_keeps_expected_replay_cleanup():
    """A durable read-only tool tail removed by stock replay cleanup stays removed."""
    live = [
        {"role": "user", "content": "check status", "_db_persisted": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
            "_db_persisted": True,
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "[Command interrupted]",
            "_db_persisted": True,
        },
    ]

    persisted, observed = _build_gateway_agent_history(live)

    assert observed is None
    assert len(persisted) < len(live)
    assert persisted == [{"role": "user", "content": "check status"}]
    assert _select_cached_agent_history(persisted, live) is persisted


def test_select_cached_history_preserves_live_when_a_real_row_is_unpersisted():
    persisted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    live = [
        {"role": "user", "content": "hello", "_db_persisted": True},
        {"role": "assistant", "content": "hi", "_db_persisted": True},
        {"role": "user", "content": "not written"},
    ]

    selected = _select_cached_agent_history(persisted, live)

    assert selected is not persisted
    assert selected == live


def test_select_cached_history_ignores_unmarked_ephemeral_scaffolding():
    """Internal retry nudges are deliberately unpersisted, not evidence of DB lag."""
    persisted = [{"role": "user", "content": "ship the fix"}]
    live = [
        {"role": "user", "content": "ship the fix", "_db_persisted": True},
        {
            "role": "user",
            "content": "Verify your work before stopping.",
            "_pre_verify_synthetic": True,
        },
    ]

    assert _select_cached_agent_history(persisted, live) is persisted
