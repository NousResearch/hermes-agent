from gateway.run import _build_gateway_agent_history, _build_resume_pending_message


def _assert_no_same_role_neighbors(messages):
    previous = None
    for message in messages:
        role = message.get("role")
        if role in {"system", "tool"}:
            continue
        assert role != previous
        previous = role


def test_resume_note_is_prepended_to_synthetic_turn_and_history_alternates():
    history = [
        {"role": "user", "content": "run the migration"},
        {
            "role": "assistant",
            "content": "Operation interrupted.",
            "finish_reason": "interrupt_close",
        },
    ]
    agent_history, _observed = _build_gateway_agent_history(history)

    message, surface_and_ask = _build_resume_pending_message(
        agent_history=agent_history,
        message="",
        reason_phrase="a gateway restart",
    )

    assert surface_and_ask is True
    assert message.startswith("[System note:")
    assert "Tell the user concisely what you had COMPLETED" in message
    assert "\n\n" not in message

    assembled = agent_history + [{"role": "user", "content": message}]
    _assert_no_same_role_neighbors(assembled)
