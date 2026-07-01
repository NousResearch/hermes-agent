from gateway.run import _build_resume_pending_message


def _flagged_interrupt_history():
    return [
        {"role": "user", "content": "run the migration"},
        {
            "role": "assistant",
            "content": "Operation interrupted.",
            "_interrupt_close": True,
        },
    ]


def test_flagged_interrupt_close_surfaces_and_asks_without_skip():
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=_flagged_interrupt_history(),
        message="",
        reason_phrase="a gateway restart",
    )

    assert surface_and_ask is True
    assert "ask whether to pick it back up" in message
    assert "skip any unfinished work" not in message
    assert "continue your in-progress work" not in message
    assert "do NOT re-execute or verify it" in message


def test_plain_assistant_tail_uses_idle_restore_branch():
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=[{"role": "assistant", "content": "done"}],
        message="",
        reason_phrase="a gateway restart",
    )

    assert surface_and_ask is False
    assert "skip any unfinished work" in message
    assert "restored" in message


def test_operation_interrupted_text_without_flag_does_not_surface_and_ask():
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=[{"role": "assistant", "content": "Operation interrupted."}],
        message="",
        reason_phrase="a gateway restart",
    )

    assert surface_and_ask is False
    assert "skip any unfinished work" in message
    assert "ask whether to pick it back up" not in message


def test_new_message_first_mentions_prior_interrupted_task():
    message, surface_and_ask = _build_resume_pending_message(
        agent_history=_flagged_interrupt_history(),
        message="what happened?",
        reason_phrase="a gateway restart",
    )

    assert surface_and_ask is False
    assert "Address the user's NEW message below FIRST" in message
    assert "prior task was interrupted by the restart" in message
    assert message.endswith("\n\nwhat happened?")
