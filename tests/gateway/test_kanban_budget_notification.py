from gateway.kanban_watchers import _format_gave_up_notification


def test_budget_exhaustion_is_not_described_as_a_spawn_failure():
    message = _format_gave_up_notification(
        board_tag="[board] ",
        tag="@worker ",
        task_id="t_budget",
        payload={
            "trigger_outcome": "timed_out",
            "budget_used": 18,
            "budget_max": 18,
            "error": "Iteration budget exhausted (18/18)",
        },
    )

    assert "iteration budget exhausted (18/18)" in message.lower()
    assert "spawn failure" not in message.lower()
