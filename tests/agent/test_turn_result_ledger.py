import json

from agent.turn_result_ledger import (
    MIN_SUBSTANTIVE_TOOL_COMPLETIONS,
    TURN_RESULT_LEDGER_MARKER,
    TurnResultLedger,
)


def _tool_pair(
    index: int,
    name: str,
    *,
    arguments: dict | None = None,
    result: str = "ok",
) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments or {}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": result,
        },
    ]


def _ledger(
    history: list[dict] | None = None,
    initial_todos: list[dict] | None = None,
) -> TurnResultLedger:
    return TurnResultLedger.start(
        turn_id="session:task:turn",
        original_user_message="Implement the reconciliation approach you recommended.",
        conversation_history=history or [],
        initial_todo_items=initial_todos or [],
    )


def test_prior_history_is_a_baseline_not_current_turn_work():
    history = []
    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS + 5):
        history.extend(_tool_pair(index, "terminal", result="historical check passed"))

    ledger = _ledger(history)
    ledger.observe_messages(history)

    assert ledger.substantive_tool_completions == 0
    assert ledger.should_finalize is False

    current = history + _tool_pair(999, "patch", result="current edit applied")
    ledger.observe_messages(current)

    assert ledger.substantive_tool_completions == 1


def test_trigger_is_completed_substantive_tools_not_housekeeping():
    ledger = _ledger()
    messages: list[dict] = []

    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS - 1):
        messages.extend(_tool_pair(index, "read_file", result=f"read {index}"))
    ledger.observe_messages(messages)
    assert ledger.substantive_tool_completions == MIN_SUBSTANTIVE_TOOL_COMPLETIONS - 1
    assert ledger.should_finalize is False

    for offset, name in enumerate(
        ("todo", "memory", "skill_manage", "session_search"), 1000
    ):
        messages.extend(_tool_pair(offset, name, result="housekeeping complete"))
    ledger.observe_messages(messages)
    assert ledger.substantive_tool_completions == MIN_SUBSTANTIVE_TOOL_COMPLETIONS - 1
    assert ledger.should_finalize is False

    messages.extend(_tool_pair(2000, "terminal", result="verification passed"))
    ledger.observe_messages(messages)
    assert ledger.substantive_tool_completions == MIN_SUBSTANTIVE_TOOL_COMPLETIONS
    assert ledger.should_finalize is True


def test_long_turn_projection_retains_early_change_and_late_verification():
    history = [
        {
            "role": "assistant",
            "content": (
                "I recommend bounded resumable reconciliation with one worker, "
                "priority for new leads, and a round-robin verification cursor."
            ),
        }
    ]
    ledger = _ledger(history)
    messages = list(history)
    messages.extend(
        _tool_pair(
            0,
            "patch",
            arguments={
                "path": "ricochet_history.py",
                "patch": "Implement one-worker priority and round-robin reconciliation.",
            },
            result="Implemented bounded resumable reconciliation in ricochet_history.py.",
        )
    )

    for index in range(1, 148):
        messages.extend(
            _tool_pair(
                index,
                "read_file",
                arguments={"path": f"fixture-{index}.txt"},
                result=f"middle inspection {index}",
            )
        )

    messages.extend(
        _tool_pair(
            148,
            "terminal",
            arguments={"command": "python -m pytest -q", "workdir": "repo"},
            result="63 passed in 4.2s; exit_code=0",
        )
    )
    messages.extend(
        _tool_pair(
            149,
            "terminal",
            arguments={"command": "ruff check .", "workdir": "repo"},
            result="All checks passed; exit_code=0",
        )
    )

    ledger.observe_messages(messages)
    projection = ledger.build_projection(
        todo_items=[
            {
                "id": "sync-3",
                "content": "Implement one-worker priority and round-robin reconciliation foundations",
                "status": "completed",
            },
            {
                "id": "sync-5",
                "content": "Run targeted and full verification",
                "status": "completed",
            },
        ],
        changed_paths=["ricochet_history.py", "ricochet_history_sync.py"],
    )

    assert ledger.substantive_tool_completions == 150
    assert ledger.should_finalize is True
    assert "bounded resumable reconciliation" in projection
    assert "one-worker priority and round-robin" in projection
    assert "63 passed" in projection
    assert "All checks passed" in projection
    assert "sync-3" in projection
    assert len(projection) <= ledger.max_projection_chars


def test_compaction_prior_context_keeps_goal_and_decisions_outside_head_tail_clip():
    compacted = (
        "[PRIOR CONTEXT — for reference only; not a new message]\n"
        "Use a persistent fairness cursor so older records are continuously revisited.\n"
        "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]\n"
        "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
        + ("historical filler " * 350)
        + "\n## Goal\n"
        + "Implement bounded resumable reconciliation with a round-robin verification cursor.\n"
        + "\n## Completed Actions\n"
        + ("completed historical detail " * 500)
        + "\n## Key Decisions\n"
        + "Use one worker and prioritize new leads before verification sweeps.\n"
        + "\n## Relevant Files\n"
        + ("irrelevant/path.py\n" * 400)
    )
    ledger = TurnResultLedger.start(
        turn_id="turn-compacted-prior-context",
        original_user_message="Implement the recommendation.",
        conversation_history=[{"role": "assistant", "content": compacted}],
    )

    prompt = ledger.build_finalizer_prompt(
        draft_response="Verification completed.",
        todo_items=[],
        changed_paths=[],
    )

    assert "bounded resumable reconciliation" in prompt
    assert "round-robin verification cursor" in prompt
    assert "persistent fairness cursor" in prompt
    assert "prioritize new leads" in prompt


def test_indirect_objective_retains_latest_prior_assistant_context():
    recommendation = (
        "Use one worker with new-lead priority and a round-robin verification cursor."
    )
    history = [{"role": "assistant", "content": recommendation}]
    ledger = _ledger(history)
    messages = list(history)
    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS):
        messages.extend(_tool_pair(index, "read_file", result=f"read {index}"))

    ledger.observe_messages(messages)
    projection = ledger.build_projection(todo_items=[], changed_paths=[])

    assert "Implement the reconciliation approach you recommended" in projection
    assert recommendation in projection


def test_ledger_survives_message_rotation_without_double_counting():
    ledger = _ledger()
    first_window: list[dict] = []
    for index in range(60):
        first_window.extend(_tool_pair(index, "read_file", result=f"read {index}"))
    ledger.observe_messages(first_window)

    rotated_window = [
        {"role": "assistant", "content": "[compressed prior context]"},
        *first_window[-20:],
    ]
    for index in range(60, 150):
        rotated_window.extend(_tool_pair(index, "read_file", result=f"read {index}"))
    ledger.observe_messages(rotated_window)

    assert ledger.substantive_tool_completions == 150
    assert ledger.should_finalize is True


def test_projection_force_redacts_opaque_secret_fields():
    ledger = _ledger()
    messages: list[dict] = []
    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS):
        messages.extend(
            _tool_pair(
                index,
                "terminal",
                arguments={
                    "command": "OPENAI_API_KEY=plain-secret-value-123456789 run-check"
                },
                result='{"apiKey":"plain-secret-value-987654321"}',
            )
        )

    ledger.observe_messages(messages)
    projection = ledger.build_projection(todo_items=[], changed_paths=[])

    assert "plain-secret-value-123456789" not in projection
    assert "plain-secret-value-987654321" not in projection


def test_projection_redacts_url_userinfo_and_signed_query_credentials():
    ledger = _ledger()
    messages: list[dict] = []
    secret_url = (
        "https://plain-user:plain-password@example.com/object?"
        "X-Amz-Credential=plain-secret-value-987654321&"
        "X-Amz-Signature=signature-secret-246813579&"
        "code=oauth-code-secret-123456&"
        "confirmation_token=confirm-secret-234567&"
        "oobCode=oob-secret-345678#"
        "access_token=fragment-secret-456789"
    )
    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS):
        messages.extend(
            _tool_pair(
                index,
                "browser_navigate",
                arguments={"url": secret_url},
                result="navigation complete",
            )
        )
    ledger.observe_messages(messages)

    prompt = ledger.build_finalizer_prompt(
        draft_response="done",
        todo_items=[],
        changed_paths=[],
    )

    assert "plain-user" not in prompt
    assert "plain-password" not in prompt
    assert "plain-secret-value-987654321" not in prompt
    assert "signature-secret-246813579" not in prompt
    assert "oauth-code-secret-123456" not in prompt
    assert "confirm-secret-234567" not in prompt
    assert "oob-secret-345678" not in prompt
    assert "fragment-secret-456789" not in prompt


def test_projection_excludes_unchanged_prior_todos_but_keeps_current_changes():
    initial = [
        {
            "id": "old",
            "content": "Deployed production billing migration",
            "status": "completed",
        },
        {
            "id": "continued",
            "content": "Finish reconciliation",
            "status": "pending",
        },
    ]
    ledger = _ledger(initial_todos=initial)
    projection = ledger.build_projection(
        todo_items=[
            initial[0],
            {
                "id": "continued",
                "content": "Finish reconciliation",
                "status": "completed",
            },
            {
                "id": "new",
                "content": "Verify round-robin cursor",
                "status": "completed",
            },
        ],
        changed_paths=[],
    )

    assert "Deployed production billing migration" not in projection
    assert "continued [completed]" in projection
    assert "new [completed]" in projection


def test_finalizer_prompt_is_bounded_and_does_not_mutate_canonical_messages():
    canonical: list[dict] = []
    for index in range(MIN_SUBSTANTIVE_TOOL_COMPLETIONS):
        canonical.extend(_tool_pair(index, "read_file", result=f"read {index}"))
    before = json.dumps(canonical, sort_keys=True)

    ledger = _ledger()
    ledger.observe_messages(canonical)
    prompt = ledger.build_finalizer_prompt(
        draft_response="Fresh verification completed.",
        todo_items=[],
        changed_paths=[],
    )

    assert json.dumps(canonical, sort_keys=True) == before
    assert TURN_RESULT_LEDGER_MARKER not in before
    assert TURN_RESULT_LEDGER_MARKER in prompt
    assert "Fresh verification completed." in prompt
    assert len(prompt) <= ledger.max_finalizer_prompt_chars
