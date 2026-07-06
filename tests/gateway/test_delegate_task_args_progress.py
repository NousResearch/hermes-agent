from gateway.run import (
    _DELEGATE_TASK_GOAL_PREVIEW_CHARS,
    _format_delegate_task_args_progress,
    _tool_progress_pipeline_enabled,
)


def _lines(cards):
    """A single card is always returned; split it into header + rows."""
    assert len(cards) == 1
    return cards[0].split("\n")


def test_delegate_task_args_keeps_pipeline_alive_when_tool_progress_off():
    """The bug: tool_progress="off" tore down the progress queue, so the
    delegate_task_args card never rendered. The pipeline MUST stay alive when
    delegate_task_args is enabled even with every other progress source off."""
    assert _tool_progress_pipeline_enabled(
        is_webhook=False,
        progress_mode="off",
        tool_completion_durations_enabled=False,
        subagent_progress_enabled=False,
        delegate_task_args_enabled=True,
    ) is True


def test_pipeline_stays_off_when_everything_disabled():
    assert _tool_progress_pipeline_enabled(
        is_webhook=False,
        progress_mode="off",
        tool_completion_durations_enabled=False,
        subagent_progress_enabled=False,
        delegate_task_args_enabled=False,
    ) is False


def test_webhook_never_gets_pipeline_even_with_delegate_args():
    assert _tool_progress_pipeline_enabled(
        is_webhook=True,
        progress_mode="off",
        tool_completion_durations_enabled=False,
        subagent_progress_enabled=False,
        delegate_task_args_enabled=True,
    ) is False


def test_single_task_roster_header_profile_left_goal_backticked_right():
    """Single delegate_task: header is bare 'Delegate task', the row leads with
    the (un-backticked) profile, then ' · ' then the goal inside a code span.
    context is NEVER shown on the row."""
    cards = _format_delegate_task_args_progress(
        {
            "goal": "Review API handlers",
            "context": "Use the project at /tmp/app and report exact files read.",
            "profile": "file-explorer",
        }
    )

    lines = _lines(cards)
    assert lines[0] == "🔀 Delegate task"
    assert len(lines) == 2
    row = lines[1]
    # profile on the LEFT, NOT backticked; goal on the right, backticked.
    assert row == "file-explorer · `Review API handlers`"
    # context is excluded entirely from the card.
    assert "context" not in cards[0]
    assert "/tmp/app" not in cards[0]


def test_extra_param_renders_between_profile_and_goal():
    """A curated non-goal/context param (role/toolsets/...) renders as a plain
    key=value cell after the profile and before the backticked goal."""
    cards = _format_delegate_task_args_progress(
        {
            "goal": "Second opinion on the sharding plan",
            "profile": "oracle",
            "role": "orchestrator",
            "toolsets": ["terminal", "file"],
        }
    )

    row = _lines(cards)[1]
    assert row == (
        "oracle · role=orchestrator · toolsets=terminal,file "
        "· `Second opinion on the sharding plan`"
    )


def test_goal_truncated_to_120_chars_with_ellipsis():
    long_goal = "g" * 200
    cards = _format_delegate_task_args_progress({"goal": long_goal, "profile": "p"})

    row = _lines(cards)[1]
    cap = _DELEGATE_TASK_GOAL_PREVIEW_CHARS
    assert cap == 120
    # Inside the backticks: cap-1 chars + ellipsis.
    expected_goal = "g" * (cap - 1) + "…"
    assert row == f"p · `{expected_goal}`"
    # Hard cap: never more than `cap` visible goal chars.
    assert ("g" * cap) not in row


def test_batch_tasks_one_row_each_with_per_task_profile_override():
    cards = _format_delegate_task_args_progress(
        {
            "profile": "dual-review",
            "tasks": [
                {"goal": "Review auth refactor", "profile": "reviewer-codex"},
                {"goal": "Audit the migration"},  # inherits call-level profile
            ],
        }
    )

    lines = _lines(cards)
    assert lines[0] == "🔀 Delegate task — 2 tasks"
    assert lines[1] == "reviewer-codex · `Review auth refactor`"
    # second task has no per-task profile -> inherits the call-level default.
    assert lines[2] == "dual-review · `Audit the migration`"


def test_batch_truncates_to_max_rows_with_more_line():
    tasks = [{"goal": f"task {i}", "profile": "p"} for i in range(13)]
    cards = _format_delegate_task_args_progress({"tasks": tasks})

    lines = _lines(cards)
    assert lines[0] == "🔀 Delegate task — 13 tasks"
    # 1 header + 10 rows + 1 "more" line
    assert len(lines) == 12
    assert lines[-1] == "… +3 more"


def test_missing_goal_renders_placeholder_and_no_crash():
    cards = _format_delegate_task_args_progress({"profile": "p"})
    row = _lines(cards)[1]
    assert row == "p · `goal`"


def test_no_profile_row_is_just_the_goal():
    cards = _format_delegate_task_args_progress({"goal": "do a thing"})
    row = _lines(cards)[1]
    assert row == "`do a thing`"


def test_backticks_in_goal_are_stripped():
    """A stray backtick would break the inline code span on Telegram."""
    cards = _format_delegate_task_args_progress({"goal": "fix `foo` bug", "profile": "p"})
    row = _lines(cards)[1]
    assert row == "p · `fix foo bug`"


def test_backticks_in_profile_and_params_are_stripped():
    """profile + param cells are plain text; a stray backtick there would pair
    with the goal cell's opening backtick and break the code span on Telegram.
    The only backticks in a row must be the goal cell's own pair."""
    cards = _format_delegate_task_args_progress(
        {
            "goal": "review diff",
            "profile": "ev`il",
            "role": "orch`estrator",
            "toolsets": ["ter`minal", "file"],
        }
    )
    row = _lines(cards)[1]
    # Exactly two backticks total — the goal cell's delimiters.
    assert row.count("`") == 2
    assert row == "evil · role=orchestrator · toolsets=terminal,file · `review diff`"


def test_empty_args_does_not_crash():
    cards = _format_delegate_task_args_progress(None)
    lines = _lines(cards)
    assert lines[0] == "🔀 Delegate task"
    assert lines[1] == "`goal`"
