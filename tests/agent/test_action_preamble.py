from agent.action_preamble import looks_like_action_preamble_stall


def test_short_action_preamble_without_tool_call_is_stall() -> None:
    assert looks_like_action_preamble_stall(
        "Let me check what's on taro and figure out the right approach.",
        "stop",
        False,
    )


def test_long_colon_ended_action_preamble_is_stall() -> None:
    content = (
        "Now I have a clear picture. The task is to add a pre-dispatch "
        "executable-state gate in subagent_dispatch_register_command that "
        "checks if the linked task is in an executable state before registering "
        "a dispatch. Currently the stale guard only runs post-dispatch during "
        "drain and settlement. The goal is to prevent non-executable dispatches "
        "from being registered in the first place. Let me look at the register "
        "command's validation section and the _validate_subagent_dispatch_record "
        "function:"
    )

    assert len(content) > 400
    assert looks_like_action_preamble_stall(content, "stop", False)


def test_completion_text_is_not_stall() -> None:
    assert not looks_like_action_preamble_stall(
        "Done. The task is complete and no further action is needed.",
        "stop",
        False,
    )


def test_explanatory_let_me_text_is_not_stall() -> None:
    assert not looks_like_action_preamble_stall(
        "Let me explain why this does not require another tool call.",
        "stop",
        False,
    )


def test_empty_visible_response_uses_empty_response_recovery() -> None:
    assert not looks_like_action_preamble_stall("", "stop", False)
    assert not looks_like_action_preamble_stall("<think>still reasoning</think>", "stop", False)


def test_tool_call_response_is_not_stall() -> None:
    assert not looks_like_action_preamble_stall(
        "Let me inspect the file:",
        "tool_calls",
        True,
    )
