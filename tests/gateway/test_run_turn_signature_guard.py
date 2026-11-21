from types import SimpleNamespace


def test_turn_runner_omits_persist_kwargs_for_older_run_conversation_signature():
    """Compatibility implementations may not accept newer persistence kwargs."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution
    from gateway.turn_context import TurnContext

    seen = []

    def run(message, conversation_history=None, task_id=None):
        seen.append((message, conversation_history, task_id))
        return {"final_response": "ok"}

    turn = TurnRunner(
        SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: []),
        TurnContext(
            message="hello",
            session_id="legacy-runner",
            session_key="api",
            source=SimpleNamespace(user_id="u", user_name="U", is_bot=False),
            persist_user_display_metadata={"notice": True},
            inbound_message_id="platform-991",
        ),
    )
    token = api_execution.set(None)
    try:
        result = turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run), [], None, None, None)
    finally:
        api_execution.reset(token)

    assert result == {"final_response": "ok"}
    assert seen == [("hello", [], "legacy-runner")]
