from gateway.platforms.base import MessageEvent, MessageType


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT)


def test_preqstation_dispatch_command_with_bot_suffix_and_newlines():
    event = _event(
        "/preqstation_dispatch@PreqHermesBot\n"
        "project_key=PQST\n"
        "task_key=PQST-154\n"
        "objective=implement\n"
        "engine=codex"
    )

    assert event.get_command() == "preqstation_dispatch"
    assert "project_key=PQST" in event.get_command_args()
    assert "task_key=PQST-154" in event.get_command_args()


def test_preqstation_dispatch_command_recovers_collapsed_bot_suffix():
    event = _event(
        "/preqstation_dispatch@PreqHermesBotproject_key=PQST "
        "task_key=PQST-154 objective=implement engine=codex"
    )

    assert event.get_command() == "preqstation_dispatch"
    assert event.get_command_args().startswith("project_key=PQST task_key=PQST-154")


def test_preqstation_dispatch_command_recovers_collapsed_fields_without_suffix():
    event = _event(
        "/preqstation_dispatchproject_key=PQST "
        "task_key=PQST-154 objective=implement engine=codex"
    )

    assert event.get_command() == "preqstation_dispatch"
    assert event.get_command_args().startswith("project_key=PQST task_key=PQST-154")
