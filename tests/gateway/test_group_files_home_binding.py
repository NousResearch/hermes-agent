"""Old file menus cannot resume after a home is reselected and accepted again."""

from dataclasses import replace

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.config import HomeChannel, Platform
from gateway.group_home_identity import acknowledgement
from tests.gateway.test_hosted_room_messaging_files import (
    consumer as consumer,
    event,
    file_state as file_state,
    publish,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("entry", ["files", "group-list"])
@pytest.mark.parametrize("reselect", [False, True])
async def test_file_menu_retains_original_home_selection(
    consumer, monkeypatch, shared, entry, reselect
):
    state, runner, adapter = consumer
    publish(state, "result.md", b"the exact shared result")
    command = event("/group 1 files" if entry == "files" else "/group")
    if shared:
        command.source.chat_type = "group"
        command.source.is_one_to_one = False
    config = runner.config.platforms[Platform.SIGNAL]
    config.extra["group_allow_admin_from"] = ["owner"]
    home = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat",
        name="Selected home",
        thread_id="topic",
        user_id="owner",
        selection_id="first-selection",
    )
    home.group_audience_ack = acknowledgement(home)
    config.home_channel = home
    assert runner._can_control_group_chats(command)
    assert await runner._handle_rooms_command(command) is None
    menu = adapter.pages[-1]

    if reselect:
        replacement = replace(home, selection_id="second-selection")
        replacement.group_audience_ack = acknowledgement(replacement)
        config.home_channel = replacement
    # Current permission remains valid: the old menu's binding must still fail.
    assert runner._can_control_group_chats(command)

    private_fetches = []
    for owner, name in (
        (state.backend, "list_files"),
        (state.backend, "read_file"),
        (state.backend.service, "status"),
    ):
        original = getattr(owner, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            private_fetches.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    result = await menu["on_choice_selected"]("chat", menu["choices"][0]["value"])
    if isinstance(result, ChoiceProgress):
        result = await result.complete()
    if reselect:
        assert private_fetches == []
        assert adapter.documents == []
        assert isinstance(result, str)
        assert state.room["name"] not in result
    elif entry == "files":
        assert [document[1] for document in adapter.documents] == [b"the exact shared result"]
    else:
        assert isinstance(result, ChoicePage)
        assert private_fetches
