"""Actual Slack/Matrix fallback faults must not produce a Files Sent result."""

from pathlib import Path

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.hosted_room_messaging_files import FilesMenu
from tests.gateway.test_hosted_room_file_access import file_state, publish  # noqa: F401
from tests.gateway.test_hosted_room_messaging_files import Runner, consumer, event  # noqa: F401
from tests.gateway.test_native_document_direct_fallback import (
    _install_failure,
    _install_success,
    native,  # noqa: F401
    receipt,
)


@pytest.mark.asyncio
async def test_actual_adapter_fallback_never_says_sent_or_replays(
    consumer, native, monkeypatch
):
    state, _, _ = consumer
    _install_failure(native)
    if native.name == "matrix":
        exists = Path.exists

        def missing_private_file(path):
            if path.parent.parent.name == "group-file-delivery-tmp":
                return False
            return exists(path)

        monkeypatch.setattr(Path, "exists", missing_private_file)

    native.adapter.config.extra["allow_admin_from"] = ["owner"]
    runner = Runner(native.adapter, platform=native.adapter.platform)
    command = event("/group 1 file exact", platform=native.adapter.platform)
    command.source.chat_id = "C123" if native.name == "slack" else "!room:isolated"
    item = publish(state, data=b"shared bytes")
    menu = FilesMenu(runner, command, state.backend, "/group")
    await menu.bind("1")
    progress = await menu.prepare_file(item)
    assert isinstance(progress, ChoiceProgress)
    result = await progress.complete()
    assert isinstance(result, ChoicePage)
    assert "Delivery wasn’t confirmed" in result.title
    assert "Sent." not in result.title
    replay = await menu.deliver(item)
    assert isinstance(replay, ChoicePage) and replay.title == result.title
    assert native.notices == []
    assert receipt(state.db) == [("unknown", 1)]
    assert not list(state.db.parent.glob("group-file-delivery-tmp/send-*"))

    # A new explicit action can succeed after the native fault is repaired.
    if native.name == "matrix":
        monkeypatch.setattr(Path, "exists", exists)
    _install_success(native)
    retried = await menu.deliver(item, retry=":explicit-again")
    assert retried == "File sent."
    assert native.uploaded[0][0] == b"shared bytes"
    assert sorted(receipt(state.db)) == [("delivered", 1), ("unknown", 1)]
    assert not list(state.db.parent.glob("group-file-delivery-tmp/send-*"))
