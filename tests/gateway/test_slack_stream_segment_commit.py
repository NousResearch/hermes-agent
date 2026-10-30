"""Only an explicitly identified draft may be committed at an overflow boundary."""

from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.test_slack_initial_streaming import _SlackClient


class _AmbiguousClient(_SlackClient):
    def __init__(self):
        super().__init__(downgrade=False)
        self.fail_update = True
        self.fail_stop = True

    async def chat_stopStream(self, **kwargs):
        result = await super().chat_stopStream(**kwargs)
        if self.fail_stop:
            self.fail_stop = False
            raise TimeoutError("stopStream acknowledged too late")
        return result

    async def chat_update(self, **kwargs):
        if self.fail_update:
            self.fail_update = False
            raise TimeoutError("chat.update temporarily unavailable")
        return await super().chat_update(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["wrong-draft", "other-thread", "rewrite", "ambiguous"])
async def test_segment_commit_owns_only_its_draft_and_retries_in_place(scenario):
    client = _AmbiguousClient() if scenario == "ambiguous" else _SlackClient(downgrade=False)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    adapter._get_client = MagicMock(return_value=client)
    adapter._running = True
    metadata = {"thread_id": "111.000", "user_id": "U123"}
    initial = "Visible text already in the draft"
    await adapter.send_draft("D1", 7, initial, metadata=metadata)
    first_id = next(iter(client.messages))
    final = "Replacement head" if scenario == "rewrite" else initial + " and its sealed tail"
    seal = dict(metadata, expect_edits=True, _finalize_draft_id=(8 if scenario == "wrong-draft" else 7))
    if scenario == "other-thread":
        seal["thread_id"] = "222.000"
    result = await adapter.send("D1", final, metadata=seal)
    if scenario == "wrong-draft":
        assert not result.success
        assert list(client.messages.values()) == [initial]
    elif scenario == "other-thread":
        assert result.success
        assert client.messages[first_id] == initial
        assert len(client.messages) == 2
        assert not any(method == "stop" for method, _ in client.calls)
    else:
        if scenario == "ambiguous":
            assert not result.success
            assert list(client.messages.values()) == [final]  # The append landed despite timeout.
            result = await adapter.send("D1", final, metadata=seal)
        assert result.success
        assert result.message_id == first_id
        assert list(client.messages.values()) == [final]
        assert not adapter._active_streams
        assert not any(method == "post" for method, _ in client.calls)
