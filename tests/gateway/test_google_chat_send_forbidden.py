"""A 403 on one Google Chat space fails that send only.

``messages.create`` answers 403 when the bot was removed from that one space; every other space
keeps receiving through Pub/Sub and answering. The adapter must report the failed send and stay
connected, not mark the whole platform fatal and disconnected.
"""

import pytest

from gateway.status import read_runtime_status
from tests.gateway.test_google_chat import GoogleChatAdapter, _base_config, _FakeHttpError


class _ChatServiceWithoutSpaceA:
    """The googleapiclient ``chat`` v1 resource surface ``send()`` drives:
    ``spaces().messages().create(**kwargs).execute(http=...)``. Space A refuses the bot."""

    def __init__(self):
        self.delivered: list[tuple[str, str]] = []

    def spaces(self):
        return self

    def messages(self):
        return self

    def create(self, **kwargs):
        service = self

        class _Request:
            def execute(self, http=None):
                if kwargs["parent"] == "spaces/A":
                    raise _FakeHttpError(status=403, reason="The caller does not have permission")
                service.delivered.append((kwargs["parent"], kwargs["body"]["text"]))
                return {"name": f"{kwargs['parent']}/messages/{len(service.delivered)}"}

        return _Request()


@pytest.mark.asyncio
async def test_forbidden_space_fails_only_that_send(tmp_path):
    adapter = GoogleChatAdapter(_base_config())
    adapter._chat_api = chat = _ChatServiceWithoutSpaceA()
    adapter._new_authed_http = lambda: None
    adapter._mark_connected()

    refused = await adapter._send_with_retry("spaces/A", "hello A")
    answered = await adapter._send_with_retry("spaces/B", "hello B")

    assert refused.success is False
    assert answered.success is True
    assert chat.delivered == [("spaces/B", "hello B")]
    assert adapter.has_fatal_error is False
    assert adapter.is_connected is True
    assert read_runtime_status()["platforms"]["google_chat"]["state"] == "connected"
