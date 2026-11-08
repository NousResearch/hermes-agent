"""An approval card whose command preview is cut never offers an approve choice for text the user could
not read: the whole command is sent just before the card, or, when it cannot be, only Deny is offered."""

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Chat(BasePlatformAdapter):
    """Concrete adapter with a 1,000-char cap that records sends and the rendered prompt."""

    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, *, send_ok: bool = True) -> None:
        self.sent, self.prompts, self._send_ok = [], [], send_ok

    def _exec_approval_cmd_budget(self, description: str, smart_denied: bool) -> int:
        return 600

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_metadata = metadata
        return SendResult(success=self._send_ok, message_id="m1")

    async def _send_exec_approval_prompt(self, prompt):
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="card")


def _choices(adapter):
    return [choice for _, choice, _ in adapter.prompts[-1].actions]


async def _ask(adapter, command):
    await adapter.send_exec_approval("c1", command, "sk", description="flagged", request_id="r1")
    return adapter.prompts[-1].text


@pytest.mark.asyncio
async def test_short_command_keeps_the_compact_card():
    adapter = _Chat()
    text = await _ask(adapter, "rm -rf /tmp/x")
    assert adapter.sent == [] and "rm -rf /tmp/x" in text and "Preview shortened" not in text
    assert _choices(adapter) == ["once", "session", "always", "deny"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "send_ok", "choices", "attempted"), [
    ("echo " + "A" * 900 + " && rm -rf ~ # TAIL", True, ["once", "session", "always", "deny"], True),
    ("echo " + "A" * 900 + " && rm -rf ~ # TAIL", False, ["deny"], True),  # the full send failed
    ("echo '```' " + "A" * 900 + " && rm -rf ~ # TAIL", True, ["deny"], False),  # cannot be fenced verbatim
])
async def test_truncated_preview_is_backed_by_the_full_command_or_deny_only(command, send_ok, choices, attempted):
    adapter = _Chat(send_ok=send_ok)
    text = await _ask(adapter, command)
    assert "TAIL" not in text and "Preview shortened" in text and str(len(command)) in text
    assert _choices(adapter) == choices
    assert adapter.sent == ([f"```\n{command}\n```"] if attempted else [])


@pytest.mark.asyncio
async def test_full_command_send_is_marked_interim_so_it_never_seals_the_live_response():
    """The full command goes out mid-turn: unmarked, a stream-is-the-message adapter (relay drafts) reads it
    as the turn-final reply and swallows every later update to the open stream."""
    adapter = _Chat()
    await adapter.send_exec_approval("c1", "echo " + "A" * 900, "sk", metadata={"thread_id": "t1"})
    assert adapter.sent_metadata == {"thread_id": "t1", "_interim_send": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_factory", ["base", "discord", "slack"])
async def test_backticks_inside_a_short_command_cannot_close_the_card_fence(adapter_factory):
    """A ``` (or longer run) in the command would end the card's markdown fence early, and the rest
    would render as markdown (spoilers, links): the fence is longer than any run in the command."""
    from gateway.platforms.base_exec_approval import render_exec_approval

    adapter = _Chat()
    if adapter_factory == "discord":
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        adapter._approval_mention_content = lambda: ""
    elif adapter_factory == "slack":
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter.__new__(SlackAdapter)
    command = "echo '```' ||spoiler|| [x](https://e.x) ````` done"
    text, truncated = render_exec_approval(adapter, command, "flagged", False, True)
    assert not truncated
    fence = "``````"  # one longer than the command's longest run (5)
    assert text.count(fence) == 2
    assert text.index(fence) < text.index(command) < text.rindex(fence)
