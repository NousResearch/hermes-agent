"""``pre_send_message``: a fail-open plugin hook in front of every adapter ``send`` (#134330).

Uses a real ``BasePlatformAdapter`` subclass (so ``__init_subclass__`` wraps its ``send``) and the
real plugin dispatch: callbacks are registered on the per-home ``PluginManager`` exactly as
``ctx.register_hook`` stores them.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

HOOK = "pre_send_message"


def _register(callback) -> None:
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager._discovered = True  # no on-disk plugins in the test home; skip discovery
    manager._hooks.setdefault(HOOK, []).append(callback)


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.wire = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.wire.append(content)
        return SendResult(success=True, message_id=str(len(self.wire)))


class _SplittingAdapter(_Adapter):
    """Sends a long payload as two chunks through its parent ``send``: the hook must fire once."""

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if len(content) > 10:
            await super().send(chat_id, content[:10], reply_to=reply_to, metadata=metadata)
            return await super().send(chat_id, content[10:], reply_to=reply_to, metadata=metadata)
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


@pytest.mark.asyncio
async def test_no_hook_sends_unchanged():
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.wire == ["hello"]


@pytest.mark.asyncio
async def test_rewrite_changes_the_text_on_the_wire_and_payload_is_complete():
    seen = []

    def hook(**kwargs):
        kwargs.pop("telemetry_schema_version", None)  # stamped by the plugin dispatcher
        seen.append(kwargs)
        return {"action": "rewrite", "text": kwargs["text"].upper()}

    _register(hook)
    adapter = _Adapter()
    await adapter.send("c1", "hello", reply_to="r9", metadata={"_interim_send": True})
    assert adapter.wire == ["HELLO"]
    assert seen == [{"platform": "telegram", "chat_id": "c1", "text": "hello", "kind": "interim",
                     "reply_to": "r9", "metadata": {"_interim_send": True}}]


@pytest.mark.asyncio
async def test_drop_beats_rewrite_and_reports_success_without_a_message_id():
    _register(lambda **_: {"action": "rewrite", "text": "x"})
    _register(lambda **_: {"action": "drop", "reason": "budget"})
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert adapter.wire == []
    assert result.success and result.message_id is None
    assert result.raw_response == {"pre_send": "dropped"}


@pytest.mark.asyncio
async def test_retry_path_does_not_resend_a_dropped_message():
    _register(lambda **_: {"action": "drop"})
    adapter = _Adapter()
    result = await adapter._send_with_retry("c1", "hello")
    assert result.success and adapter.wire == []


@pytest.mark.asyncio
async def test_nested_sends_fire_the_hook_once_for_the_outer_message():
    calls = []

    def hook(text, **_):
        calls.append(text)
        return {"action": "rewrite", "text": text + "!"}

    _register(hook)
    adapter = _SplittingAdapter()
    await adapter.send("c1", "hello world, twice")
    assert calls == ["hello world, twice"]
    assert "".join(adapter.wire) == "hello world, twice!"


@pytest.mark.asyncio
async def test_streamed_previews_skip_the_hook():
    calls = []
    _register(lambda **kw: calls.append(kw) or {"action": "drop"})
    adapter = _Adapter()
    await adapter.send("c1", "partial", metadata={"expect_edits": True})
    assert calls == [] and adapter.wire == ["partial"]


@pytest.mark.asyncio
async def test_a_raising_hook_fails_open():
    def boom(**_):
        raise RuntimeError("plugin bug")

    _register(boom)
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.wire == ["hello"]


@pytest.mark.asyncio
async def test_keyword_only_send_signatures_are_rewritten_too():
    class _KwAdapter(_Adapter):
        async def send(self, **kwargs):
            self.wire.append(kwargs["content"])
            return SendResult(success=True, message_id="1")

    _register(lambda text, **_: {"action": "rewrite", "text": text + "?"})
    adapter = _KwAdapter()
    await adapter.send(chat_id="c1", content="hello", metadata=None)
    assert adapter.wire == ["hello?"]
