"""``pre_platform_send``: the outbound gate every adapter text send passes.

Covers the return contract (None / rewrite / cancel, cancel beats rewrite), one firing per
outermost send when a subclass delegates to ``super().send``, ``_send_with_retry`` treating a
cancel as final, fail-open on a dispatch failure, the no-subscriber fast path, and the hook's
registration (valid hook, refused for shell hooks).
"""

from typing import Any, Dict, List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SEND_ERROR_KINDS, BasePlatformAdapter, SendResult


class _Adapter(BasePlatformAdapter):
    """Records every text that reaches the platform; sends succeed unless scripted otherwise."""

    def __init__(self, results=None):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self._results = list(results or [])
        self.sent: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return self._results.pop(0) if self._results else SendResult(success=True, message_id="ok")


class _Delegating(_Adapter):
    """A subclass whose ``send`` delegates to ``super().send`` (both are wrapped)."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


@pytest.fixture
def hook(monkeypatch):
    """Subscribe a scripted ``pre_platform_send`` and record every call it receives."""
    calls: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {"results": []}

    async def _ainvoke(name, **kwargs):
        assert name == "pre_platform_send"
        calls.append(kwargs)
        return list(state["results"])

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "pre_platform_send")
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _ainvoke)
    state["calls"] = calls
    return state


@pytest.mark.asyncio
async def test_none_sends_unchanged_with_documented_kwargs(hook):
    adapter = _Adapter()
    result = await adapter.send("c1", "hello", metadata={"thread_id": "t"})
    assert result.success and adapter.sent == ["hello"]
    (call,) = hook["calls"]
    assert call == {"adapter": adapter, "platform": "telegram", "chat_id": "c1", "text": "hello",
                    "metadata": {"thread_id": "t"}}


@pytest.mark.asyncio
async def test_rewrite_replaces_text(hook):
    hook["results"] = [{"action": "rewrite", "text": "redacted"}, {"action": "rewrite", "text": "second"}]
    adapter = _Adapter()
    result = await adapter.send("c1", "secret")
    assert result.success and adapter.sent == ["redacted"]


@pytest.mark.asyncio
async def test_cancel_sends_nothing_and_beats_rewrite(hook):
    hook["results"] = [{"action": "rewrite", "text": "x"}, {"action": "cancel", "reason": "banned"}]
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert adapter.sent == []
    assert not result.success and result.error_kind == "cancelled"
    assert result.error == "cancelled by pre_platform_send: banned"
    assert "cancelled" in SEND_ERROR_KINDS


@pytest.mark.asyncio
async def test_super_send_delegation_fires_once(hook):
    hook["results"] = [{"action": "rewrite", "text": "once"}]
    adapter = _Delegating()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.sent == ["once"]
    assert len(hook["calls"]) == 1


@pytest.mark.asyncio
async def test_send_with_retry_treats_cancel_as_final(hook, monkeypatch):
    slept = []

    async def _sleep(d):
        slept.append(d)
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _sleep)
    hook["results"] = [{"action": "cancel", "reason": "no"}]
    adapter = _Adapter()
    result = await adapter._send_with_retry("c1", "hello")
    assert result.error_kind == "cancelled"
    assert len(hook["calls"]) == 1  # no retry, no plain-text fallback, no failure notice
    assert adapter.sent == [] and slept == []


@pytest.mark.asyncio
async def test_dispatch_failure_fails_open(monkeypatch):
    async def _boom(name, **kwargs):
        raise RuntimeError("dispatch broke")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _boom)
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.sent == ["hello"]


@pytest.mark.asyncio
async def test_no_subscriber_skips_dispatch(monkeypatch):
    async def _never(name, **kwargs):
        raise AssertionError("invoked without a subscriber")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _never)
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.sent == ["hello"]


@pytest.mark.asyncio
async def test_plugin_callback_raising_counts_as_none(monkeypatch):
    """End to end through a real PluginManager: a raising callback is isolated, the send proceeds."""
    from hermes_cli import plugins

    manager = plugins.PluginManager()

    def _raises(**kwargs):
        raise ValueError("guard bug")
    manager._hooks["pre_platform_send"] = [_raises, lambda **kw: {"action": "rewrite", "text": "ok"}]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.lifecycle._observe", lambda *a, **k: None)
    adapter = _Adapter()
    result = await adapter.send("c1", "hello")
    assert result.success and adapter.sent == ["ok"]


def test_hook_registered_and_refused_for_shell_hooks():
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS

    assert "pre_platform_send" in VALID_HOOKS
    assert "pre_platform_send" in SHELL_UNSUPPORTED_HOOKS
