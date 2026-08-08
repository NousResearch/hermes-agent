"""Tests for the structural send gate (``platforms.<name>.extra.send_gate``).

The point of the gate is that it is *structural*: it holds for adapters nobody
remembered to update. So the tests below deliberately include coverage that
fails when a new adapter or a new ``send_*`` helper is added without a gate
check, rather than only testing the adapters that exist today.
"""

import asyncio
import importlib
import os
from typing import Optional

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, install_send_gate
from gateway.send_gate import (
    SEND_GATE_EXEMPT_METHODS,
    SendGateDisabledError,
    is_send_blocked,
    platform_name_of,
)

PLUGIN_PLATFORM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins",
    "platforms",
)


class _FakeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter; records whether the transport was reached."""

    def __init__(self, config: PlatformConfig):
        self.config = config
        self.platform = Platform.SLACK
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id: str):
        return {}


def _config(send_gate: Optional[str]) -> PlatformConfig:
    extra = {} if send_gate is None else {"send_gate": send_gate}
    return PlatformConfig(enabled=True, extra=extra)


# --------------------------------------------------------------------------
# Config reading — only an explicit "disabled" blocks; everything else opens.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,blocked",
    [
        ("disabled", True),
        ("DISABLED", True),
        ("  disabled  ", True),
        ("enabled", False),
        ("", False),
        ("off", False),
        ("true", False),
        (None, False),
    ],
)
def test_only_explicit_disabled_blocks(value, blocked):
    assert is_send_blocked(_config(value)) is blocked


def test_unreadable_config_fails_open():
    """A gate that fails closed on a malformed config is an outage."""

    class Hostile:
        @property
        def extra(self):
            raise RuntimeError("boom")

    assert is_send_blocked(Hostile()) is False
    assert is_send_blocked(None) is False
    assert is_send_blocked(object()) is False
    assert is_send_blocked({"extra": "not-a-dict"}) is False


def test_dict_shaped_config_supported():
    assert is_send_blocked({"extra": {"send_gate": "disabled"}}) is True
    assert is_send_blocked({"extra": {}}) is False


# --------------------------------------------------------------------------
# Adapter layer.
# --------------------------------------------------------------------------


def test_send_blocked_when_disabled():
    adapter = _FakeAdapter(_config("disabled"))
    with pytest.raises(SendGateDisabledError) as excinfo:
        asyncio.run(adapter.send("c1", "hello"))
    # The error must tell an operator how to undo it.
    assert "send_gate" in str(excinfo.value)
    assert "slack" in str(excinfo.value)
    assert adapter.sent == [], "transport was reached despite the gate"


def test_send_allowed_by_default():
    adapter = _FakeAdapter(_config(None))
    result = asyncio.run(adapter.send("c1", "hello"))
    assert result.success is True
    assert adapter.sent == [("c1", "hello")]


def test_send_allowed_when_explicitly_enabled():
    adapter = _FakeAdapter(_config("enabled"))
    assert asyncio.run(adapter.send("c1", "hi")).success is True


def test_adapter_without_init_fails_open():
    """Gateway tests build adapters via object.__new__(); that must not crash."""
    adapter = object.__new__(_FakeAdapter)
    adapter.sent = []
    result = asyncio.run(adapter.send("c1", "hello"))
    assert result.success is True


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("send_image", {"image_url": "http://x/i.png"}),
        ("send_image_file", {"image_path": "/tmp/i.png"}),
        ("send_voice", {"audio_path": "/tmp/a.ogg"}),
        ("send_video", {"video_path": "/tmp/v.mp4"}),
        ("send_document", {"file_path": "/tmp/d.pdf"}),
    ],
)
def test_native_media_methods_are_gated(method, kwargs):
    """cron/scheduler.py calls these directly, bypassing send()."""
    adapter = _FakeAdapter(_config("disabled"))
    with pytest.raises(SendGateDisabledError):
        asyncio.run(getattr(adapter, method)(chat_id="c1", **kwargs))
    assert adapter.sent == []


def test_presence_signals_are_exempt():
    """send_typing carries no content and runs in a loop that can't take a raise."""
    adapter = _FakeAdapter(_config("disabled"))
    asyncio.run(adapter.send_typing("c1"))  # must not raise
    for name in SEND_GATE_EXEMPT_METHODS:
        method = getattr(BasePlatformAdapter, name, None)
        if method is not None:
            assert not getattr(method, "__send_gate_wrapped__", False)


def test_gate_is_not_applied_twice():
    """Double-wrapping would run the check twice and break functools.wraps identity."""
    install_send_gate(_FakeAdapter)
    install_send_gate(_FakeAdapter)
    adapter = _FakeAdapter(_config(None))
    assert asyncio.run(adapter.send("c1", "x")).success is True


def test_mixin_contributed_send_is_gated():
    """Adapters like WhatsAppAdapter inherit send helpers from a plain mixin."""

    class _Mixin:
        async def send_broadcast(self, chat_id, content):
            return "sent"

    class _MixedAdapter(_Mixin, _FakeAdapter):
        pass

    adapter = _MixedAdapter(_config("disabled"))
    with pytest.raises(SendGateDisabledError):
        asyncio.run(adapter.send_broadcast("c1", "x"))


def _iter_plugin_adapter_classes():
    for name in sorted(os.listdir(PLUGIN_PLATFORM_DIR)):
        if not os.path.exists(os.path.join(PLUGIN_PLATFORM_DIR, name, "adapter.py")):
            continue
        module = importlib.import_module(f"plugins.platforms.{name}.adapter")
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePlatformAdapter)
                and attr is not BasePlatformAdapter
            ):
                yield name, attr


def test_every_plugin_adapter_is_gated():
    """Regression guard: a newly added adapter inherits the gate for free.

    This is the test that fails if someone reintroduces per-adapter opt-in.
    """
    discovered = list(_iter_plugin_adapter_classes())
    assert len(discovered) >= 20, f"adapter discovery broke: found {len(discovered)}"
    ungated = [
        f"{mod}.{cls.__name__}"
        for mod, cls in discovered
        if not getattr(cls.send, "__send_gate_wrapped__", False)
    ]
    assert ungated == [], f"adapters missing the send gate: {ungated}"


def test_send_with_retry_degrades_without_retrying(monkeypatch):
    """A closed gate is permanent: no backoff loop, no plain-text fallback.

    _send_with_retry must return a failed SendResult rather than propagate, so
    the caller closes its delivery-ledger obligation instead of stranding it in
    "attempting" and replaying it on the next restart.
    """
    adapter = _FakeAdapter(_config("disabled"))
    slept = []

    async def _no_sleep(delay, *args, **kwargs):
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    result = asyncio.run(adapter._send_with_retry(chat_id="c1", content="hello"))

    assert result.success is False
    assert result.retryable is False
    assert "send_gate" in (result.error or "")
    # A backoff sleep means the gate block was treated as transient.
    assert slept == [], f"gate block entered the retry loop: slept {slept}"
    # The plain-text fallback would have reached the transport.
    assert adapter.sent == []


def test_send_with_retry_unaffected_when_gate_open():
    adapter = _FakeAdapter(_config(None))
    result = asyncio.run(adapter._send_with_retry(chat_id="c1", content="hello"))
    assert result.success is True
    assert adapter.sent == [("c1", "hello")]


BUILTIN_ADAPTER_MODULES = [
    "gateway.platforms.signal",
    "gateway.platforms.webhook",
    "gateway.platforms.weixin",
    "gateway.platforms.whatsapp_cloud",
    "gateway.platforms.bluebubbles",
    "gateway.platforms.yuanbao",
    "gateway.platforms.qqbot.adapter",
    "gateway.platforms.msgraph_webhook",
    "gateway.platforms.api_server",
    "gateway.relay.adapter",
]


@pytest.mark.parametrize("module_name", BUILTIN_ADAPTER_MODULES)
def test_builtin_adapters_are_gated(module_name):
    """Adapters outside plugins/ -- including RelayAdapter and APIServerAdapter."""
    module = importlib.import_module(module_name)
    found = [
        attr
        for attr in vars(module).values()
        if isinstance(attr, type)
        and issubclass(attr, BasePlatformAdapter)
        and attr is not BasePlatformAdapter
        and attr.__module__ == module_name
    ]
    assert found, f"no adapter class found in {module_name}"
    for cls in found:
        assert getattr(cls.send, "__send_gate_wrapped__", False), (
            f"{module_name}.{cls.__name__}.send is not gated"
        )


def test_non_overriding_adapter_keeps_base_method_identity():
    """The gate must not make every adapter look like it overrides everything.

    Code (and tests such as tests/gateway/test_signal.py) distinguishes "this
    adapter has a native media implementation" from "it inherits the default"
    by comparing the bound function against BasePlatformAdapter's. Installing
    a per-subclass wrapper unconditionally would silently break that; the
    already-wrapped check in install_send_gate is what preserves it.
    """
    from gateway.platforms.signal import SignalAdapter

    ntfy = importlib.import_module("plugins.platforms.ntfy.adapter").NtfyAdapter
    assert ntfy.send_image is BasePlatformAdapter.send_image
    assert SignalAdapter.send_image is not BasePlatformAdapter.send_image


def test_platform_name_of():
    assert platform_name_of(Platform.SLACK) == "slack"
    assert platform_name_of(_FakeAdapter(_config(None))) == "slack"
    assert platform_name_of(None) is None


# --------------------------------------------------------------------------
# Standalone / out-of-process sender layer.
# --------------------------------------------------------------------------


def test_send_message_tool_blocks_before_dispatch(monkeypatch):
    """_send_to_platform must not reach any transport when the gate is closed."""
    from tools import send_message_tool

    called = []
    monkeypatch.setattr(
        send_message_tool,
        "_send_weixin",
        lambda *a, **kw: called.append("weixin"),
    )
    monkeypatch.setattr(
        send_message_tool,
        "_send_via_adapter",
        lambda *a, **kw: called.append("adapter"),
    )

    result = asyncio.run(
        send_message_tool._send_to_platform(
            Platform.SLACK, _config("disabled"), "c1", "hello"
        )
    )
    assert "error" in result
    assert "send_gate" in result["error"]
    assert called == []


def test_send_message_tool_blocks_weixin_native_path(monkeypatch):
    """Weixin short-circuits above the adapter imports; the gate sits above it."""
    from tools import send_message_tool

    called = []
    monkeypatch.setattr(
        send_message_tool,
        "_send_weixin",
        lambda *a, **kw: called.append("weixin"),
    )
    result = asyncio.run(
        send_message_tool._send_to_platform(
            Platform.WEIXIN, _config("disabled"), "c1", "hello"
        )
    )
    assert "error" in result
    assert called == []


# --------------------------------------------------------------------------
# End-to-end: real config.yaml under a temporary HERMES_HOME.
# --------------------------------------------------------------------------


def test_e2e_config_yaml_closes_the_gate(monkeypatch, tmp_path):
    """A real on-disk config must produce a blocking adapter."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      send_gate: disabled\n"
        "  telegram:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()
    slack_cfg = config.platforms[Platform.SLACK]
    telegram_cfg = config.platforms[Platform.TELEGRAM]

    assert is_send_blocked(slack_cfg) is True
    # The gate is per-platform: closing Slack must not close Telegram.
    assert is_send_blocked(telegram_cfg) is False

    blocked = _FakeAdapter(slack_cfg)
    with pytest.raises(SendGateDisabledError):
        asyncio.run(blocked.send("c1", "hello"))
    assert blocked.sent == []

    allowed = _FakeAdapter(telegram_cfg)
    assert asyncio.run(allowed.send("c1", "hello")).success is True


def test_e2e_receiving_is_unaffected(monkeypatch, tmp_path):
    """The gate blocks sending only; the adapter still connects and reads."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  slack:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      send_gate: disabled\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = _FakeAdapter(load_gateway_config().platforms[Platform.SLACK])
    assert asyncio.run(adapter.connect()) is True
    assert asyncio.run(adapter.get_chat_info("c1")) == {}
    asyncio.run(adapter.disconnect())
