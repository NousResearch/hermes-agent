import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_owner_gate_bridge_initializes_dormant():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    assert adapter._owner_gate_pending_task is None
    assert adapter._owner_gate_notified_gate_ids == set()


def test_owner_gate_redaction_masks_credentials():
    raw = (
        "token=abc secret:xyz password=p api_key=q "
        "Authorization=Bearer value sk-abcdefghijk"
    )
    redacted = TelegramAdapter._owner_gate_redact_text(raw)
    assert "abc" not in redacted
    assert "xyz" not in redacted
    assert "password=p" not in redacted
    assert "api_key=q" not in redacted
    assert "sk-abcdefghijk" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.parametrize(
    "command,secret",
    [
        ("deploy --token supersecret", "supersecret"),
        ("deploy --api-key supersecret", "supersecret"),
        ('deploy --json \'{"api_key": "supersecret"}\'', "supersecret"),
        ("deploy Authorization=Bearer supersecret", "supersecret"),
    ],
)
def test_owner_gate_entry_summary_never_includes_raw_command(command, secret):
    summary = TelegramAdapter._owner_gate_entry_summary(
        {"tool_name": "terminal", "tool_args": {"command": command}}
    )
    assert secret not in summary
    assert command not in summary
    assert "command=[REDACTED]" in summary


def test_owner_gate_owner_chat_id_reads_only_named_key(tmp_path, monkeypatch):
    env_file = tmp_path / "owner.env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=must-not-return\nTELEGRAM_CHAT_ID=12345\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_OWNER_GATE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.setenv("HERMES_OWNER_GATE_TELEGRAM_ENV_FILE", str(env_file))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    assert adapter._owner_gate_owner_chat_id() == "12345"


def test_owner_gate_owner_chat_id_ignores_generic_chat_env(monkeypatch):
    monkeypatch.delenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_OWNER_GATE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "shared-chat")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "shared-home")
    monkeypatch.setenv("HERMES_OWNER_GATE_TELEGRAM_ENV_FILE", "/nonexistent")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    assert adapter._owner_gate_owner_chat_id() is None


def test_owner_gate_owner_chat_file_rejects_unsafe_mode(tmp_path, monkeypatch):
    env_file = tmp_path / "owner.env"
    env_file.write_text("TELEGRAM_CHAT_ID=12345\n", encoding="utf-8")
    env_file.chmod(0o666)
    monkeypatch.delenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_OWNER_GATE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("HERMES_OWNER_GATE_TELEGRAM_ENV_FILE", str(env_file))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    assert adapter._owner_gate_owner_chat_id() is None


def test_owner_gate_bridge_requires_explicit_flag_and_owner_binding(monkeypatch):
    monkeypatch.delenv("HERMES_OWNER_GATE_BRIDGE_ENABLED", raising=False)
    monkeypatch.delenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_OWNER_GATE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("HERMES_OWNER_GATE_TELEGRAM_ENV_FILE", "/nonexistent")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert adapter._owner_gate_bridge_enabled() is False
    monkeypatch.setenv("HERMES_OWNER_GATE_BRIDGE_ENABLED", "1")
    assert adapter._owner_gate_bridge_enabled() is False
    monkeypatch.setenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", "12345")
    assert adapter._owner_gate_bridge_enabled() is True


def test_owner_gate_commands_require_private_owner_sender(monkeypatch):
    monkeypatch.setenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", "12345")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    private_owner = SimpleNamespace(
        chat=SimpleNamespace(id="12345", type="private"),
        from_user=SimpleNamespace(id="12345"),
    )
    shared_chat = SimpleNamespace(
        chat=SimpleNamespace(id="12345", type="group"),
        from_user=SimpleNamespace(id="12345"),
    )
    different_sender = SimpleNamespace(
        chat=SimpleNamespace(id="12345", type="private"),
        from_user=SimpleNamespace(id="67890"),
    )

    assert adapter._owner_gate_is_owner_chat(private_owner) is True
    assert adapter._owner_gate_is_owner_chat(shared_chat) is False
    assert adapter._owner_gate_is_owner_chat(different_sender) is False


@pytest.mark.asyncio
async def test_owner_gate_pending_poll_sends_once(tmp_path, monkeypatch):
    (tmp_path / "pending.json").write_text("{}", encoding="utf-8")
    receiver = types.ModuleType("ultra_instinkt.telegram_owner_gate_receiver")
    receiver.list_pending = lambda _store: [{"gate_id": "gate-1"}]
    package = types.ModuleType("ultra_instinkt")
    package.telegram_owner_gate_receiver = receiver
    monkeypatch.setitem(sys.modules, "ultra_instinkt", package)
    monkeypatch.setitem(
        sys.modules,
        "ultra_instinkt.telegram_owner_gate_receiver",
        receiver,
    )
    monkeypatch.setenv("HERMES_OWNER_GATE_PENDING_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_OWNER_GATE_BRIDGE_ENABLED", "1")
    monkeypatch.setenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", "12345")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._owner_gate_send_pending_notice = AsyncMock(return_value=True)

    assert await adapter._owner_gate_poll_pending_once() == 1
    assert await adapter._owner_gate_poll_pending_once() == 0
    adapter._owner_gate_send_pending_notice.assert_awaited_once_with(
        {"gate_id": "gate-1"}
    )


@pytest.mark.asyncio
async def test_owner_gate_approve_routes_through_scoped_receiver(tmp_path, monkeypatch):
    calls = {}
    receiver = types.ModuleType("ultra_instinkt.telegram_owner_gate_receiver")
    receiver.parse_command = lambda _text: SimpleNamespace(
        action="approve", gate_id="gate-1"
    )

    def process_update(update, state, **kwargs):
        calls["update"] = update
        calls["state"] = state
        calls["kwargs"] = kwargs
        return {"handled": True, "granted": True}

    receiver.process_update = process_update
    receiver.list_pending = lambda _store: []
    receiver.format_pending_list = lambda _held: "No pending owner_gate actions."
    package = types.ModuleType("ultra_instinkt")
    package.telegram_owner_gate_receiver = receiver
    monkeypatch.setitem(sys.modules, "ultra_instinkt", package)
    monkeypatch.setitem(
        sys.modules,
        "ultra_instinkt.telegram_owner_gate_receiver",
        receiver,
    )
    monkeypatch.setenv("HERMES_OWNER_GATE_PENDING_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_OWNER_GATE_BRIDGE_ENABLED", "1")
    monkeypatch.setenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", "12345")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    msg = SimpleNamespace(
        text="/approve gate-1",
        chat=SimpleNamespace(id="12345", type="private"),
        from_user=SimpleNamespace(id="12345"),
    )

    assert await adapter._maybe_handle_owner_gate_command(msg) is True
    assert calls["update"] == {
        "message": {"chat": {"id": "12345"}, "text": "/approve gate-1"}
    }
    assert calls["state"] == {}
    assert calls["kwargs"]["store_dir"] == tmp_path
    assert calls["kwargs"]["owner_chat_id"] == "12345"
    assert calls["kwargs"]["dry_run"] is True
    assert calls["kwargs"]["credential"] == "bridge-dry-run-no-credential"
    adapter.send.assert_awaited_once()
    assert "gate-1 was granted within" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_owner_gate_command_ignores_group_and_different_sender(monkeypatch):
    monkeypatch.setenv("HERMES_OWNER_GATE_BRIDGE_ENABLED", "1")
    monkeypatch.setenv("ULTRA_INSTINKT_TELEGRAM_OWNER_CHAT_ID", "12345")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

    group = SimpleNamespace(
        text="/approve gate-1",
        chat=SimpleNamespace(id="12345", type="group"),
        from_user=SimpleNamespace(id="12345"),
    )
    stranger = SimpleNamespace(
        text="/approve gate-1",
        chat=SimpleNamespace(id="12345", type="private"),
        from_user=SimpleNamespace(id="67890"),
    )

    assert await adapter._maybe_handle_owner_gate_command(group) is False
    assert await adapter._maybe_handle_owner_gate_command(stranger) is False
    adapter.send.assert_not_awaited()
