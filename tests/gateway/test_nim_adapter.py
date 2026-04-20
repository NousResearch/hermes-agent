from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig, _default_nim_bridge_command, load_nim_config, load_nim_instances
from gateway.platforms.nim import MultiNimAdapter, NimAdapter, _check_nim_instance_requirements, check_nim_requirements
from gateway.run import GatewayRunner


class FakeBridge:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.config = None
        self.event_handler = None
        self.sent = []

    async def start(self, config, *, event_handler=None):
        self.started = True
        self.config = config
        self.event_handler = event_handler

    async def stop(self):
        self.stopped = True

    async def health(self):
        return {"connected": True}

    async def send_text(self, *, chat_id, text, session_type, reply_to=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "session_type": session_type,
                "reply_to": reply_to,
            }
        )
        return {"message_id": "server-msg-1", "client_message_id": "client-msg-1"}


@pytest.mark.asyncio
async def test_connect_and_disconnect():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"}),
        bridge=bridge,
    )

    assert await adapter.connect() is True
    assert adapter.is_connected is True
    assert bridge.started is True

    await adapter.disconnect()
    assert adapter.is_connected is False
    assert bridge.stopped is True


@pytest.mark.asyncio
async def test_send_team_message_uses_team_session_type():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"}),
        bridge=bridge,
    )

    result = await adapter.send("team:123", "hello", reply_to="reply-1")

    assert result.success is True
    assert result.message_id == "server-msg-1"
    assert bridge.sent == [
        {
            "chat_id": "team:123",
            "text": "hello",
            "session_type": "team",
            "reply_to": "reply-1",
        }
    ]


@pytest.mark.asyncio
async def test_send_qchat_message_uses_qchat_session_type():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"}),
        bridge=bridge,
    )

    result = await adapter.send("qchat:server-1:channel-2", "hello qchat")

    assert result.success is True
    assert bridge.sent == [
        {
            "chat_id": "qchat:server-1:channel-2",
            "text": "hello qchat",
            "session_type": "qchat",
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_send_splits_long_messages():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
                "advanced": {"textChunkLimit": 32},
            },
        ),
        bridge=bridge,
    )

    content = ("a" * 31) + "\n" + ("b" * 32)

    result = await adapter.send("user:123", content)

    assert result.success is True
    assert [item["text"] for item in bridge.sent] == [
        ("a" * 31) + "\n",
        "b" * 32,
    ]
    assert all(len(item["text"]) <= 32 for item in bridge.sent)


@pytest.mark.asyncio
async def test_inbound_dm_respects_allowlist():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
                "p2p": {"policy": "allowlist", "allowFrom": ["allowed-user"]},
            },
        ),
        bridge=bridge,
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "session_type": "p2p",
                "sender_id": "blocked-user",
                "sender_name": "Blocked",
                "text": "hello",
                "message_type": "text",
            },
        }
    )
    adapter.handle_message.assert_not_awaited()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "message_id": "m-1",
                "session_type": "p2p",
                "sender_id": "allowed-user",
                "sender_name": "Allowed",
                "text": "hello",
                "message_type": "text",
            },
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "user:allowed-user"
    assert event.source.chat_type == "dm"
    assert event.text == "hello"


@pytest.mark.asyncio
async def test_inbound_non_online_message_is_ignored():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
            },
        ),
        bridge=bridge,
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "message_id": "m-offline-1",
                "message_source": 3,
                "session_type": "p2p",
                "sender_id": "allowed-user",
                "sender_name": "Allowed",
                "text": "history hello",
                "message_type": "text",
            },
        }
    )

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_message_requires_allowed_team_and_mention():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
                "team": {
                    "policy": "allowlist",
                    "allowFrom": ["team-1"],
                },
            },
        ),
        bridge=bridge,
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "session_type": "team",
                "target_id": "team-1",
                "sender_id": "user-1",
                "text": "hello",
                "message_type": "text",
                "mentioned": False,
            },
        }
    )
    adapter.handle_message.assert_not_awaited()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "message_id": "m-2",
                "session_type": "team",
                "target_id": "team-1",
                "sender_id": "user-1",
                "text": "hello @bot",
                "message_type": "text",
                "mentioned": True,
            },
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "team:team-1"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_qchat_message_requires_allowlist_match_and_mention():
    bridge = FakeBridge()
    adapter = NimAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
                "qchat": {
                    "policy": "allowlist",
                    "allowFrom": ["server-1|channel-2|allowed-user"],
                },
            },
        ),
        bridge=bridge,
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "session_type": "qchat",
                "server_id": "server-1",
                "channel_id": "channel-2",
                "target_id": "server-1:channel-2",
                "sender_id": "blocked-user",
                "text": "@bot hello",
                "message_type": "text",
                "mentioned": True,
            },
        }
    )
    adapter.handle_message.assert_not_awaited()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "message_id": "m-qchat-1",
                "session_type": "qchat",
                "server_id": "server-1",
                "channel_id": "channel-2",
                "target_id": "server-1:channel-2",
                "sender_id": "allowed-user",
                "text": "@bot hello",
                "message_type": "text",
                "mentioned": True,
            },
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "qchat:server-1:channel-2"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_inbound_message_prefixes_chat_id_for_multi_instance():
    bridge = FakeBridge()
    resolved = load_nim_config(
        PlatformConfig(
            enabled=True,
            extra={
                "instance_name": "work",
                "nim_token": "app|bot|secret",
            },
        )
    )
    resolved.route_prefix = "work/"
    adapter = NimAdapter(
        PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"}),
        bridge=bridge,
        resolved=resolved,
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_bridge_event(
        {
            "event": "message",
            "payload": {
                "message_id": "m-3",
                "session_type": "p2p",
                "sender_id": "allowed-user",
                "sender_name": "Allowed",
                "text": "hello",
                "message_type": "text",
            },
        }
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "work/user:allowed-user"


@pytest.mark.asyncio
async def test_multi_instance_routes_send_to_matching_bridge():
    config = PlatformConfig(
        enabled=True,
        extra={
            "nim_token": "app|default-bot|secret-default",
            "instances": [
                {
                    "instance_name": "work",
                    "nim_token": "app|work-bot|secret-work",
                }
            ],
        },
    )
    instances = load_nim_instances(config)
    bridges = {}

    def bridge_factory(resolved):
        bridge = FakeBridge()
        bridges[resolved.instance_name] = bridge
        return bridge

    adapter = MultiNimAdapter(
        config,
        resolved_instances=instances,
        bridge_factory=bridge_factory,
    )

    assert await adapter.connect() is True

    await adapter.send("work/user:200", "hello work")
    await adapter.send("user:300", "hello default")

    assert bridges["work"].sent == [
        {
            "chat_id": "user:200",
            "text": "hello work",
            "session_type": "p2p",
            "reply_to": None,
        }
    ]
    assert bridges["default"].sent == [
        {
            "chat_id": "user:300",
            "text": "hello default",
            "session_type": "p2p",
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_multi_instance_rejects_unknown_route_prefix():
    config = PlatformConfig(
        enabled=True,
        extra={
            "nim_token": "app|default-bot|secret-default",
            "instances": [
                {
                    "instance_name": "work",
                    "nim_token": "app|work-bot|secret-work",
                }
            ],
        },
    )
    instances = load_nim_instances(config)
    bridges = {}

    def bridge_factory(resolved):
        bridge = FakeBridge()
        bridges[resolved.instance_name] = bridge
        return bridge

    adapter = MultiNimAdapter(
        config,
        resolved_instances=instances,
        bridge_factory=bridge_factory,
    )

    assert await adapter.connect() is True

    with pytest.raises(RuntimeError, match="unavailable"):
        await adapter.send("other/user:200", "hello unknown")

    assert bridges["default"].sent == []
    assert bridges["work"].sent == []


def test_multi_instance_dedupes_duplicate_qchat_events():
    config = PlatformConfig(
        enabled=True,
        extra={
            "instances": [
                {
                    "instance_name": "main",
                    "nim_token": "app|main-bot|secret-main",
                }
            ],
        },
    )
    instances = load_nim_instances(config)
    adapter = MultiNimAdapter(config, resolved_instances=instances)

    event = SimpleNamespace(
        message_id="qchat-server-msg-1",
        text="@bot 你是谁",
        source=SimpleNamespace(user_id="user-1"),
        raw_message={
            "session_type": "qchat",
            "server_id": "server-1",
            "channel_id": "channel-2",
            "sender_id": "user-1",
            "message_id": "qchat-server-msg-1",
        },
    )

    assert adapter._should_drop_duplicate_qchat_event(event) is False
    assert adapter._should_drop_duplicate_qchat_event(event) is True


def test_runner_uses_multi_adapter_for_single_explicit_instance(monkeypatch):
    config = PlatformConfig(
        enabled=True,
        extra={
            "instances": [
                {
                    "instance_name": "main",
                    "nim_token": "app|main-bot|secret-main",
                }
            ],
        },
    )
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)

    import gateway.platforms.nim as nim_module

    monkeypatch.setattr(nim_module, "check_nim_requirements", lambda _config: True)

    adapter = GatewayRunner._create_adapter(runner, Platform.NIM, config)
    assert isinstance(adapter, MultiNimAdapter)


def test_load_nim_config_hardcodes_packaged_bridge_command():
    resolved = load_nim_config(
        PlatformConfig(
            enabled=True,
            extra={
                "nim_token": "app|bot|secret",
                "bridge_command": ["node", "/tmp/custom/index.mjs"],
            },
        ),
        {"NIM_BRIDGE_COMMAND": "node /tmp/ignored/index.mjs"},
    )

    assert resolved.bridge_command == _default_nim_bridge_command()


def test_check_nim_requirements_supports_bundled_bridge(monkeypatch, tmp_path):
    bridge_dir = tmp_path / "nim_bot_py" / "bridge_js"
    bridge_dir.mkdir(parents=True)
    bridge_script = bridge_dir / "index.mjs"
    bridge_script.write_text("console.log('ok')\n", encoding="utf-8")

    monkeypatch.setattr("gateway.platforms.nim._default_nim_bridge_command", lambda: ["node", str(bridge_script)])
    monkeypatch.setattr("gateway.config._default_nim_bridge_command", lambda: ["node", str(bridge_script)])
    monkeypatch.setattr("gateway.platforms.nim.shutil.which", lambda name: f"/usr/bin/{name}")

    class FakePackagedBridge:
        def __init__(self, command):
            self.command = command

        def ensure_runtime(self):
            return None

    monkeypatch.setattr("gateway.platforms.nim.PackagedNodeBridge", FakePackagedBridge)

    assert check_nim_requirements(
        PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"})
    ) is True


def test_check_nim_instance_requirements_fails_when_packaged_bridge_runtime_is_unavailable(monkeypatch, tmp_path):
    bridge_dir = tmp_path / "nim_bot_py" / "bridge_js"
    bridge_dir.mkdir(parents=True)
    bridge_script = bridge_dir / "index.mjs"
    bridge_script.write_text("console.log('ok')\n", encoding="utf-8")

    monkeypatch.setattr("gateway.platforms.nim._default_nim_bridge_command", lambda: ["node", str(bridge_script)])
    monkeypatch.setattr("gateway.config._default_nim_bridge_command", lambda: ["node", str(bridge_script)])
    monkeypatch.setattr("gateway.platforms.nim.shutil.which", lambda name: f"/usr/bin/{name}")

    class FakePackagedBridge:
        def __init__(self, command):
            self.command = command

        def ensure_runtime(self):
            from nim_bot_py.bridge import NimBridgeError

            raise NimBridgeError("npm not found")

    monkeypatch.setattr("gateway.platforms.nim.PackagedNodeBridge", FakePackagedBridge)

    resolved = load_nim_config(PlatformConfig(enabled=True, extra={"nim_token": "app|bot|secret"}), {})
    assert resolved.bridge_command == ["node", str(bridge_script)]
    assert _check_nim_instance_requirements(resolved) is False
