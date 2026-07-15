"""Behavior and integration tests for the bundled Delta Chat platform."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_deltachat = load_plugin_adapter("deltachat")

DeltaChatAdapter = _deltachat.DeltaChatAdapter
DeltaChatRPC = _deltachat.DeltaChatRPC
DeltaChatRPCError = _deltachat.DeltaChatRPCError
_create_chatmail_account = _deltachat._create_chatmail_account
_wait_until_ready = _deltachat._wait_until_ready
is_connected = _deltachat.is_connected
mentions_bot = _deltachat.mentions_bot
parse_email_setting = _deltachat.parse_email_setting
register = _deltachat.register
validate_config = _deltachat.validate_config
_apply_yaml_config = _deltachat._apply_yaml_config


class FakeRPC:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def call(self, method, *params, timeout=30.0):
        self.calls.append((method, params, timeout))
        response = self.responses.get(method)
        if callable(response):
            return response(*params)
        return response


class LifecycleRPC(FakeRPC):
    def __init__(self, responses=None):
        super().__init__(responses)
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def call(self, method, *params, timeout=30.0):
        if method == "wait_next_msgs":
            await asyncio.Event().wait()
        return await super().call(method, *params, timeout=timeout)


def _config(**extra):
    from gateway.config import PlatformConfig

    defaults = {
        "email": "hermes@example.org",
        "display_name": "Hermes Agent",
        "dm_policy": "pairing",
        "group_policy": "disabled",
    }
    defaults.update(extra)
    return PlatformConfig(enabled=True, extra=defaults)


def test_platform_enum_resolves_via_bundled_plugin_scan():
    from gateway.config import Platform

    platform = Platform("deltachat")
    assert platform.value == "deltachat"
    assert Platform("deltachat") is platform


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Hermes@Example.ORG", ("hermes@example.org", False)),
        ("@nine.testrun.org", ("nine.testrun.org", True)),
    ],
)
def test_parse_email_setting(value, expected):
    assert parse_email_setting(value) == expected


@pytest.mark.parametrize("value", ["", "@", "@bad relay", "not-an-email"])
def test_parse_email_setting_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_email_setting(value)


def test_mention_detection_has_token_boundaries():
    assert mentions_bot("Hermes Agent, help", "Hermes Agent", "hermes@example.org")
    assert mentions_bot("hey @hermes!", "Hermes Agent", "hermes@example.org")
    assert not mentions_bot(
        "a hermes agentic workflow", "Hermes Agent", "hermes@example.org"
    )
    assert not mentions_bot("mail @hermes2", "Hermes Agent", "hermes@example.org")


def test_validate_and_connected_accept_configured_or_bootstrap_address():
    from gateway.config import PlatformConfig

    assert validate_config(_config()) is True
    assert is_connected(_config()) is True
    assert is_connected(_config(email="@nine.testrun.org")) is True
    assert validate_config(PlatformConfig(enabled=True, extra={})) is False
    assert (
        is_connected(PlatformConfig(enabled=False, extra={"email": "bot@example.org"}))
        is False
    )


def test_adapter_defaults_to_pairing_dms_and_disabled_groups():
    adapter = DeltaChatAdapter(_config())

    assert adapter.enforces_own_access_policy is True
    assert adapter._sender_allowed("alice@example.org", is_group=False) is True
    assert adapter._sender_allowed("alice@example.org", is_group=True) is False


def test_pairing_policy_never_opens_group_messages():
    adapter = DeltaChatAdapter(_config(group_policy="pairing"))

    assert adapter._sender_allowed("alice@example.org", is_group=True) is False


def test_adapter_allowlist_is_case_insensitive_for_dm_and_group():
    adapter = DeltaChatAdapter(
        _config(
            dm_policy="allowlist",
            group_policy="allowlist",
            allow_from=["Alice@Example.ORG"],
        )
    )

    assert adapter._sender_allowed("alice@example.org", is_group=False) is True
    assert adapter._sender_allowed("ALICE@example.org", is_group=True) is True
    assert adapter._sender_allowed("mallory@example.org", is_group=False) is False


def test_inbound_dm_builds_event_then_marks_message_seen():
    adapter = DeltaChatAdapter(
        _config(dm_policy="allowlist", allow_from=["alice@example.org"])
    )
    adapter._account_id = 7
    adapter._mark_connected()
    adapter.handle_message = AsyncMock()
    adapter._rpc = FakeRPC({
        "get_message": {
            "id": 91,
            "chatId": 42,
            "text": "hello",
            "timestamp": 1_700_000_000,
            "sender": {
                "address": "alice@example.org",
                "displayName": "Alice",
            },
        },
        "get_full_chat_by_id": {
            "id": 42,
            "name": "Alice",
            "chatType": "Single",
            "isContactRequest": True,
            "isDeviceChat": False,
        },
        "accept_chat": None,
        "markseen_msgs": None,
    })

    asyncio.run(adapter._handle_inbound_message(91))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.message_id == "91"
    assert event.source.platform.value == "deltachat"
    assert event.source.chat_id == "42"
    assert event.source.user_id == "alice@example.org"
    methods = [call[0] for call in adapter._rpc.calls]
    assert methods[-2:] == ["accept_chat", "markseen_msgs"]


def test_group_requires_mention_and_strips_it_before_dispatch():
    adapter = DeltaChatAdapter(
        _config(
            group_policy="allowlist",
            group_allow_from=["alice@example.org"],
            require_mention=True,
        )
    )
    adapter._account_id = 7
    adapter._mark_connected()
    adapter.handle_message = AsyncMock()
    messages = {
        1: {
            "id": 1,
            "chatId": 55,
            "text": "general chatter",
            "sender": {"address": "alice@example.org"},
        },
        2: {
            "id": 2,
            "chatId": 55,
            "text": "@hermes, summarize this",
            "sender": {"address": "alice@example.org"},
        },
    }
    adapter._rpc = FakeRPC({
        "get_message": lambda _account_id, message_id: messages[message_id],
        "get_full_chat_by_id": {
            "id": 55,
            "name": "Project",
            "chatType": "Group",
            "isContactRequest": False,
            "isDeviceChat": False,
        },
        "markseen_msgs": None,
    })

    asyncio.run(adapter._handle_inbound_message(1))
    adapter.handle_message.assert_not_awaited()

    asyncio.run(adapter._handle_inbound_message(2))
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "summarize this"
    assert event.source.chat_type == "group"


def test_send_uses_misc_send_msg_and_preserves_reply_anchor():
    adapter = DeltaChatAdapter(_config())
    adapter._account_id = 7
    adapter._mark_connected()
    adapter._rpc = FakeRPC({"misc_send_msg": [123, {"id": 123}]})

    result = asyncio.run(adapter.send("42", "answer", reply_to="91"))

    assert result.success is True
    assert result.message_id == "123"
    method, params, _timeout = adapter._rpc.calls[-1]
    assert method == "misc_send_msg"
    assert params == (7, 42, "answer", None, None, None, 91)


def test_connect_selects_account_starts_io_and_disconnects(tmp_path, monkeypatch):
    rpc = LifecycleRPC({
        "is_configured": True,
        "select_account": None,
        "batch_set_config": None,
        "start_io": None,
        "get_chat_securejoin_qr_code": "https://i.delta.chat/#invite",
        "stop_io": None,
    })

    async def _ready(_rpc):
        return None

    async def _find(_rpc, _email):
        return 7

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(_deltachat, "DeltaChatRPC", lambda *_args: rpc)
    monkeypatch.setattr(
        _deltachat, "_configured_rpc_path", lambda _extra=None: "/bin/true"
    )
    monkeypatch.setattr(_deltachat, "_wait_until_ready", _ready)
    monkeypatch.setattr(_deltachat, "_find_account", _find)
    monkeypatch.setattr(_deltachat, "_print_invite", lambda *_args: None)
    adapter = DeltaChatAdapter(
        _config(data_dir=str(tmp_path / "accounts"), show_invite_link=True)
    )

    async def _run():
        assert await adapter.connect() is True
        assert adapter.is_connected is True
        await adapter.disconnect()

    asyncio.run(_run())

    assert rpc.started is True
    assert rpc.closed is True
    methods = [call[0] for call in rpc.calls]
    assert methods[:4] == [
        "is_configured",
        "select_account",
        "batch_set_config",
        "batch_set_config",
    ]
    assert "start_io" in methods
    assert "get_chat_securejoin_qr_code" in methods
    assert "stop_io" in methods


def test_bootstrap_marker_creates_account_then_stops_for_config_update(
    tmp_path, monkeypatch
):
    rpc = LifecycleRPC()

    async def _ready(_rpc):
        return None

    async def _create(*_args):
        return 7, "generated@nine.testrun.org"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(_deltachat, "DeltaChatRPC", lambda *_args: rpc)
    monkeypatch.setattr(
        _deltachat, "_configured_rpc_path", lambda _extra=None: "/bin/true"
    )
    monkeypatch.setattr(_deltachat, "_wait_until_ready", _ready)
    monkeypatch.setattr(_deltachat, "_create_chatmail_account", _create)
    adapter = DeltaChatAdapter(
        _config(email="@nine.testrun.org", data_dir=str(tmp_path / "accounts"))
    )

    assert asyncio.run(adapter.connect()) is False

    assert rpc.closed is True
    assert adapter.has_fatal_error is True
    assert adapter._fatal_error_code == "deltachat_account_created"
    assert adapter._fatal_error_retryable is False
    assert "generated@nine.testrun.org" in adapter._fatal_error_message


def test_email_recipient_resolution_creates_contact_and_chat():
    adapter = DeltaChatAdapter(_config())
    adapter._account_id = 7
    adapter._mark_connected()
    adapter._rpc = FakeRPC({
        "lookup_contact_id_by_addr": 0,
        "create_contact": 33,
        "get_chat_id_by_contact_id": 0,
        "create_chat_by_contact_id": 44,
    })

    chat_id = asyncio.run(adapter._resolve_chat_id("Alice@Example.ORG"))

    assert chat_id == 44
    calls = [(method, params) for method, params, _timeout in adapter._rpc.calls]
    assert calls == [
        ("lookup_contact_id_by_addr", (7, "alice@example.org")),
        ("create_contact", (7, "alice@example.org", None)),
        ("get_chat_id_by_contact_id", (7, 33)),
        ("create_chat_by_contact_id", (7, 33)),
    ]


def test_unique_contact_name_wins_over_its_duplicate_chat_name():
    adapter = DeltaChatAdapter(_config())
    adapter._account_id = 7
    adapter._mark_connected()
    adapter._rpc = FakeRPC({
        "get_contacts": [
            {
                "id": 33,
                "address": "alice@example.org",
                "displayName": "Alice",
            }
        ],
        "get_chat_id_by_contact_id": 44,
        # A contact normally has a same-named single chat. Recipient
        # resolution must not count that as a second ambiguous identity.
        "get_chatlist_entries": [44],
    })

    chat_id = asyncio.run(adapter._resolve_chat_id("Alice"))

    assert chat_id == 44
    assert [call[0] for call in adapter._rpc.calls] == [
        "get_contacts",
        "get_chat_id_by_contact_id",
    ]


def test_inbound_attachment_is_copied_to_hermes_media_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    attachment = tmp_path / "source" / "report.txt"
    attachment.parent.mkdir()
    attachment.write_bytes(b"delta chat attachment")

    adapter = DeltaChatAdapter(
        _config(dm_policy="allowlist", allow_from=["alice@example.org"])
    )
    adapter._account_id = 7
    adapter._mark_connected()
    adapter.handle_message = AsyncMock()
    adapter._rpc = FakeRPC({
        "get_message": {
            "id": 92,
            "chatId": 42,
            "text": "",
            "file": str(attachment),
            "fileName": "report.txt",
            "fileMime": "text/plain",
            "sender": {"address": "alice@example.org"},
        },
        "get_full_chat_by_id": {
            "id": 42,
            "name": "Alice",
            "chatType": "Single",
            "isContactRequest": False,
            "isDeviceChat": False,
        },
        "markseen_msgs": None,
    })

    asyncio.run(adapter._handle_inbound_message(92))

    event = adapter.handle_message.await_args.args[0]
    assert event.text == ""
    assert event.media_types == ["text/plain"]
    assert len(event.media_urls) == 1
    cached = Path(event.media_urls[0])
    assert cached.is_file()
    assert cached.read_bytes() == b"delta chat attachment"


def test_animation_uses_base_contract_keyword_and_native_file_send(tmp_path):
    animation = tmp_path / "clip.gif"
    animation.write_bytes(b"GIF89a" + b"\x00" * 16)
    adapter = DeltaChatAdapter(_config())
    adapter._account_id = 7
    adapter._mark_connected()
    adapter._rpc = FakeRPC({"send_msg": 93})

    result = asyncio.run(
        adapter.send_animation(
            chat_id="42",
            animation_url=str(animation),
            caption="demo",
        )
    )

    assert result.success is True
    method, params, _timeout = adapter._rpc.calls[-1]
    assert method == "send_msg"
    assert params[0:2] == (7, 42)
    assert params[2]["file"] == str(animation.resolve())
    assert params[2]["filename"] == "clip.gif"
    assert "viewtype" not in params[2]


def test_voice_send_forces_voice_viewtype(tmp_path):
    audio = tmp_path / "reply.ogg"
    audio.write_bytes(b"OggS" + b"\x00" * 16)
    adapter = DeltaChatAdapter(_config())
    adapter._account_id = 7
    adapter._mark_connected()
    adapter._rpc = FakeRPC({"send_msg": 94})

    result = asyncio.run(adapter.send_voice("42", str(audio)))

    assert result.success is True
    _method, params, _timeout = adapter._rpc.calls[-1]
    assert params[2]["viewtype"] == "Voice"


def test_register_exposes_full_platform_hooks():
    ctx = MagicMock()

    register(ctx)

    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "deltachat"
    assert kwargs["label"] == "Delta Chat"
    assert kwargs["required_env"] == []
    assert callable(kwargs["setup_fn"])
    assert callable(kwargs["apply_yaml_config_fn"])
    assert kwargs["cron_deliver_env_var"] == "DELTACHAT_HOME_CHANNEL"
    assert callable(kwargs["standalone_sender_fn"])
    assert kwargs["pii_safe"] is True
    assert "attachments" in kwargs["platform_hint"]


def test_yaml_config_loads_plugin_and_bridges_cron_home_channel(tmp_path, monkeypatch):
    """Exercise plugin discovery + real config propagation end to end."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DELTACHAT_HOME_CHANNEL", raising=False)
    (tmp_path / "config.yaml").write_text(
        """
platforms:
  deltachat:
    enabled: true
    dm_policy: allowlist
    allow_from:
      - alice@example.org
    home_channel:
      platform: deltachat
      chat_id: "42"
      name: Notifications
    extra:
      email: hermes@example.org
      data_dir: /tmp/hermes-deltachat-test
""".lstrip(),
        encoding="utf-8",
    )

    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform = Platform("deltachat")
    delta_config = config.platforms[platform]

    assert delta_config.enabled is True
    assert delta_config.extra["email"] == "hermes@example.org"
    assert delta_config.extra["dm_policy"] == "allowlist"
    assert delta_config.extra["allow_from"] == ["alice@example.org"]
    assert delta_config.home_channel.chat_id == "42"
    assert os.environ["DELTACHAT_HOME_CHANNEL"] == "42"
    assert platform in config.get_connected_platforms()

    from cron.scheduler import cron_delivery_targets

    targets = {target["id"]: target for target in cron_delivery_targets()}
    assert targets["deltachat"]["home_target_set"] is True
    assert targets["deltachat"]["home_env_var"] == "DELTACHAT_HOME_CHANNEL"


def test_real_rpc_process_round_trip(tmp_path):
    """Exercise the actual subprocess/stdio correlation path when installed."""

    binary = shutil.which("deltachat-rpc-server")
    if not binary:
        pytest.skip("deltachat-rpc-server is not installed")

    async def _run():
        rpc = DeltaChatRPC(binary, tmp_path / "accounts")
        (tmp_path / "accounts").mkdir()
        await rpc.start()
        try:
            await _wait_until_ready(rpc)
            system_info, accounts = await asyncio.gather(
                rpc.call("get_system_info"),
                rpc.call("get_all_accounts"),
            )
            assert isinstance(system_info, dict)
            assert accounts == []
        finally:
            await rpc.close()

    asyncio.run(_run())


def test_rpc_error_response_is_raised(tmp_path):
    binary = shutil.which("deltachat-rpc-server")
    if not binary:
        pytest.skip("deltachat-rpc-server is not installed")

    async def _run():
        accounts_dir = tmp_path / "accounts"
        accounts_dir.mkdir()
        rpc = DeltaChatRPC(binary, accounts_dir)
        await rpc.start()
        try:
            await _wait_until_ready(rpc)
            with pytest.raises(DeltaChatRPCError):
                await rpc.call("method_that_does_not_exist")
        finally:
            await rpc.close()

    asyncio.run(_run())


def test_failed_chatmail_bootstrap_removes_pending_account():
    def _fail_transport(*_params):
        raise DeltaChatRPCError("relay rejected account")

    rpc = FakeRPC({
        "add_account": 7,
        "add_transport_from_qr": _fail_transport,
        "stop_ongoing_process": None,
        "remove_account": None,
    })

    with pytest.raises(DeltaChatRPCError, match="relay rejected"):
        asyncio.run(
            _create_chatmail_account(
                rpc,
                "nine.testrun.org",
                "Hermes Agent",
                "",
            )
        )

    assert [call[0] for call in rpc.calls] == [
        "add_account",
        "add_transport_from_qr",
        "stop_ongoing_process",
        "remove_account",
    ]
