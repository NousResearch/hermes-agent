"""Signal native receipts through the registered plugin and gateway boundaries."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from gateway.config import Platform, PlatformConfig
from tests.gateway.test_signal import _make_signal_adapter, _stub_rpc, _reset_signal_scheduler

@pytest.fixture(autouse=True)
def _discover_signal_plugin():
    from hermes_cli.plugins import discover_plugins
    discover_plugins()

def _make_receipt_event(
    adapter,
    *,
    chat_id="+15551239999",
    chat_type="dm",
    sender="+15551239999",
    sender_uuid="05668cf3-8ffa-467e-9b24-f5eefa5cf475",
    timestamp_ms=1777600696077,
    is_note_to_self=False,
):
    """Build a Signal event carrying the real receipt-target metadata."""
    from gateway.platforms.base import MessageEvent

    source = adapter.build_source(
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=sender,
        user_name=sender,
    )
    return MessageEvent(
        text="hello",
        source=source,
        raw_message={
            "sender": sender,
            "sender_uuid": sender_uuid,
            "timestamp_ms": timestamp_ms,
            "is_note_to_self": is_note_to_self,
        },
    )


class TestSignalReadReceipts:
    @pytest.fixture(autouse=True)
    def _disable_progress_reactions(self, monkeypatch):
        """Keep receipt lifecycle tests isolated from reaction RPCs."""
        monkeypatch.setenv("SIGNAL_REACTIONS", "false")

    @pytest.mark.asyncio
    async def test_send_read_receipt_uses_signal_cli_contract(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter._rpc, captured = _stub_rpc(None)

        await adapter.send_read_receipt(
            "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
            1777600696077,
        )

        assert captured == [{
            "method": "sendReceipt",
            "params": {
                "account": "+15551234567",
                "recipient": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "type": "read",
                "targetTimestamps": [1777600696077],
            },
        }]

    @pytest.mark.asyncio
    async def test_authorized_dm_is_scheduled_without_blocking(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.handle_message = AsyncMock()
        release = asyncio.Event()

        async def delayed_receipt(recipient, timestamp):
            await release.wait()

        adapter.send_read_receipt = AsyncMock(side_effect=delayed_receipt)
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15551239999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "timestamp": 1777600696000,
                "dataMessage": {
                    "message": "hello",
                    # Receipt targeting must prefer the data-message timestamp.
                    "timestamp": 1777600696077,
                },
            },
        })
        await asyncio.sleep(0)

        adapter.handle_message.assert_awaited_once()
        adapter.send_read_receipt.assert_awaited_once_with(
            "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
            1777600696077,
        )
        assert any(not task.done() for task in adapter._background_tasks)

        release.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_group_receipt_targets_author_not_group(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        auth_calls = []
        adapter.set_authorization_check(
            lambda user_id, chat_type, chat_id: auth_calls.append(
                (user_id, chat_type, chat_id)
            ) or True
        )
        adapter.send_read_receipt = AsyncMock()
        event = _make_receipt_event(
            adapter,
            chat_id="group:abc123=",
            chat_type="group",
        )

        adapter._schedule_read_receipt(event)
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_awaited_once_with(
            "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
            1777600696077,
        )
        assert auth_calls == [(
            "+15551239999",
            "group",
            "group:abc123=",
        )]

    @pytest.mark.asyncio
    async def test_phone_number_is_receipt_recipient_without_uuid(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.send_read_receipt = AsyncMock()

        adapter._schedule_read_receipt(
            _make_receipt_event(adapter, sender_uuid=None)
        )
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_awaited_once_with(
            "+15551239999",
            1777600696077,
        )

    @pytest.mark.asyncio
    async def test_disabled_receipts_do_not_schedule(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=False)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.send_read_receipt = AsyncMock()

        adapter._schedule_read_receipt(_make_receipt_event(adapter))
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("authorized", [False, None])
    async def test_unapproved_or_unknown_sender_is_not_acknowledged(
        self, monkeypatch, authorized
    ):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        if authorized is not None:
            adapter.set_authorization_check(
                lambda user_id, chat_type, chat_id: authorized
            )
        adapter.send_read_receipt = AsyncMock()

        adapter._schedule_read_receipt(_make_receipt_event(adapter))
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejected_profile_route_is_not_acknowledged(self, monkeypatch):
        """Match GatewayRunner's pre-auth fail-closed multiplexing gate."""
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.send_read_receipt = AsyncMock()
        event = _make_receipt_event(adapter)
        event.source.profile_route_rejected = True

        adapter._schedule_read_receipt(event)
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_not_awaited()
        assert not adapter._background_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("route_profile", "served_profiles", "expected_profile", "rejected", "receipted"),
        [
            ("work", ["default"], None, True, False),
            ("work", ["default", "work"], "work", False, True),
            ("default", ["default"], "default", False, True),
        ],
    )
    async def test_factory_routes_before_authorizing_read_receipts(
        self,
        monkeypatch,
        route_profile,
        served_profiles,
        expected_profile,
        rejected,
        receipted,
    ):
        """The real Signal factory must route before applying receipt policy."""
        from gateway.profile_routing import ProfileRoute
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from gateway.run import GatewayRunner

        config = PlatformConfig()
        config.enabled = True
        config.extra = {
            "http_url": "http://localhost:8080",
            "account": "+15551234567",
            "send_read_receipts": True,
        }
        runner = object.__new__(GatewayRunner)
        runner.config = SimpleNamespace(
            group_sessions_per_user=False,
            thread_sessions_per_user=False,
            multiplex_profiles=True,
            profile_routes=[
                ProfileRoute(
                    name="signal-chat",
                    platform="signal",
                    profile=route_profile,
                    chat_id="+15551239999",
                )
            ],
        )

        with (
            patch("plugins.platforms.signal.adapter.check_signal_requirements", return_value=True),
            patch("plugins.platforms.signal.adapter.validate_signal_config", return_value=True),
            patch(
                "gateway.run._multiplex_profile_homes",
                return_value=[
                    (name, Path(f"/profiles/{name}")) for name in served_profiles
                ],
            ),
            patch("hermes_cli.profiles.get_active_profile_name", return_value="default"),
        ):
            adapter = runner._create_adapter(Platform.SIGNAL, config)
            assert adapter is not None
            runner.adapters = {Platform.SIGNAL: adapter}
            runner._profile_adapters = {}
            authorized_profiles = []
            runner._is_user_authorized_for_source = lambda source: (
                authorized_profiles.append(source.profile) or True)
            adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))
            adapter.handle_message = AsyncMock()
            adapter.send_read_receipt = AsyncMock()

            await adapter._handle_envelope({
                "envelope": {
                    "sourceNumber": "+15551239999",
                    "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                    "timestamp": 1777600696000,
                    "dataMessage": {
                        "message": "hello",
                        "timestamp": 1777600696077,
                    },
                },
            })
            await asyncio.sleep(0)

        assert adapter.gateway_runner is runner
        assert authorized_profiles == ([expected_profile] if receipted else [])
        if receipted:
            adapter.send_read_receipt.assert_awaited_once()
        else:
            adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.parametrize(
        ("primary_env", "primary_value"),
        [
            ("SIGNAL_ALLOWED_USERS", "+15551239999"),
            ("SIGNAL_ALLOW_ALL_USERS", "true"),
            ("GATEWAY_ALLOWED_USERS", "+15551239999"),
            ("GATEWAY_ALLOW_ALL_USERS", "true"),
        ],
    )
    def test_secondary_receipt_auth_does_not_borrow_primary_env(
        self, monkeypatch, tmp_path, primary_env, primary_value
    ):
        """The callback installs its owner scope instead of borrowing primary env."""
        from agent.secret_scope import (
            set_multiplex_active,
        )
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv(primary_env, primary_value)
        profile_home = tmp_path / "work"
        profile_home.mkdir()
        (profile_home / ".env").write_text(
            "UNRELATED=secondary-only\n",
            encoding="utf-8",
        )
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {}
        runner._profile_adapters = {"work": {Platform.SIGNAL: adapter}}
        runner._active_profile_name = lambda: "default"
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {"work": MagicMock()}
        runner.pairing_stores["work"].is_approved.return_value = False
        adapter.gateway_runner = runner
        with patch(
            "hermes_cli.profiles.get_profile_dir",
            return_value=profile_home,
        ):
            adapter.set_authorization_check(
                runner._make_adapter_auth_check(
                    Platform.SIGNAL,
                    profile_name="work",
                )
            )

        event = _make_receipt_event(adapter)
        set_multiplex_active(True)
        try:
            assert adapter._read_receipt_target(event) is None
        finally:
            set_multiplex_active(False)


    @pytest.mark.parametrize(
        ("stale_env", "stale_value"),
        [
            ("SIGNAL_ALLOWED_USERS", "+15551239999"),
            ("SIGNAL_ALLOW_ALL_USERS", "true"),
            ("GATEWAY_ALLOWED_USERS", "+15551239999"),
            ("GATEWAY_ALLOW_ALL_USERS", "true"),
        ],
    )
    def test_primary_receipt_auth_does_not_borrow_stale_process_env(
        self, monkeypatch, tmp_path, stale_env, stale_value
    ):
        """Primary receipt auth reloads the active profile before ingress auth."""
        from agent.secret_scope import set_multiplex_active
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv(stale_env, stale_value)
        profile_home = tmp_path / "custom-hermes-home"
        profile_home.mkdir()
        (profile_home / ".env").write_text(
            "SIGNAL_ALLOWED_USERS=someone-else\n",
            encoding="utf-8",
        )
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {Platform.SIGNAL: adapter}
        runner._profile_adapters = {}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {}
        adapter.gateway_runner = runner

        with patch("gateway.run.get_hermes_home", return_value=profile_home):
            adapter.set_authorization_check(
                runner._make_adapter_auth_check(Platform.SIGNAL)
            )

        set_multiplex_active(True)
        try:
            assert adapter._read_receipt_target(_make_receipt_event(adapter)) is None
        finally:
            set_multiplex_active(False)

    def test_primary_receipt_auth_reloads_live_profile_allowlist(
        self, monkeypatch, tmp_path
    ):
        """A listener callback sees allowlist grants and revocations after connect."""
        from agent.secret_scope import set_multiplex_active
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        for key in (
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_ALLOW_ALL_USERS",
            "GATEWAY_ALLOWED_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(key, raising=False)
        profile_home = tmp_path / "custom-hermes-home"
        profile_home.mkdir()
        env_path = profile_home / ".env"
        env_path.write_text(
            "SIGNAL_ALLOWED_USERS=+15551239999\n",
            encoding="utf-8",
        )
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {Platform.SIGNAL: adapter}
        runner._profile_adapters = {}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {}
        adapter.gateway_runner = runner

        with patch("gateway.run.get_hermes_home", return_value=profile_home):
            adapter.set_authorization_check(
                runner._make_adapter_auth_check(Platform.SIGNAL)
            )

        event = _make_receipt_event(adapter)
        set_multiplex_active(True)
        try:
            assert adapter._read_receipt_target(event) == (
                "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                1777600696077,
            )
            env_path.write_text(
                "SIGNAL_ALLOWED_USERS=someone-else\n",
                encoding="utf-8",
            )
            assert adapter._read_receipt_target(event) is None
        finally:
            set_multiplex_active(False)

    def test_single_profile_receipt_auth_keeps_process_env(self, monkeypatch):
        """The legacy single-profile environment contract is unchanged."""
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15551239999")
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=False)
        runner.adapters = {Platform.SIGNAL: adapter}
        runner._profile_adapters = {}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        adapter.gateway_runner = runner
        adapter.set_authorization_check(
            runner._make_adapter_auth_check(Platform.SIGNAL)
        )

        assert adapter._read_receipt_target(_make_receipt_event(adapter)) == (
            "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
            1777600696077,
        )

    @pytest.mark.asyncio
    async def test_note_to_self_is_not_acknowledged(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.send_read_receipt = AsyncMock()

        adapter._schedule_read_receipt(
            _make_receipt_event(adapter, is_note_to_self=True)
        )
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_note_to_self_envelope_is_handled_without_receipt(self, monkeypatch):
        """The live sync-envelope path must carry the privacy exclusion."""
        adapter = _make_signal_adapter(
            monkeypatch,
            account="+155****4567",
            send_read_receipts=True,
        )
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.handle_message = AsyncMock()
        adapter.send_read_receipt = AsyncMock()

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 2000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****4567",
                        "destination": "+155****4567",
                        "timestamp": 2000000000,
                        "message": "note to self: buy milk",
                    }
                },
            }
        })
        await asyncio.sleep(0)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.raw_message["is_note_to_self"] is True
        adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timestamp", [None, 0, -1, True, "not-a-timestamp"])
    async def test_invalid_timestamp_is_not_acknowledged(
        self, monkeypatch, timestamp
    ):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.send_read_receipt = AsyncMock()

        adapter._schedule_read_receipt(
            _make_receipt_event(adapter, timestamp_ms=timestamp)
        )
        await asyncio.sleep(0)

        adapter.send_read_receipt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_signal_filters_run_before_receipt_scheduling(self, monkeypatch):
        """Stories, empty messages, and rejected groups never reach receipts."""
        filtered_cases = [
            (
                _make_signal_adapter(
                    monkeypatch,
                    send_read_receipts=True,
                    ignore_stories=True,
                ),
                {
                    "envelope": {
                        "sourceNumber": "+15551239999",
                        "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                        "timestamp": 1777600696077,
                        "storyMessage": {"allowsReplies": True},
                        "dataMessage": {"message": "story"},
                    },
                },
            ),
            (
                _make_signal_adapter(monkeypatch, send_read_receipts=True),
                {
                    "envelope": {
                        "sourceNumber": "+15551239999",
                        "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                        "timestamp": 1777600696077,
                        "dataMessage": {"message": ""},
                    },
                },
            ),
            (
                _make_signal_adapter(monkeypatch, send_read_receipts=True),
                {
                    "envelope": {
                        "sourceNumber": "+15551239999",
                        "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                        "timestamp": 1777600696077,
                        "dataMessage": {
                            "message": "hello group",
                            "groupInfo": {"groupId": "not-allowed"},
                        },
                    },
                },
            ),
            (
                _make_signal_adapter(
                    monkeypatch,
                    send_read_receipts=True,
                    group_allowed="allowed-group",
                    require_mention=True,
                ),
                {
                    "envelope": {
                        "sourceNumber": "+15551239999",
                        "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                        "timestamp": 1777600696077,
                        "dataMessage": {
                            "message": "not mentioning the bot",
                            "groupInfo": {"groupId": "allowed-group"},
                        },
                    },
                },
            ),
        ]

        for adapter, envelope in filtered_cases:
            adapter._schedule_read_receipt = MagicMock()
            adapter.handle_message = AsyncMock()

            await adapter._handle_envelope(envelope)

            adapter._schedule_read_receipt.assert_not_called()
            adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_receipt_rpc_failure_does_not_block_message_handling(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.handle_message = AsyncMock()
        adapter._rpc = AsyncMock(side_effect=RuntimeError("daemon unavailable"))

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15551239999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "timestamp": 1777600696077,
                "dataMessage": {"message": "hello"},
            },
        })
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        adapter.handle_message.assert_awaited_once()
        adapter._rpc.assert_awaited_once()
        assert adapter._rpc.await_args.kwargs["log_failures"] is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tracked_receipt_task(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        blocked = asyncio.Event()

        async def wait_forever(recipient, timestamp):
            await blocked.wait()

        adapter.send_read_receipt = AsyncMock(side_effect=wait_forever)
        adapter._schedule_read_receipt(_make_receipt_event(adapter))
        await asyncio.sleep(0)
        tasks = list(adapter._background_tasks)

        await adapter.cancel_background_tasks()

        assert len(tasks) == 1
        assert tasks[0].cancelled()
        assert adapter._background_tasks == set()
