"""Tests for Signal messenger platform adapter."""
import asyncio
import base64
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """The attachment scheduler is process-wide; drop it between tests
    so a fresh token bucket greets each case."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _stub_rpc(return_value):
    """Return an async mock for SignalAdapter._rpc that captures call params."""
    captured = []

    async def mock_rpc(method, params, rpc_id=None, **_kwargs):
        captured.append({"method": method, "params": dict(params)})
        return return_value

    return mock_rpc, captured


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


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSignalConfigLoading:
    def test_apply_env_overrides_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL in config.platforms
        sc = config.platforms[Platform.SIGNAL]
        assert sc.enabled is True
        assert sc.extra["http_url"] == "http://localhost:9090"
        assert sc.extra["account"] == "+15551234567"


    @pytest.mark.parametrize(
        ("yaml_value", "expected"),
        [("true", True), ('"false"', False)],
    )
    def test_config_yaml_bridges_read_receipts(
        self, monkeypatch, tmp_path, yaml_value, expected
    ):
        """The documented top-level setting reaches the adapter as a bool."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            f"signal:\n  send_read_receipts: {yaml_value}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)

        from gateway.config import load_gateway_config
        from gateway.platforms.signal import SignalAdapter

        config = load_gateway_config()
        signal_config = config.platforms[Platform.SIGNAL]

        assert signal_config.extra["send_read_receipts"] is expected
        assert SignalAdapter(signal_config).send_read_receipts is expected

# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------

class TestSignalAdapterInit:
    def test_init_parses_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="group123,group456")
        assert adapter.http_url == "http://localhost:8080"
        assert adapter.account == "+15551234567"
        assert "group123" in adapter.group_allow_from

    def test_read_receipts_are_opt_in(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter.send_read_receipts is False

    def test_read_receipts_can_be_enabled_in_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        assert adapter.send_read_receipts is True

    def test_quoted_false_does_not_enable_read_receipts(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts="false")
        assert adapter.send_read_receipts is False


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
            ("work", ["default", "work"], "work", False, False),
            ("default", ["default"], "default", False, True),
        ],
    )
    async def test_factory_scopes_receipts_to_signal_transport_profile(
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
            patch("gateway.platforms.signal.check_signal_requirements", return_value=True),
            patch("gateway.platforms.signal.validate_signal_config", return_value=True),
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
            adapter.set_authorization_check(
                lambda user_id, chat_type, chat_id: True
            )
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
        assert adapter._signal_transport_profile_name == "default"
        event = adapter.handle_message.await_args.args[0]
        assert event.source.profile == expected_profile
        assert event.source.profile_route_rejected is rejected
        if receipted:
            adapter.send_read_receipt.assert_awaited_once()
        else:
            adapter.send_read_receipt.assert_not_awaited()

    def test_secondary_signal_transport_is_profile_stamped(self, monkeypatch):
        """Secondary reconnects use the same owner-profile receipt boundary."""
        from gateway.run import GatewayRunner

        runner = MagicMock()
        adapter = _make_signal_adapter(monkeypatch)

        GatewayRunner._configure_profile_adapter(
            runner,
            adapter,
            "work",
            Platform.SIGNAL,
        )

        assert adapter._signal_transport_profile_name == "work"

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
        adapter._signal_transport_profile_name = "work"
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

    def test_secondary_receipt_auth_denies_when_profile_home_is_unresolved(
        self, monkeypatch
    ):
        """A broken owner resolver must not fall back to the process allowlist."""
        from agent.secret_scope import set_multiplex_active
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15551239999")
        adapter = _make_signal_adapter(monkeypatch, send_read_receipts=True)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {}
        runner._profile_adapters = {"work": {Platform.SIGNAL: adapter}}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {"work": MagicMock()}
        runner.pairing_stores["work"].is_approved.return_value = False
        adapter.gateway_runner = runner

        with patch(
            "hermes_cli.profiles.get_profile_dir",
            side_effect=RuntimeError("profile home unavailable"),
        ):
            adapter.set_authorization_check(
                runner._make_adapter_auth_check(
                    Platform.SIGNAL,
                    profile_name="work",
                )
            )

        set_multiplex_active(True)
        try:
            assert adapter._read_receipt_target(_make_receipt_event(adapter)) is None
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

class TestSignalConnectCleanup:
    """Regression coverage for failed connect() cleanup."""

    @pytest.mark.asyncio
    async def test_releases_lock_and_closes_client_on_healthcheck_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))
        mock_client.aclose = AsyncMock()

        with patch("gateway.platforms.signal.httpx.AsyncClient", return_value=mock_client), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        mock_client.aclose.assert_awaited_once()
        mock_release.assert_called_once_with("signal-phone", "+15551234567")
        assert adapter.client is None
        assert adapter._platform_lock_identity is None


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+155****4567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("") == "<none>"

    def test_parse_comma_list(self):
        from gateway.platforms.signal import _parse_comma_list
        assert _parse_comma_list("+1234, +5678 , +9012") == ["+1234", "+5678", "+9012"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []


    def test_guess_extension_wav_routes_to_audio_cache(self):
        """A detected WAV must route to the audio cache, not the document cache.

        ``.wav`` is already in ``_is_audio_ext``; the bug was purely that
        ``_guess_extension`` never produced ``.wav`` for raw bytes, so the
        attachment was treated as a document and STT never received it.
        """
        from gateway.platforms.signal import _is_audio_ext, _guess_extension
        wav = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 100
        ext = _guess_extension(wav)
        assert ext == ".wav"
        assert _is_audio_ext(ext) is True


    def test_guess_extension_m4a_audio_brand(self):
        """iOS Signal voice notes are MP4-container AAC with an M4A ftyp brand.

        Classifying them as ``.mp4`` sent them to the document cache and made
        STT reject the upload ("Invalid file format") even though the bytes
        were valid audio. Audio brands must resolve to ``.m4a``.
        """
        from gateway.platforms.signal import _guess_extension, _is_audio_ext
        for brand in (b"M4A ", b"M4B ", b"m4a "):
            data = b"\x00\x00\x00\x1cftyp" + brand + b"\x00" * 100
            assert _guess_extension(data) == ".m4a", brand
            assert _is_audio_ext(_guess_extension(data)) is True


    def test_remux_aac_to_m4a_round_trip(self):
        """A real ADTS AAC stream remuxes to a valid MP4 (.m4a) container.

        Generates a short ADTS AAC sample with ffmpeg at runtime so the
        end-to-end remux path actually exercises in CI (skipped only when
        ffmpeg is unavailable), rather than depending on a machine-specific
        file.
        """
        import shutil
        import subprocess
        import tempfile
        from gateway.platforms.signal import _remux_aac_to_m4a

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import pytest
            pytest.skip("ffmpeg not available in this env")

        # Synthesize 0.5s of silence encoded as raw ADTS AAC.
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            adts_path = tmp.name
        try:
            gen = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5",
                 "-c:a", "aac", "-f", "adts", adts_path],
                capture_output=True, timeout=30,
            )
            if gen.returncode != 0:
                import pytest
                pytest.skip("ffmpeg could not produce an ADTS AAC sample")
            with open(adts_path, "rb") as f:
                aac_data = f.read()
        finally:
            try:
                import os
                os.unlink(adts_path)
            except OSError:
                pass

        result = _remux_aac_to_m4a(aac_data)
        assert result is not None
        m4a_bytes, ext = result
        assert ext == ".m4a"
        # MP4 files start with a 4-byte size, then ``ftyp`` at offset 4.
        assert m4a_bytes[4:8] == b"ftyp", \
            f"expected MP4 ftyp box, got {m4a_bytes[:12]!r}"
        # File must be at least as long as the input (MP4 has overhead).
        assert len(m4a_bytes) >= len(aac_data) * 0.5


    def test_is_image_ext(self):
        from gateway.platforms.signal import _is_image_ext
        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False


    def test_check_requirements(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        assert check_signal_requirements() is True

    def test_render_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello \uFFFC, how are you?"
        mentions = [{"start": 6, "length": 1, "number": "+15559999999"}]
        result = _render_mentions(text, mentions)
        assert "@+15559999999" in result
        assert "\uFFFC" not in result


    def test_validate_signal_config_accepts_platform_values(self, monkeypatch):
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        from gateway.platforms.signal import validate_signal_config

        config = PlatformConfig(
            enabled=True,
            extra={
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
            },
        )
        assert validate_signal_config(config) is True


# ---------------------------------------------------------------------------
# SSE URL Encoding (Bug Fix: phone numbers with + must be URL-encoded)
# ---------------------------------------------------------------------------

class TestSignalSSEUrlEncoding:
    """Verify that phone numbers with + are URL-encoded in the SSE endpoint."""

    def test_sse_url_encodes_plus_in_account(self):
        """The + in E.164 phone numbers must be percent-encoded in the SSE query string."""
        encoded = quote("+31612345678", safe="")
        assert encoded == "%2B31612345678"


# ---------------------------------------------------------------------------
# Attachment Fetch (Bug Fix: parameter must be "id" not "attachmentId")
# ---------------------------------------------------------------------------

class TestSignalAttachmentFetch:
    """Verify that _fetch_attachment uses the correct RPC parameter name."""

    @pytest.mark.asyncio
    async def test_fetch_attachment_uses_id_parameter(self, monkeypatch):
        """RPC getAttachment must use 'id', not 'attachmentId' (signal-cli requirement)."""
        adapter = _make_signal_adapter(monkeypatch)

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()

        adapter._rpc, captured = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/test.png"):
            await adapter._fetch_attachment("attachment-123")

        call = captured[0]
        assert call["method"] == "getAttachment"
        assert call["params"]["id"] == "attachment-123"
        assert "attachmentId" not in call["params"], "Must NOT use 'attachmentId' — causes NullPointerException in signal-cli"
        assert call["params"]["account"] == "+15551234567"


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestSignalSessionSource:

    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:xyz",
            chat_type="group",
            user_id="+15551234567",
            user_id_alt="uuid:abc",
            chat_id_alt="xyz",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.user_id_alt == "uuid:abc"
        assert restored.chat_id_alt == "xyz"
        assert restored.platform == Platform.SIGNAL


# ---------------------------------------------------------------------------
# Phone Redaction in agent/redact.py
# ---------------------------------------------------------------------------

class TestSignalPhoneRedaction:
    @pytest.fixture(autouse=True)
    def _ensure_redaction_enabled(self, monkeypatch):
        # agent.redact snapshots _REDACT_ENABLED at import time from the
        # HERMES_REDACT_SECRETS env var. monkeypatch.delenv is too late —
        # the module was already imported during test collection with
        # whatever value was in the env then. Force the flag directly.
        # See skill: xdist-cross-test-pollution Pattern 5.
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    def test_us_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Call +15551234567 now")
        assert "+15551234567" not in result
        assert "+155" in result  # Prefix preserved
        assert "4567" in result  # Suffix preserved


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------

class TestSignalAuthorization:
    def test_signal_in_allowlist_maps(self):
        """Signal should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIGNAL
        source.user_id = "+15559999999"

        # No allowlists set — should check GATEWAY_ALLOW_ALL_USERS
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# send_image_file method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendImageFile:
    @pytest.mark.asyncio
    async def test_send_image_file_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_image_file should send image as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "chart.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is True
        assert len(captured) == 1
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["account"] == adapter.account
        assert captured[0]["params"]["recipient"] == ["+155****4567"]
        assert captured[0]["params"]["attachments"] == [str(img_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        # Typing indicator must be stopped before sending
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        # Timestamp must be tracked for echo-back prevention
        assert 1234567890 in adapter._recent_sent_timestamps


    @pytest.mark.asyncio
    async def test_send_image_file_too_large(self, monkeypatch, tmp_path):
        """send_image_file should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "huge.png"
        img_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024  # 200 MB
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "too large" in result.error.lower()


class TestSignalRecipientResolution:

    @pytest.mark.asyncio
    async def test_send_looks_up_uuid_via_list_contacts(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return [{
                    "recipient": "351935789098",
                    "number": "+15551230000",
                    "uuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                    "isRegistered": True,
                }]
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "listContacts"
        assert captured[1]["method"] == "send"
        assert captured[1]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]


# ---------------------------------------------------------------------------
# send_voice method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVoice:
    @pytest.mark.asyncio
    async def test_send_voice_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_voice should send audio as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(audio_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps


    @pytest.mark.asyncio
    async def test_send_voice_too_large(self, monkeypatch, tmp_path):
        """send_voice should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "huge.ogg"
        audio_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "too large" in result.error.lower()


# ---------------------------------------------------------------------------
# send_video method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_video should send video as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"\x00" * 100)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(vid_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps


# ---------------------------------------------------------------------------
# MEDIA: tag extraction integration
# ---------------------------------------------------------------------------

class TestSignalMediaExtraction:
    """Verify the full pipeline: MEDIA: tag → extract → send_image_file/send_voice."""

    def test_extract_media_finds_image_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: image paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "Here's the chart.\nMEDIA:/tmp/price_graph.png"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/price_graph.png"
        assert "MEDIA:" not in cleaned


# ---------------------------------------------------------------------------
# Inbound attachment message type classification
# ---------------------------------------------------------------------------

def _make_dm_envelope(sender: str, attachments: list, text: str = "") -> dict:
    """Build a minimal signal-cli DM envelope with the given attachments."""
    return {
        "envelope": {
            "sourceNumber": sender,
            "sourceName": "Test User",
            "sourceUuid": "aaaaaaaa-0000-0000-0000-000000000001",
            "timestamp": 1700000000000,
            "dataMessage": {
                "timestamp": 1700000000000,
                "message": text,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "attachments": attachments,
            },
        }
    }


class TestSignalInboundMessageTypeClassification:
    """_handle_envelope must set MessageType.DOCUMENT for application/* and text/* attachments.

    Before the fix, PDFs and other documents left msg_type as MessageType.TEXT,
    so run.py's document-context injection (which gates on MessageType.DOCUMENT)
    silently dropped the file and the agent never saw it.
    """

    async def _dispatch_single_attachment(self, monkeypatch, content_type: str,
                                          att_id: str, fetch_path: str, fetch_ext: str):
        """Helper: run _handle_envelope with one attachment and return the dispatched event."""
        envelope = _make_dm_envelope(
            sender="+15559876543",
            attachments=[{
                "contentType": content_type,
                "id": att_id,
                "size": 1024,
                "filename": None,
                "width": None,
                "height": None,
                "caption": None,
                "uploadTimestamp": 1700000000000,
            }],
        )
        adapter = _make_signal_adapter(monkeypatch)
        adapter._rpc, _ = _stub_rpc(None)
        dispatched = []

        async def _fake_handle_message(event):
            dispatched.append(event)

        adapter.handle_message = _fake_handle_message
        adapter._fetch_attachment = AsyncMock(return_value=(fetch_path, fetch_ext))
        await adapter._handle_envelope(envelope)
        assert dispatched, "_handle_envelope did not dispatch any event"
        return dispatched[0]

    @pytest.mark.asyncio
    async def test_pdf_attachment_sets_document_type(self, monkeypatch):
        """A PDF attachment (application/pdf) must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="application/pdf",
            att_id="6zLO3b-6Yf3zVWeLDctA.pdf",
            fetch_path="/tmp/report.pdf",
            fetch_ext=".pdf",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "PDFs must be classified as DOCUMENT so run.py injects file context."
        )
        assert "/tmp/report.pdf" in event.media_urls

    @pytest.mark.asyncio
    async def test_text_plain_attachment_sets_document_type(self, monkeypatch):
        """A text/plain attachment must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="text/plain",
            att_id="notes.txt",
            fetch_path="/tmp/notes.txt",
            fetch_ext=".txt",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "text/plain must be classified as DOCUMENT so run.py injects file context."
        )


# ---------------------------------------------------------------------------
# send_document now routes through _send_attachment (#5105 bonus)
# ---------------------------------------------------------------------------

class TestSignalSendDocumentViaHelper:
    """Verify send_document gained size check and path-in-error via _send_attachment."""

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, monkeypatch, tmp_path):
        """send_document should now reject files over 100MB (was previously missing)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        doc_path = tmp_path / "huge.pdf"
        doc_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_document(chat_id="+155****4567", file_path=str(doc_path))

        assert result.success is False
        assert "too large" in result.error.lower()


# ---------------------------------------------------------------------------
# Signal streaming edit capability / message_id behavior
# ---------------------------------------------------------------------------

class TestSignalStreamingCapabilities:
    """Signal must opt out of edit-based streaming behavior."""

    def test_signal_declares_no_message_editing(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestSignalSendReturnsMessageId:
    """Signal send() should not pretend sent messages are editable."""

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_even_with_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({"timestamp": 1712345678000})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None


class TestSignalSendResultValidation:
    """Verify that send() validates recipient-level delivery results."""


    @pytest.mark.asyncio
    async def test_send_failure_when_results_has_success_false(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "success": False,
                    "failure": "Some connection error"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is False
        assert result.error == "Some connection error"


# ---------------------------------------------------------------------------
# stop_typing() delegates to _stop_typing_indicator (#4647)
# ---------------------------------------------------------------------------

class TestSignalStopTyping:
    """Signal must expose a public stop_typing() so base adapter's
    _keep_typing finally block can clean up platform-level typing tasks."""

    @pytest.mark.asyncio
    async def test_stop_typing_calls_private_method(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.stop_typing("+155****4567")

        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")


# ---------------------------------------------------------------------------
# Typing-indicator backoff on repeated failures (Signal RPC spam fix)
# ---------------------------------------------------------------------------

class TestSignalTypingBackoff:
    """When base.py's _keep_typing refresh loop calls send_typing every ~2s
    and the recipient is unreachable (NETWORK_FAILURE), the adapter must:

    - log WARNING only for the first failure (subsequent failures use DEBUG
      via log_failures=False on the _rpc call)
    - after 3 consecutive failures, skip the RPC entirely during an
      exponential cooldown window instead of hammering signal-cli every 2s
    - reset counters on a successful sendTyping
    - reset counters when _stop_typing_indicator() is called for the chat
    """

    @pytest.mark.asyncio
    async def test_first_failure_logs_at_warning_subsequent_at_debug(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        calls = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            calls.append({"log_failures": log_failures})
            return None  # simulate NETWORK_FAILURE

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")

        assert len(calls) == 2
        assert calls[0]["log_failures"] is True   # first failure — warn
        assert calls[1]["log_failures"] is False  # subsequent — debug

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_trigger_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        call_count = {"n": 0}

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_count["n"] += 1
            return None

        adapter._rpc = _fake_rpc

        # Three failures engage the cooldown.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3
        assert "+155****4567" in adapter._typing_skip_until

        # Fourth, fifth, ... calls during the cooldown window are short-
        # circuited — the RPC is not issued at all.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# _stop_typing_indicator sends explicit sendTyping(stop=True) RPC
# ---------------------------------------------------------------------------

class TestSignalStopTypingExplicitRPC:
    """Cancelling the typing indicator must issue an explicit
    sendTyping(stop=True) RPC so the recipient's device drops the indicator
    immediately, instead of waiting for Signal's built-in ~5s timeout.

    The stop RPC is best-effort: any failure must not prevent the per-chat
    backoff state from being cleared.
    """


    @pytest.mark.asyncio
    async def test_stop_typing_indicator_best_effort_on_rpc_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._resolve_recipient = AsyncMock(return_value="uuid-recipient")

        # Drive the chat into backoff so we can confirm cleanup still happens
        # even when the stop RPC itself fails.
        async def _noop(method, params, rpc_id=None, **kwargs):
            return None

        adapter._rpc = _noop
        for _ in range(3):
            await adapter.send_typing("+155****0000")

        assert adapter._typing_failures.get("+155****0000") == 3
        assert "+155****0000" in adapter._typing_skip_until

        # Now make the stop RPC raise — backoff state must still be cleared.
        async def failing_rpc(method, params, rpc_id=None, **kwargs):
            raise RuntimeError("signal-cli unreachable")

        adapter._rpc = failing_rpc

        await adapter._stop_typing_indicator("+155****0000")

        assert "+155****0000" not in adapter._typing_failures
        assert "+155****0000" not in adapter._typing_skip_until


# ---------------------------------------------------------------------------
# Reply quote extraction
# ---------------------------------------------------------------------------

class TestSignalQuoteExtraction:
    """Verify Signal reply quote fields are propagated to MessageEvent."""

    @pytest.mark.asyncio
    async def test_handle_envelope_sets_reply_context_from_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "yes I agree",
                    "quote": {
                        "id": 99,
                        "text": "want to grab lunch?",
                        "author": "other-author",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.text == "yes I agree"
        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "want to grab lunch?"
        assert event.reply_to_author_id == "other-author"
        assert event.reply_to_is_own_message is False


    @pytest.mark.asyncio
    async def test_track_sent_timestamp_keeps_reply_detection_cache_after_echo_discard(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._track_sent_timestamp({"timestamp": 111222333})
        # Echo suppression consumes the entry from the recent-sent ring; the
        # separate reply-detection cache must still retain it.
        adapter._consume_sent_timestamp(111222333)

        assert "111222333" in adapter._sent_message_timestamps
        assert adapter._quote_references_own_message("111222333", None) is True


# ---------------------------------------------------------------------------
# _rpc rate-limit detection
# ---------------------------------------------------------------------------

class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response — only what _rpc touches."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def _install_fake_client(adapter, json_data):
    """Replace adapter.client.post with an async fn returning json_data."""
    from types import SimpleNamespace

    async def _post(url, json=None, timeout=None):
        return _FakeHttpResponse(json_data)

    adapter.client = SimpleNamespace(post=_post)


class TestSignalRpcRateLimit:
    """_rpc opt-in 429 detection and SignalRateLimitError propagation."""


    @pytest.mark.asyncio
    async def test_default_swallows_rate_limit_returns_none(self, monkeypatch):
        """Without opt-in, 429 stays swallowed — preserves backwards compat."""
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "[429] Rate Limited"},
        })

        result = await adapter._rpc("send", {})
        assert result is None


    @pytest.mark.asyncio
    async def test_raises_with_retry_after_from_v0_14_3_payload(self, monkeypatch):
        """signal-cli ≥ v0.14.3 surfaces server Retry-After under
        ``error.data.response.results[*].retryAfterSeconds`` — _rpc
        carries that value through SignalRateLimitError.retry_after."""
        from gateway.platforms.signal_rate_limit import (
            SignalRateLimitError, SIGNAL_RPC_ERROR_RATELIMIT,
        )

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "Failed to send message due to rate limiting",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [
                            {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 90},
                        ],
                    }
                },
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 90.0


# ---------------------------------------------------------------------------
# send_multiple_images — chunking, pacing, rate-limit retry
# ---------------------------------------------------------------------------


def _make_image_files(tmp_path, count, prefix="img"):
    """Materialize `count` tiny PNG files and return file:// URIs for them."""
    uris = []
    for i in range(count):
        p = tmp_path / f"{prefix}_{i}.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 32)
        uris.append((f"file://{p}", ""))
    return uris


def _stub_rpc_responses(responses):
    """Build an _rpc replacement that pops a response per call.

    Each entry in `responses` is either:
      * a return value (dict / None) → returned to the caller, or
      * an Exception subclass instance → raised.
    Captures (params, kwargs) per call for inspection.
    """
    captured = []
    queue = list(responses)

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params), "kwargs": kwargs})
        await asyncio.sleep(0)
        if not queue:
            raise AssertionError("Unexpected extra _rpc call")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return mock_rpc, captured


def _patch_scheduler_sleep(monkeypatch, capture: list):
    """Capture sleeps inside the scheduler so tests don't actually wait.
    Zero-second sleeps (e.g. event-loop yields from mock RPCs) are
    delegated to the real asyncio.sleep so they don't pollute the
    capture list."""
    _real_sleep = asyncio.sleep
    offset = [0.0]

    async def fake_sleep(seconds):
        if seconds > 0:
            capture.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset[0]
    )


class TestSignalSendMultipleImages:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(chat_id="+155****4567", images=[])

        assert captured == []
        adapter._stop_typing_indicator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_bad_files_no_rpc(self, monkeypatch, tmp_path):
        """If every image is missing/invalid, no RPC fires."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=[(f"file://{tmp_path}/missing_a.png", ""),
                    (f"file://{tmp_path}/missing_b.png", "")],
        )

        assert captured == []

    @pytest.mark.asyncio
    async def test_single_batch_under_limit(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        images = _make_image_files(tmp_path, 5)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        params = captured[0]["params"]
        assert params["recipient"] == ["+155****4567"]
        assert params["message"] == ""
        assert len(params["attachments"]) == 5
        # raise_on_rate_limit must be opted into so the retry loop sees 429s
        assert captured[0]["kwargs"].get("raise_on_rate_limit") is True


    @pytest.mark.asyncio
    async def test_429_without_retry_after_uses_default_rate(
        self, monkeypatch, tmp_path
    ):
        """signal-cli < v0.14.3 doesn't surface Retry-After. The
        scheduler keeps its default refill rate (1 token / 4s), so a
        retry of n=3 waits 12s."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
            SignalRateLimitError,
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("[429] Rate Limited", retry_after=None),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=_make_image_files(tmp_path, 3),
        )

        assert len(captured) == 2
        assert sleep_calls == [
            pytest.approx(3 * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER, abs=1.0)
        ]


class TestSignalRateLimitDetection:
    """Coverage for the typed-code + substring detection helpers."""


    def test_extract_retry_after_from_results(self):
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -5,
            "message": "Failed to send message due to rate limiting",
            "data": {
                "response": {
                    "timestamp": 0,
                    "results": [
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 30},
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 45},
                    ],
                }
            },
        }
        assert _extract_retry_after_seconds(err) == 45.0


    def test_detect_retry_later_exception_substring(self):
        """libsignal-net's RetryLaterException leaks through as
        AttachmentInvalidException → UnexpectedErrorException when the
        rate-limit fires inside attachment upload. Detect it by substring."""
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _is_signal_rate_limit_error(err) is True


class TestSignalSendTimeout:
    """Timeout scaling for batched attachment sends."""


    def test_scales_with_batch_size(self):
        from gateway.platforms.signal import _signal_send_timeout
        # 32 attachments × 5s = 160s; ought to comfortably outlast a
        # serial upload of an attachment-heavy batch.
        assert _signal_send_timeout(32) == 160.0


# ---------------------------------------------------------------------------
# Contentless Envelope Filtering (profile key updates, empty messages)
# ---------------------------------------------------------------------------

class TestSignalContentlessEnvelope:
    """Verify that profile key updates and empty Signal messages are skipped."""

    @pytest.mark.asyncio
    async def test_skips_profile_key_update_no_message_field(self, monkeypatch):
        """Profile key updates may carry a dataMessage without 'message' field.
        Must be skipped to avoid triggering agent turns for metadata."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Profile key update: dataMessage exists but has no "message" field
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    # No "message" field — profile key update metadata only
                    "profileKey": "some-profile-key-data",
                },
            }
        })

        assert "event" not in captured, "Profile key update should be skipped"


    @pytest.mark.asyncio
    async def test_allows_message_with_attachment_no_text(self, monkeypatch):
        """Messages with attachments but no text should still be processed."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Mock attachment fetch to return a cached image
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()
        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/img.png"):
            await adapter._handle_envelope({
                "envelope": {
                    "sourceNumber": "+155****9999",
                    "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                    "sourceName": "Elliott McManis",
                    "timestamp": 1777600696077,
                    "dataMessage": {
                        "message": "",  # No text
                        "attachments": [{"id": "att-123", "size": 200}],
                    },
                }
            })

        assert "event" in captured, "Message with attachment should NOT be skipped"
        assert captured["event"].media_urls == ["/tmp/img.png"]


class TestSignalSyncMessageHandling:
    """signal-cli running as a linked secondary device receives the user's
    own messages as ``syncMessage.sentMessage`` envelopes. Two cases must
    be handled:

      1. Note to Self (destination == self): promote to dataMessage so the
         user can talk to the agent in their own self-chat.
      2. Group sync-sent (destination is None, groupInfo set): promote so
         single-user / personal groups work.

    In both cases, the bot's own outbound replies bounce back as
    sync-sents and must be suppressed via the recently-sent timestamp ring.
    """

    @pytest.mark.asyncio
    async def test_note_to_self_echo_of_own_reply_is_suppressed(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        # Simulate that the bot just sent a reply with timestamp 3000000000
        adapter._track_sent_timestamp({"timestamp": 3000000000})
        called = []

        async def fake_handle(event):
            called.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 3000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****4567",
                        "destination": "+155****4567",
                        "timestamp": 3000000000,
                        "message": "this is the bot's own reply echo",
                    }
                },
            }
        })

        assert called == [], "Echo of bot's own reply must be suppressed"
        # Consumed: timestamp must be removed from the ring
        assert 3000000000 not in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_group_sync_sent_promoted_to_inbound(self, monkeypatch):
        """User sends a message in a group from their primary phone; the
        linked device receives it as a sync-sent with destination=None and
        a groupInfo block. It must be treated as inbound so the agent can
        respond in groups when the user is the only human participant."""
        adapter = _make_signal_adapter(
            monkeypatch, account="+155****4567", group_allowed="abc123=="
        )
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 4000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": None,
                        "destination": None,
                        "timestamp": 4000000000,
                        "message": "ping the group",
                        "groupInfo": {
                            "groupId": "abc123==",
                            "type": "DELIVER",
                        },
                    }
                },
            }
        })

        assert "event" in captured, "Group sync-sent must reach handle_message"
        assert captured["event"].text == "ping the group"
        assert captured["event"].source.chat_id == "group:abc123=="


class TestRecentSentTimestampRing:
    """Verify the LRU+TTL behaviour of the echo-suppression ring."""


    def test_ttl_evicts_stale_entries(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._recent_sent_ttl_seconds = 100.0

        # Drive time.monotonic deterministically.
        import gateway.platforms.signal as sig_mod
        fake_now = [1000.0]
        monkeypatch.setattr(sig_mod.time, "monotonic", lambda: fake_now[0])

        adapter._track_sent_timestamp({"timestamp": 1})
        fake_now[0] = 1050.0
        adapter._track_sent_timestamp({"timestamp": 2})
        fake_now[0] = 1200.0  # 200s elapsed since ts=1 (>TTL), 150s since ts=2 (>TTL)
        adapter._track_sent_timestamp({"timestamp": 3})
        # Both 1 and 2 should be evicted on TTL, only 3 remains
        assert list(adapter._recent_sent_timestamps.keys()) == [3]
