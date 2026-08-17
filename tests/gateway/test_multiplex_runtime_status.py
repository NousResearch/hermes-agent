"""Regression coverage for multiplex profile runtime-status ownership (#88047)."""

from unittest.mock import MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _StatusAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        raise NotImplementedError

    async def get_chat_info(self, chat_id: str) -> dict:
        return {}


def _profile_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = object()
    runner._busy_text_mode = "queue"
    runner._busy_text_modes_by_profile = {}
    runner._recover_telegram_topic_thread_id = MagicMock()
    runner._handle_reaction_event = MagicMock()
    runner._make_profile_message_handler = MagicMock(return_value=MagicMock())
    runner._make_profile_fatal_error_handler = MagicMock(return_value=MagicMock())
    runner._make_profile_busy_session_handler = MagicMock(return_value=MagicMock())
    runner._make_adapter_auth_check = MagicMock(return_value=MagicMock())
    runner._make_profile_platform_event_handler = MagicMock(return_value=MagicMock())
    return runner


def test_secondary_profile_fatal_does_not_overwrite_primary_status(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    primary = _StatusAdapter()
    primary._mark_connected()

    secondary = _StatusAdapter()
    _profile_runner()._configure_profile_adapter(
        secondary, "secondary", Platform.WHATSAPP
    )
    secondary._set_fatal_error(
        "whatsapp_not_paired",
        "secondary profile has no session",
        retryable=False,
    )

    runtime = read_runtime_status()
    assert runtime is not None
    assert runtime["platforms"]["whatsapp"]["state"] == "connected"


def test_single_profile_adapter_still_publishes_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _StatusAdapter()

    adapter._mark_connected()
    connected = read_runtime_status()
    assert connected is not None
    assert connected["platforms"]["whatsapp"]["state"] == "connected"

    adapter._set_fatal_error("auth_failed", "bad token", retryable=False)
    fatal = read_runtime_status()
    assert fatal is not None
    assert fatal["platforms"]["whatsapp"]["state"] == "fatal"
    assert fatal["platforms"]["whatsapp"]["error_code"] == "auth_failed"
