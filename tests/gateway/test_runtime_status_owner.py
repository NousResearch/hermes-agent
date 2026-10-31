"""``gateway_state.json`` holds the status of the gateway runner's own adapters only.

Senders that connect a throwaway platform adapter for one message (Matrix fallback, BlueBubbles, the
Feishu/WeCom standalone senders used by out-of-process cron delivery) are not the gateway. Their
adapters' connect/disconnect status must not re-stamp the running gateway's record with the
sender's pid/argv, reset ``gateway_state``, drop the other platforms' entries or mark the live
platform ``disconnected``.
"""

import json

import pytest

from gateway import status
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


_GATEWAY_RECORD = {
    "pid": 4242, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
    "start_time": 1234, "gateway_state": "running", "active_agents": 0,
    "platforms": {"telegram": {"state": "connected"}},
}


class _Adapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, *_args, **_kwargs):
        return None

    async def get_chat_info(self, *_args, **_kwargs):
        return {}


def _adapter() -> BasePlatformAdapter:
    return _Adapter(PlatformConfig(enabled=True), Platform.MATRIX)


@pytest.fixture
def gateway_state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "gateway_state.json"
    path.write_text(json.dumps(_GATEWAY_RECORD), encoding="utf-8")
    return path


def test_throwaway_sender_adapter_leaves_gateway_status_alone(gateway_state_file):
    adapter = _adapter()

    adapter._mark_connected()
    adapter._mark_disconnected()
    adapter._set_fatal_error("matrix_auth", "bad token", retryable=False)

    assert status.flush_runtime_status(timeout=2.0)
    assert json.loads(gateway_state_file.read_text(encoding="utf-8")) == _GATEWAY_RECORD


def test_adapter_wired_by_the_runner_publishes(gateway_state_file):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = None
    runner._busy_text_mode = "interrupt"
    adapter = _adapter()

    runner._wire_adapter_handlers(adapter)
    adapter._mark_connected()

    assert status.flush_runtime_status(timeout=2.0)
    assert status.read_runtime_status()["platforms"]["matrix"]["state"] == "connected"


def test_adapter_built_without_init_does_not_publish(gateway_state_file):
    """The ownership default lives on the class, so an adapter that skipped ``__init__`` (tests,
    plugin shims) is a non-owner rather than an AttributeError."""
    adapter = object.__new__(_Adapter)
    assert adapter._runtime_status_owned is False
