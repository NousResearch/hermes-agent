"""Local/staging E2E coverage for read-only /dobby gateway commands."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


AUTHORIZED_USER = "user-stage-9a"
AUTHORIZED_CHANNEL = "channel-stage-9a"
OPENAI_SECRET = "sk-" + ("A" * 32)
DISCORD_TOKEN = "M" + ("D" * 23) + "." + ("E" * 6) + "." + ("F" * 32)
WEBHOOK_SECRET = "whsec_" + ("W" * 32)


class _FakeGatewayAdapter(BasePlatformAdapter):
    """Concrete local adapter that records sends without connecting anywhere."""

    def __init__(self) -> None:
        super().__init__(
            PlatformConfig(
                enabled=True,
                token=DISCORD_TOKEN,
                extra={"group_sessions_per_user": True},
            ),
            Platform.DISCORD,
        )
        self.sent: list[dict[str, object]] = []
        self._allowed_user_ids = {AUTHORIZED_USER}
        self._allowed_role_ids = set()

    async def connect(self) -> bool:
        raise AssertionError("local E2E must not connect to Discord")

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict[str, str]:
        return {"id": chat_id, "type": "synthetic"}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"local-send-{len(self.sent)}")

    async def _keep_typing(self, chat_id: str, metadata: dict | None = None) -> None:
        await asyncio.sleep(3600)

    async def stop_typing(self, chat_id: str) -> None:
        return None


def _loaded_config_shape(hermes_home) -> dict:
    config_text = f"""
profile_name: staging-e2e
platforms:
  discord:
    enabled: true
    token: "{DISCORD_TOKEN}"
    api_key: "{OPENAI_SECRET}"
    extra:
      allowed_users:
        - "{AUTHORIZED_USER}"
      allowed_channels:
        - "{AUTHORIZED_CHANNEL}"
      operator_note: "{WEBHOOK_SECRET}"
  webhook:
    enabled: true
    extra:
      routes:
        dobby:
          secret: "{WEBHOOK_SECRET}"
          prompt: event
          require_signature: true
          signature_algorithm: hmac-sha256
          signature_header: X-Dobby-Signature
          timestamp_header: X-Dobby-Timestamp
          replay_window_seconds: 300
browser:
  enabled: false
memory:
  memory_enabled: false
  user_profile_enabled: false
  provider: ""
honcho: {{}}
external_memory_providers:
  enabled: false
  api_key: "{OPENAI_SECRET}"
"""
    config_path = hermes_home / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=AUTHORIZED_USER,
        user_name="Stage 9A User",
        chat_id=AUTHORIZED_CHANNEL,
        chat_name="stage-e2e",
        chat_type="dm",
    )


def _make_event(command: str) -> MessageEvent:
    return MessageEvent(
        text=command,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id=f"message-{command.rsplit(' ', 1)[-1]}",
    )


def _make_session_store_spy() -> MagicMock:
    session_store = MagicMock()
    for method_name in (
        "get_or_create_session",
        "append_to_transcript",
        "update_session",
    ):
        getattr(session_store, method_name).side_effect = AssertionError(
            f"/dobby local E2E reached session_store.{method_name}"
        )
    return session_store


def _wire_local_gateway(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", AUTHORIZED_USER)
    for env_name in (
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(env_name, raising=False)

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = object.__new__(gateway_run.GatewayRunner)
    adapter = _FakeGatewayAdapter()
    config = _loaded_config_shape(hermes_home)
    session_key = build_session_key(_make_source())
    stale_started_at = time.time() - 999999
    stale_agent = MagicMock()
    stale_agent.get_activity_summary.return_value = {
        "seconds_since_activity": 999999,
        "last_activity_desc": "synthetic stale fixture",
        "api_call_count": 0,
        "max_iterations": 0,
    }

    runner.config = config
    runner.adapters = {
        Platform.DISCORD: adapter,
        Platform.WEBHOOK: SimpleNamespace(
            _routes=config["platforms"]["webhook"]["extra"]["routes"],
            _global_secret="",
        ),
    }
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = _make_session_store_spy()
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    runner._running_agents = {session_key: stale_agent}
    runner._running_agents_ts = {session_key: stale_started_at}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._draining = False
    runner._session_key_for_source = lambda source: build_session_key(source)
    runner._release_running_agent_state = MagicMock(
        side_effect=AssertionError("/dobby local E2E released running-agent state")
    )
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("/dobby local E2E reached chat/session path")
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/dobby local E2E reached agent path")
    )
    runner._is_user_authorized = gateway_run.GatewayRunner._is_user_authorized.__get__(
        runner,
        gateway_run.GatewayRunner,
    )
    adapter.set_message_handler(runner._handle_message)
    return runner, adapter, session_key, stale_agent, stale_started_at


async def _drain_adapter(adapter: _FakeGatewayAdapter) -> None:
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_heading"),
    [
        ("/dobby status", "Dobby Package Status"),
        ("/dobby help", "Dobby command center"),
    ],
)
async def test_dobby_status_help_local_gateway_e2e_is_read_only(
    monkeypatch,
    tmp_path,
    command,
    expected_heading,
):
    runner, adapter, session_key, stale_agent, stale_started_at = _wire_local_gateway(
        monkeypatch,
        tmp_path,
    )

    await adapter.handle_message(_make_event(command))
    await _drain_adapter(adapter)

    assert len(adapter.sent) == 1
    sent = adapter.sent[0]
    assert sent["chat_id"] == AUTHORIZED_CHANNEL
    assert sent["reply_to"] == f"message-{command.rsplit(' ', 1)[-1]}"
    content = str(sent["content"])
    assert expected_heading in content
    if command.endswith("status"):
        assert "- Hermes home: present" in content
        assert "- Discord allowlist: present" in content
        assert "- Webhook strict policy: present" in content
    for secret in (OPENAI_SECRET, DISCORD_TOKEN, WEBHOOK_SECRET):
        assert secret not in content

    runner.hooks.emit.assert_not_called()
    runner._release_running_agent_state.assert_not_called()
    runner._handle_message_with_agent.assert_not_called()
    runner._run_agent.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()
    runner.session_store.append_to_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()
    stale_agent.interrupt.assert_not_called()
    assert runner._running_agents[session_key] is stale_agent
    assert runner._running_agents_ts[session_key] == stale_started_at
    assert runner._pending_messages == {}
