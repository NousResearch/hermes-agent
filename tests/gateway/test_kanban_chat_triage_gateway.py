import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb
from tests.gateway.restart_test_helpers import RestartTestAdapter


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants

        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


@pytest.fixture
def inline_to_thread(monkeypatch):
    async def run_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("gateway.platforms.base.asyncio.to_thread", run_inline)


def _source(message_id: str | None = "m1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="u1",
        user_name="Sean",
        thread_id="17585",
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_configured_gateway_routes_triggered_message_to_kanban_without_agent(
    fresh_home,
    inline_to_thread,
):
    kb.create_board("inbox")
    adapter = RestartTestAdapter()
    adapter.config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "kanban_chat_triage": {
                "enabled": True,
                "trigger_prefixes": ["todo:"],
                "fallback_board": "inbox",
                "create_missing_boards": False,
            }
        },
    )
    called = False
    sent: list[str] = []

    async def handler(_event):
        nonlocal called
        called = True
        return "agent response"

    async def send_ack(*, chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="ack1")

    adapter.set_message_handler(handler)
    adapter._send_with_retry = send_ack
    event = MessageEvent(
        text="todo: fix gateway triage handoff",
        source=_source(),
        message_id="m1",
    )

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    assert called is False
    assert sent and "Queued for triage: board=inbox task=" in sent[0]
    assert adapter._active_sessions == {}
    with kb.connect(board="inbox") as conn:
        rows = conn.execute("SELECT title, status FROM tasks").fetchall()
    assert [(row["title"], row["status"]) for row in rows] == [
        ("fix gateway triage handoff", "triage")
    ]


@pytest.mark.asyncio
async def test_gateway_kanban_chat_triage_is_inert_without_matching_config(fresh_home):
    kb.create_board("inbox")
    adapter = RestartTestAdapter()
    adapter.config = PlatformConfig(
        enabled=True,
        token="***",
        extra={"kanban_chat_triage": {"enabled": True, "trigger_prefixes": ["todo:"]}},
    )
    routed = await adapter._maybe_route_kanban_chat_triage(
        MessageEvent(text="ordinary chat", source=_source(), message_id="m1")
    )

    assert routed is False


@pytest.mark.asyncio
async def test_gateway_kanban_chat_triage_redacts_failure_reply(
    fresh_home, inline_to_thread, caplog
):
    secret = "sk-abc1234567890supersecret"
    adapter = RestartTestAdapter()
    adapter.config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "kanban_chat_triage": {
                "enabled": True,
                "trigger_prefixes": ["todo:"],
                "fallback_board": f"inbox-{secret}",
                "create_missing_boards": False,
            }
        },
    )
    sent: list[str] = []

    async def send_ack(*, chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="ack1")

    adapter._send_with_retry = send_ack
    adapter.set_message_handler(lambda _event: "agent response")

    await adapter.handle_message(
        MessageEvent(text="todo: route this", source=_source(message_id=None))
    )

    assert sent
    assert sent[0].startswith("Kanban triage failed:")
    assert secret not in sent[0]
    assert secret not in caplog.text
