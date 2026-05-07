import pytest
from types import SimpleNamespace

from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner


class _Adapter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append((chat_id, content, metadata))
        return self.result


@pytest.mark.asyncio
async def test_kanban_notification_sendresult_failure_raises(monkeypatch):
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=False, error="Not connected"))
    sub = {"task_id": "t_fail", "platform": "telegram", "chat_id": "123"}
    mirror_calls = []
    monkeypatch.setattr(
        "gateway.mirror.mirror_to_session",
        lambda *args, **kwargs: mirror_calls.append((args, kwargs)) or True,
    )

    with pytest.raises(RuntimeError, match="Not connected"):
        await runner._send_kanban_notification(adapter, sub, "done", {})

    assert adapter.calls == [("123", "done", {})]
    assert mirror_calls == []


@pytest.mark.asyncio
async def test_kanban_notification_sendresult_success_does_not_raise():
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=True, message_id="m1"))
    sub = {"task_id": "t_ok", "platform": "telegram", "chat_id": "123"}

    await runner._send_kanban_notification(adapter, sub, "done", {"thread_id": "7"})

    assert adapter.calls == [("123", "done", {"thread_id": "7"})]


@pytest.mark.asyncio
async def test_kanban_notification_synthesize_mode_sends_synthesized_text(monkeypatch):
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=True, message_id="m1"))
    sub = {
        "task_id": "t_syn",
        "platform": "telegram",
        "chat_id": "123",
        "notification_mode": "synthesize",
        "origin_context": "user asked for a concise result",
    }
    ev = SimpleNamespace(kind="completed", payload={"summary": "worker handoff"}, run_id=7)

    async def fake_synthesize(**kwargs):
        assert kwargs["sub"] is sub
        assert kwargs["event"] is ev
        assert kwargs["direct_message"] == "direct fallback"
        return "synthesized reply"

    monkeypatch.setattr(runner, "_synthesize_kanban_notification", fake_synthesize)

    await runner._send_kanban_notification(
        adapter, sub, "direct fallback", {}, event=ev, task=None, board="default"
    )

    assert adapter.calls == [("123", "synthesized reply", {})]


@pytest.mark.asyncio
async def test_kanban_notification_synthesis_failure_falls_back_direct(monkeypatch):
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=True, message_id="m1"))
    sub = {
        "task_id": "t_syn_fail",
        "platform": "telegram",
        "chat_id": "123",
        "notification_mode": "synthesize",
    }
    ev = SimpleNamespace(kind="completed", payload={"summary": "worker handoff"}, run_id=7)

    async def fake_synthesize(**kwargs):
        raise RuntimeError("synthesis unavailable")

    monkeypatch.setattr(runner, "_synthesize_kanban_notification", fake_synthesize)

    await runner._send_kanban_notification(
        adapter, sub, "direct fallback", {"thread_id": "9"}, event=ev, task=None
    )

    assert adapter.calls == [("123", "direct fallback", {"thread_id": "9"})]


@pytest.mark.asyncio
async def test_kanban_notification_silent_mode_noops():
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=True, message_id="m1"))
    sub = {
        "task_id": "t_silent",
        "platform": "telegram",
        "chat_id": "123",
        "notification_mode": "silent",
    }

    await runner._send_kanban_notification(adapter, sub, "direct fallback", {})

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_kanban_notification_success_mirrors_into_origin_session(monkeypatch):
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter(SendResult(success=True, message_id="m1"))
    sub = {
        "task_id": "t_ctx",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "7",
        "user_id": "u1",
    }
    mirror_calls = []

    def fake_mirror(platform, chat_id, message_text, source_label="cli", thread_id=None, user_id=None):
        mirror_calls.append({
            "platform": platform,
            "chat_id": chat_id,
            "message_text": message_text,
            "source_label": source_label,
            "thread_id": thread_id,
            "user_id": user_id,
        })
        return True

    monkeypatch.setattr("gateway.mirror.mirror_to_session", fake_mirror)

    await runner._send_kanban_notification(
        adapter, sub, "done text", {"thread_id": "7"}
    )

    assert adapter.calls == [("123", "done text", {"thread_id": "7"})]
    assert mirror_calls == [{
        "platform": "telegram",
        "chat_id": "123",
        "message_text": "done text",
        "source_label": "kanban",
        "thread_id": "7",
        "user_id": "u1",
    }]


def test_kanban_notifier_defaults_to_dispatch_owner(monkeypatch):
    """Secondary gateways with dispatch disabled must not consume shared subs."""
    from hermes_cli import config as hermes_config

    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", raising=False)
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )

    assert runner._kanban_notify_in_gateway_enabled() is False


def test_kanban_notifier_explicit_config_overrides_dispatch(monkeypatch):
    from hermes_cli import config as hermes_config

    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", raising=False)
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False, "notify_in_gateway": True}},
    )

    assert runner._kanban_notify_in_gateway_enabled() is True


def test_kanban_notifier_env_override(monkeypatch):
    from hermes_cli import config as hermes_config

    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )

    monkeypatch.setenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", "false")
    assert runner._kanban_notify_in_gateway_enabled() is False

    monkeypatch.setenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", "true")
    assert runner._kanban_notify_in_gateway_enabled() is True


def test_kanban_synthesis_prompt_hides_internal_workflow_by_default():
    task = SimpleNamespace(
        title="answer factual question",
        body="User asked: how many airports does Guangzhou have?",
    )
    event = SimpleNamespace(payload={"summary": "Guangzhou has one operating passenger airport."})
    sub = {
        "task_id": "t_internal123",
        "origin_context": "how many airports does Guangzhou have?",
        "origin_session_id": "telegram:chat",
        "origin_profile": "default",
    }

    prompt = GatewayRunner._build_kanban_synthesis_prompt(
        sub=sub,
        event=event,
        task=task,
        board="default",
        worker_summary="Guangzhou has one operating passenger airport.",
        worker_metadata={"assignee": "worker-research"},
        direct_message="✔ @worker-research Kanban t_internal123 done — answer factual question",
    )

    assert "Do NOT mention Kanban" in prompt
    assert "task ids" in prompt
    assert "assignees" in prompt
    assert "worker/dispatcher" in prompt
    assert "Internal debug context" in prompt
    assert "t_internal123" in prompt  # available for debug context, not for normal prose