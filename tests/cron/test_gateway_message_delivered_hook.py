from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

import cron.scheduler as scheduler
from cron.scheduler import _deliver_result, _emit_gateway_message_delivered
from gateway.config import GatewayConfig, Platform, PlatformConfig


def _job(*, emit_hook=False):
    job = {
        "id": "daily-brief",
        "execution_id": "exec-123",
        "deliver": "origin",
        "origin": {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "42",
        },
    }
    if emit_hook:
        job["_gateway_message_delivered_hook"] = True
    return job


def _run_coro(coro, _loop):
    future = Future()
    future.set_result(asyncio.run(coro))
    return future


def test_delivery_hook_uses_only_the_documented_payload():
    job = _job()

    with (
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        _emit_gateway_message_delivered(
            job,
            platform="telegram",
            chat_id="-1001",
            thread_id="42",
            message_id="9001",
        )

    invoke.assert_called_once_with(
        "gateway_message_delivered",
        source="cron",
        execution_id="exec-123",
        job_id="daily-brief",
        platform="telegram",
        chat_id="-1001",
        thread_id="42",
        message_id="9001",
    )


def test_delivery_hook_does_nothing_without_a_subscriber():
    job = MagicMock()

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        _emit_gateway_message_delivered(
            job,
            platform="telegram",
            chat_id="-1001",
            thread_id="42",
            message_id="9001",
        )

    invoke.assert_not_called()
    job.get.assert_not_called()


def test_delivery_hook_normalizes_the_platform_name():
    with (
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        _emit_gateway_message_delivered(
            _job(),
            platform="Telegram",
            chat_id="-1001",
            thread_id=None,
            message_id="9001",
        )

    assert invoke.call_args.kwargs["platform"] == "telegram"


def test_live_telegram_delivery_uses_the_confirmed_message_id():
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(
        success=True, message_id="live-9002", raw_response=None
    )
    loop = MagicMock()
    loop.is_running.return_value = True

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch(
            "cron.scheduler.load_config",
            return_value={"cron": {"wrap_response": False}},
        ),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coro),
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        error = _deliver_result(
            _job(emit_hook=True),
            "Daily brief",
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
        )

    assert error is None
    invoke.assert_called_once()
    assert invoke.call_args.kwargs["message_id"] == "live-9002"


def test_live_telegram_delivery_uses_the_resolved_topic_id():
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    adapter = AsyncMock()
    adapter.ensure_dm_topic.return_value = "38049"
    adapter.send.return_value = MagicMock(
        success=True,
        message_id="live-9003",
        raw_response={"requested_thread_id": 38049, "thread_fallback": False},
    )
    loop = MagicMock()
    loop.is_running.return_value = True
    job = _job(emit_hook=True)
    job["origin"]["thread_id"] = "Feedback routine"

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch(
            "cron.scheduler.load_config",
            return_value={"cron": {"wrap_response": False}},
        ),
        patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coro),
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        error = _deliver_result(
            job, "Daily brief", adapters={Platform.TELEGRAM: adapter}, loop=loop
        )

    assert error is None
    assert invoke.call_args.kwargs["thread_id"] == "38049"


def test_standalone_delivery_does_not_emit_the_hook():
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )

    with (
        patch("gateway.config.load_gateway_config", return_value=config),
        patch(
            "cron.scheduler.load_config",
            return_value={"cron": {"wrap_response": False}},
        ),
        patch(
            "tools.send_message_tool._send_to_platform",
            new=AsyncMock(return_value={"success": True, "message_id": "9001"}),
        ),
        patch("hermes_cli.plugins.has_hook", return_value=True),
        patch("hermes_cli.plugins.invoke_hook") as invoke,
    ):
        error = _deliver_result(_job(emit_hook=True), "Daily brief")

    assert error is None
    invoke.assert_not_called()


def _delivery_flags(monkeypatch, result, job=None):
    calls = []
    monkeypatch.setattr(
        scheduler, "create_execution", lambda *_a, **_kw: {"id": "exec-flag"}
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(scheduler, "run_job", lambda *_a, **_kw: result)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_kw: "/tmp/out.txt")
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, _content, **_kwargs: (
            calls.append(bool(job.get("_gateway_message_delivered_hook"))) or None
        ),
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_a, **_kw: None)

    scheduler.run_one_job(job or {"id": "daily-brief", "deliver": "telegram"})
    return calls


def test_run_one_job_marks_only_successful_generated_text(monkeypatch, tmp_path):
    assert _delivery_flags(monkeypatch, (True, "raw", "Daily brief", None)) == [True]

    media = tmp_path / "report.pdf"
    media.write_bytes(b"%PDF-1.4 test")
    assert _delivery_flags(monkeypatch, (True, "raw", f"MEDIA:{media}", None)) == [
        False
    ]


def test_run_one_job_removes_persisted_delivery_hook_flag_on_failure(monkeypatch):
    assert _delivery_flags(
        monkeypatch,
        (False, "raw", "", "provider failed"),
        {
            "id": "daily-brief",
            "deliver": "telegram",
            "_gateway_message_delivered_hook": True,
        },
    ) == [False]
