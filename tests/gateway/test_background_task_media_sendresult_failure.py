"""Background-task (``/bg``) ``MEDIA:`` delivery must honour the ``SendResult`` contract.

``GatewayRunner._run_background_task_inner`` sends the task's text, then routes
each ``MEDIA:`` file to ``send_voice`` / ``send_video`` / ``send_image_file`` /
``send_document`` under ``suppress(Exception)``. A ``SendResult(success=False)``
returned without raising was treated exactly like a delivered file: no log, no
user notice. Same defect shape as the post-stream lane
(``tests/gateway/test_post_stream_media_sendresult_failure.py``); the
non-streaming lane in ``gateway/platforms/base.py`` already handles it.

#91335 (open) adds ``_notify_media_delivery_failure`` to the ``except`` branch
of this loop; it explicitly leaves the ``success=False`` shape out. No network,
no tokens: the agent and the adapter are mocks.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

SIMULATED_ERROR = "simulated document delivery failure"


def _make_runner():
    """Bare GatewayRunner, same shape as ``tests/gateway/test_background_command.py``."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    mock_store = MagicMock()
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store
    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()
    return runner


def _allowed_media(tmp_path, monkeypatch, name: str):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


def _mock_adapter(document_result: SendResult):
    adapter = AsyncMock()
    adapter.name = "fake"
    adapter.platform = Platform.TELEGRAM
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
    adapter.extract_media = MagicMock(side_effect=BasePlatformAdapter.extract_media)
    adapter.extract_images = MagicMock(side_effect=BasePlatformAdapter.extract_images)
    adapter.send_document = AsyncMock(return_value=document_result)
    adapter._notify_media_delivery_failure = AsyncMock(return_value=None)
    return adapter


async def _run_bg(runner, source, response: str):
    mock_result = {"final_response": response, "messages": []}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent") as MockAgent:
        agent = MagicMock()
        agent.shutdown_memory_provider = MagicMock()
        agent.close = MagicMock()
        agent.run_conversation.return_value = mock_result
        MockAgent.return_value = agent
        await runner._run_background_task("make the report", source, "bg_test")


@pytest.mark.asyncio
async def test_background_task_send_document_success_false_runs_failure_path(tmp_path, monkeypatch, caplog):
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    runner = _make_runner()
    adapter = _mock_adapter(SendResult(success=False, error=SIMULATED_ERROR))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _run_bg(runner, source, f"Report ready.\nMEDIA:{pdf}")

    # text went out (the "Background task complete" header), the upload was attempted ...
    adapter.send.assert_awaited_once()
    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(pdf)
    # ... and success=False is NOT a confirmed delivery: the failure path runs
    adapter._notify_media_delivery_failure.assert_awaited_once()
    args, kwargs = adapter._notify_media_delivery_failure.await_args
    assert args[0] == "67890"
    assert args[1] == str(pdf)
    assert kwargs.get("is_voice") is False
    assert any(SIMULATED_ERROR in rec.getMessage() for rec in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_background_task_send_document_success_true_does_not_notify(tmp_path, monkeypatch):
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    runner = _make_runner()
    adapter = _mock_adapter(SendResult(success=True, message_id="doc"))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")

    await _run_bg(runner, source, f"Report ready.\nMEDIA:{pdf}")

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_task_legacy_none_result_is_tolerated(tmp_path, monkeypatch):
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    runner = _make_runner()
    adapter = _mock_adapter(None)
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")

    await _run_bg(runner, source, f"Report ready.\nMEDIA:{pdf}")

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_task_exception_takes_suppress_branch_not_sendresult_handler(tmp_path, monkeypatch):
    """A raised exception is the EXCEPTION shape: the pre-existing ``suppress(Exception)``
    swallows it, the task does not fail, the next file is still attempted, and the
    ``success=False`` handler is never entered. Whether that branch should also notify is
    #91335's call; this pins only that the two branches are exclusive."""
    pdf = _allowed_media(tmp_path, monkeypatch, "report.pdf")
    csv = pdf.parent / "data.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    runner = _make_runner()
    adapter = _mock_adapter(SendResult(success=True, message_id="doc"))
    adapter.send_document = AsyncMock(
        side_effect=[RuntimeError("simulated transport crash"), SendResult(success=True, message_id="doc2")])
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")
    handler_spy = AsyncMock(return_value=None)
    monkeypatch.setattr("gateway.run_notifications._report_media_send_failure", handler_spy)

    await _run_bg(runner, source, f"Report ready.\nMEDIA:{pdf}\nMEDIA:{csv.resolve()}")

    # only the completion header went out: no "❌ Background task ... failed" message
    adapter.send.assert_awaited_once()
    assert "Background task complete" in adapter.send.await_args.kwargs["content"]
    # the exception did not escape and the next file was still attempted
    assert adapter.send_document.await_count == 2
    assert adapter.send_document.await_args_list[1].kwargs["file_path"] == str(csv.resolve())
    # the SendResult(success=False) handler was never entered for the exception
    handler_spy.assert_not_awaited()
