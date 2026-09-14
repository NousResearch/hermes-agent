"""Background task answer delivery requires a positive persistence receipt."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.run import GatewayRunner


class _FakeAgent:
    result: Any = None

    def __init__(self, **_kwargs):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.base_url = ""
        self.session_id = "bg-test"

    def run_conversation(self, **_kwargs):
        return type(self).result


class _FakeAdapter:
    def __init__(self):
        self.send = AsyncMock()
        self.send_image = AsyncMock()
        self.send_voice = AsyncMock()
        self.send_video = AsyncMock()
        self.send_document = AsyncMock()
        self.send_image_file = AsyncMock()

    def extract_media(self, response):
        return [], response

    def extract_images(self, response):
        return [], response


@pytest.fixture
def runner(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _FakeAdapter()
    source = SimpleNamespace(
        platform="discord",
        chat_id="chat-1",
        thread_id=None,
        user_id=None,
        user_id_alt=None,
        user_name=None,
        chat_name=None,
        chat_type=None,
    )
    runner._adapter_for_source = Mock(return_value=adapter)
    runner._thread_metadata_for_source = Mock(return_value=None)
    runner._resolve_session_agent_runtime = Mock(return_value=("fake-model", {"api_key": "present"}))
    runner._resolve_turn_toolsets = Mock(return_value=(None, None))
    runner._resolve_session_reasoning_config = Mock(return_value=None)
    runner._resolve_session_service_tier = Mock(return_value=None)
    runner._resolve_turn_agent_config = Mock(
        return_value={"model": "fake-model", "runtime": {}, "request_overrides": None}
    )
    runner._refresh_fallback_model = Mock(return_value=None)
    runner._cleanup_agent_resources = Mock()
    runner._provider_routing = {}
    runner._session_db = None

    async def _run_in_executor(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = _run_in_executor
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _config: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 3)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._platform_config_key", lambda _platform: "discord")
    return runner, adapter, source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "SECRET", "persistence_confirmed": False, "completed": True},
        {"final_response": "SECRET", "completed": True},
        {"final_response": "SECRET", "persistence_confirmed": None, "completed": True},
        {
            "final_response": "SECRET",
            "persistence_confirmed": True,
            "completed": True,
            "failed": True,
        },
        {
            "final_response": "SECRET",
            "persistence_confirmed": True,
            "completed": True,
            "interrupted": True,
        },
        {
            "final_response": "SECRET",
            "persistence_confirmed": True,
            "completed": True,
            "failed": None,
            "interrupted": False,
        },
        {
            "final_response": "SECRET",
            "persistence_confirmed": True,
            "completed": True,
            "failed": False,
            "interrupted": None,
        },
        {"final_response": "SECRET", "persistence_confirmed": True, "failed": False, "interrupted": False},
        {"final_response": "SECRET", "persistence_confirmed": True, "completed": None, "failed": False, "interrupted": False},
        {"final_response": "SECRET", "persistence_confirmed": True, "completed": False, "failed": False, "interrupted": False},
        {"final_response": "SECRET", "persistence_confirmed": True, "completed": True, "failed": False},
        False,
        [],
    ],
)
async def test_background_task_suppresses_unconfirmed_answer(runner, result):
    gateway_runner, adapter, source = runner
    _FakeAgent.result = result

    await gateway_runner._run_background_task_inner("prompt", source, "task-1")

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.kwargs["content"]
    assert "SECRET" not in sent
    assert "canonical session persistence was not confirmed" in sent
    adapter.send_image.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_task_delivers_only_positive_receipt(runner):
    gateway_runner, adapter, source = runner
    _FakeAgent.result = {
        "final_response": "SAFE ANSWER",
        "messages": [],
        "persistence_confirmed": True,
        "completed": True,
        "failed": False,
        "interrupted": False,
    }

    await gateway_runner._run_background_task_inner("prompt", source, "task-2")

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.kwargs["content"]
    assert "SAFE ANSWER" in sent
    assert "canonical session persistence was not confirmed" not in sent


@pytest.mark.asyncio
async def test_background_task_delivers_all_answer_media_after_positive_receipt(runner, tmp_path):
    gateway_runner, adapter, source = runner
    _FakeAgent.result = {
        "final_response": "SAFE ANSWER",
        "messages": [],
        "persistence_confirmed": True,
        "completed": True,
        "failed": False,
        "interrupted": False,
    }
    media_paths = {
        "audio": tmp_path / "audio.mp3",
        "video": tmp_path / "video.mp4",
        "image": tmp_path / "image.png",
        "document": tmp_path / "file.pdf",
    }
    for path in media_paths.values():
        path.write_bytes(b"fixture")
    adapter.extract_images = Mock(return_value=([("https://example.test/image.png", "alt")], "SAFE ANSWER"))
    adapter.extract_media = Mock(
        return_value=(
            [
                (str(media_paths["audio"]), True),
                (str(media_paths["video"]), False),
                (str(media_paths["image"]), False),
                (str(media_paths["document"]), False),
            ],
            "SAFE ANSWER",
        )
    )

    await gateway_runner._run_background_task_inner("prompt", source, "task-media")

    adapter.send.assert_awaited_once()
    adapter.send_image.assert_awaited_once()
    adapter.send_voice.assert_awaited_once()
    adapter.send_video.assert_awaited_once()
    adapter.send_image_file.assert_awaited_once()
    adapter.send_document.assert_awaited_once()

@pytest.mark.asyncio
async def test_background_task_does_not_promote_error_to_answer(runner):
    gateway_runner, adapter, source = runner
    sentinel = "UNTRUSTED_ERROR_PAYLOAD"
    _FakeAgent.result = {
        "final_response": "",
        "error": sentinel,
        "messages": [],
        "persistence_confirmed": True,
        "completed": True,
        "failed": False,
        "interrupted": False,
    }

    await gateway_runner._run_background_task_inner("prompt", source, "task-error")

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.kwargs["content"]
    assert sentinel not in sent
    assert "(No response generated)" in sent


@pytest.mark.asyncio
async def test_background_task_exception_uses_sanitized_notice(runner):
    gateway_runner, adapter, source = runner
    sentinel = "UNTRUSTED_PROVIDER_PAYLOAD"

    async def _raise(_func, *args):
        raise RuntimeError(sentinel)

    gateway_runner._run_in_executor_with_context = _raise

    await gateway_runner._run_background_task_inner("prompt", source, "task-3")

    adapter.send.assert_awaited_once()
    sent = adapter.send.await_args.kwargs["content"]
    assert sentinel not in sent
    assert "failed before a safe answer could be delivered" in sent


def test_background_result_predicate_is_strict():
    assert GatewayRunner._background_result_is_deliverable(
        {"persistence_confirmed": True, "completed": True, "failed": False, "interrupted": False}
    )
    assert not GatewayRunner._background_result_is_deliverable(None)
    assert not GatewayRunner._background_result_is_deliverable(
        {"persistence_confirmed": 1, "completed": True, "failed": False, "interrupted": False}
    )
    assert not GatewayRunner._background_result_is_deliverable(
        {"persistence_confirmed": True, "completed": True, "failed": None, "interrupted": False}
    )
