import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource

from tests.gateway.test_discord_slash_commands import _ensure_discord_mock


_ensure_discord_mock()
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _AutoTitleAgent:
    def __init__(self, *args, **kwargs):
        self.tools = []
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.base_url = "https://example.invalid"
        self.api_key = "***"
        self.api_mode = "chat_completions"

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
        persist_user_timestamp=None,
    ):
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "ok"},
            ],
            "api_calls": 1,
            "completed": True,
        }


class _DiscordRunnerAdapter:
    supports_code_blocks = False

    def __init__(self):
        self.rename_thread = AsyncMock()
        self.sent = []
        self._active_sessions = {}

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs))
        return SimpleNamespace(success=True, message_id="m1")

    def get_pending_message(self, session_key):
        return None


@pytest.fixture
def adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _make_source(**overrides):
    values = dict(
        platform=Platform.DISCORD,
        user_id="42",
        chat_id="999",
        user_name="tester",
        chat_type="thread",
        thread_id="999",
        parent_chat_id="800",
    )
    values.update(overrides)
    return SessionSource(**values)


def _make_runner(*, auto_rename_threads=None, include_adapter=True):
    runner = object.__new__(gateway_run.GatewayRunner)
    discord_cfg = PlatformConfig(enabled=True, token="***")
    if auto_rename_threads is not None:
        discord_cfg.extra["auto_rename_threads"] = auto_rename_threads
    runner.config = GatewayConfig(platforms={Platform.DISCORD: discord_cfg})
    adapter = _DiscordRunnerAdapter()
    runner.adapters = {Platform.DISCORD: adapter} if include_adapter else {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = SimpleNamespace(_db=object())
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _AutoTitleAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


async def _run_agent_and_capture_auto_title_kwargs(monkeypatch, tmp_path, runner, source):
    _install_fake_agent(monkeypatch)
    captured = {}

    def fake_maybe_auto_title(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    import agent.title_generator as title_generator
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(title_generator, "maybe_auto_title", fake_maybe_auto_title)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"tool_progress": "off"}},
    )
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "fake-model")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***", "api_mode": "chat_completions"},
    )
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key="agent:main:discord:thread:999",
    )
    return captured["kwargs"]


@pytest.mark.asyncio
async def test_config_disabled_does_not_install_discord_title_callback(monkeypatch, tmp_path):
    runner = _make_runner(
        auto_rename_threads={"enabled": False, "mode": "session_title", "max_length": 100}
    )

    kwargs = await _run_agent_and_capture_auto_title_kwargs(
        monkeypatch,
        tmp_path,
        runner,
        _make_source(),
    )

    assert "title_callback" not in kwargs


@pytest.mark.asyncio
async def test_enabled_discord_thread_auto_title_installs_callback(monkeypatch, tmp_path):
    runner = _make_runner(
        auto_rename_threads={"enabled": True, "mode": "session_title", "max_length": 100}
    )
    runner._schedule_discord_thread_title_rename = MagicMock()
    source = _make_source()

    kwargs = await _run_agent_and_capture_auto_title_kwargs(
        monkeypatch,
        tmp_path,
        runner,
        source,
    )

    callback = kwargs.get("title_callback")
    assert callable(callback)
    callback("Investigate Discord Thread Titles")
    runner._schedule_discord_thread_title_rename.assert_called_once_with(
        source,
        "Investigate Discord Thread Titles",
    )


@pytest.mark.asyncio
async def test_auto_generated_title_renames_discord_thread():
    runner = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(),
        "Investigate Discord Thread Titles",
    )

    runner.adapters[Platform.DISCORD].rename_thread.assert_awaited_once_with(
        thread_id="999",
        name="Investigate Discord Thread Titles",
    )


@pytest.mark.asyncio
async def test_discord_thread_rename_ignores_non_discord_source():
    runner = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(platform=Platform.TELEGRAM),
        "Ignored",
    )

    runner.adapters[Platform.DISCORD].rename_thread.assert_not_called()


@pytest.mark.asyncio
async def test_discord_thread_rename_ignores_non_thread_source():
    runner = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(chat_type="channel", thread_id=None),
        "Ignored",
    )

    runner.adapters[Platform.DISCORD].rename_thread.assert_not_called()


@pytest.mark.asyncio
async def test_discord_thread_rename_ignores_missing_adapter():
    runner = _make_runner(include_adapter=False)

    await runner._rename_discord_thread_for_session_title(
        _make_source(),
        "Ignored",
    )


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_edits_resolved_thread(adapter):
    thread = SimpleNamespace(edit=AsyncMock())
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=thread))

    await adapter.rename_thread(thread_id="999", name="  Build   Discord Search Titles  ")

    thread.edit.assert_awaited_once_with(name="Build Discord Search Titles")


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_fetches_when_cache_misses(adapter):
    thread = SimpleNamespace(edit=AsyncMock())
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=thread),
    )

    await adapter.rename_thread(thread_id="999", name="Fetched Thread Title")

    adapter._client.get_channel.assert_called_once_with(999)
    adapter._client.fetch_channel.assert_awaited_once_with(999)
    thread.edit.assert_awaited_once_with(name="Fetched Thread Title")


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_invalid_id_returns_quietly(adapter):
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(),
        fetch_channel=AsyncMock(),
    )

    await adapter.rename_thread(thread_id="not-a-snowflake", name="Ignored")

    adapter._client.get_channel.assert_not_called()
    adapter._client.fetch_channel.assert_not_called()


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_missing_client_returns_quietly(adapter):
    adapter._client = None

    await adapter.rename_thread(thread_id="999", name="Ignored")


@pytest.mark.asyncio
async def test_discord_adapter_rename_thread_caps_title_without_ellipsis_churn(adapter):
    thread = SimpleNamespace(edit=AsyncMock())
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=thread))

    await adapter.rename_thread(thread_id="999", name=("A" * 100) + "...")

    thread.edit.assert_awaited_once_with(name="A" * 100)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [PermissionError("forbidden"), RuntimeError("archived")])
async def test_discord_adapter_rename_thread_errors_do_not_raise(adapter, exc):
    thread = SimpleNamespace(edit=AsyncMock(side_effect=exc))
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=thread))

    await adapter.rename_thread(thread_id="999", name="Best Effort Title")

    thread.edit.assert_awaited_once_with(name="Best Effort Title")


@pytest.mark.parametrize("attr_name", ["Forbidden", "HTTPException"])
def test_discord_adapter_expected_rename_failure_recognizes_discord_api_errors(
    monkeypatch,
    attr_name,
):
    class FakeDiscordRenameError(Exception):
        pass

    monkeypatch.setattr(
        sys.modules["discord"],
        attr_name,
        FakeDiscordRenameError,
        raising=False,
    )

    assert DiscordAdapter._is_thread_rename_expected_failure(
        FakeDiscordRenameError("api denied")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attr_name", ["Forbidden", "HTTPException"])
async def test_discord_adapter_rename_thread_real_discord_exceptions_do_not_raise(
    monkeypatch,
    adapter,
    attr_name,
):
    class FakeDiscordRenameError(Exception):
        pass

    monkeypatch.setattr(
        sys.modules["discord"],
        attr_name,
        FakeDiscordRenameError,
        raising=False,
    )
    thread = SimpleNamespace(
        edit=AsyncMock(side_effect=FakeDiscordRenameError("api denied"))
    )
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=thread))

    await adapter.rename_thread(thread_id="999", name="Best Effort Title")

    thread.edit.assert_awaited_once_with(name="Best Effort Title")
