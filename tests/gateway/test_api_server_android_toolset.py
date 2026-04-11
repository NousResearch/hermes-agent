from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig


class TestHermesAndroidAppToolset:
    def test_toolset_exists_and_is_narrow(self):
        from toolsets import get_toolset, resolve_toolset

        toolset = get_toolset("hermes-android-app")
        assert toolset is not None

        resolved = resolve_toolset("hermes-android-app")
        for expected in [
            "terminal",
            "process",
            "android_device_status",
            "android_shared_folder_list",
            "android_shared_folder_read",
            "android_shared_folder_write",
            "android_ui_snapshot",
            "android_ui_action",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "web_search",
            "web_extract",
            "vision_analyze",
            "skills_list",
            "skill_view",
            "skill_manage",
            "todo",
            "memory",
            "session_search",
        ]:
            assert expected in resolved

        for blocked in [
            "image_generate",
            "execute_code",
            "delegate_task",
            "cronjob",
        ]:
            assert blocked not in resolved


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_forces_android_default_when_bootstrap_env_and_no_config(monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())
    monkeypatch.setenv("HERMES_ANDROID_BOOTSTRAP", "1")

    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model") as mock_model, \
         patch("gateway.run._load_gateway_config") as mock_config, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "***",
            "base_url": None,
            "provider": None,
            "api_mode": None,
            "command": None,
            "args": [],
        }
        mock_model.return_value = "test/model"
        mock_config.return_value = {}
        mock_agent_cls.return_value = MagicMock()

        adapter._create_agent()

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["enabled_toolsets"] == ["hermes-android-app"]


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_respects_valid_android_config_override(monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())
    monkeypatch.setenv("HERMES_ANDROID_BOOTSTRAP", "1")

    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model") as mock_model, \
         patch("gateway.run._load_gateway_config") as mock_config, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "***",
            "base_url": None,
            "provider": None,
            "api_mode": None,
            "command": None,
            "args": [],
        }
        mock_model.return_value = "test/model"
        mock_config.return_value = {"platform_toolsets": {"api_server": ["web", "terminal"]}}
        mock_agent_cls.return_value = MagicMock()

        adapter._create_agent()

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert sorted(call_kwargs["enabled_toolsets"]) == ["terminal", "web"]


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_create_agent_forces_android_default_for_invalid_override(monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())
    monkeypatch.setenv("HERMES_ANDROID_BOOTSTRAP", "1")

    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model") as mock_model, \
         patch("gateway.run._load_gateway_config") as mock_config, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "***",
            "base_url": None,
            "provider": None,
            "api_mode": None,
            "command": None,
            "args": [],
        }
        mock_model.return_value = "test/model"
        mock_config.return_value = {"platform_toolsets": {"api_server": ["does-not-exist"]}}
        mock_agent_cls.return_value = MagicMock()

        adapter._create_agent()

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["enabled_toolsets"] == ["hermes-android-app"]
