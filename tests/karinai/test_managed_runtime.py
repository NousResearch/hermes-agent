"""KarinAI managed-runtime integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import tomllib

import pytest

from agent.system_prompt import build_system_prompt_parts
from karinai.runtime.config import ManagedRuntimeConfig, ManagedRuntimeConfigError
from karinai.runtime.managed import (
    compose_ephemeral_system_prompt,
    prepare_managed_runtime_filesystem,
    write_managed_model_gateway_config,
)
from karinai.runtime.prompts import TemplateRenderError, render_template_text
from karinai.runtime.tool_policy import (
    BETA_DISABLED_TOOLSETS,
    BETA_ENABLED_TOOLSETS,
    FORBIDDEN_BETA_TOOLS,
    beta_policy_summary,
    effective_tool_names,
)


def managed_env(**overrides: str) -> dict[str, str]:
    env = {
        "KARINAI_MANAGED_RUNTIME": "true",
        "KARINAI_USER_ID": "user_123",
        "KARINAI_WORKSPACE_ID": "workspace_123",
        "KARINAI_WORKSPACE_DIR": "/workspace",
        "KARINAI_RUNTIME_STATE_DIR": "/hermes",
        "API_SERVER_KEY": "runtime-secret",
        "API_SERVER_HOST": "127.0.0.1",
        "API_SERVER_PORT": "8000",
    }
    env.update(overrides)
    return env


def make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_manager=None,
        model="",
        provider="",
        platform="api_server",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value="User-editable SOUL should not win"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def test_managed_config_requires_internal_api_key():
    env = managed_env(API_SERVER_KEY="")
    with pytest.raises(ManagedRuntimeConfigError, match="API_SERVER_KEY"):
        ManagedRuntimeConfig.from_env(env)


def test_managed_config_requires_runtime_token_when_model_gateway_is_configured():
    with pytest.raises(ManagedRuntimeConfigError, match="KARINAI_RUNTIME_TOKEN"):
        ManagedRuntimeConfig.from_env(
            managed_env(KARINAI_MODEL_GATEWAY_URL="http://model-gateway.internal/v1")
        )


def test_managed_config_rejects_local_cron_plugin_install_and_dashboard():
    with pytest.raises(ManagedRuntimeConfigError, match="LOCAL_CRON"):
        ManagedRuntimeConfig.from_env(managed_env(KARINAI_LOCAL_CRON_ENABLED="true"))
    with pytest.raises(ManagedRuntimeConfigError, match="PLUGIN_INSTALL"):
        ManagedRuntimeConfig.from_env(managed_env(KARINAI_PLUGIN_INSTALL_ENABLED="true"))
    with pytest.raises(ManagedRuntimeConfigError, match="DASHBOARD"):
        ManagedRuntimeConfig.from_env(managed_env(KARINAI_DASHBOARD_ENABLED="true"))


def test_managed_config_requires_absolute_workspace_and_state_paths():
    with pytest.raises(ManagedRuntimeConfigError, match="KARINAI_WORKSPACE_DIR"):
        ManagedRuntimeConfig.from_env(managed_env(KARINAI_WORKSPACE_DIR="workspace"))
    with pytest.raises(ManagedRuntimeConfigError, match="KARINAI_RUNTIME_STATE_DIR"):
        ManagedRuntimeConfig.from_env(managed_env(KARINAI_RUNTIME_STATE_DIR="hermes"))


def test_managed_gateway_env_scopes_runtime_state_and_workspace_writes():
    cfg = ManagedRuntimeConfig.from_env(managed_env())
    gateway_env = cfg.gateway_env()
    assert gateway_env["API_SERVER_ENABLED"] == "true"
    assert gateway_env["HERMES_HOME"] == "/hermes"
    assert gateway_env["HOME"] == "/hermes/home"
    assert gateway_env["TERMINAL_CWD"] == "/workspace"
    assert gateway_env["HERMES_WRITE_SAFE_ROOT"] == "/workspace"
    assert gateway_env["HERMES_DASHBOARD"] == "false"


def test_prepare_managed_runtime_filesystem_creates_workspace_state_and_home(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    cfg = ManagedRuntimeConfig.from_env(
        managed_env(
            KARINAI_WORKSPACE_DIR=str(workspace),
            KARINAI_RUNTIME_STATE_DIR=str(state),
        )
    )
    prepare_managed_runtime_filesystem(cfg)
    assert workspace.is_dir()
    assert state.is_dir()
    assert (state / "home").is_dir()


def test_write_managed_model_gateway_config_uses_key_env_not_raw_token(tmp_path):
    yaml = pytest.importorskip("yaml")
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    cfg = ManagedRuntimeConfig.from_env(
        managed_env(
            KARINAI_WORKSPACE_DIR=str(workspace),
            KARINAI_RUNTIME_STATE_DIR=str(state),
            KARINAI_MODEL_GATEWAY_URL="http://model-gateway.internal",
            KARINAI_MODEL_GATEWAY_MODEL="karinai/test-model",
            KARINAI_RUNTIME_TOKEN="scoped-runtime-token",
        )
    )
    prepare_managed_runtime_filesystem(cfg)
    config_path = write_managed_model_gateway_config(cfg)

    assert config_path is not None
    assert config_path == state / "config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["model"] == {
        "default": "karinai/test-model",
        "provider": "custom:karinai-model-gateway",
        "base_url": "http://model-gateway.internal/v1",
        "api_mode": "chat_completions",
    }
    provider = data["providers"]["karinai-model-gateway"]
    assert provider["api"] == "http://model-gateway.internal/v1"
    assert provider["key_env"] == "KARINAI_RUNTIME_TOKEN"
    assert provider["default_model"] == "karinai/test-model"
    assert "scoped-runtime-token" not in config_path.read_text(encoding="utf-8")


def test_start_managed_prepares_env_chdirs_and_runs_gateway(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    for key, value in managed_env(
        KARINAI_WORKSPACE_DIR=str(workspace),
        KARINAI_RUNTIME_STATE_DIR=str(state),
    ).items():
        monkeypatch.setenv(key, value)
    start_cwd = Path.cwd()
    fake_gateway_main = MagicMock()

    from karinai.runtime import start_managed

    monkeypatch.setattr(start_managed, "_run_gateway_main", fake_gateway_main)
    try:
        start_managed.main()
        assert Path.cwd() == workspace
        assert os.environ["HERMES_HOME"] == str(state)
        assert os.environ["HOME"] == str(state / "home")
        assert os.environ["TERMINAL_CWD"] == str(workspace)
        assert os.environ["HERMES_WRITE_SAFE_ROOT"] == str(workspace)
        assert os.environ["HERMES_DASHBOARD"] == "false"
        fake_gateway_main.assert_called_once_with()
    finally:
        os.chdir(start_cwd)


def test_beta_tool_policy_exposes_only_sandbox_basics():
    exposed = effective_tool_names(BETA_ENABLED_TOOLSETS, BETA_DISABLED_TOOLSETS)
    assert {"read_file", "write_file", "patch", "search_files"}.issubset(exposed)
    assert {"terminal", "process", "execute_code", "web_search", "web_extract"}.issubset(exposed)
    assert not (exposed & FORBIDDEN_BETA_TOOLS)
    summary = beta_policy_summary()
    assert summary["mode"] == "beta"
    assert "cronjob" in summary["disabled_toolsets"]


def test_prompt_template_rendering_fails_on_missing_variables():
    with pytest.raises(TemplateRenderError, match="workspace_id"):
        render_template_text("Workspace {{ workspace_id }}", {})


def test_managed_prompt_renders_karinai_identity_without_upstream_identity():
    cfg = ManagedRuntimeConfig.from_env(managed_env())
    prompt = compose_ephemeral_system_prompt(None, cfg)
    assert "You are KarinAI" in prompt
    assert "KarinAI agent" in prompt
    assert "workspace_123" in prompt
    assert "You are Hermes Agent" not in prompt
    assert "You run on Hermes Agent" not in prompt


def test_managed_policy_is_appended_after_client_instructions():
    cfg = ManagedRuntimeConfig.from_env(managed_env())
    prompt = compose_ephemeral_system_prompt("Talk like a pirate.", cfg)
    assert prompt.index("Talk like a pirate") < prompt.index("KarinAI managed runtime instructions")
    assert prompt.rstrip().endswith("implementation details unless the user explicitly asks for technical internals.")


def test_managed_system_prompt_replaces_default_identity_and_soul(monkeypatch):
    for key, value in managed_env().items():
        monkeypatch.setenv(key, value)
    prompt = stable_prompt(make_agent())
    assert "You are KarinAI" in prompt
    assert "KarinAI agent" in prompt
    assert "User-editable SOUL should not win" not in prompt
    assert "You are Hermes Agent" not in prompt
    assert "You run on Hermes Agent" not in prompt
    assert "Active Hermes profile" not in prompt


def test_api_server_uses_managed_tool_policy(monkeypatch):
    for key, value in managed_env().items():
        monkeypatch.setenv(key, value)

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())

    with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
         patch("gateway.run._resolve_gateway_model") as mock_model, \
         patch("gateway.run._load_gateway_config") as mock_config, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_kwargs.return_value = {
            "api_key": "test-key",
            "base_url": None,
            "provider": None,
            "api_mode": None,
            "command": None,
            "args": [],
        }
        mock_model.return_value = "test/model"
        mock_config.return_value = {"platform_toolsets": {"api_server": ["hermes-api-server"]}}
        mock_agent_cls.return_value = MagicMock()

        adapter._create_agent(ephemeral_system_prompt="Client instruction")

        call_kwargs = mock_agent_cls.call_args.kwargs
        assert call_kwargs["enabled_toolsets"] == list(BETA_ENABLED_TOOLSETS)
        assert call_kwargs["disabled_toolsets"] == list(BETA_DISABLED_TOOLSETS)
        assert call_kwargs["ephemeral_system_prompt"] == "Client instruction"
        assert call_kwargs["platform"] == "api_server"


def test_tool_policy_yaml_matches_runtime_constants():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / "karinai" / "config" / "tool-policy.beta.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert tuple(data["enabled_toolsets"]) == BETA_ENABLED_TOOLSETS
    assert tuple(data["disabled_toolsets"]) == BETA_DISABLED_TOOLSETS


def test_karinai_runtime_is_packaged_for_installed_entrypoint():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["karinai-agent-managed"] == "karinai.runtime.start_managed:main"
    assert "karinai" in data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "karinai.*" in data["tool"]["setuptools"]["packages"]["find"]["include"]
    package_data = data["tool"]["setuptools"]["package-data"]["karinai"]
    assert "prompts/*.j2" in package_data
    assert "config/*.yaml" in package_data
    assert "config/*.example" in package_data
    assert "docker/*.sh" in package_data
