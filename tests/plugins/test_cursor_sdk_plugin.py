from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from plugins.cursor_agent_sdk import register
from plugins.cursor_agent_sdk import tools as cursor_tools
from plugins.cursor_agent_sdk.tools import (
    CURSOR_AGENT_SCHEMA,
    DEFAULT_TIMEOUT_SECONDS,
    check_cursor_sdk_available,
    handle_cursor_agent,
)


@pytest.fixture(autouse=True)
def _disable_cursor_sdk_lazy_install(monkeypatch):
    from tools import lazy_deps

    _FakeCursorClient.instances.clear()
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("CURSOR_SDK_BRIDGE_URL", raising=False)
    monkeypatch.delenv("CURSOR_SDK_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_SDK_BRIDGE_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(lazy_deps, "ensure", lambda feature, prompt=False: None)


class _FakeCtx:
    def __init__(self) -> None:
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


class _FakeCursorClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.closed = False
        self.launched = False
        type(self).instances.append(self)

    @classmethod
    def launch_bridge(cls, *args, **kwargs):
        client = cls(*args, **kwargs)
        client.launched = True
        return client

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


def test_register_exposes_cursor_agent_tool():
    ctx = _FakeCtx()

    register(ctx)

    assert len(ctx.tools) == 1
    tool = ctx.tools[0]
    assert tool["name"] == "cursor_agent"
    assert tool["toolset"] == "cursor_sdk"
    assert tool["requires_env"] == ["CURSOR_API_KEY"]
    assert tool["max_result_size_chars"] == 100_000
    assert tool["schema"]["parameters"]["required"] == ["prompt"]
    assert (
        tool["schema"]["parameters"]["properties"]["timeout_seconds"]["default"]
        == DEFAULT_TIMEOUT_SECONDS
    )


def test_plugin_package_does_not_shadow_cursor_sdk_package():
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "plugins" / "cursor_sdk").exists()


def test_register_real_context_sets_result_size_cap():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    manifest = PluginManifest(name="cursor_sdk", key="cursor_sdk")
    ctx = PluginContext(manifest, PluginManager())

    try:
        register(ctx)
        assert registry.get_max_result_size("cursor_agent") == 100_000
    finally:
        registry.deregister("cursor_agent")


def test_register_tool_positional_override_compatibility():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    registry.register(
        name="cursor_agent",
        toolset="builtin",
        schema={
            "name": "cursor_agent",
            "description": "Built-in",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: "builtin",
    )
    manifest = PluginManifest(name="cursor_sdk", key="cursor_sdk")
    ctx = PluginContext(manifest, PluginManager())

    try:
        ctx.register_tool(
            "cursor_agent",
            "cursor_sdk",
            CURSOR_AGENT_SCHEMA,
            lambda args, **kw: "plugin",
            None,
            None,
            False,
            "",
            "C",
            True,
        )
        assert registry.get_toolset_for_tool("cursor_agent") == "cursor_sdk"
    finally:
        registry.deregister("cursor_agent")


def test_check_cursor_sdk_available_requires_key(monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    assert check_cursor_sdk_available() is False

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    assert check_cursor_sdk_available() is True


def test_cursor_sdk_toolset_is_default_off(monkeypatch):
    from hermes_cli import tools_config

    monkeypatch.setattr(
        tools_config,
        "_get_plugin_toolset_keys",
        lambda: {"cursor_sdk"},
    )

    assert "cursor_sdk" not in tools_config._get_platform_tools({}, "cli")
    assert "cursor_sdk" in tools_config._DEFAULT_OFF_TOOLSETS


def test_cursor_sdk_toolset_can_be_explicitly_enabled(monkeypatch):
    from hermes_cli import tools_config

    monkeypatch.setattr(
        tools_config,
        "_get_plugin_toolset_keys",
        lambda: {"cursor_sdk"},
    )
    config = {"platform_toolsets": {"cli": ["hermes-cli", "cursor_sdk"]}}

    assert "cursor_sdk" in tools_config._get_platform_tools(config, "cli")


def test_handle_cursor_agent_missing_key_returns_json_error(monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    result = json.loads(handle_cursor_agent({"prompt": "summarize"}))

    assert result["success"] is False
    assert "CURSOR_API_KEY" in result["error"]


def test_handle_cursor_agent_missing_sdk_returns_json_error(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(sys.modules, "cursor_sdk", None)

    result = json.loads(handle_cursor_agent({"prompt": "summarize"}))

    assert result["success"] is False
    assert "cursor-sdk" in result["error"]


def test_handle_cursor_agent_local_options_error_returns_json(monkeypatch):
    def bad_local_options(**kwargs):
        raise TypeError("bad local option")

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=object(),
            LocalAgentOptions=bad_local_options,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is False
    assert result["error"] == "Invalid cursor_agent arguments"
    assert "bad local option" in result["detail"]


def test_handle_cursor_agent_rejects_nonlocal_terminal_backend(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=object(),
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is False
    assert result["error"] == "Invalid cursor_agent arguments"
    assert "TERMINAL_ENV='docker'" in result["detail"]


def test_handle_cursor_agent_rejects_invalid_timeout(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=object(),
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(
        handle_cursor_agent({"prompt": "work", "timeout_seconds": 0})
    )

    assert result["success"] is False
    assert result["error"] == "Invalid cursor_agent arguments"
    assert "timeout_seconds" in result["detail"]


def test_handle_cursor_agent_local_run_invokes_sdk(monkeypatch, tmp_path):
    calls = {}

    class FakeLocalAgentOptions:
        def __init__(self, **kwargs):
            calls["local_options"] = kwargs

    class FakeCloudAgentOptions:
        def __init__(self, **kwargs):
            calls["cloud_options"] = kwargs

    class FakeCloudRepository:
        def __init__(self, **kwargs):
            calls["cloud_repository"] = kwargs

    class FakeRun:
        id = "run_123"

        def text(self):
            return "cursor response"

    class FakeAgentInstance:
        agent_id = "agent_123"

        def send(self, prompt):
            calls["prompt"] = prompt
            return FakeRun()

        def close(self):
            calls["closed"] = True

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return FakeAgentInstance()

    fake_module = types.SimpleNamespace(
        AsyncAgent=FakeAgent,
        LocalAgentOptions=FakeLocalAgentOptions,
        CloudAgentOptions=FakeCloudAgentOptions,
        CloudRepository=FakeCloudRepository,
        AsyncClient=_FakeCursorClient,
    )
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_module)
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")

    result = json.loads(
        handle_cursor_agent(
            {
                "prompt": "summarize this repo",
                "cwd": str(tmp_path),
                "model": "composer-test",
                "sandbox": True,
                "timeout_seconds": 12,
            }
        )
    )

    assert result == {
        "success": True,
        "model": "composer-test",
        "agent_id": "agent_123",
        "run_id": "run_123",
        "status": "",
        "text": "cursor response",
        "runtime": "local",
        "cwd": str(tmp_path.resolve()),
    }
    assert calls["create"]["api_key"] == "cursor-key"
    assert calls["create"]["model"] == "composer-test"
    assert calls["create"]["client"].launched is True
    assert calls["create"]["client"].kwargs == {
        "workspace": str(tmp_path.resolve()),
        "timeout": pytest.approx(12, rel=0.01),
        "client_timeout": pytest.approx(12, rel=0.01),
        "max_retries": 0,
        "allow_api_key_env_fallback": True,
    }
    assert calls["create"]["client"].closed is True
    assert calls["prompt"] == "summarize this repo"
    assert calls["local_options"] == {
        "cwd": str(tmp_path.resolve()),
        "sandbox_options": {"enabled": True},
    }
    assert calls["closed"] is True


def test_handle_cursor_agent_uses_terminal_cwd_default(monkeypatch, tmp_path):
    calls = {}

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.chdir("/")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is True
    assert result["cwd"] == str(tmp_path.resolve())
    assert calls["create"]["local"]["cwd"] == str(tmp_path.resolve())


def test_handle_cursor_agent_resolves_relative_cwd_from_terminal_cwd(
    monkeypatch,
    tmp_path,
):
    calls = {}
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.chdir("/")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work", "cwd": "project"}))

    assert result["success"] is True
    assert result["cwd"] == str(project.resolve())
    assert calls["create"]["local"]["cwd"] == str(project.resolve())


def test_handle_cursor_agent_ignores_api_key_tool_arg(monkeypatch, tmp_path):
    calls = {}

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "env-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(
        handle_cursor_agent(
            {
                "prompt": "work",
                "cwd": str(tmp_path),
                "api_key": "model-controlled-key",
            }
        )
    )

    assert result["success"] is True
    assert calls["create"]["api_key"] == "env-key"


def test_handle_cursor_agent_non_finished_status_reports_failure(monkeypatch):
    class FakeRun:
        id = "run_error"

        def wait(self):
            return types.SimpleNamespace(status="error", result="partial output")

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is False
    assert result["status"] == "error"
    assert result["text"] == "partial output"
    assert "status: error" in result["error"]


def test_handle_cursor_agent_wait_error_returns_runtime_failure(monkeypatch):
    calls = {}

    class FakeRun:
        id = "run_timeout"

        def wait(self):
            raise TimeoutError("wait timed out")

        def cancel(self):
            calls["cancelled"] = True

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is False
    assert result["error"] == "Cursor SDK agent run failed"
    assert "wait timed out" in result["detail"]
    assert calls["cancelled"] is True


def test_handle_cursor_agent_timeout_cancels_run(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        id = "run_slow"

        async def wait(self):
            await asyncio.sleep(1)
            return types.SimpleNamespace(status="finished", result="late")

        async def cancel(self):
            calls["cancelled"] = True

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setattr(cursor_tools, "_coerce_timeout_seconds", lambda value: 0.01)
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(
        handle_cursor_agent({"prompt": "work", "cwd": str(tmp_path)})
    )

    assert result["success"] is False
    assert result["error"] == "Cursor SDK agent run failed"
    assert "timeout_seconds=0.01" in result["detail"]
    assert calls["cancelled"] is True


def test_handle_cursor_agent_runs_inside_active_event_loop(monkeypatch, tmp_path):
    class FakeAgent:
        @classmethod
        async def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    async def call_handler():
        return handle_cursor_agent({"prompt": "work", "cwd": str(tmp_path)})

    result = json.loads(asyncio.run(call_handler()))

    assert result["success"] is True
    assert result["text"] == "ok"


def test_handle_cursor_agent_awaits_async_run_text_fallback(monkeypatch):
    class FakeRun:
        id = "run"

        async def wait(self):
            return types.SimpleNamespace(status="finished", result="")

        async def text(self):
            return "async run text"

    class FakeAgent:
        @classmethod
        async def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is True
    assert result["text"] == "async run text"


def test_handle_cursor_agent_uses_bridge_env_client(monkeypatch, tmp_path):
    calls = {}

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("CURSOR_SDK_BRIDGE_URL", "http://127.0.0.1:12345")
    monkeypatch.setenv("CURSOR_SDK_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(
        handle_cursor_agent(
            {"prompt": "work", "cwd": str(tmp_path), "timeout_seconds": 34}
        )
    )

    assert result["success"] is True
    assert calls["create"]["client"].launched is False
    assert calls["create"]["client"].kwargs == {
        "base_url": "http://127.0.0.1:12345",
        "auth_token": "bridge-token",
        "timeout": pytest.approx(34, rel=0.01),
        "unary_timeout": pytest.approx(34, rel=0.01),
        "stream_timeout": pytest.approx(34, rel=0.01),
        "max_retries": 0,
        "allow_api_key_env_fallback": False,
    }
    assert calls["create"]["client"].closed is True


def test_handle_cursor_agent_close_failure_does_not_mask_success(monkeypatch):
    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            def close():
                raise RuntimeError("cleanup failed")

            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: types.SimpleNamespace(id="run", text=lambda: "ok"),
                close=close,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is True
    assert result["text"] == "ok"


def test_handle_cursor_agent_wait_text_method_is_called(monkeypatch):
    class WaitResult:
        status = "finished"

        def text(self):
            return "wait text"

    class FakeRun:
        id = "run"

        def wait(self):
            return WaitResult()

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is True
    assert result["status"] == "finished"
    assert result["text"] == "wait text"


def test_handle_cursor_agent_text_none_returns_empty_string(monkeypatch):
    class FakeRun:
        id = "run"

        def text(self):
            return None

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            return types.SimpleNamespace(
                agent_id="agent",
                send=lambda prompt: FakeRun(),
                close=lambda: None,
            )

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=lambda **kwargs: kwargs,
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work"}))

    assert result["success"] is True
    assert result["text"] == ""


def test_handle_cursor_agent_cloud_run_requires_repo_or_pr(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=object(),
            LocalAgentOptions=object(),
            CloudAgentOptions=object(),
            CloudRepository=object(),
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(handle_cursor_agent({"prompt": "work", "runtime": "cloud"}))

    assert result["success"] is False
    assert result["error"] == "Invalid cursor_agent arguments"
    assert "cloud_repo_url or cloud_pr_url" in result["detail"]


def test_handle_cursor_agent_cloud_pr_derives_repo_url(monkeypatch):
    calls = {}

    class FakeCloudRepository:
        def __init__(self, **kwargs):
            calls["repository"] = kwargs

    class FakeCloudAgentOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["cloud"] = kwargs

    class FakeAgentInstance:
        agent_id = "agent_cloud"

        def send(self, prompt):
            calls["prompt"] = prompt
            return types.SimpleNamespace(id="run_cloud", text=lambda: "cloud ok")

        def close(self):
            calls["closed"] = True

    class FakeAgent:
        @classmethod
        def create(cls, **kwargs):
            calls["create"] = kwargs
            return FakeAgentInstance()

    monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setitem(
        sys.modules,
        "cursor_sdk",
        types.SimpleNamespace(
            AsyncAgent=FakeAgent,
            LocalAgentOptions=object(),
            CloudAgentOptions=FakeCloudAgentOptions,
            CloudRepository=FakeCloudRepository,
            AsyncClient=_FakeCursorClient,
        ),
    )

    result = json.loads(
        handle_cursor_agent(
            {
                "prompt": "review this PR",
                "runtime": "cloud",
                "cloud_pr_url": "https://github.com/acme/widgets/pull/42",
            }
        )
    )

    assert result["success"] is True
    assert result["cloud_repo_url"] == "https://github.com/acme/widgets"
    assert calls["repository"] == {
        "url": "https://github.com/acme/widgets",
        "pr_url": "https://github.com/acme/widgets/pull/42",
    }
    assert calls["create"]["client"].launched is True
    assert calls["create"]["client"].kwargs["workspace"] is None
    assert calls["create"]["cloud"].kwargs is calls["cloud"]
    assert calls["closed"] is True
