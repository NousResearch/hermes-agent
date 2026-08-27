import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import OpenVikingMemoryProvider, quick_local


def _preflight(tmp_path: Path) -> quick_local.QuickLocalPreflight:
    return quick_local.QuickLocalPreflight(
        paths=quick_local.managed_paths(tmp_path),
        reusable_endpoint=None,
    )


def test_build_server_config_uses_profile_scoped_storage_and_local_embedding(
    tmp_path,
):
    paths = quick_local.managed_paths(tmp_path)

    config = quick_local.build_server_config(
        paths,
        {
            "provider": "openai",
            "model": "model-1",
            "api_key": "secret",
            "api_base": "https://llm.example/v1",
        },
        port=1941,
    )

    assert config == {
        "server": {"host": "127.0.0.1", "port": 1941},
        "storage": {"workspace": str(tmp_path / "openviking" / "data")},
        "embedding": {
            "dense": {
                "provider": "local",
                "model": "bge-small-zh-v1.5-f16",
                "dimension": 512,
                "cache_dir": str(tmp_path / "openviking" / "models"),
            }
        },
        "vlm": {
            "provider": "openai",
            "model": "model-1",
            "api_key": "secret",
            "api_base": "https://llm.example/v1",
        },
    }


def test_resolve_vlm_uses_one_persisted_hermes_model_source(monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"model": {"provider": "custom:test", "default": "saved-model"}},
    )
    resolver = MagicMock(
        return_value={
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://llm.example/v1",
            "api_key": "secret",
            "source": "custom_provider:test",
            "extra_headers": {"X-Tenant": "tenant"},
            "request_overrides": {"extra_body": {"thinking": {"type": "disabled"}}},
        }
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolver)

    vlm = quick_local.resolve_hermes_vlm_config()

    resolver.assert_called_once_with(
        requested="custom:test",
        target_model="saved-model",
    )
    assert vlm == {
        "provider": "openai",
        "model": "saved-model",
        "api_key": "secret",
        "api_base": "https://llm.example/v1",
        "extra_headers": {"X-Tenant": "tenant"},
        "extra_request_body": {"thinking": {"type": "disabled"}},
        "temperature": 0.0,
        "max_retries": 2,
    }


def test_resolve_vlm_maps_anthropic_transport(monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"model": {"provider": "anthropic", "default": "claude-sonnet"}},
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "secret",
            "source": "env",
            "model": "claude-sonnet",
        },
    )

    vlm = quick_local.resolve_hermes_vlm_config()

    assert vlm["provider"] == "litellm"
    assert vlm["model"] == "anthropic/claude-sonnet"
    assert vlm["api_base"] == "https://api.anthropic.com"


def test_resolve_vlm_accepts_structured_persisted_default(monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "model": {"default": {"provider": "custom:corp", "model": "corp-model"}}
        },
    )
    resolver = MagicMock(
        return_value={
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://llm.example/v1",
            "api_key": "secret",
            "source": "custom_provider:corp",
        }
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolver)

    vlm = quick_local.resolve_hermes_vlm_config()

    resolver.assert_called_once_with(
        requested="custom:corp",
        target_model="corp-model",
    )
    assert vlm["model"] == "corp-model"


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        (
            {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "short-lived",
                "source": "oauth",
            },
            "cannot be copied safely",
        ),
        (
            {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": "https://llm.example/v1",
                "api_key": lambda: pytest.fail("Setup must not execute key commands"),
                "source": "key_cmd",
            },
            "cannot be copied safely",
        ),
        (
            {
                "provider": "copilot",
                "api_mode": "chat_completions",
                "base_url": "https://api.githubcopilot.com",
                "api_key": "short-lived-exchanged-token",
                "source": "GH_TOKEN",
            },
            "cannot be copied safely",
        ),
        (
            {
                "provider": "future-oauth-provider",
                "api_mode": "chat_completions",
                "base_url": "https://llm.example/v1",
                "api_key": "short-lived",
                "source": "future-credential-store",
            },
            "cannot be copied safely",
        ),
        (
            {
                "provider": "openai-api",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "secret",
                "source": "OPENAI_API_KEY",
            },
            "transport is not supported",
        ),
        (
            {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": "https://llm.example/v1",
                "api_key": False,
                "source": "env/config",
            },
            "cannot be copied safely",
        ),
        (
            {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": {"url": "https://llm.example/v1"},
                "api_key": "secret",
                "source": "env/config",
            },
            "API base URL must be a string",
        ),
        (
            {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": False,
                "api_key": "secret",
                "source": "env/config",
            },
            "API base URL must be a string",
        ),
    ],
)
def test_resolve_vlm_rejects_credentials_or_transports_openviking_cannot_reuse(
    runtime,
    message,
    monkeypatch,
):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"model": {"provider": "openai", "default": "model-1"}},
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: runtime,
    )

    with pytest.raises(quick_local.QuickLocalSetupError, match=message):
        quick_local.resolve_hermes_vlm_config()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("api_key", "did not resolve reusable static credentials"),
        ("base_url", "did not resolve an API base URL"),
    ],
)
@pytest.mark.parametrize("missing_value", [None, "", " \t "])
def test_resolve_vlm_reports_missing_fields_before_classifying_credentials(
    field, message, missing_value, monkeypatch
):
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"model": {"provider": "custom", "default": "model-1"}},
    )
    runtime = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://llm.example/v1",
        "api_key": "secret",
        "source": "env/config",
        field: missing_value,
    }
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_provider", lambda **_kwargs: runtime
    )
    classify = MagicMock(wraps=quick_local._has_copyable_static_credentials)
    monkeypatch.setattr(quick_local, "_has_copyable_static_credentials", classify)

    with pytest.raises(quick_local.QuickLocalSetupError, match=message):
        quick_local.resolve_hermes_vlm_config()

    classify.assert_not_called()


def test_openviking_install_is_bounded_and_isolated(tmp_path, monkeypatch):
    from hermes_cli import managed_uv

    installed = iter([False, True])
    monkeypatch.setattr(
        quick_local,
        "openviking_install_satisfies_requirement",
        lambda _paths: next(installed),
    )
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: "/usr/local/bin/uv")
    calls = []

    def run_isolated(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["UV_NATIVE_TLS"] == "true"
        assert kwargs["env"]["UV_SYSTEM_CERTS"] == "true"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(quick_local.subprocess, "run", run_isolated)
    setup = quick_local.QuickLocalSetup(health_check=lambda _endpoint: (True, ""))
    paths = quick_local.managed_paths(tmp_path)

    assert setup._ensure_openviking_installed(paths) is True

    assert calls[0][0] == [
        "/usr/local/bin/uv",
        "venv",
        str(paths.runtime),
        "--python",
        quick_local.sys.executable,
    ]
    assert calls[1][0] == [
        "/usr/local/bin/uv",
        "pip",
        "install",
        "--python",
        str(paths.runtime_python),
        "openviking[local-embed]>=0.4.16,<0.6",
    ]
    assert all(kwargs["cwd"] == paths.root for _command, kwargs in calls)
    assert all(kwargs["stdin"] is subprocess.DEVNULL for _command, kwargs in calls)


def test_openviking_install_failure_is_not_activated(tmp_path, monkeypatch):
    from hermes_cli import managed_uv

    monkeypatch.setattr(
        quick_local,
        "openviking_install_satisfies_requirement",
        lambda _paths: False,
    )
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: "/usr/local/bin/uv")
    results = iter([SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)])
    monkeypatch.setattr(
        quick_local.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )
    setup = quick_local.QuickLocalSetup(health_check=lambda _endpoint: (True, ""))

    with pytest.raises(quick_local.QuickLocalSetupError, match="compatible OpenViking"):
        setup._ensure_openviking_installed(quick_local.managed_paths(tmp_path))


def test_existing_openviking_without_local_embedding_runtime_is_not_reused(
    tmp_path,
    monkeypatch,
):
    paths = quick_local.managed_paths(tmp_path)
    paths.runtime_python.parent.mkdir(parents=True)
    paths.runtime_python.touch()
    paths.server_command.touch()
    run = MagicMock(
        return_value=SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'llama_cpp'",
        )
    )
    monkeypatch.setattr(quick_local.subprocess, "run", run)

    assert quick_local.openviking_install_satisfies_requirement(paths) is False
    assert "import llama_cpp" in run.call_args.args[0][2]


def test_existing_openviking_with_local_embedding_runtime_is_reused(
    tmp_path,
    monkeypatch,
):
    paths = quick_local.managed_paths(tmp_path)
    paths.runtime_python.parent.mkdir(parents=True)
    paths.runtime_python.touch()
    paths.server_command.touch()
    monkeypatch.setattr(
        quick_local.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="0.4.16\n",
            stderr="",
        ),
    )

    assert quick_local.openviking_install_satisfies_requirement(paths) is True


def test_validation_server_does_not_inherit_stdin(tmp_path, monkeypatch):
    server_command = tmp_path / "openviking-server"
    server_command.write_text("", encoding="utf-8")
    config_path = tmp_path / "ov.conf"
    config_path.write_text("{}", encoding="utf-8")
    process = MagicMock()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(quick_local.subprocess, "Popen", popen)
    monkeypatch.setattr(quick_local, "_can_bind_local_port", lambda *_args: True)

    result = quick_local._start_validation_server(
        "http://127.0.0.1:1933",
        config_path,
        tmp_path,
        server_command,
    )

    assert result is process
    assert popen.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_validation_server_rechecks_selected_port_before_start(tmp_path, monkeypatch):
    server_command = tmp_path / "openviking-server"
    server_command.write_text("", encoding="utf-8")
    config_path = tmp_path / "ov.conf"
    config_path.write_text("{}", encoding="utf-8")
    popen = MagicMock()
    monkeypatch.setattr(quick_local.subprocess, "Popen", popen)
    monkeypatch.setattr(quick_local, "_can_bind_local_port", lambda *_args: False)

    with pytest.raises(quick_local.QuickLocalSetupError, match="became unavailable"):
        quick_local._start_validation_server(
            "http://127.0.0.1:1933",
            config_path,
            tmp_path,
            server_command,
        )

    popen.assert_not_called()


def test_reuse_rechecks_runtime_and_refreshes_saved_vlm(tmp_path, monkeypatch):
    paths = quick_local.managed_paths(tmp_path)
    paths.root.mkdir(parents=True)
    old_vlm = {
        "provider": "openai",
        "model": "old-model",
        "api_key": "old-secret",
        "api_base": "https://old.example/v1",
    }
    quick_local.atomic_json_write(
        paths.server_config,
        quick_local.build_server_config(paths, old_vlm, port=1938),
        mode=0o600,
    )
    quick_local.atomic_json_write(
        paths.ovcli_config,
        {"url": "http://127.0.0.1:1938", "actor_peer_id": "hermes"},
        mode=0o600,
    )
    new_vlm = {
        "provider": "openai",
        "model": "new-model",
        "api_key": "new-secret",
        "api_base": "https://new.example/v1",
    }
    resolve = MagicMock(return_value=new_vlm)
    monkeypatch.setattr(quick_local, "resolve_hermes_vlm_config", resolve)
    setup = quick_local.QuickLocalSetup(
        health_check=lambda _endpoint: (True, ""),
    )
    ensure_runtime = MagicMock(return_value=False)
    monkeypatch.setattr(setup, "_ensure_openviking_installed", ensure_runtime)

    result = setup.provision(hermes_home=tmp_path)

    assert result.reused is True
    assert result.endpoint == "http://127.0.0.1:1938"
    assert result.server_restart_required is True
    assert json.loads(paths.server_config.read_text(encoding="utf-8")) == (
        quick_local.build_server_config(paths, new_vlm, port=1938)
    )
    resolve.assert_called_once_with()
    ensure_runtime.assert_called_once_with(paths)


@pytest.mark.parametrize("runtime_current", [False, True])
def test_reuse_with_unchanged_config_requires_restart_only_after_runtime_upgrade(
    tmp_path, monkeypatch, runtime_current
):
    from hermes_cli import managed_uv

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        json.dumps({
            "model": {"provider": "custom:quick-local-test", "default": "model-1"},
            "custom_providers": [
                {
                    "name": "quick-local-test",
                    "base_url": "https://llm.example/v1",
                    "api_key": "secret",
                    "api_mode": "chat_completions",
                }
            ],
        }),
        encoding="utf-8",
    )
    paths = quick_local.managed_paths(tmp_path)
    paths.runtime_python.parent.mkdir(parents=True)
    paths.runtime_python.touch()
    paths.server_command.touch()
    vlm = quick_local.resolve_hermes_vlm_config()
    assert vlm["model"] == "model-1"
    assert vlm["api_key"] == "secret"
    server_config = quick_local.build_server_config(paths, vlm, port=1938)
    quick_local.atomic_json_write(paths.server_config, server_config, mode=0o600)
    quick_local.atomic_json_write(
        paths.ovcli_config,
        {"url": "http://127.0.0.1:1938", "actor_peer_id": "hermes"},
        mode=0o600,
    )
    compatible = MagicMock(side_effect=[runtime_current, True])
    monkeypatch.setattr(
        quick_local, "openviking_install_satisfies_requirement", compatible
    )
    ensure_uv = MagicMock(return_value="/usr/local/bin/uv")
    monkeypatch.setattr(managed_uv, "ensure_uv", ensure_uv)
    install = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(quick_local.subprocess, "run", install)
    write = MagicMock(wraps=quick_local.atomic_json_write)
    monkeypatch.setattr(quick_local, "atomic_json_write", write)
    setup = quick_local.QuickLocalSetup(health_check=lambda _endpoint: (True, ""))

    result = setup.provision(hermes_home=tmp_path)

    assert result.reused is True
    assert result.endpoint == "http://127.0.0.1:1938"
    assert result.server_restart_required is (not runtime_current)
    assert json.loads(paths.server_config.read_text(encoding="utf-8")) == server_config
    write.assert_not_called()
    if runtime_current:
        ensure_uv.assert_not_called()
        install.assert_not_called()
    else:
        ensure_uv.assert_called_once_with()
        install.assert_called_once()


def test_fresh_provision_validates_before_writing_active_config(tmp_path, monkeypatch):
    events = []
    validation_configs = []
    process = MagicMock()
    setup = quick_local.QuickLocalSetup(
        health_check=lambda _endpoint: (True, ""),
        progress=lambda event: events.append(event.stage),
    )
    monkeypatch.setattr(
        quick_local,
        "resolve_hermes_vlm_config",
        lambda: {
            "provider": "openai",
            "model": "model-1",
            "api_key": "secret",
            "api_base": "https://llm.example/v1",
        },
    )
    monkeypatch.setattr(setup, "_ensure_openviking_installed", lambda _paths: True)
    monkeypatch.setattr(quick_local, "find_available_port", lambda **_kwargs: 1937)

    def start(endpoint, config_path, hermes_home, server_command):
        paths = quick_local.managed_paths(tmp_path)
        assert endpoint == "http://127.0.0.1:1937"
        assert hermes_home == tmp_path
        assert server_command == paths.server_command
        assert not paths.server_config.exists()
        assert not paths.ovcli_config.exists()
        validation_configs.append(json.loads(config_path.read_text(encoding="utf-8")))
        return process

    monkeypatch.setattr(quick_local, "_start_validation_server", start)
    monkeypatch.setattr(quick_local, "_wait_for_health", lambda *args, **kwargs: True)
    stop = MagicMock(return_value=True)
    monkeypatch.setattr(quick_local, "_stop_process", stop)

    result = setup.provision(
        hermes_home=tmp_path,
        preflight=_preflight(tmp_path),
    )

    paths = result.paths
    assert result.endpoint == "http://127.0.0.1:1937"
    assert result.reused is False
    assert result.server_restart_required is False
    final_config = json.loads(paths.server_config.read_text(encoding="utf-8"))
    assert final_config["storage"]["workspace"] == str(paths.workspace)
    assert final_config["server"]["port"] == 1937
    assert final_config["embedding"]["dense"]["cache_dir"] == str(paths.model_cache)
    assert validation_configs[0]["storage"]["workspace"] != str(paths.workspace)
    assert validation_configs[0]["embedding"]["dense"]["cache_dir"] == str(
        paths.model_cache
    )
    assert json.loads(paths.ovcli_config.read_text(encoding="utf-8")) == {
        "url": "http://127.0.0.1:1937",
        "actor_peer_id": "hermes",
    }
    if os.name != "nt":
        assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.model_cache.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.server_config.stat().st_mode) == 0o600
        assert stat.S_IMODE(paths.ovcli_config.stat().st_mode) == 0o600
    stop.assert_called_once_with(process)
    assert quick_local.QuickLocalStage.PREPARE_EMBEDDING in events
    assert quick_local.QuickLocalStage.VALIDATE in events
    assert events[-2:] == [
        quick_local.QuickLocalStage.WRITE_CONFIG,
        quick_local.QuickLocalStage.COMPLETE,
    ]


def test_failed_validation_stops_child_and_leaves_profile_inactive(
    tmp_path, monkeypatch
):
    process = MagicMock()
    process.poll.return_value = None
    setup = quick_local.QuickLocalSetup(health_check=lambda _endpoint: (False, "down"))
    monkeypatch.setattr(
        quick_local,
        "resolve_hermes_vlm_config",
        lambda: {
            "provider": "openai",
            "model": "model-1",
            "api_key": "secret",
            "api_base": "https://llm.example/v1",
        },
    )
    monkeypatch.setattr(setup, "_ensure_openviking_installed", lambda _paths: True)
    monkeypatch.setattr(quick_local, "find_available_port", lambda **_kwargs: 1933)
    monkeypatch.setattr(
        quick_local,
        "_start_validation_server",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(quick_local, "_wait_for_health", lambda *args, **kwargs: False)
    stop = MagicMock(return_value=True)
    monkeypatch.setattr(quick_local, "_stop_process", stop)

    with pytest.raises(
        quick_local.QuickLocalSetupError, match="did not become reachable"
    ):
        setup.provision(
            hermes_home=tmp_path,
            preflight=_preflight(tmp_path),
        )

    paths = quick_local.managed_paths(tmp_path)
    assert not paths.server_config.exists()
    assert not paths.ovcli_config.exists()
    stop.assert_called_once_with(process)


def test_validation_preserves_primary_error_when_cleanup_also_fails(
    tmp_path, monkeypatch
):
    paths = quick_local.managed_paths(tmp_path)
    paths.root.mkdir(parents=True)
    process = MagicMock()
    process.poll.return_value = None
    setup = quick_local.QuickLocalSetup(health_check=lambda _endpoint: (False, "down"))
    monkeypatch.setattr(
        quick_local,
        "_start_validation_server",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(quick_local, "_wait_for_health", lambda *args, **kwargs: False)
    monkeypatch.setattr(quick_local, "_stop_process", lambda _process: False)

    with pytest.raises(
        quick_local.QuickLocalSetupError,
        match="did not become reachable",
    ) as exc_info:
        setup._validate_generated_config(
            paths=paths,
            endpoint="http://127.0.0.1:1933",
            server_config=quick_local.build_server_config(
                paths,
                {
                    "provider": "openai",
                    "model": "model-1",
                    "api_key": "secret",
                    "api_base": "https://llm.example/v1",
                },
            ),
        )

    assert any(
        "could not be stopped" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_preflight_reuses_only_healthy_profile_with_managed_workspace(tmp_path):
    paths = quick_local.managed_paths(tmp_path)
    paths.root.mkdir(parents=True)
    quick_local.atomic_json_write(
        paths.server_config,
        {"storage": {"workspace": str(paths.workspace)}},
        mode=0o600,
    )
    quick_local.atomic_json_write(
        paths.ovcli_config,
        {"url": "http://127.0.0.1:1938", "actor_peer_id": "hermes"},
        mode=0o600,
    )
    health = MagicMock(return_value=(True, ""))
    setup = quick_local.QuickLocalSetup(health_check=health)

    preflight = setup.preflight(tmp_path)

    assert preflight.reusable_endpoint == "http://127.0.0.1:1938"
    health.assert_called_once_with("http://127.0.0.1:1938")

    quick_local.atomic_json_write(
        paths.server_config,
        {"storage": {"workspace": str(tmp_path / "other-data")}},
        mode=0o600,
    )
    health.reset_mock()
    assert setup.preflight(tmp_path).reusable_endpoint is None
    health.assert_not_called()


def test_preferred_configured_port_is_reused_when_available(monkeypatch):
    bound = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def bind(self, address):
            bound.append(address)

    monkeypatch.setattr(
        quick_local.socket, "socket", lambda *args, **kwargs: FakeSocket()
    )

    port = quick_local.find_available_port(
        preferred_endpoint="http://127.0.0.1:1940",
    )

    assert port == 1940
    assert bound == [("127.0.0.1", 1940)]


def test_validation_stop_force_kills_only_after_graceful_timeout():
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 10), 0]

    assert quick_local._stop_process(process) is True

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()


def test_health_wait_budget_covers_openviking_model_download():
    assert (
        quick_local._HEALTH_TIMEOUT_SECONDS
        > quick_local._MODEL_PREPARATION_TIMEOUT_SECONDS
    )


def test_health_wait_stops_as_soon_as_validation_process_exits():
    process = MagicMock()
    process.poll.return_value = 1
    health = MagicMock(return_value=(False, "down"))

    assert (
        quick_local._wait_for_health(
            "http://127.0.0.1:1933",
            health,
            process=process,
            timeout_seconds=60,
        )
        is False
    )

    health.assert_called_once_with("http://127.0.0.1:1933")


def test_setup_menu_keeps_cloud_custom_paths_and_adds_quick_local(
    tmp_path, monkeypatch
):
    seen = []

    def select(title, options, **_kwargs):
        seen.append((title, options))
        return 1

    quick_setup = MagicMock(return_value=True)
    monkeypatch.setattr(openviking_module, "_run_quick_local_setup", quick_setup)
    config = {"memory": {}}
    provider_config = {}

    result = openviking_module._run_create_profile_setup(
        prompt=MagicMock(),
        select=select,
        cancelled=-1,
        config=config,
        provider_config=provider_config,
        env_path=tmp_path / ".env",
    )

    assert result is True
    assert seen == [
        (
            "  OpenViking connection",
            [
                (
                    "OpenViking Service (VolcEngine Cloud)",
                    "Managed cloud service; API key required",
                ),
                (
                    "Quick Local Setup",
                    "Set up OpenViking with built-in local embeddings",
                ),
                (
                    "Connect to an existing server",
                    "Use a self-managed custom server (Remote/Local)",
                ),
            ],
        )
    ]
    quick_setup.assert_called_once()


def test_quick_local_cli_links_private_profile_and_runtime_config(
    tmp_path, monkeypatch
):
    paths = quick_local.managed_paths(tmp_path)

    class FakeSetup:
        def __init__(self, **_kwargs):
            pass

        def preflight(self, _home):
            return quick_local.QuickLocalPreflight(
                paths=paths,
                reusable_endpoint=None,
            )

        def provision(self, **_kwargs):
            return quick_local.QuickLocalSetupResult(
                paths=paths,
                endpoint="http://127.0.0.1:1933",
                reused=False,
            )

    monkeypatch.setattr(quick_local, "QuickLocalSetup", FakeSetup)
    config = {"memory": {}}
    provider_config = {}

    result = openviking_module._run_quick_local_setup(
        config=config,
        provider_config=provider_config,
        env_path=tmp_path / ".env",
    )

    assert result is True
    assert config["memory"]["provider"] == "openviking"
    assert provider_config == {
        "use_ovcli_config": True,
        "ovcli_config_path": str(paths.ovcli_config),
        "deployment": "quick_local",
        "server_config_path": str(paths.server_config),
        "server_command_path": str(paths.server_command),
    }


def test_quick_local_cli_reports_when_running_server_needs_restart(
    tmp_path, monkeypatch, capsys
):
    paths = quick_local.managed_paths(tmp_path)

    class FakeSetup:
        def __init__(self, **_kwargs):
            pass

        def preflight(self, _home):
            return quick_local.QuickLocalPreflight(
                paths=paths,
                reusable_endpoint="http://127.0.0.1:1933",
            )

        def provision(self, **_kwargs):
            return quick_local.QuickLocalSetupResult(
                paths=paths,
                endpoint="http://127.0.0.1:1933",
                reused=True,
                server_restart_required=True,
            )

    monkeypatch.setattr(quick_local, "QuickLocalSetup", FakeSetup)

    assert (
        openviking_module._run_quick_local_setup(
            config={"memory": {}},
            provider_config={},
            env_path=tmp_path / ".env",
        )
        is True
    )

    output = capsys.readouterr().out
    assert "Quick Local restart required" in output
    assert "updated runtime or Hermes LLM settings" in output
    assert "Stop the Quick Local server at http://127.0.0.1:1933" in output
    assert "Hermes will restart it with the updated runtime and settings" in output
    assert "OpenViking memory is ready" not in output


def test_linking_a_self_managed_profile_clears_quick_local_runtime_settings(tmp_path):
    provider_config = {
        "deployment": "quick_local",
        "server_config_path": "/old/ov.conf",
        "server_command_path": "/old/openviking-server",
    }

    openviking_module._link_ovcli_profile(
        config={"memory": {}},
        provider_config=provider_config,
        env_path=tmp_path / ".env",
        ovcli_path=tmp_path / "ovcli.conf.remote",
    )

    assert "deployment" not in provider_config
    assert "server_config_path" not in provider_config
    assert "server_command_path" not in provider_config


def test_quick_local_profile_is_not_overridden_by_process_environment(
    tmp_path, monkeypatch
):
    paths = quick_local.managed_paths(tmp_path)
    paths.root.mkdir(parents=True)
    quick_local.atomic_json_write(
        paths.ovcli_config,
        {"url": "http://127.0.0.1:1938", "actor_peer_id": "hermes"},
        mode=0o600,
    )
    external_profile = tmp_path / "external-ovcli.conf"
    quick_local.atomic_json_write(
        external_profile,
        {"url": "http://127.0.0.1:1999", "actor_peer_id": "other"},
        mode=0o600,
    )
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(external_profile))
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1998")
    monkeypatch.setenv("OPENVIKING_AGENT", "other")

    settings = openviking_module._resolve_connection_settings({
        "use_ovcli_config": True,
        "ovcli_config_path": str(paths.ovcli_config),
        "deployment": quick_local.DEPLOYMENT,
    })

    assert settings["endpoint"] == "http://127.0.0.1:1938"
    assert settings["agent"] == "hermes"


def test_quick_local_status_hides_only_ignored_environment_overrides(
    tmp_path, monkeypatch
):
    paths = quick_local.managed_paths(tmp_path)
    paths.root.mkdir(parents=True)
    quick_local.atomic_json_write(
        paths.ovcli_config,
        {"url": "http://127.0.0.1:1938", "actor_peer_id": "hermes"},
        mode=0o600,
    )
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1998")

    status = OpenVikingMemoryProvider().get_status_config({
        "use_ovcli_config": True,
        "ovcli_config_path": str(paths.ovcli_config),
        "deployment": quick_local.DEPLOYMENT,
    })

    assert status["endpoint"] == "http://127.0.0.1:1938"
    assert "env_overrides" not in status

    self_managed_status = OpenVikingMemoryProvider().get_status_config({
        "use_ovcli_config": True,
        "ovcli_config_path": str(paths.ovcli_config),
    })

    assert self_managed_status["endpoint"] == "http://127.0.0.1:1998"
    assert self_managed_status["env_overrides"] == "OPENVIKING_ENDPOINT"


def test_incomplete_quick_local_profile_does_not_fall_back_to_environment(
    tmp_path, monkeypatch
):
    external_profile = tmp_path / "external-ovcli.conf"
    quick_local.atomic_json_write(
        external_profile,
        {"url": "http://127.0.0.1:1999", "actor_peer_id": "other"},
        mode=0o600,
    )
    monkeypatch.setenv("OPENVIKING_CLI_CONFIG_FILE", str(external_profile))

    with pytest.raises(
        openviking_module._OpenVikingEndpointError,
        match="profile link is missing",
    ):
        openviking_module._resolve_connection_settings({
            "use_ovcli_config": True,
            "deployment": quick_local.DEPLOYMENT,
        })


def test_runtime_start_passes_private_config_only_for_quick_local(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "ov.conf"
    config_path.write_text("{}", encoding="utf-8")
    server_command = tmp_path / "openviking-server"
    server_command.write_text("", encoding="utf-8")
    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        openviking_module,
        "_local_openviking_port_is_open",
        lambda _host, _port: False,
    )
    monkeypatch.setattr(openviking_module.subprocess, "Popen", popen)

    state, _message = openviking_module._start_local_openviking_server(
        "http://127.0.0.1:1939",
        config_path=config_path,
        server_command_path=server_command,
    )

    assert state == openviking_module._LOCAL_SERVER_STARTED
    assert popen.call_args.args[0] == [
        str(server_command),
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        "1939",
    ]


def test_runtime_start_refuses_missing_quick_local_config(tmp_path, monkeypatch):
    server_command = tmp_path / "openviking-server"
    server_command.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        openviking_module,
        "_local_openviking_port_is_open",
        lambda _host, _port: False,
    )
    popen = MagicMock(side_effect=AssertionError("missing config must not spawn"))
    monkeypatch.setattr(openviking_module.subprocess, "Popen", popen)

    state, message = openviking_module._start_local_openviking_server(
        "http://127.0.0.1:1939",
        config_path=tmp_path / "missing-ov.conf",
        server_command_path=server_command,
    )

    assert state == openviking_module._LOCAL_SERVER_FAILED
    assert "server config was not found" in message
    popen.assert_not_called()


def test_runtime_unreachable_uses_quick_local_config_marker(tmp_path, monkeypatch):
    config_path = tmp_path / "ov.conf"
    start = MagicMock(
        return_value=(openviking_module._LOCAL_SERVER_FAILED, "expected test failure")
    )
    monkeypatch.setattr(
        openviking_module,
        "_load_hermes_openviking_config",
        lambda: {
            "deployment": "quick_local",
            "server_config_path": str(config_path),
            "server_command_path": "/bin/ov",
        },
    )
    monkeypatch.setattr(openviking_module, "_start_local_openviking_server", start)
    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1933"

    provider._handle_runtime_openviking_unreachable()

    start.assert_called_once_with(
        "http://127.0.0.1:1933",
        config_path=config_path,
        server_command_path=Path("/bin/ov"),
    )


def test_runtime_does_not_fall_back_when_quick_local_markers_are_incomplete(
    monkeypatch,
):
    start = MagicMock(side_effect=AssertionError("must not start a PATH server"))
    monkeypatch.setattr(
        openviking_module,
        "_load_hermes_openviking_config",
        lambda: {"deployment": quick_local.DEPLOYMENT},
    )
    monkeypatch.setattr(openviking_module, "_start_local_openviking_server", start)
    warnings = []
    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1933"

    provider._handle_runtime_openviking_unreachable(warning_callback=warnings.append)

    start.assert_not_called()
    assert len(warnings) == 1
    assert "private server configuration is incomplete" in warnings[0]
