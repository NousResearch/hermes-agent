"""Regression tests for ``hermes -z`` isolation env gates."""

from __future__ import annotations

import sys
import types

from hermes_cli import config as config_mod
from hermes_cli import oneshot


SENTINEL_MODEL = "sentinel-user-config-model-73191"
SENTINEL_SECRET = "sentinel-user-config-secret-73191"


class CapturingAgent:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def run_conversation(self, prompt):
        return {
            "final_response": f"model={self.kwargs.get('model')}",
            "completed": True,
            "failed": False,
            "model": self.kwargs.get("model"),
            "provider": self.kwargs.get("provider"),
        }

    def shutdown_memory_provider(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _write_user_config(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "\n".join([
            "model:",
            f"  default: {SENTINEL_MODEL}",
            "  provider: openrouter",
            "toolsets:",
            "  cli:",
            "    - shell",
            "agent:",
            f"  system_prompt: {SENTINEL_SECRET}",
            "fallback:",
            "  enabled: true",
            "  chain:",
            "    - provider: openrouter",
            f"      model: {SENTINEL_MODEL}-fallback",
            "",
        ]),
        encoding="utf-8",
    )


def _install_fakes(monkeypatch):
    CapturingAgent.calls.clear()
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)

    import hermes_cli.runtime_provider as runtime_provider
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "fake-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or "openrouter",
            "requested_provider": kwargs.get("requested") or "openrouter",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        tools_config, "_get_platform_tools", lambda cfg, platform: {"files"}
    )


def _clear_config_caches():
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def test_oneshot_ignore_rules_env_passes_skip_flags(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    _clear_config_caches()

    oneshot._run_agent(
        "hello",
        model="explicit/model",
        provider="openrouter",
        toolsets=["files"],
        use_config_toolsets=False,
    )

    kwargs = CapturingAgent.calls[-1]
    assert kwargs["skip_context_files"] is True
    assert kwargs["skip_memory"] is True


def test_oneshot_without_ignore_rules_passes_false_skip_flags(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    _clear_config_caches()

    oneshot._run_agent(
        "hello",
        model="explicit/model",
        provider="openrouter",
        toolsets=["files"],
        use_config_toolsets=False,
    )

    kwargs = CapturingAgent.calls[-1]
    assert kwargs["skip_context_files"] is False
    assert kwargs["skip_memory"] is False


def test_oneshot_ignore_user_config_skips_cached_user_config(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    _write_user_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    _clear_config_caches()

    assert config_mod.load_config()["model"]["default"] == SENTINEL_MODEL

    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    response, _result = oneshot._run_agent("hello")

    kwargs = CapturingAgent.calls[-1]
    assert kwargs["model"] != SENTINEL_MODEL
    assert SENTINEL_MODEL not in response
    assert all(SENTINEL_MODEL not in repr(call) for call in CapturingAgent.calls)


def test_oneshot_explicit_runtime_options_survive_isolation(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    _write_user_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    _clear_config_caches()

    oneshot._run_agent(
        "hello",
        model="explicit/model",
        provider="openrouter",
        toolsets="files,shell",
        use_config_toolsets=False,
    )

    kwargs = CapturingAgent.calls[-1]
    assert kwargs["model"] == "explicit/model"
    assert kwargs["provider"] == "openrouter"
    assert kwargs["enabled_toolsets"] == ["files", "shell"]
    assert kwargs["skip_context_files"] is True
    assert kwargs["skip_memory"] is True


def test_oneshot_isolation_does_not_leak_user_config_sentinel(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_fakes(monkeypatch)
    _write_user_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    _clear_config_caches()

    rc = oneshot.run_oneshot("hello")

    captured = capsys.readouterr()
    assert rc == 0
    assert SENTINEL_MODEL not in captured.out
    assert SENTINEL_MODEL not in captured.err
    assert SENTINEL_SECRET not in captured.out
    assert SENTINEL_SECRET not in captured.err
