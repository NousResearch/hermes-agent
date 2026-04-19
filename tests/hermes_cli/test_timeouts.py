from __future__ import annotations

import textwrap

from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def test_model_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: 30
            models:
              claude-opus-4.6:
                timeout_seconds: 120
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") == 120.0


def test_provider_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ollama-local:
            request_timeout_seconds: 300
        """,
    )

    assert get_provider_request_timeout("ollama-local", "qwen3:32b") == 300.0


def test_model_stale_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: 600
            models:
              gpt-5.4:
                stale_timeout_seconds: 1800
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") == 1800.0


def test_provider_stale_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: 900
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") == 900.0


def test_missing_timeout_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            models:
              claude-opus-4.6:
                context_length: 200000
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("missing-provider", "claude-opus-4.6") is None


def test_invalid_timeout_values_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: "fast"
            models:
              claude-opus-4.6:
                timeout_seconds: -5
          ollama-local:
            request_timeout_seconds: -1
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("anthropic", "claude-sonnet-4.5") is None
    assert get_provider_request_timeout("ollama-local") is None


def test_invalid_stale_timeout_values_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: "slow"
            models:
              gpt-5.4:
                stale_timeout_seconds: -1
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") is None
    assert get_provider_stale_timeout("openai-codex", "gpt-5.5") is None


def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the 900s default read timeout."""
    pytest = __import__("pytest")
    anthropic = pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Default stays at 900s; custom overrides; invalid falls back to default
    assert c_default.timeout.read == 900.0
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == 900.0
    # Connect timeout always stays at 10s regardless
    assert c_default.timeout.connect == 10.0
    assert c_custom.timeout.connect == 10.0


def test_resolved_api_call_timeout_priority(monkeypatch, tmp_path):
    """AIAgent._resolved_api_call_timeout() honors config > env > default priority."""
    # Isolate HERMES_HOME
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    # Case A: config wins over env var
    _write_config(tmp_path, """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
        """)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # Per-model override wins
    assert agent._resolved_api_call_timeout() == 42.0

    # Provider-level (different model, no per-model override)
    agent.model = "some/other-model"
    assert agent._resolved_api_call_timeout() == 77.0

    # Case B: no config → env wins
    _write_config(tmp_path, "")
    # Clear the cached config load
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="some/model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_timeout() == 999.0

    # Case C: no config, no env → 1800.0 default
    monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_timeout() == 1800.0


def test_resolved_api_call_stale_timeout_priority(monkeypatch, tmp_path):
    """AIAgent stale timeout honors config > env > default priority."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    _write_config(tmp_path, """\
        providers:
          openai-codex:
            stale_timeout_seconds: 600
            models:
              gpt-5.4:
                stale_timeout_seconds: 1800
        """)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="gpt-5.4",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent._resolved_api_call_stale_timeout_base() == (1800.0, False)

    agent.model = "gpt-5.5"
    assert agent._resolved_api_call_stale_timeout_base() == (600.0, False)

    _write_config(tmp_path, "")
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="gpt-5.4",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_stale_timeout_base() == (999.0, False)

    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_stale_timeout_base() == (300.0, True)


def test_default_non_stream_stale_timeout_auto_disables_for_local_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    from run_agent import AIAgent
    agent = AIAgent(
        model="qwen3:32b",
        provider="ollama-local",
        api_key="sk-dummy",
        base_url="http://127.0.0.1:11434/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert agent._compute_non_stream_stale_timeout([]) == float("inf")


def test_explicit_non_stream_stale_timeout_is_honored_for_local_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "300")

    from run_agent import AIAgent
    agent = AIAgent(
        model="qwen3:32b",
        provider="ollama-local",
        api_key="sk-dummy",
        base_url="http://127.0.0.1:11434/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert agent._compute_non_stream_stale_timeout([]) == 300.0
