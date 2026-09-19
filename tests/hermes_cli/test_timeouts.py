from __future__ import annotations

import textwrap

from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")










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






# ---------------------------------------------------------------------------
# Named custom providers resolve at runtime as provider="custom" — the config
# key ("mlx-lm") is NOT what AIAgent.provider carries.  Timeout lookups must
# therefore fall back to matching the provider entry by base_url, exactly like
# get_custom_provider_context_length() does for context_length (#15779).
# Observed 2026-08-21: providers.mlx-lm.stale_timeout_seconds silently ignored.
# ---------------------------------------------------------------------------

_NAMED_CUSTOM_CONFIG = """\
    providers:
      mlx-lm:
        name: mlx-lm (local, fast)
        base_url: http://127.0.0.1:8001/v1
        api_key: local
        request_timeout_seconds: 1200
        stale_timeout_seconds: 900
        models:
          slow-model:
            stale_timeout_seconds: 1500
            timeout_seconds: 1600
    """


def _isolate(monkeypatch, tmp_path, body):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, body)
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    return importlib.reload(to_mod)


def test_named_custom_provider_timeouts_resolve_by_base_url(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    url = "http://127.0.0.1:8001/v1"
    assert to.get_provider_stale_timeout("custom", "any-model", base_url=url) == 900.0
    assert to.get_provider_request_timeout("custom", "any-model", base_url=url) == 1200.0


def test_named_custom_provider_per_model_override_by_base_url(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    url = "http://127.0.0.1:8001/v1"
    assert to.get_provider_stale_timeout("custom", "slow-model", base_url=url) == 1500.0
    assert to.get_provider_request_timeout("custom", "slow-model", base_url=url) == 1600.0


def test_base_url_match_tolerates_trailing_slash_and_case(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    assert to.get_provider_stale_timeout("custom", "m", base_url="HTTP://127.0.0.1:8001/v1/") == 900.0


def test_unmatched_base_url_yields_none(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:9999/v1") is None
    assert to.get_provider_stale_timeout("custom", "m") is None  # no base_url → no guess


def test_explicit_provider_key_still_wins_over_base_url(monkeypatch, tmp_path):
    # Same base_url on two entries: an explicit provider id must be honored,
    # not overridden by a base_url scan.
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          mlx-lm:
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 900
          other:
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 5
        """)
    url = "http://127.0.0.1:8001/v1"
    assert to.get_provider_stale_timeout("other", "m", base_url=url) == 5.0
    assert to.get_provider_stale_timeout("mlx-lm", "m", base_url=url) == 900.0


def test_literal_providers_custom_entry_does_not_shadow_active_route(monkeypatch, tmp_path):
    # A user may literally name an entry "custom"; at runtime provider_id is also
    # "custom" for every named provider, so the ACTIVE URL must win over the key.
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          custom:
            base_url: http://127.0.0.1:5555/v1
            stale_timeout_seconds: 5
          mlx-lm:
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 900
        """)
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:8001/v1") == 900.0
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:5555/v1") == 5.0
    # No base_url → the literal key is still honored (unchanged behavior).
    assert to.get_provider_stale_timeout("custom", "m") == 5.0


def test_disabled_entries_are_skipped(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          old-mlx:
            enabled: false
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 5
          mlx-lm:
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 900
        """)
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:8001/v1") == 900.0


def test_disabled_keyed_entry_is_not_resolved_by_key(monkeypatch, tmp_path):
    # A disabled entry must be invisible to BOTH resolution paths — the URL
    # scan skipping it must not let the keyed fallback return it anyway.
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          openrouter:
            enabled: false
            stale_timeout_seconds: 5
          custom:
            enabled: false
            base_url: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 7
        """)
    assert to.get_provider_stale_timeout("openrouter", "m") is None
    assert to.get_provider_stale_timeout("custom", "m") is None
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:8001/v1") is None


def test_url_key_aliases_are_honored(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          via-api-key:
            api: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 900
          via-url-key:
            url: http://127.0.0.1:8002/v1
            request_timeout_seconds: 77
        """)
    assert to.get_provider_stale_timeout("custom", "m", base_url="http://127.0.0.1:8001/v1") == 900.0
    assert to.get_provider_request_timeout("custom", "m", base_url="http://127.0.0.1:8002/v1") == 77.0


def test_builtin_provider_never_inherits_from_custom_entry_sharing_url(monkeypatch, tmp_path):
    to = _isolate(monkeypatch, tmp_path, """\
        providers:
          my-proxy:
            base_url: https://openrouter.ai/api/v1
            stale_timeout_seconds: 5
        """)
    # provider_id is a built-in, not custom-like → key-only resolution → None.
    assert to.get_provider_stale_timeout("openrouter", "m", base_url="https://openrouter.ai/api/v1") is None


def test_custom_profile_aliases_resolve_by_url(monkeypatch, tmp_path):
    # ``ollama`` / ``vllm`` are registry aliases of the custom profile.
    to = _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    assert to.get_provider_stale_timeout("vllm", "m", base_url="http://127.0.0.1:8001/v1") == 900.0


def test_agent_e2e_named_custom_provider_stale_timeout(monkeypatch, tmp_path):
    """Real AIAgent constructed as the runtime does for a named custom provider
    (provider='custom' + the entry's base_url): config must beat both the
    reasoning-model floor and the env var."""
    _isolate(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "123")
    import importlib, run_agent as ra_mod
    importlib.reload(ra_mod)
    agent = ra_mod.AIAgent(
        model="/models/Qwen3.8-27B-MLX-4bit",  # reasoning floor alone would give 180
        provider="custom",
        api_key="local",
        base_url="http://127.0.0.1:8001/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent.provider == "custom"
    assert agent._resolved_api_call_stale_timeout_base() == (900.0, False)
    assert agent._resolved_api_call_timeout() == 1200.0
