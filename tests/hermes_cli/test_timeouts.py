from __future__ import annotations

import textwrap



def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")










def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the default read timeout."""
    pytest = __import__("pytest")
    pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Custom overrides the read timeout; invalid falls back to the default;
    # the connect timeout is unaffected by the override.
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == c_default.timeout.read != 45.0
    assert c_custom.timeout.connect == c_default.timeout.connect


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




_NAMED_CUSTOM_CONFIG = """\
    providers:
      my-vllm:
        api: http://127.0.0.1:8001/v1
        request_timeout_seconds: 7200
        stale_timeout_seconds: 1800
        models:
          big-model:
            timeout_seconds: 9000
      backup-vllm:
        api: http://127.0.0.1:8002/v1
        request_timeout_seconds: 600
    """


def _isolated_home(monkeypatch, tmp_path, config: str) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    for var in ("HERMES_API_TIMEOUT", "HERMES_API_CALL_STALE_TIMEOUT", "HERMES_STREAM_STALE_TIMEOUT",
                "HERMES_STREAM_READ_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    _write_config(tmp_path, config)


def _named_custom_agent(requested_provider="my-vllm", model="served-model", **kwargs):
    from run_agent import AIAgent

    return AIAgent(
        model=model, provider="custom", requested_provider=requested_provider,
        api_key="sk-dummy", base_url="http://127.0.0.1:8001/v1",
        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli", **kwargs,
    )


def test_timeout_provider_id_maps_named_custom_providers_only():
    from hermes_cli.timeouts import timeout_provider_id

    assert timeout_provider_id("custom", "lab-box") == "lab-box"
    assert timeout_provider_id("custom", "custom:lab-box") == "lab-box"
    assert timeout_provider_id("custom", " Lab-Box ") == "Lab-Box"
    assert timeout_provider_id("custom:lab-box", "custom:lab-box") == "lab-box"
    # Bare ``custom`` (OPENAI_BASE_URL endpoints, a ``providers.custom`` entry) keeps its own id.
    assert timeout_provider_id("custom", None) == "custom"
    assert timeout_provider_id("custom", "") == "custom"
    assert timeout_provider_id("custom", "custom") == "custom"
    assert timeout_provider_id("custom", "custom:") == "custom"
    # Every other provider is looked up exactly as before, whatever was requested.
    assert timeout_provider_id("openrouter", "lab-box") == "openrouter"
    assert timeout_provider_id("openrouter", None) == "openrouter"
    assert timeout_provider_id(None, "lab-box") is None


def test_named_custom_provider_timeouts_apply_to_the_agent(monkeypatch, tmp_path):
    """A named ``providers.<id>`` entry resolves to runtime provider ``custom`` with
    ``requested_provider=<id>``; its configured timeouts used to be silently ignored because every
    lookup keyed on ``custom``."""
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    agent = _named_custom_agent()

    assert agent._resolved_api_call_timeout() == 7200.0
    assert agent._resolved_api_call_stale_timeout_base() == (1800.0, False)
    assert agent._stale_timeout_is_explicit() is True
    # The configured request timeout reaches the OpenAI client built at init.
    assert agent._client_kwargs.get("timeout") == 7200.0
    # Per-model override on the named entry wins, like it does for built-in providers.
    agent.model = "big-model"
    assert agent._resolved_api_call_timeout() == 9000.0


def test_named_custom_lookup_mirrors_runtime_name_matching(monkeypatch, tmp_path):
    """Runtime resolution matches names case-insensitively, by ``custom:<key>`` slug and by display
    name (and ``AIAgent`` lowercases ``requested_provider``); the timeout lookup must succeed
    wherever resolution does."""
    _isolated_home(monkeypatch, tmp_path, """\
        providers:
          My-VLLM:
            name: Lab Box
            api: http://127.0.0.1:8001/v1
            request_timeout_seconds: 4242
        """)
    for requested in ("My-VLLM", "my-vllm", "custom:my-vllm", "Lab Box", "lab-box"):
        agent = _named_custom_agent(requested_provider=requested)
        assert agent._resolved_api_call_timeout() == 4242.0, requested


def test_named_custom_lookup_keeps_existing_custom_block_as_fallback(monkeypatch, tmp_path):
    """No regression for configs that relied on a ``providers.custom`` block: a named id without
    its own value still falls back to it, and bare ``custom`` behaves exactly as before."""
    _isolated_home(monkeypatch, tmp_path, """\
        providers:
          custom:
            request_timeout_seconds: 333
            stale_timeout_seconds: 444
          my-vllm:
            api: http://127.0.0.1:8001/v1
            stale_timeout_seconds: 1800
        """)
    named = _named_custom_agent()
    assert named._resolved_api_call_timeout() == 333.0  # not configured on my-vllm
    assert named._resolved_api_call_stale_timeout_base() == (1800.0, False)  # my-vllm wins
    unknown = _named_custom_agent(requested_provider="ollama")
    assert unknown._resolved_api_call_timeout() == 333.0
    assert unknown._resolved_api_call_stale_timeout_base() == (444.0, False)
    bare = _named_custom_agent(requested_provider="custom")
    assert bare._resolved_api_call_timeout() == 333.0


def test_non_custom_provider_ignores_requested_named_entry(monkeypatch, tmp_path):
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout

    assert get_provider_request_timeout("openrouter", "m", requested_provider="my-vllm") is None
    assert get_provider_stale_timeout("openrouter", "m", requested_provider="my-vllm") is None
    assert get_provider_request_timeout("custom", "m", requested_provider="my-vllm") == 7200.0
    assert get_provider_request_timeout("custom", "m") is None


def test_named_custom_stream_timeouts_use_named_entry(monkeypatch, tmp_path):
    """Streaming chat_completions socket timeouts and the stream stale detector read the same
    named entry (the streaming path is the default for chat)."""
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    from agent import chat_completion_helpers as cch

    agent = _named_custom_agent()
    call = object.__new__(cch._StreamingCall)
    call.agent = agent
    call._stream_stale_timeout = None
    assert call._stream_timeouts() == (7200.0, 7200.0, 60.0)
    assert cch._configured_stale_base(agent) == 1800.0
    assert cch._cloud_stale_timeout_for(agent, {}) == 1800.0


def test_fallback_to_named_custom_uses_fallback_entry(monkeypatch, tmp_path):
    """Fallback activation re-keys provider/requested_provider to the fallback entry: its own
    timeouts apply, never the primary's named values."""
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    from unittest.mock import MagicMock, patch

    fb_client = MagicMock()
    fb_client.base_url = "http://127.0.0.1:8002/v1"
    fb_client.api_key = "fb-key"
    fb_client._custom_headers = None
    agent = _named_custom_agent(fallback_model=[{"provider": "backup-vllm", "model": "fb-model"}])
    assert agent._resolved_api_call_timeout() == 7200.0
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, "fb-model")), \
            patch.object(agent, "_replace_primary_openai_client"):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.requested_provider) == ("backup-vllm", "backup-vllm")
    assert agent._client_kwargs.get("timeout") == 600.0
    assert agent._resolved_api_call_timeout() == 600.0
    # backup-vllm configures no stale timeout: the primary's 1800 must not leak through.
    assert agent._stale_timeout_is_explicit() is False


def test_fallback_to_builtin_does_not_inherit_named_timeouts(monkeypatch, tmp_path):
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    from unittest.mock import MagicMock, patch

    fb_client = MagicMock()
    fb_client.base_url = "https://openrouter.ai/api/v1"
    fb_client.api_key = "fb-key"
    fb_client._custom_headers = None
    agent = _named_custom_agent(fallback_model=[{"provider": "openrouter", "model": "openai/gpt-4o-mini"}])
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, "openai/gpt-4o-mini")):
        assert agent._try_activate_fallback() is True
    assert agent.provider == "openrouter"
    assert "timeout" not in agent._client_kwargs
    assert agent._resolved_api_call_timeout() == 1800.0
    assert agent._stale_timeout_is_explicit() is False


def test_switch_model_to_named_custom_slug_uses_its_entry(monkeypatch, tmp_path):
    """``/model`` can switch to a ``custom:<id>`` slug; switch_model sets provider and
    requested_provider to it, and the rebuilt client must carry that entry's timeout."""
    _isolated_home(monkeypatch, tmp_path, _NAMED_CUSTOM_CONFIG)
    agent = _named_custom_agent()
    agent.switch_model("fb-model", "custom:backup-vllm", api_key="sk-dummy",
                       base_url="http://127.0.0.1:8002/v1", api_mode="chat_completions")
    assert agent.provider == "custom:backup-vllm"
    assert agent._client_kwargs.get("timeout") == 600.0
    assert agent._resolved_api_call_timeout() == 600.0
    assert agent._stale_timeout_is_explicit() is False
