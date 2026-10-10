"""Regression tests for the gateway /model picker on a bare "custom" provider storing an empty
session api_key, which then made the NEXT turn build a keyless agent.

Chain:
  1. ``_ModelSwitchContext.read_config`` fills ``current_api_key`` only from an existing session
     override; on the FIRST switch after boot there is none, so it is ``""``.
  2. The picker always passes ``explicit_provider="custom"`` for a bare-custom row, which routes
     through ``_creds_for_switched_provider``'s bare-custom branch even though the provider did
     not change, carrying the empty key straight into ``ModelSwitchResult.api_key``.
  3. ``GatewayRunner._apply_session_model_override`` applied every override field that
     ``is not None``, so ``""`` replaced the api_key ``_resolve_runtime_agent_kwargs_for_provider``
     had already resolved for the turn.

A blank override field keeps the resolved runtime value ONLY when the runtime was resolved for
the override's own provider and the override does not name another endpoint; anywhere else (a
different host, a channel override's provider) the resolved key belongs to another route, so the
override applies exactly as before and that key never travels to the override's endpoint.

This file covers step 3 directly (unit) and the whole chain end-to-end (AIAgent must build with
the real key). ``tests/hermes_cli/test_model_switch_bare_custom_empty_session_key.py`` covers
step 2.
"""

from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner


def test_apply_session_model_override_empty_api_key_and_base_url_keep_resolved_runtime():
    """Unit test for the run_agent_cache.py fix: an override recording ``api_key=""`` /
    ``base_url=""`` (the bare-custom first-switch shape) must not clobber values the runtime
    resolution already put in ``runtime_kwargs``."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "openai/gpt-4.1-mini",
            "provider": "custom",
            "api_key": "",
            "base_url": "   ",  # whitespace-only counts as absent too
            "api_mode": "chat_completions",
        },
    }
    runtime_kwargs = {
        "api_key": "sk-already-resolved",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "custom",
    }

    model, runtime = runner._apply_session_model_override("sess-1", "old-model", runtime_kwargs)

    assert model == "openai/gpt-4.1-mini"
    assert runtime["api_key"] == "sk-already-resolved"
    assert runtime["base_url"] == "https://openrouter.ai/api/v1"
    # Non-blank fields still apply normally.
    assert runtime["api_mode"] == "chat_completions"


def test_apply_session_model_override_non_empty_api_key_still_applies():
    """Sanity: a genuine api_key/base_url on the override still wins, as before the fix."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "m2",
            "provider": "custom:other",
            "api_key": "sk-new-endpoint",
            "base_url": "https://other.example.test/v1",
        },
    }
    runtime_kwargs = {"api_key": "sk-old", "base_url": "https://old.example.test/v1"}

    model, runtime = runner._apply_session_model_override("sess-1", "old-model", runtime_kwargs)

    assert model == "m2"
    assert runtime["api_key"] == "sk-new-endpoint"
    assert runtime["base_url"] == "https://other.example.test/v1"


def test_resolve_session_agent_runtime_falls_through_to_resolved_key_on_empty_override(monkeypatch):
    """Integration through ``_resolve_session_agent_runtime`` (gateway/run_turn.py): the fast path
    bails on an empty override api_key (already falsy in Python), the slow path re-resolves via
    ``_resolve_runtime_agent_kwargs_for_provider``, and the SAME re-application
    (``_apply_session_model_override``) that followed used to blank the freshly resolved key back
    out. Fixing run_agent_cache.py alone makes this path consistent too."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "openai/gpt-4.1-mini",
            "provider": "custom",
            "api_key": "",
            "base_url": "",
        },
    }
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda _uc=None: "default-model")
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider, target_model=None: {
            "api_key": "sk-resolved-from-config",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "custom",
            "api_mode": "chat_completions",
        },
    )

    model, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert model == "openai/gpt-4.1-mini"
    assert runtime["api_key"] == "sk-resolved-from-config"
    assert runtime["base_url"] == "https://openrouter.ai/api/v1"


@pytest.fixture
def repro_home(tmp_path, monkeypatch):
    """The exact reproduction config: bare-custom provider fronting openrouter.ai, api_key
    resolved from an env-var template, matching the picker's first /model switch after boot."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: openai/gpt-4o-mini\n"
        "  provider: custom\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  api_mode: chat_completions\n"
        "  api_key: ${OPENAI_API_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return home


_MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}


def test_end_to_end_picker_first_switch_builds_a_keyed_agent(repro_home):
    """Mirrors the reported reproduction: the picker's own "is_current" bare-custom slug, the
    real ``switch_model`` pipeline, the real ``_apply_session_model_override``, and a real
    ``AIAgent`` construction. Before the fix this raised "No LLM provider configured" because the
    resolved api_key was overwritten with the empty session key."""
    from gateway.run import GatewayRunner, _resolve_runtime_agent_kwargs_for_provider
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import switch_model
    from hermes_cli.model_switch_providers import list_picker_providers
    from run_agent import AIAgent

    ctx = _ModelSwitchContext(
        session_key="s", source=None, config_path=repro_home / "config.yaml", persist_global=False)
    ctx.read_config()
    assert ctx.current_api_key == "", "sanity: the first switch really has no session override yet"

    providers = list_picker_providers(
        current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
        current_model=ctx.current_model, user_providers=ctx.user_provs, custom_providers=ctx.custom_provs,
        max_models=5, non_blocking_catalogs=True, probe_custom_providers=False,
        probe_current_custom_provider=False,
    )
    current_rows = [p for p in providers if p.get("is_current")]
    assert current_rows, "the current bare-custom endpoint must appear as a picker row"
    button_slug = current_rows[0]["slug"]

    with patch("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION), \
         patch("hermes_cli.model_switch.get_model_info", lambda *a, **k: None), \
         patch("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None):
        result = switch_model(
            raw_input="openai/gpt-4.1-mini", explicit_provider=button_slug,
            current_provider=ctx.current_provider, current_model=ctx.current_model,
            current_base_url=ctx.current_base_url, current_api_key=ctx.current_api_key,
            user_providers=ctx.user_provs, custom_providers=ctx.custom_provs,
        )
    assert result.success, result.error_message

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "s": {
            "model": result.new_model, "provider": result.target_provider, "api_key": result.api_key,
            "base_url": result.base_url, "api_mode": result.api_mode,
        },
    }
    base_runtime = _resolve_runtime_agent_kwargs_for_provider(result.target_provider, target_model=result.new_model)
    model, kwargs = runner._apply_session_model_override("s", ctx.current_model, base_runtime)
    for k in ("model", "request_overrides", "capabilities"):
        kwargs.pop(k, None)

    agent = AIAgent(model=model, quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)
    assert agent.client.api_key == "sk-test-not-real"


@pytest.mark.parametrize(
    "override_provider, override_base_url, runtime_provider, runtime_base_url",
    [
        # Same provider, the override names a DIFFERENT endpoint than the one resolved for.
        ("custom", "https://session.example.test/v1", "custom", "https://other.example.test/v1"),
        # A different provider resolved the runtime (e.g. a channel override's), blank override URL.
        ("custom", "", "openrouter", "https://openrouter.ai/api/v1"),
        # A different provider resolved the runtime, override naming its own endpoint.
        ("custom", "https://session.example.test/v1", "openrouter", "https://openrouter.ai/api/v1"),
    ],
)
def test_blank_override_key_for_another_route_applies_as_before(
        override_provider, override_base_url, runtime_provider, runtime_base_url):
    """The resolved key belongs to another route here: the override applies exactly as it always
    did (blank key and all), so that key is never paired with the override's endpoint."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {"model": "m2", "provider": override_provider, "api_key": "", "base_url": override_base_url},
    }
    runtime_kwargs = {"provider": runtime_provider, "api_key": "sk-OTHER", "base_url": runtime_base_url}

    _, runtime = runner._apply_session_model_override("sess-1", "old-model", runtime_kwargs)

    assert runtime["api_key"] == ""
    assert runtime["base_url"] == override_base_url
    assert runtime["provider"] == override_provider


def test_blank_override_key_same_endpoint_modulo_trailing_slash_keeps_resolved_key():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {"model": "m2", "provider": "custom", "api_key": "", "base_url": "https://ep.example.test/v1/"},
    }
    runtime_kwargs = {"provider": "custom", "api_key": "sk-ep", "base_url": "https://ep.example.test/v1"}

    _, runtime = runner._apply_session_model_override("sess-1", "old-model", runtime_kwargs)

    assert runtime["api_key"] == "sk-ep"


def test_channel_override_provider_key_never_reaches_the_session_endpoint():
    """gateway/run_turn.py resolves runtime_kwargs for a channel override's provider before the
    session override is layered on. That provider's key must not be sent to the session's own
    (different) endpoint: the blank session key applies as before."""
    from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "m2", "provider": "custom", "api_key": "",
            "base_url": "https://session.example.test/v1",
        },
    }
    runner.config = GatewayConfig(platforms={
        Platform.DISCORD: PlatformConfig(
            enabled=True, channel_overrides={"chan_1": ChannelOverride(provider="openrouter")}),
    })
    source = SessionSource(platform=Platform.DISCORD, chat_id="chan_1", user_id="u1")

    def _for_provider(provider, target_model=None):
        if provider == "openrouter":
            return {"provider": "openrouter", "api_key": "sk-or-channel",
                    "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"}
        return {"provider": "custom", "api_key": "", "base_url": "https://session.example.test/v1",
                "api_mode": "chat_completions"}

    with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=_for_provider):
        _, runtime = runner._resolve_session_agent_runtime(source=source, session_key="sess-1")

    assert runtime["base_url"] == "https://session.example.test/v1"
    assert runtime["api_key"] != "sk-or-channel"


def test_whitespace_only_override_key_takes_the_resolving_path(monkeypatch):
    """The fast path and _apply_session_model_override agree: a whitespace-only key is no key, so
    the turn resolves credentials for the override's provider instead of sending "   "."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {"model": "m2", "provider": "custom", "api_key": "   ",
                   "base_url": "https://ep.example.test/v1"},
    }
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda _uc=None: "default-model")
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider, target_model=None: {
            "provider": "custom", "api_key": "sk-ep", "base_url": "https://ep.example.test/v1",
            "api_mode": "chat_completions",
        },
    )

    _, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert runtime["api_key"] == "sk-ep"


@pytest.fixture
def other_endpoint_home(tmp_path, monkeypatch):
    """Config points bare custom at other.example.test with its own literal key."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: some-model\n"
        "  provider: custom\n"
        "  base_url: https://other.example.test/v1\n"
        "  api_mode: chat_completions\n"
        "  api_key: sk-OTHER\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
                "CUSTOM_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return home


def test_end_to_end_other_endpoints_key_is_not_sent_to_the_session_endpoint(other_endpoint_home):
    """A session override on custom @ session.example.test with no key, while config's bare
    custom resolves to other.example.test with sk-OTHER: the agent must never be built for
    session.example.test carrying sk-OTHER."""
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    from run_agent import AIAgent

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "s": {"model": "some-model-2", "provider": "custom", "api_key": "",
              "base_url": "https://session.example.test/v1", "api_mode": "chat_completions"},
    }
    base_runtime = _resolve_runtime_agent_kwargs_for_provider("custom", target_model="some-model-2")
    assert base_runtime["api_key"] == "sk-OTHER", "sanity: config resolves the other endpoint's key"
    model, kwargs = runner._apply_session_model_override("s", "some-model", base_runtime)
    for k in ("model", "request_overrides", "capabilities"):
        kwargs.pop(k, None)

    assert kwargs["base_url"] == "https://session.example.test/v1"
    assert kwargs["api_key"] == ""
    try:
        agent = AIAgent(model=model, quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)
    except Exception:
        return  # no agent at all: nothing was sent anywhere
    assert not (
        "session.example.test" in str(agent.client.base_url) and agent.client.api_key == "sk-OTHER"
    )
