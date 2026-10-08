"""Behavior contract for OpenCode Zen's explicitly keyless model route."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.model_switch import switch_model
from hermes_cli.runtime_provider import resolve_runtime_provider
from providers import get_provider_profile
from run_agent import AIAgent


def test_space_bunny_is_keyless_only_on_the_official_zen_route(monkeypatch):
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    profile = get_provider_profile("opencode-zen")

    # The keyless subset is non-empty and the free chat model is in it. Asserting the exact
    # contents would be a change-detector: the set is expected to grow with the lineup.
    assert "space-bunny-free" in profile.keyless_model_ids
    assert profile.supports_anonymous_access(
        model="opencode-zen/space-bunny-free",
        base_url="https://opencode.ai/zen/v1",
    )
    assert not profile.supports_anonymous_access(
        model="mimo-v2.5-free", base_url="https://opencode.ai/zen/v1"
    )
    assert not profile.supports_anonymous_access(
        model="space-bunny-free", base_url="https://proxy.example/v1"
    )

    runtime = resolve_runtime_provider(
        requested="opencode-zen", target_model="space-bunny-free"
    )
    assert runtime["provider"] == "opencode-zen"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == "https://opencode.ai/zen/v1"
    assert runtime["api_key"] == ""
    assert runtime["source"] == "anonymous-model"

    with pytest.raises(AuthError) as paid:
        resolve_runtime_provider(requested="opencode-zen", target_model="gpt-5.5")
    assert paid.value.code == "missing_api_key"

    selected = switch_model(
        "space-bunny-free",
        current_provider="custom",
        current_model="stealth/union-alpha",
        current_base_url="https://openrouter.ai/api/v1",
        explicit_provider="opencode-zen",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    assert selected.success is True
    # The resolved id is the auth-registry id, not the label the user typed: Zen's aliases
    # ("opencode", "zen") normalize onto one registry entry. Assert the route it resolved to
    # rather than the spelling, so an alias change is not a test failure.
    assert selected.target_provider in {"opencode-zen", "opencode"}
    assert selected.base_url == "https://opencode.ai/zen/v1"
    assert selected.api_key == ""

    paid_switch = switch_model(
        "gpt-5.5",
        current_provider="opencode-zen",
        current_model="space-bunny-free",
        current_base_url="https://opencode.ai/zen/v1",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    assert paid_switch.success is False
    assert "requires an API key" in paid_switch.error_message


def test_keyless_zen_is_explicit_in_the_ui_and_omits_sdk_authorization():
    ctx = ConfigContext(
        current_provider="custom",
        current_model="stealth/union-alpha",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    payload = build_models_payload(
        ctx,
        explicit_only=True,
        picker_hints=True,
        non_blocking_catalogs=True,
        probe_custom_providers=False,
    )
    row = next(r for r in payload["providers"] if r.get("slug") == "opencode-zen")
    # The keyless model is offered without a credential, so it must appear in the picker;
    # the exact list grows with the lineup and is not asserted.
    assert "space-bunny-free" in row["models"]
    assert row["authenticated"] is True

    agent = AIAgent(
        api_key="",
        base_url="https://opencode.ai/zen/v1",
        model="space-bunny-free",
        provider="opencode-zen",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-test",
    )
    try:
        assert agent.api_key == ""
        assert agent.client.api_key == ""
        assert agent.client.auth_headers == {}
        assert agent.client.default_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
        assert agent.client.default_headers["X-Title"] == "Hermes Agent"
    finally:
        agent.client.close()


def test_keyless_grant_is_scoped_to_the_endpoint_actually_used(monkeypatch, tmp_path):
    """The resolver is where a wrong endpoint would put a credential-less client on the wire, so
    the refusal is asserted through it rather than only against the profile method: a config
    ``model.base_url`` or an ``OPENCODE_ZEN_BASE_URL`` aimed elsewhere must fail closed, and an
    endpoint the caller never supplied must not be filled in from the profile default."""
    from hermes_cli import config as config_mod

    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    profile = get_provider_profile("opencode-zen")

    # No endpoint to validate: refuse rather than substitute our own.
    assert not profile.supports_anonymous_access(model="space-bunny-free", base_url=None)
    assert not profile.supports_anonymous_access(model="space-bunny-free", base_url="")
    assert not profile.supports_anonymous_access(model="space-bunny-free", base_url="   ")

    def _write_config(base_url):
        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: opencode-zen\n  default: space-bunny-free\n"
            f"  base_url: {base_url}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # The loader caches on file signature; several endpoints are written to one path inside
        # this test, so each read has to start from a cold cache.
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()

    # A config endpoint that is not the official relay must not yield an anonymous runtime.
    # Note: a same-family path (``/zen/go/v1`` under a Zen provider) is deliberately rewritten to
    # ``/zen/v1`` by normalize_opencode_base_url before the check, so it is correct for that one
    # to resolve — the grant is decided on the endpoint that will actually be used.
    for hostile in ("https://attacker.example/v1", "https://evil.opencode.ai/zen/v1",
                    "http://opencode.ai/zen/v1", "https://opencode.ai:8443/zen/v1",
                    "https://opencode.ai/zen/v1?x=1",
                    "https://opencode.ai/zen/v1/../go"):
        _write_config(hostile)
        try:
            leaked = resolve_runtime_provider(requested="opencode-zen", target_model="space-bunny-free")
        except AuthError as exc:
            assert exc.code == "missing_api_key", hostile
        else:
            pytest.fail("hostile endpoint granted an anonymous runtime: %s -> %r" % (hostile, leaked))

    # The env override is the same lever and must be refused identically.
    _write_config("")
    monkeypatch.setenv("OPENCODE_ZEN_BASE_URL", "https://attacker.example/v1")
    with pytest.raises(AuthError):
        resolve_runtime_provider(requested="opencode-zen", target_model="space-bunny-free")
    monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)

    # And the official endpoint still resolves anonymously, so the refusals above are specific.
    _write_config("")
    granted = resolve_runtime_provider(requested="opencode-zen", target_model="space-bunny-free")
    assert granted["api_key"] == ""
    assert granted["base_url"] == "https://opencode.ai/zen/v1"


def test_keyless_grant_requires_the_id_that_will_be_sent(monkeypatch):
    """The wire normalizer strips only this provider's own prefix; a model spelled any other way
    reaches the relay with the prefix still attached, so it must not inherit the grant. Otherwise
    the id that was authorized and the id that is sent are different strings."""
    from hermes_cli.models import normalize_opencode_model_id

    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    profile = get_provider_profile("opencode-zen")
    url = "https://opencode.ai/zen/v1"

    for spelling in ("space-bunny-free", "opencode-zen/space-bunny-free", "zen/space-bunny-free"):
        granted = profile.supports_anonymous_access(model=spelling, base_url=url)
        sent = normalize_opencode_model_id("opencode-zen", spelling)
        assert granted is (sent == "space-bunny-free"), spelling

    # Prefixes the normalizer does not strip, and deeper paths, get no grant.
    for spelling in ("other/space-bunny-free", "gpt-5.5/space-bunny-free",
                     "vendor/extra/space-bunny-free"):
        assert not profile.supports_anonymous_access(model=spelling, base_url=url), spelling


def test_aux_client_routes_a_keyless_task_without_manufacturing_a_credential(monkeypatch):
    """The auxiliary branch builds its own client instead of routing through the resolver, so it
    needs its own contract: a keyless task on the official endpoint must resolve to a credential-less
    client rather than being reported as unconfigured, and the sibling cases (a paid model, or the
    keyless model aimed at another host) must keep resolving to nothing."""
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    from agent.auxiliary_client import resolve_provider_client

    official = "https://opencode.ai/zen/v1"

    with patch("agent.auxiliary_client._create_openai_client") as make_client:
        make_client.return_value = MagicMock()
        client, model = resolve_provider_client("opencode-zen", model="space-bunny-free")
    assert client is not None, "a keyless aux task must resolve instead of being dropped"
    assert make_client.call_args.kwargs["api_key"] == ""
    assert make_client.call_args.kwargs["base_url"] == official

    # A paid model on the same credential-free provider is still unconfigured.
    client, model = resolve_provider_client("opencode-zen", model="gpt-5.5")
    assert (client, model) == (None, None)

    # And the keyless grant does not follow the model to a host it was never granted for: this
    # branch builds the client from the endpoint it is handed, so a wrong one must resolve to
    # nothing rather than produce a credential-less client pointed at it.
    client, model = resolve_provider_client(
        "opencode-zen", model="space-bunny-free", explicit_base_url="https://attacker.example/v1")
    assert (client, model) == (None, None)


def test_switching_onto_the_keyless_route_drops_the_previous_providers_key():
    """A switch onto an anonymous route must not carry the previous provider's credential to the
    new endpoint. `agent.switch_model` historically treated a falsy `api_key` as "caller said
    nothing" and kept `agent.api_key`, which paired the old provider's bearer token with the new
    provider's base_url — so an OpenRouter key reached opencode.ai. `None` still means "no
    decision" (best-effort credential refresh); `""` is a decision to send no Authorization."""
    agent = AIAgent(
        api_key="sk-or-v1-previous-provider-key",
        base_url="https://openrouter.ai/api/v1",
        model="stealth/union-alpha",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-switch-test",
    )
    try:
        agent.switch_model(
            new_model="space-bunny-free",
            new_provider="opencode-zen",
            api_key="",
            base_url="https://opencode.ai/zen/v1",
        )

        assert agent.base_url == "https://opencode.ai/zen/v1"
        assert agent.api_key == ""
        assert agent._client_kwargs["api_key"] == ""
        assert agent._client_kwargs["base_url"] == "https://opencode.ai/zen/v1"
        # The rebuilt client must carry no Authorization header at all, not a stale bearer.
        assert agent.client.auth_headers == {}
    finally:
        agent.client.close()


def test_api_key_none_still_means_no_decision_on_a_plain_provider_switch():
    """The guard above must not break the best-effort credential refresh: `None` keeps the
    session's existing key so a same-provider re-select still authenticates."""
    agent = AIAgent(
        api_key="existing-key",
        base_url="https://openrouter.ai/api/v1",
        model="stealth/union-alpha",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-none-test",
    )
    try:
        agent.switch_model(
            new_model="stealth/union-alpha",
            new_provider="custom",
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
        )

        assert agent.api_key == "existing-key"
        assert agent._client_kwargs["api_key"] == "existing-key"
    finally:
        agent.client.close()
