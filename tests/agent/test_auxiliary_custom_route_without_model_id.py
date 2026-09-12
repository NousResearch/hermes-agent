"""``auxiliary.*.provider: auto`` must fall back to a configured custom endpoint (#107509).

A self-hosted-only install (``model.provider: custom`` + a local ``model.base_url``, no external
provider configured anywhere) has exactly one reachable route: the endpoint the user configured.
When no model id is configured, ``_try_main_provider_route()`` vetoed that route outright and
``_discovery_chain_allowed()`` then refused the built-in chain ("refusing to guess another
logged-in provider"), so every ``auxiliary.*`` subsystem (kanban_decomposer, curator, vision,
title_generation, compression, ...) hard-failed with
``RuntimeError: No LLM provider configured for task=... provider=auto`` instead of using the only
provider that actually exists.

These tests pin: (1) the custom route is retried with the endpoint/entry-supplied model,
(2) an explicit fallback policy still wins, (3) main-first routing is untouched when an external
provider IS configured, and (4) installs with a selected non-custom main provider still refuse to
guess other logged-in accounts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent import auxiliary_client as aux

_EXTERNAL_ENV = (
    "OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY", "XAI_API_KEY", "DEEPINFRA_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
)


def _selfhosted_home(tmp_path, monkeypatch, model_block: str, extra: str = "") -> None:
    """Hermetic HERMES_HOME with only the given model block + no external provider creds."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n" + model_block + extra, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in _EXTERNAL_ENV:
        monkeypatch.delenv(var, raising=False)
    aux.clear_runtime_main()
    aux._aux_unhealthy_until.clear()


def _client_base_url(client) -> str:
    for attr in ("base_url",):
        value = getattr(client, attr, None)
        if value:
            return str(value)
    inner = getattr(client, "_client", None)
    return str(getattr(inner, "base_url", "") or "")


def test_auto_uses_configured_custom_endpoint_when_no_model_id_is_set(tmp_path, monkeypatch):
    """model.provider: custom + base_url, no model id, zero external providers → route to it."""
    _selfhosted_home(
        tmp_path, monkeypatch,
        "  provider: custom\n  base_url: 'https://local.example/v1'\n  api_key: local-key\n",
    )

    client, model, provider = aux._resolve_auto_route(main_runtime=None, task="kanban_decomposer")

    assert client is not None, "auto dead-ended on the configured custom endpoint"
    assert provider == "custom"
    assert _client_base_url(client).rstrip("/") == "https://local.example/v1"
    assert model, "auto must carry a model for the resolved endpoint"


def test_auto_uses_named_custom_entry_model_when_model_id_is_unset(tmp_path, monkeypatch):
    """A ``custom:<name>`` entry supplies its own model; a missing model.default must not veto it."""
    _selfhosted_home(
        tmp_path, monkeypatch,
        "  provider: 'custom:selfhosted'\n  base_url: ''\n",
        "custom_providers:\n"
        "  - name: selfhosted\n"
        "    base_url: 'https://entry.example/v1'\n"
        "    model: local-qwen\n"
        "    api_key: entry-key\n",
    )

    client, model, provider = aux._resolve_auto_route(main_runtime=None, task="title_generation")

    assert client is not None, "named custom entry was vetoed by the missing model id"
    assert model == "local-qwen"
    assert _client_base_url(client).rstrip("/") == "https://entry.example/v1"
    assert provider == "custom:selfhosted"


def test_configured_fallback_policy_still_wins_over_the_custom_retry(tmp_path, monkeypatch):
    """Precedence is unchanged: an explicit fallback chain is honoured before the custom retry."""
    _selfhosted_home(
        tmp_path, monkeypatch,
        "  provider: custom\n  base_url: 'https://local.example/v1'\n  api_key: local-key\n",
    )
    fallback_client = MagicMock(name="fallback-client")

    with patch.object(aux, "_try_main_fallback_chain",
                      return_value=(fallback_client, "fb-model", "fallback_providers[0](openrouter)")):
        client, model, provider = aux._resolve_auto_route(main_runtime=None, task="compression")

    assert client is fallback_client
    assert model == "fb-model"
    assert provider == "fallback_providers[0](openrouter)"


def test_external_provider_configured_keeps_main_first_routing(tmp_path, monkeypatch):
    """With an external provider available, ``auto`` still runs on the main model (unchanged)."""
    _selfhosted_home(
        tmp_path, monkeypatch,
        "  provider: custom\n  model: local-qwen\n  base_url: 'https://local.example/v1'\n  api_key: local-key\n",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # external provider IS configured

    with patch.object(aux, "_try_discovery_chain") as discovery:
        client, model, provider = aux._resolve_auto_route(main_runtime=None, task="title_generation")

    assert client is not None and model == "local-qwen" and provider == "custom"
    discovery.assert_not_called()


def test_selected_non_custom_main_provider_still_refuses_to_guess(tmp_path, monkeypatch):
    """The anti-"bill another account" guard is preserved for non-custom main providers."""
    _selfhosted_home(tmp_path, monkeypatch, "  model: local-qwen\n")
    runtime = {"provider": "xai-oauth", "model": "grok-4.6",
               "base_url": "https://api.x.ai/v1", "api_key": "dead"}

    with patch.object(aux, "_try_main_provider_route", return_value=None), \
         patch.object(aux, "_try_discovery_chain", return_value=(MagicMock(), "guessed", "nous")) as discovery:
        assert aux._resolve_auto_route(main_runtime=runtime, task="compression") == (None, None, "")
    discovery.assert_not_called()


def test_custom_route_without_any_endpoint_still_fails_cleanly(tmp_path, monkeypatch):
    """No endpoint configured anywhere → no route (never a silent API-key-provider guess)."""
    _selfhosted_home(tmp_path, monkeypatch, "  provider: custom\n")

    with patch.object(aux, "_try_discovery_chain") as discovery:
        assert aux._resolve_auto_route(main_runtime=None, task="compression") == (None, None, "")
    discovery.assert_not_called()
