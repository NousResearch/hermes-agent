"""Auxiliary Anthropic client on an Azure Foundry endpoint must use a key bound to that endpoint,
never the Claude Code OAuth token the ``anthropic`` pool (or the legacy resolver) holds (#154242).

Real pool: the test's HERMES_HOME holds config.yaml and auth.json and a temporary
CLAUDE_CONFIG_DIR holds the Claude Code login ``load_pool("anthropic")`` seeds; only the SDK client
constructor is replaced, to record the (token, base_url) pair a request would carry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.auxiliary_client as aux

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
OAUTH_TOKEN = "sk-ant-" + "oat01-" + "c" * 24  # concatenated: push protection rejects token-shaped literals


@pytest.fixture
def built(tmp_path, monkeypatch):
    """Claude Code login present, no ambient keys; returns the last built client's (token, base_url)."""
    for name in ("ANTHROPIC_API_KEY", "AZURE_ANTHROPIC_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": 4102444800000}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    calls = {}

    def _fake_build(token, base_url):
        calls.update(token=token, base_url=base_url)
        return MagicMock()

    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", _fake_build)
    return calls


def _home(*, rows=(), **model):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    (home / "config.yaml").write_text(json.dumps({"model": model} if model else {}))
    pool_rows = [{"id": f"row{i}", "label": f"row{i}", "auth_type": "api_key", "priority": i, "source": "manual", **row}
                 for i, row in enumerate(rows)]
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": pool_rows}}))


def test_foundry_model_api_key_wins_over_the_claude_code_login(built):
    _home(provider="anthropic", base_url=FOUNDRY_URL, api_key="KAZ")
    client, _ = aux._try_anthropic()
    assert client is not None
    assert built == {"token": "KAZ", "base_url": FOUNDRY_URL}


def test_foundry_azure_env_key_wins_over_the_claude_code_login(built, monkeypatch):
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    _home(provider="anthropic", base_url=FOUNDRY_URL)
    aux._try_anthropic()
    assert built == {"token": "KAZ-ENV", "base_url": FOUNDRY_URL}


def test_explicit_foundry_url_never_gets_the_oauth_token(built):
    _home(provider="anthropic")
    client, model = aux._try_anthropic(explicit_base_url=FOUNDRY_URL)
    assert (client, model) == (None, None)
    assert built == {}


def test_plain_anthropic_endpoint_still_uses_the_pool(built):
    _home(provider="anthropic")
    aux._try_anthropic()
    assert built["token"] == OAUTH_TOKEN


def test_other_providers_inline_key_never_goes_to_an_explicit_foundry_url(built, monkeypatch):
    """An aux/fallback entry ``anthropic`` + Foundry URL while ``model`` belongs to OpenRouter: the
    OpenRouter key in ``model.api_key`` is not the Foundry key; AZURE_ANTHROPIC_KEY is."""
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    _home(provider="openrouter", api_key="OR-SECRET")
    aux.resolve_provider_client("anthropic", model="claude-haiku-4-5", explicit_base_url=FOUNDRY_URL)
    assert built == {"token": "KAZ-ENV", "base_url": FOUNDRY_URL}


def test_foundry_bound_pool_key_serves_foundry_even_behind_the_oauth_row(built):
    """``hermes auth add anthropic --api-key <azure key>`` recorded for the Foundry URL; an OAuth row ranks first."""
    _home(provider="anthropic", base_url=FOUNDRY_URL, rows=[
        {"access_token": OAUTH_TOKEN, "auth_type": "oauth", "source": "manual:hermes_pkce"},
        {"access_token": "KAZ-POOL", "base_url": FOUNDRY_URL}])
    aux._try_anthropic()
    assert built == {"token": "KAZ-POOL", "base_url": FOUNDRY_URL}


def test_pool_key_recorded_for_another_origin_does_not_outrank_the_foundry_model_key(built):
    """A Console key (recorded for api.anthropic.com, or unrecorded) is not the Foundry key."""
    _home(provider="anthropic", base_url=FOUNDRY_URL, api_key="KAZ", rows=[
        {"access_token": "CONSOLE", "base_url": "https://api.anthropic.com"}, {"access_token": "UNRECORDED"}])
    aux._try_anthropic()
    assert built == {"token": "KAZ", "base_url": FOUNDRY_URL}


def test_credential_rotation_never_swaps_the_oauth_row_onto_a_foundry_session(built):
    """``_swap_credential`` is where every rotation (429/401 recovery, revert, restore, a delegate's
    lease) adopts a row; an unbound row would keep the session's Foundry URL and carry its token there."""
    from agent.client_lifecycle import ClientLifecycleMixin
    agent = SimpleNamespace(provider="anthropic", model="claude-x", api_mode="anthropic_messages", base_url=FOUNDRY_URL,
                            api_key="KAZ", _anthropic_api_key="KAZ", _anthropic_base_url=FOUNDRY_URL,
                            _anthropic_client=MagicMock(), _build_direct_anthropic_client=MagicMock(),
                            _anthropic_oauth_flag=lambda _t: False)
    oauth_row = SimpleNamespace(id="cc", source="claude_code", auth_type="oauth", access_token=OAUTH_TOKEN,
                                runtime_api_key=OAUTH_TOKEN, base_url="")
    assert ClientLifecycleMixin._swap_credential(agent, oauth_row) is False
    assert (agent.api_key, agent.base_url) == ("KAZ", FOUNDRY_URL)
    agent._build_direct_anthropic_client.assert_not_called()

    foundry_row = SimpleNamespace(id="k2", source="manual", auth_type="api_key", access_token="KAZ-2",
                                  runtime_api_key="KAZ-2", base_url=FOUNDRY_URL)
    assert ClientLifecycleMixin._swap_credential(agent, foundry_row) is True
    assert (agent.api_key, agent.base_url) == ("KAZ-2", FOUNDRY_URL)


@pytest.mark.parametrize(("base_url", "expected"), [
    (FOUNDRY_URL, {"token": "KAZ-ENV", "base_url": FOUNDRY_URL}),
    ("", {"token": OAUTH_TOKEN, "base_url": "https://api.anthropic.com"}),
])
def test_cached_aux_client_takes_the_oauth_row_only_off_foundry(built, monkeypatch, base_url, expected):
    """``call_llm(provider="anthropic")`` goes through ``_get_cached_client``, whose pool peek must not
    hand the OAuth row to ``_try_anthropic`` as a caller-supplied key (@BearHuddleston's repro)."""
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    _home(provider="anthropic", default="claude-x", **({"base_url": base_url} if base_url else {}))
    monkeypatch.setattr(aux, "_client_cache", {})
    client, _model = aux._get_cached_client("anthropic")
    assert client is not None
    assert built == expected
