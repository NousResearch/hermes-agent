"""Native ``anthropic`` with a config-stored Azure Foundry ``model.base_url`` must resolve a key bound
to that Foundry endpoint, never the Claude Code OAuth token the ``anthropic`` credential pool seeds
(#154242), neither at resolution nor when a 429/401 rotates the attached pool.

Real pool: the test's HERMES_HOME holds config.yaml and auth.json, and a temporary
CLAUDE_CONFIG_DIR holds the Claude Code login that ``load_pool("anthropic")`` seeds.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import runtime_provider as rp

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
OAUTH_TOKEN = "sk-ant-" + "oat01-" + "c" * 24  # concatenated: push protection rejects token-shaped literals


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The test HERMES_HOME with a Claude Code login present and no ambient Anthropic/Azure keys."""
    from hermes_constants import get_hermes_home
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "AZURE_ANTHROPIC_KEY",
                 "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": 4102444800000}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    return get_hermes_home()


def write_home(home, *, rows=(), extra=None, **model):
    """config.yaml with ``model`` = anthropic + ``model`` kwargs, auth.json with manual api-key ``rows``."""
    cfg = {"model": {"provider": "anthropic", "default": "claude-x", **model}, **(extra or {})}
    (home / "config.yaml").write_text(json.dumps(cfg))  # JSON is YAML
    pool_rows = [{"id": f"row{i}", "label": f"row{i}", "auth_type": "api_key", "priority": i, "source": "manual", **row}
                 for i, row in enumerate(rows)]
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": pool_rows}}))


def _pool_keys(pool):
    return {e.runtime_api_key for e in pool.entries()} if pool is not None else set()


def test_claude_code_login_is_seeded_into_the_real_pool(home):
    """Precondition for every case below: the pool really holds the OAuth row."""
    from agent.credential_pool import load_pool
    write_home(home)
    assert OAUTH_TOKEN in _pool_keys(load_pool("anthropic"))


def test_foundry_model_api_key_wins_over_the_claude_code_login(home):
    write_home(home, base_url=FOUNDRY_URL, api_key="KAZ")
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert (runtime["api_key"], runtime["base_url"], runtime["api_mode"]) == ("KAZ", FOUNDRY_URL, "anthropic_messages")
    assert OAUTH_TOKEN not in _pool_keys(runtime.get("credential_pool"))


def test_foundry_azure_env_key_wins_without_an_explicit_request(home, monkeypatch):
    """``hermes chat`` passes no ``--provider``: the saved provider takes the same path."""
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    write_home(home, base_url=FOUNDRY_URL)
    runtime = rp.resolve_runtime_provider()
    assert (runtime["api_key"], runtime["base_url"]) == ("KAZ-ENV", FOUNDRY_URL)


def test_foundry_endpoint_without_any_azure_key_fails_loudly(home):
    write_home(home, base_url=FOUNDRY_URL)
    with pytest.raises(rp.AuthError, match="Azure Anthropic API key"):
        rp.resolve_runtime_provider(requested="anthropic")


def test_foundry_bound_pool_row_is_chosen_even_when_the_oauth_row_ranks_first(home):
    """Manual rows rank first, so put the OAuth row ahead explicitly: selection must still reach the Foundry key."""
    write_home(home, base_url=FOUNDRY_URL, rows=[
        {"access_token": OAUTH_TOKEN, "auth_type": "oauth", "source": "manual:hermes_pkce"},
        {"access_token": "KAZ-POOL", "base_url": FOUNDRY_URL}])
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert (runtime["api_key"], runtime["base_url"]) == ("KAZ-POOL", FOUNDRY_URL)


def test_recovery_pool_attached_to_a_foundry_session_holds_only_foundry_rows(home):
    """A rate limit rotates the attached pool: it must never hand back the Claude Code token."""
    write_home(home, base_url=FOUNDRY_URL, rows=[{"access_token": "KAZ-1", "base_url": FOUNDRY_URL},
                                                 {"access_token": "KAZ-2", "base_url": FOUNDRY_URL}])
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    pool = runtime["credential_pool"]
    assert _pool_keys(pool) == {"KAZ-1", "KAZ-2"}
    rotated = pool.mark_exhausted_and_rotate(status_code=429, api_key_hint=runtime["api_key"])
    assert rotated is not None and rotated.runtime_api_key == ({"KAZ-1", "KAZ-2"} - {runtime["api_key"]}).pop()
    assert pool.mark_exhausted_and_rotate(status_code=429, api_key_hint=rotated.runtime_api_key) is None


def test_pool_key_recorded_for_another_origin_never_follows_model_base_url_to_foundry(home):
    """``hermes auth add anthropic`` records api.anthropic.com (an unrecorded row means it too):
    a later Foundry ``model.base_url`` is another host."""
    write_home(home, base_url=FOUNDRY_URL, api_key="KAZ", rows=[
        {"access_token": "CONSOLE", "base_url": "https://api.anthropic.com"}, {"access_token": "UNRECORDED"}])
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_key"] == "KAZ"
    assert _pool_keys(runtime.get("credential_pool")).isdisjoint({"CONSOLE", "UNRECORDED", OAUTH_TOKEN})


def test_plain_anthropic_endpoint_still_uses_the_claude_code_login(home):
    """Control: the pool IS the right answer for api.anthropic.com."""
    write_home(home)
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert (runtime["api_key"], runtime["base_url"]) == (OAUTH_TOKEN, "https://api.anthropic.com")


def test_providers_entry_owning_the_foundry_url_supplies_its_own_key(home):
    """A ``providers:`` entry that owns ``model.base_url`` keeps supplying the key."""
    write_home(home, base_url=FOUNDRY_URL, extra={"providers": {"myfoundry": {"base_url": FOUNDRY_URL, "api_key": "KAZ-ENTRY"}}})
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert (runtime["api_key"], runtime["source"], runtime["base_url"]) == ("KAZ-ENTRY", "config:providers", FOUNDRY_URL)


@pytest.mark.parametrize("requested", ["anthropic", "claude", "claude-code"])
def test_explicit_foundry_base_url_without_an_azure_key_fails_loudly(home, requested):
    """An empty key would be filled with the OAuth token downstream, so the resolver refuses,
    under every alias of the provider (the aliases skip the requested-name shortcut)."""
    write_home(home)
    with pytest.raises(rp.AuthError, match="Azure Anthropic API key"):
        rp.resolve_runtime_provider(requested=requested, explicit_base_url=FOUNDRY_URL)


@pytest.mark.parametrize("requested", ["anthropic", "claude", "claude-code"])
def test_explicit_foundry_base_url_uses_the_azure_env_key_under_every_alias(home, monkeypatch, requested):
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    write_home(home)
    runtime = rp.resolve_runtime_provider(requested=requested, explicit_base_url=FOUNDRY_URL)
    assert (runtime["provider"], runtime["api_key"], runtime["base_url"]) == ("anthropic", "KAZ-ENV", FOUNDRY_URL)


def test_delegation_with_a_claude_alias_and_a_foundry_base_url_never_gets_the_oauth_token(home, monkeypatch):
    """``delegation.provider: claude`` + ``delegation.base_url: <foundry>`` reaches the same resolver."""
    from tools.delegate_tool_config import _runtime_provider_credentials
    write_home(home)
    with pytest.raises(ValueError, match="Azure Anthropic API key"):
        _runtime_provider_credentials({"provider": "claude", "model": "claude-x", "base_url": FOUNDRY_URL}, None, FOUNDRY_URL)
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    creds = _runtime_provider_credentials({"provider": "claude", "model": "claude-x", "base_url": FOUNDRY_URL}, None, FOUNDRY_URL)
    assert (creds["api_key"], creds["base_url"]) == ("KAZ-ENV", FOUNDRY_URL)


def test_explicit_non_foundry_base_url_under_the_claude_alias_keeps_the_oauth_token(home):
    """Control: the Foundry guard is host-scoped; an Anthropic-compatible proxy keeps main's behaviour."""
    write_home(home)
    runtime = rp.resolve_runtime_provider(requested="claude", explicit_base_url="https://api.anthropic.com")
    assert runtime["api_key"] == OAUTH_TOKEN
