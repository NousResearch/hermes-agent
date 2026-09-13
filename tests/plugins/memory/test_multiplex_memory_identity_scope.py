"""Multiplex invariant: memory-provider identity/tenant/endpoint never comes from the default profile.

Under ``gateway.multiplex_profiles`` ``os.environ`` is the DEFAULT profile's ``.env``. When a secondary
profile's scope does not define MEM0_USER_ID / SUPERMEMORY_CONTAINER_TAG / RETAINDB_PROJECT /
OPENVIKING_* / HINDSIGHT_BANK_ID / HERMES_HONCHO_HOST, the provider must fall back to its own default
(per-profile partition), NOT write the secondary's memories into the default profile's account.
"""
from __future__ import annotations

import pytest

from agent import secret_scope

_DEFAULT_ENV = {
    "MEM0_USER_ID": "user-default", "MEM0_AGENT_ID": "agent-default", "MEM0_HOST": "http://mem0.default",
    "MEM0_MODE": "self_hosted",
    "SUPERMEMORY_CONTAINER_TAG": "tag-default", "SUPERMEMORY_BASE_URL": "https://sm.default",
    "RETAINDB_PROJECT": "proj-default", "RETAINDB_BASE_URL": "https://rdb.default",
    "OPENVIKING_API_KEY": "ov-default", "OPENVIKING_ACCOUNT": "acct-default", "OPENVIKING_USER": "user-default",
    "OPENVIKING_AGENT": "agent-default", "OPENVIKING_ENDPOINT": "http://ov.default",
    "HINDSIGHT_BANK_ID": "bank-default", "HINDSIGHT_MODE": "local_external", "HINDSIGHT_API_URL": "http://hs.default",
    "HINDSIGHT_RETAIN_TAGS": "tag-default", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES": "per_tag",
    "HINDSIGHT_RETAIN_SOURCE": "source-default", "HINDSIGHT_RETAIN_USER_PREFIX": "UserDefault",
    "HINDSIGHT_RETAIN_ASSISTANT_PREFIX": "AssistantDefault",
    "HERMES_HONCHO_HOST": "host-default", "HONCHO_BASE_URL": "https://honcho.default",
    "OPENAI_API_KEY": "sk-default", "OPENAI_BASE_URL": "https://openai.default/v1",
}


@pytest.fixture
def secondary_profile(monkeypatch, tmp_path):
    """Multiplex ON; default profile's values in environ; secondary profile `b` scope with only its
    own API keys (no identity/tenant/endpoint vars)."""
    for k, v in _DEFAULT_ENV.items():
        monkeypatch.setenv(k, v)
    home = tmp_path / ".hermes"
    prof_b = home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(prof_b))
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"RETAINDB_API_KEY": "rdb-b", "SUPERMEMORY_API_KEY": "sm-b",
                                           "HONCHO_API_KEY": "honcho-b", "MEM0_API_KEY": "mem0-b"})
    try:
        yield prof_b
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_secondary_profile_memory_identity_never_inherits_default_environ(secondary_profile):
    import plugins.memory.hindsight as hindsight
    import plugins.memory.mem0 as mem0
    import plugins.memory.openviking as openviking
    import plugins.memory.retaindb as retaindb
    import plugins.memory.supermemory as supermemory
    from plugins.memory.honcho import client as honcho_client

    cfg = mem0._load_config()
    assert "user_id" not in cfg  # falls back to the gateway-native id, not the default's user
    assert (cfg["agent_id"], cfg["host"], cfg["mode"]) == ("hermes", "", "platform")

    assert supermemory._resolve_container_tag("cfg_tag", "id") == "cfg_tag"
    assert "default" not in supermemory._resolve_base_url("")

    provider = retaindb.RetainDBMemoryProvider()
    provider.initialize("s1", hermes_home=str(secondary_profile))
    assert provider._client.project == "hermes-b"
    assert "default" not in provider._client.base_url

    settings = openviking._resolve_connection_settings({})
    assert (settings["api_key"], settings["account"], settings["user"]) == ("", "", "")
    assert "default" not in settings["endpoint"]
    client = openviking._VikingClient("http://x", "k")
    assert (client._account, client._user) == ("default", "default")  # the built-in tenant, not acct-default

    hcfg = hindsight._load_config()
    assert (hcfg["banks"]["hermes"]["bankId"], hcfg["mode"]) == ("hermes", "cloud")

    assert honcho_client.resolve_active_host() != "host-default"
    assert honcho_client._env_base_url() is None


def test_mem0_oss_llm_never_borrows_default_profile_openai_key(secondary_profile):
    pytest.importorskip("mem0")
    from mem0.configs.llms.openai import OpenAIConfig

    from plugins.memory.mem0._openai_llm import DirectOpenAILLM

    with pytest.raises(ValueError, match="OpenAI API key is required"):
        DirectOpenAILLM(OpenAIConfig(model="gpt-5-mini"))
    # The secondary's own scoped key + base URL are used when present.
    token = secret_scope.set_secret_scope({"OPENAI_API_KEY": "sk-b", "OPENAI_BASE_URL": "https://openai.b/v1"})
    try:
        llm = DirectOpenAILLM(OpenAIConfig(model="gpt-5-mini"))
    finally:
        secret_scope.reset_secret_scope(token)
    assert llm.client.api_key == "sk-b"
    assert str(llm.client.base_url).startswith("https://openai.b/v1")


def test_hindsight_retain_shaping_is_not_re_read_from_the_default_profile_environ(secondary_profile):
    """The provider must retain with ITS OWN shaping, not the default profile's.

    ``_load_config`` resolves retain shaping through the secret scope, but the values it produces
    pass through ``_apply_retain_settings``, whose ``cfg_or_env`` half re-read ``os.environ`` — so
    every scoped miss came back as the default profile's tag, scope, source and speaker prefixes.
    ``metadata.source`` is opt-in by AGENTS.md, and tags are the retrieval partition, so a secondary
    profile's memories were both mislabelled and selectable by the default profile's tag filters.
    """
    import plugins.memory.hindsight as hindsight
    from plugins.memory.hindsight.settings import _DEFAULT_RETAIN_SOURCE

    cfg = hindsight._load_config()
    assert (cfg["retain_source"], cfg["retain_user_prefix"], cfg["retain_assistant_prefix"]) == (
        _DEFAULT_RETAIN_SOURCE, "User", "Assistant")

    provider = hindsight.HindsightMemoryProvider()
    provider._apply_retain_settings(cfg)
    assert provider._retain_tags == []
    assert provider._observation_scopes is None
    assert provider._retain_source == _DEFAULT_RETAIN_SOURCE
    assert (provider._retain_user_prefix, provider._retain_assistant_prefix) == ("User", "Assistant")

    # A config.json install never reaches ``_load_config``'s scoped read at all, so the fallback
    # inside ``_apply_retain_settings`` is the only gate for it.
    provider._apply_retain_settings({})
    assert provider._retain_tags == []
    assert provider._retain_source == _DEFAULT_RETAIN_SOURCE

    # The profile's OWN scoped values still win — this is isolation, not a blindfold.
    token = secret_scope.set_secret_scope({"HINDSIGHT_RETAIN_TAGS": "tag-b", "HINDSIGHT_RETAIN_SOURCE": "source-b"})
    try:
        provider._apply_retain_settings({})
    finally:
        secret_scope.reset_secret_scope(token)
    assert provider._retain_tags == ["tag-b"]
    assert provider._retain_source == "source-b"


def test_hindsight_retain_shaping_still_reads_the_process_env_for_a_single_profile(monkeypatch, tmp_path):
    """No multiplexing: the process env IS this profile's own .env, so it must keep being read."""
    import plugins.memory.hindsight as hindsight

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HINDSIGHT_RETAIN_TAGS", "solo-tag")
    monkeypatch.setenv("HINDSIGHT_RETAIN_SOURCE", "solo-source")
    monkeypatch.setenv("HINDSIGHT_RETAIN_USER_PREFIX", "Operator")

    provider = hindsight.HindsightMemoryProvider()
    provider._apply_retain_settings({})
    assert provider._retain_tags == ["solo-tag"]
    assert provider._retain_source == "solo-source"
    assert provider._retain_user_prefix == "Operator"
