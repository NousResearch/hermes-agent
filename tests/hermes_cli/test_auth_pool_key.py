"""A pooled API key keeps the endpoint stored on its credential-pool entry.

Regression for #42269. ``hermes auth add deepseek --type api-key`` stores the key with the
endpoint it was added for (``providers.deepseek.base_url``, else the registry default). The
runtime pool rung already sends that key to that endpoint, but
``resolve_api_key_provider_credentials`` (auxiliary client, model catalogs, ``hermes status``)
took the key from the pool and paired it with the registry default, so a key issued for a
DeepSeek-compatible host was sent to api.deepseek.com. Key and endpoint now come from one entry.
"""

import uuid

from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool

PROXY = "https://deepseek-proxy.example.com/v1"
PROXY_KEY = "sk-proxy-0123456789abcdef0123456789abcdef"


def _add_deepseek_key(key: str, base_url: str) -> None:
    """The row ``hermes auth add deepseek --type api-key`` writes to auth.json."""
    load_pool("deepseek").add_entry(PooledCredential(
        provider="deepseek", id=uuid.uuid4().hex[:6], label=f"key-{key[-4:]}",
        auth_type=AUTH_TYPE_API_KEY, priority=0, source=SOURCE_MANUAL, access_token=key,
        base_url=base_url))


def test_runtime_provider_sends_pool_key_to_its_endpoint():
    """The CLI/gateway path from the issue report."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _add_deepseek_key(PROXY_KEY, PROXY)
    runtime = resolve_runtime_provider(requested="deepseek")
    assert (runtime["api_key"], runtime["base_url"]) == (PROXY_KEY, PROXY)


def test_api_key_resolvers_keep_pool_entry_endpoint():
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.auth import get_api_key_provider_status, resolve_api_key_provider_credentials

    _add_deepseek_key(PROXY_KEY, PROXY)
    creds = resolve_api_key_provider_credentials("deepseek")
    assert (creds["api_key"], creds["base_url"]) == (PROXY_KEY, PROXY)
    assert get_api_key_provider_status("deepseek")["base_url"] == PROXY
    client, _model = resolve_provider_client("deepseek", model="deepseek-chat")
    assert (client.api_key, str(client.base_url).rstrip("/")) == (PROXY_KEY, PROXY)


def test_key_and_endpoint_come_from_one_pool_entry():
    """The pool's first pick holds a pasted placeholder, so the next entry supplies the key: its
    endpoint must come with it, never the skipped entry's."""
    from hermes_cli.auth import resolve_api_key_provider_credentials

    _add_deepseek_key("your-api-key", "https://placeholder-host.example.com/v1")
    _add_deepseek_key(PROXY_KEY, PROXY)
    creds = resolve_api_key_provider_credentials("deepseek")
    assert (creds["api_key"], creds["base_url"]) == (PROXY_KEY, PROXY)


def test_base_url_env_override_still_wins(monkeypatch):
    from hermes_cli.auth import resolve_api_key_provider_credentials

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://override.example.com/v1")
    _add_deepseek_key(PROXY_KEY, PROXY)
    assert resolve_api_key_provider_credentials("deepseek")["base_url"] == "https://override.example.com/v1"
