from types import SimpleNamespace

from agent.agent_init import _bootstrap_primary_credential_pool


class _Entry:
    runtime_api_key = "pool-key"
    access_token = ""


class _Pool:
    def __init__(self):
        self.selected = False

    def has_credentials(self):
        return True

    def select(self):
        self.selected = True
        return _Entry()


def test_named_custom_provider_uses_pool_key_before_first_request(monkeypatch):
    pool = _Pool()

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: pool if provider == "custom:nvidia-rotating" else None,
    )
    agent = SimpleNamespace(provider="custom:nvidia-rotating", _credential_pool=None)

    key = _bootstrap_primary_credential_pool(agent, "static-key")

    assert key == "pool-key"
    assert agent._credential_pool is pool
    assert pool.selected


def test_non_custom_provider_keeps_existing_key(monkeypatch):
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError(provider)),
    )
    agent = SimpleNamespace(provider="openrouter", _credential_pool=None)

    assert _bootstrap_primary_credential_pool(agent, "static-key") == "static-key"
    assert agent._credential_pool is None
