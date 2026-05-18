import agent.auxiliary_client as aux


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")


def test_openrouter_uses_env_key_when_pool_entries_exhausted(monkeypatch):
    """An exhausted OpenRouter pool must not block a freshly loaded env key."""
    monkeypatch.setattr(aux, "_select_pool_entry", lambda provider: (True, None))
    monkeypatch.setattr(aux, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-live-test-key")

    client, model = aux._try_openrouter(model="google/gemini-3-flash-preview")

    assert isinstance(client, _FakeOpenAI)
    assert client.api_key == "sk-or-v1-live-test-key"
    assert client.base_url == aux.OPENROUTER_BASE_URL
    assert model == "google/gemini-3-flash-preview"
