import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def test_moa_stream_recovery_rebuilds_virtual_client_without_openai_api_key(monkeypatch):
    from agent.moa_loop import MoAClient

    callback = lambda *_a, **_k: None
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.provider = "moa"
    agent.model = "coding"
    agent.base_url = "moa://local"
    agent._client_kwargs = {}
    agent.client = MoAClient("coding", reference_callback=callback)

    def forbidden_openai_rebuild(*_a, **_k):
        raise AssertionError("MoA recovery must not construct an OpenAI SDK client")

    monkeypatch.setattr(agent, "_create_openai_client", forbidden_openai_rebuild)

    old_client = agent.client
    assert agent._replace_primary_openai_client(reason="stream retry") is True
    assert agent.client is not old_client
    assert agent.client.chat.completions.reference_callback is callback
