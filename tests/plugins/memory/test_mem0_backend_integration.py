"""Integration coverage for Hermes' pinned Mem0 OSS boundary."""

import copy
import os
from types import SimpleNamespace

import pytest


pytest.importorskip("mem0", reason="requires the existing mem0 extra")


def test_openai_backend_uses_real_mem0_config_and_factory(monkeypatch, tmp_path):
    mem0_dir = tmp_path / "mem0"
    monkeypatch.setenv("MEM0_DIR", str(mem0_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "environment-openai-sentinel")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-sentinel")

    import openai
    from mem0.memory import main as memory_main
    from mem0.utils.factory import LlmFactory

    from plugins.memory.mem0._backend import OSSBackend
    from plugins.memory.mem0._openai_llm import DirectOpenAILLM

    clients = []
    requests = []

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )
            clients.append(self)

        @staticmethod
        def _create(**params):
            requests.append(params)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="direct answer",
                            tool_calls=None,
                        )
                    )
                ]
            )

    class DummyVectorStore:
        pass

    class DummyDB:
        def __init__(self, _path):
            pass

    monkeypatch.setattr(
        LlmFactory,
        "provider_to_class",
        dict(LlmFactory.provider_to_class),
    )
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        memory_main.EmbedderFactory,
        "create",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        memory_main.VectorStoreFactory,
        "create",
        lambda *_args, **_kwargs: DummyVectorStore(),
    )
    monkeypatch.setattr(memory_main, "SQLiteManager", DummyDB)
    monkeypatch.setattr(memory_main, "MEM0_TELEMETRY", False)
    monkeypatch.setattr(memory_main, "capture_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        OSSBackend,
        "_recreate_collection_if_dims_changed",
        staticmethod(lambda *_args, **_kwargs: None),
    )

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-5-mini",
                "api_key": "configured-openai-sentinel",
                "openai_base_url": "https://openai.example/v1",
                "models": ["router-model"],
                "route": "lowest-latency",
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "ollama_base_url": "http://ollama.example:11434",
                "embedding_dims": 768,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": str(tmp_path / "qdrant"),
            },
        },
    }
    original_config = copy.deepcopy(config)
    environment = dict(os.environ)

    backend = OSSBackend(config)
    result = backend._memory.llm.generate_response(
        [{"role": "user", "content": "remember tea"}]
    )

    assert isinstance(backend._memory.llm, DirectOpenAILLM)
    assert len(clients) == 1
    assert clients[0].api_key == "configured-openai-sentinel"
    assert clients[0].base_url == "https://openai.example/v1"
    assert requests == [
        {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "remember tea"}],
        }
    ]
    assert result == "direct answer"
    assert config == original_config
    assert dict(os.environ) == environment
