"""Tests for Mem0Backend abstraction — PlatformBackend, OSSBackend, SelfHostedBackend."""

import copy
import importlib
import json
import os
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from plugins.memory.mem0._backend import (
    Mem0Backend,
    PlatformBackend,
    OSSBackend,
    SelfHostedBackend,
)


class FakePlatformClient:
    """Fake MemoryClient for PlatformBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"count": 1, "next": None, "results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"status": "PENDING", "event_id": "evt-1"}

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"id": kwargs["memory_id"], "text": kwargs["text"]}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


class TestPlatformBackend:

    def _make(self):
        client = FakePlatformClient()
        backend = PlatformBackend.__new__(PlatformBackend)
        backend._client = client
        return backend, client

    def test_search_forwards_params(self):
        backend, client = self._make()
        result = backend.search("test query", filters={"user_id": "u1"}, top_k=5)
        assert client.calls[0][0] == "search"
        assert client.calls[0][1] == "test query"
        assert client.calls[0][2]["filters"] == {"user_id": "u1"}
        assert client.calls[0][2]["top_k"] == 5


    def test_add_forwards_kwargs(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        result = backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        call = client.calls[0]
        assert call[2]["user_id"] == "u1"
        assert call[2]["infer"] is False
        # metadata kwarg should be omitted entirely when not provided so we
        # don't surprise older mem0 client versions with an unknown kwarg.
        assert "metadata" not in call[2]


    def test_update_forwards(self):
        backend, client = self._make()
        backend.update("m1", "new text")
        assert client.calls[0][1] == {"memory_id": "m1", "text": "new text"}

    def test_delete_forwards(self):
        backend, client = self._make()
        backend.delete("m1")
        assert client.calls[0][1] == {"memory_id": "m1"}


class FakeOSSMemory:
    """Fake mem0.Memory for OSSBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "event": "ADD"}]}

    def update(self, memory_id, **kwargs):
        self.calls.append(("update", memory_id, kwargs))
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"message": "Memory deleted successfully!"}


@dataclass
class _FakeMem0State:
    factory_registrations: list = field(default_factory=list)
    from_config_calls: int = 0
    clients: list = field(default_factory=list)
    requests: list = field(default_factory=list)


def _install_fake_mem0(monkeypatch):
    """Install a small mem0 2.0.10-shaped surface for OSS backend tests."""

    state = _FakeMem0State()

    class BaseLlmConfig:
        def __init__(
            self,
            model=None,
            temperature=0.1,
            api_key=None,
            max_tokens=2000,
            top_p=0.1,
            top_k=1,
            enable_vision=False,
            vision_details="auto",
            reasoning_effort=None,
            http_client_proxies=None,
            is_reasoning_model=None,
            **kwargs,
        ):
            self.model = model
            self.temperature = temperature
            self.api_key = api_key
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.top_k = top_k
            self.enable_vision = enable_vision
            self.vision_details = vision_details
            self.reasoning_effort = reasoning_effort
            self.http_client_proxies = http_client_proxies
            self.is_reasoning_model = is_reasoning_model
            for name, value in kwargs.items():
                setattr(self, name, value)

    class OpenAIConfig(BaseLlmConfig):
        def __init__(
            self,
            model=None,
            temperature=0.1,
            api_key=None,
            max_tokens=2000,
            top_p=0.1,
            top_k=1,
            enable_vision=False,
            vision_details="auto",
            reasoning_effort=None,
            http_client_proxies=None,
            is_reasoning_model=None,
            openai_base_url=None,
            models=None,
            route="fallback",
            openrouter_base_url=None,
            site_url=None,
            app_name=None,
            store=None,
            response_callback=None,
        ):
            super().__init__(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                enable_vision=enable_vision,
                vision_details=vision_details,
                reasoning_effort=reasoning_effort,
                http_client_proxies=http_client_proxies,
                is_reasoning_model=is_reasoning_model,
            )
            self.openai_base_url = openai_base_url
            self.models = models
            self.route = route
            self.openrouter_base_url = openrouter_base_url
            self.site_url = site_url
            self.app_name = app_name
            self.store = store
            self.response_callback = response_callback

    class LLMBase:
        def __init__(self, config=None):
            self.config = config or BaseLlmConfig()
            if not hasattr(self.config, "model"):
                raise ValueError("Configuration must have a 'model' attribute")

        def _get_supported_params(self, **kwargs):
            if self.config.is_reasoning_model:
                return {
                    name: kwargs[name]
                    for name in ("messages", "response_format", "tools", "tool_choice")
                    if name in kwargs
                }
            params = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
            params.update(kwargs)
            return params

    class OpenAILLM(LLMBase):
        @staticmethod
        def _parse_response(response, tools):
            if not tools:
                return response.choices[0].message.content
            parsed = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            for tool_call in response.choices[0].message.tool_calls or []:
                parsed["tool_calls"].append(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                )
            return parsed

    class Factory:
        provider_to_class = {
            "openai": ("mem0.llms.openai.OpenAILLM", OpenAIConfig),
            "ollama": ("mem0.llms.openai.OpenAILLM", BaseLlmConfig),
        }

        @classmethod
        def register_provider(cls, name, class_path, config_class=None):
            cls.provider_to_class[name] = (
                class_path,
                config_class or BaseLlmConfig,
            )
            state.factory_registrations.append((name, class_path, config_class))

        @classmethod
        def create(cls, provider_name, config=None, **kwargs):
            class_path, config_class = cls.provider_to_class[provider_name]
            if config is None:
                config = config_class(**kwargs)
            elif isinstance(config, dict):
                config = config_class(**config)
            module_name, class_name = class_path.rsplit(".", 1)
            llm_class = getattr(importlib.import_module(module_name), class_name)
            return llm_class(config)

    class MemoryConfig:
        def __init__(self, **config):
            llm = config["llm"]
            if llm["provider"] not in {"openai", "ollama"}:
                raise ValueError(
                    f"Unsupported LLM provider: {llm['provider']}"
                )
            self.llm = SimpleNamespace(
                provider=llm["provider"],
                config=copy.deepcopy(llm.get("config", {})),
            )
            embedder = config["embedder"]
            self.embedder = SimpleNamespace(
                provider=embedder["provider"],
                config=copy.deepcopy(embedder.get("config", {})),
            )
            vector_store = config["vector_store"]
            self.vector_store = SimpleNamespace(
                provider=vector_store["provider"],
                config=copy.deepcopy(vector_store.get("config", {})),
            )
            self.version = config.get("version", "v1.1")

    class Memory:
        instances = []

        def __init__(self, config):
            self.config = config
            self.llm = Factory.create(config.llm.provider, config.llm.config)
            self.embedding_model = SimpleNamespace(
                provider=config.embedder.provider,
                config=config.embedder.config,
            )
            self.vector_store = SimpleNamespace(
                provider=config.vector_store.provider,
                config=config.vector_store.config,
            )
            type(self).instances.append(self)

        @classmethod
        def from_config(cls, config):
            # This mirrors mem0 2.0.10: validation rejects the private provider
            # before the factory gets a chance to resolve its registration.
            state.from_config_calls += 1
            return cls(MemoryConfig(**config))

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            state.clients.append(self)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **params):
            state.requests.append(params)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="direct answer",
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="remember",
                                        arguments='{"fact": "tea"}',
                                    )
                                )
                            ],
                        )
                    )
                ]
            )

    package_names = {
        "mem0": types.ModuleType("mem0"),
        "mem0.configs": types.ModuleType("mem0.configs"),
        "mem0.configs.llms": types.ModuleType("mem0.configs.llms"),
        "mem0.llms": types.ModuleType("mem0.llms"),
        "mem0.utils": types.ModuleType("mem0.utils"),
        "mem0.configs.base": types.ModuleType("mem0.configs.base"),
        "mem0.configs.llms.base": types.ModuleType("mem0.configs.llms.base"),
        "mem0.configs.llms.openai": types.ModuleType("mem0.configs.llms.openai"),
        "mem0.llms.base": types.ModuleType("mem0.llms.base"),
        "mem0.llms.openai": types.ModuleType("mem0.llms.openai"),
        "mem0.utils.factory": types.ModuleType("mem0.utils.factory"),
        "openai": types.ModuleType("openai"),
    }
    setattr(package_names["mem0"], "Memory", Memory)
    setattr(package_names["mem0.configs.base"], "MemoryConfig", MemoryConfig)
    setattr(package_names["mem0.configs.llms.base"], "BaseLlmConfig", BaseLlmConfig)
    setattr(package_names["mem0.configs.llms.openai"], "OpenAIConfig", OpenAIConfig)
    setattr(package_names["mem0.llms.base"], "LLMBase", LLMBase)
    setattr(package_names["mem0.llms.openai"], "OpenAILLM", OpenAILLM)
    setattr(package_names["mem0.utils.factory"], "LlmFactory", Factory)
    setattr(package_names["openai"], "OpenAI", FakeOpenAI)
    for name, module in package_names.items():
        if name in {"mem0", "mem0.configs", "mem0.configs.llms", "mem0.llms", "mem0.utils"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    # The class-path registration imports this module after the fake mem0
    # surface is installed, so it binds to the test doubles above.
    monkeypatch.delitem(
        sys.modules, "plugins.memory.mem0._openai_llm", raising=False
    )
    return state, Memory, Factory


class TestOSSBackend:

    def _make(self):
        memory = FakeOSSMemory()
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = memory
        return backend, memory


    def test_legacy_api_base_aliases_are_normalized_before_mem0_init(self, monkeypatch):
        state, Memory, factory = _install_fake_mem0(monkeypatch)
        raw = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-5-mini",
                    "api_key": "openai-sentinel",
                    "api_base": "https://llm.example/v1",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {"model": "nomic-embed-text", "api_base": "http://ollama:11434"},
            },
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        before = copy.deepcopy(raw)
        environment = dict(os.environ)

        OSSBackend(raw)

        assert len(Memory.instances) == 1
        captured = Memory.instances[0].config
        assert captured.llm.provider == "hermes_openai"
        assert captured.llm.config["openai_base_url"] == "https://llm.example/v1"
        assert captured.embedder.provider == "ollama"
        assert captured.embedder.config["ollama_base_url"] == "http://ollama:11434"
        assert "api_base" not in captured.llm.config
        assert "api_base" not in captured.embedder.config
        assert factory.provider_to_class["hermes_openai"][1].__name__ == "OpenAIConfig"
        assert len(state.factory_registrations) == 1
        assert state.from_config_calls == 0
        assert raw == before
        assert dict(os.environ) == environment

    def test_direct_openai_uses_openai_credentials_and_request_shape(self, monkeypatch):
        state, _, factory = _install_fake_mem0(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "router-sentinel")
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-sentinel")

        module = importlib.import_module("plugins.memory.mem0._openai_llm")
        callback_calls = []
        config = factory.provider_to_class["openai"][1](
            model="gpt-5-mini",
            api_key="configured-openai-sentinel",
            openai_base_url="https://openai.example/v1",
            models=["router-model"],
            route="lowest-latency",
            site_url="https://hermes.example",
            app_name="Hermes",
            store=True,
            response_callback=lambda *args: callback_calls.append(args),
        )
        adapter = module.DirectOpenAILLM(config)
        assert adapter.config.is_reasoning_model is True
        tools = [
            {
                "type": "function",
                "function": {"name": "remember", "parameters": {}},
            }
        ]

        result = adapter.generate_response(
            [{"role": "user", "content": "remember tea"}],
            response_format={"type": "json_object"},
            tools=tools,
            tool_choice="required",
        )

        assert len(state.clients) == 1
        client = state.clients[0]
        assert client.api_key == "configured-openai-sentinel"
        assert client.base_url == "https://openai.example/v1"
        request = state.requests[0]
        assert request["model"] == "gpt-5-mini"
        assert request["tools"] == tools
        assert request["tool_choice"] == "required"
        assert request["response_format"] == {"type": "json_object"}
        assert request["store"] is True
        assert "models" not in request
        assert "route" not in request
        assert "extra_headers" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "max_tokens" not in request
        assert result == {
            "content": "direct answer",
            "tool_calls": [{"name": "remember", "arguments": {"fact": "tea"}}],
        }
        assert len(callback_calls) == 1
        assert callback_calls[0][0] is adapter
        assert callback_calls[0][2] == request

    def test_direct_openai_preserves_explicit_non_reasoning_override(self, monkeypatch):
        state, _, factory = _install_fake_mem0(monkeypatch)
        config = factory.provider_to_class["openai"][1](
            model="gpt-5-mini",
            api_key="configured-openai-sentinel",
            is_reasoning_model=False,
        )

        module = importlib.import_module("plugins.memory.mem0._openai_llm")
        adapter = module.DirectOpenAILLM(config)
        adapter.generate_response([{"role": "user", "content": "remember tea"}])

        assert adapter.config.is_reasoning_model is False
        request = state.requests[0]
        assert request["temperature"] == 0.1
        assert request["top_p"] == 0.1
        assert request["max_tokens"] == 2000

    def test_direct_openai_defaults_missing_model_to_reasoning_safe_mini(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "environment-openai-sentinel")
        _install_fake_mem0(monkeypatch)

        module = importlib.import_module("plugins.memory.mem0._openai_llm")
        adapter = module.DirectOpenAILLM()

        assert adapter.config.model == "gpt-5-mini"
        assert adapter.config.is_reasoning_model is True

    def test_direct_openai_uses_openai_environment_when_config_omits_values(self, monkeypatch):
        state, _, factory = _install_fake_mem0(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "router-sentinel")
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-sentinel")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env-openai.example/v1")

        module = importlib.import_module("plugins.memory.mem0._openai_llm")
        config = factory.provider_to_class["openai"][1](model="gpt-5-mini")
        adapter = module.DirectOpenAILLM(config)

        assert len(state.clients) == 1
        assert state.clients[0].api_key == "env-openai-sentinel"
        assert state.clients[0].base_url == "https://env-openai.example/v1"

    def test_missing_openai_key_fails_before_client_and_hides_router_secret(self, monkeypatch):
        state, _, factory = _install_fake_mem0(monkeypatch)
        router_secret = "router-secret-sentinel"
        monkeypatch.setenv("OPENROUTER_API_KEY", router_secret)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        module = importlib.import_module("plugins.memory.mem0._openai_llm")
        config = factory.provider_to_class["openai"][1](
            model="gpt-5-mini",
            api_key=None,
        )

        with pytest.raises(ValueError) as exc_info:
            module.DirectOpenAILLM(config)

        assert "OpenAI API key" in str(exc_info.value)
        assert router_secret not in str(exc_info.value)
        assert state.clients == []
        assert state.requests == []

    def test_registration_is_idempotent_and_clients_keep_instance_config(self, monkeypatch):
        state, Memory, factory = _install_fake_mem0(monkeypatch)
        first = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-5-mini",
                    "api_key": "first-openai-sentinel",
                    "openai_base_url": "https://first.example/v1",
                },
            },
            "embedder": {"provider": "ollama", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        second = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-5-mini",
                    "api_key": "second-openai-sentinel",
                    "openai_base_url": "https://second.example/v1",
                },
            },
            "embedder": {"provider": "ollama", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        first_before = copy.deepcopy(first)
        second_before = copy.deepcopy(second)

        OSSBackend(first)
        OSSBackend(second)

        assert len(state.factory_registrations) == 1
        assert factory.provider_to_class["hermes_openai"][0].endswith(
            "_openai_llm.DirectOpenAILLM"
        )
        assert [
            (client.api_key, client.base_url) for client in state.clients
        ] == [
            ("first-openai-sentinel", "https://first.example/v1"),
            ("second-openai-sentinel", "https://second.example/v1"),
        ]
        assert len(Memory.instances) == 2
        assert state.from_config_calls == 0
        assert first == first_before
        assert second == second_before

    def test_ollama_bypasses_direct_openai_adapter(self, monkeypatch):
        state, Memory, factory = _install_fake_mem0(monkeypatch)
        raw = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.1:8b",
                    "api_base": "http://ollama:11434",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "api_base": "http://ollama:11434",
                },
            },
            "vector_store": {"provider": "qdrant", "config": {}},
        }
        before = copy.deepcopy(raw)

        OSSBackend(raw)

        assert len(Memory.instances) == 1
        assert state.from_config_calls == 1
        assert Memory.instances[0].config.llm.provider == "ollama"
        assert Memory.instances[0].config.embedder.provider == "ollama"
        assert "hermes_openai" not in factory.provider_to_class
        assert state.clients == []
        assert raw == before


httpx = pytest.importorskip("httpx")


class _StubServer:
    """Records requests and serves the real self-hosted server's response shapes."""

    def __init__(self, rows=10):
        self.requests = []
        self._rows = [{"id": f"m{i}", "memory": f"f{i}"} for i in range(rows)]

    def handler(self, request):
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/search" and method == "POST":
            return httpx.Response(200, json={"results": [{"id": "m1", "memory": "tea", "score": 0.9}]})
        if path == "/memories" and method == "GET":
            top_k = int(request.url.params.get("top_k", len(self._rows)))
            return httpx.Response(200, json={"results": self._rows[:top_k]})
        if path == "/memories" and method == "POST":
            return httpx.Response(200, json={"results": [{"id": "new", "memory": "stored", "event": "ADD"}]})
        if path.startswith("/memories/") and method in ("PUT", "DELETE"):
            if path.endswith("/missing"):  # server 404s unknown ids
                return httpx.Response(404, json={"detail": "Memory not found"})
            verb = "updated" if method == "PUT" else "Memory deleted successfully"
            return httpx.Response(200, json={"message": verb})
        return httpx.Response(404, json={"detail": "not found"})


def _backend(server, api_key="adminkey", host="http://sh:8888"):
    """Build a SelfHostedBackend routed through the stub transport.

    Uses the real __init__ (via the injectable ``transport`` kwarg) so the
    constructor's header/base_url setup is exercised by every test here.
    """
    return SelfHostedBackend(
        api_key, host, transport=httpx.MockTransport(server.handler)
    )


class TestSelfHostedBackend:
    # --- constructor / auth setup (the crux of the bug) -------------------

    def test_init_uses_x_api_key_not_token_auth(self):
        b = SelfHostedBackend("adminkey", "http://sh:8888")
        assert b._client.headers["x-api-key"] == "adminkey"
        assert "authorization" not in b._client.headers  # NOT the cloud 'Token' scheme


    # --- search ----------------------------------------------------------


    # --- add / update / delete ------------------------------------------


    # --- error propagation (feeds the plugin's circuit breaker) ----------

    def test_http_error_raises(self):
        s = _StubServer()
        with pytest.raises(httpx.HTTPStatusError):
            _backend(s).delete("missing")  # 404 -> raise_for_status; 'not found' won't trip breaker


class _FakeQdrantClient:
    """Stands in for qdrant_client.QdrantClient: records construction and
    serves a stale-dimension collection."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.deleted = []
        _FakeQdrantClient.instances.append(self)

    def collection_exists(self, name):
        return True

    def get_collection(self, name):
        class _V:
            size = 768  # stale: expected 1536
        class _Params:
            vectors = _V()
        class _Info:
            config = type("C", (), {"params": _Params()})
        return _Info()

    def delete_collection(self, name):
        self.deleted.append(name)

    def close(self):
        pass


class TestDimChangeGuardQdrantShapes:
    """#91034: the guard must reach the dimension check for every valid
    Qdrant config shape, not only path/url."""

    def _guard(self, config):
        import plugins.memory.mem0._backend as backend_mod

        saved = backend_mod.__dict__.get("QdrantClient")
        _FakeQdrantClient.instances = []
        import sys, types

        fake_mod = types.ModuleType("qdrant_client")
        fake_mod.QdrantClient = _FakeQdrantClient
        sys.modules["qdrant_client"] = fake_mod
        try:
            backend_mod.OSSBackend._recreate_collection_if_dims_changed(
                "qdrant", config, expected_dims=1536,
            )
        finally:
            if saved is not None:
                backend_mod.__dict__["QdrantClient"] = saved
            del sys.modules["qdrant_client"]
        return _FakeQdrantClient.instances

    def test_host_port_config_reaches_dimension_check(self):
        clients = self._guard({
            "collection_name": "hermes_mem0",
            "host": "127.0.0.1",
            "port": 6333,
        })
        assert len(clients) == 1, "host+port config must not be skipped"
        assert clients[0].kwargs["host"] == "127.0.0.1"
        assert clients[0].kwargs["port"] == 6333
        assert clients[0].deleted == ["hermes_mem0"], "stale dims must be dropped"

    def test_url_config_still_works(self):
        clients = self._guard({
            "collection_name": "hermes_mem0",
            "url": "https://q.example.com:6333",
        })
        assert len(clients) == 1
        assert clients[0].deleted == ["hermes_mem0"]

    def test_path_config_still_works(self):
        clients = self._guard({
            "collection_name": "hermes_mem0",
            "path": "/tmp/qdrant-embed",
        })
        assert len(clients) == 1
        assert clients[0].deleted == ["hermes_mem0"]

    def test_unrecognized_shape_still_skips(self):
        clients = self._guard({"collection_name": "hermes_mem0"})
        assert clients == [], "no client should be built for an unknown shape"
