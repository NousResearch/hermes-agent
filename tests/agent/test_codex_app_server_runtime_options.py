"""Generic provider settings reach the Codex app-server without Core backend policy."""

from types import SimpleNamespace

from agent.codex_runtime import _codex_app_server_runtime_options
from agent.transports.codex_app_server_session import CodexAppServerSession
from providers.base import ProviderProfile


class _RuntimeProvider(ProviderProfile):
    def codex_app_server_runtime_options(self, *, model_config: dict) -> dict:
        return {"codex_home": model_config.get("codex_home"),
                "thread_config": {"model_context_window": model_config.get("window")}}


def test_provider_options_flow_through_generic_seam(monkeypatch):
    import providers
    provider = _RuntimeProvider(name="runtime-test")
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: provider)
    options = _codex_app_server_runtime_options(
        SimpleNamespace(provider="runtime-test"), {"model": {"codex_home": "/profile/codex", "window": 98304}})
    assert options == {"codex_home": "/profile/codex", "thread_config": {"model_context_window": 98304}}


def test_named_custom_model_provider_falls_back_to_codex_runtime_plugin(monkeypatch):
    import providers
    provider = _RuntimeProvider(name="openai-codex")
    custom = ProviderProfile(name="custom")
    monkeypatch.setattr(providers, "get_provider_profile",
                        lambda name: custom if name == "custom" else provider)
    options = _codex_app_server_runtime_options(
        SimpleNamespace(provider="custom"), {"model": {"codex_home": "/profile/codex", "window": 4096}})
    assert options["codex_home"] == "/profile/codex"
    assert options["thread_config"] == {"model_context_window": 4096}


def test_app_server_forwards_provider_thread_config_on_start_and_resume():
    class Client:
        def __init__(self):
            self.requests = []
        def initialize(self, **kwargs):
            pass
        def request(self, method, params, timeout):
            self.requests.append((method, params))
            return {"thread": {"id": params.get("threadId", "thread-1")}}

    config = {"model_context_window": 98304, "features": {"goals": False}}
    for resume_id in (None, "stored-1"):
        client = Client()
        session = CodexAppServerSession(
            client_factory=lambda **kw: client, thread_config=config,
            resume_thread_id=resume_id,
        )
        session.ensure_started()
        assert client.requests[0][1]["config"] == config


def test_provider_without_runtime_options_keeps_default():
    assert _codex_app_server_runtime_options(SimpleNamespace(provider="openrouter"), {"model": {}}) == {
        "codex_home": None, "thread_config": None,
    }
