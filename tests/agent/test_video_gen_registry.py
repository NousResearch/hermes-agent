from __future__ import annotations

from typing import Any

from agent.video_gen_provider import VideoGenProvider
from agent import video_gen_registry


class DummyVideoProvider(VideoGenProvider):
    def __init__(self, name: str, available: bool):
        self._name = name
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - not used here
        return {"success": True, "prompt": prompt, "provider": self.name}


def setup_function() -> None:
    video_gen_registry._reset_for_tests()


def teardown_function() -> None:
    video_gen_registry._reset_for_tests()


def test_single_unavailable_provider_is_not_auto_selected(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"video_gen": {"provider": ""}})
    provider = DummyVideoProvider("dummy", available=False)
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_active_provider() is None


def test_configured_unavailable_provider_is_returned(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"video_gen": {"provider": "dummy"}})
    provider = DummyVideoProvider("dummy", available=False)
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_active_provider() is provider


def test_single_available_provider_is_auto_selected(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"video_gen": {"provider": ""}})
    provider = DummyVideoProvider("dummy", available=True)
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_active_provider() is provider
