"""Tests for agent/video_gen_registry.py — provider registration & active lookup."""

from __future__ import annotations

import pytest

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


class _FakeProvider(VideoGenProvider):
    def __init__(self, name: str, available: bool = True):
        self._name = name
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, **kw):
        return {"success": True, "video": f"{self._name}://{prompt}"}


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


class TestRegisterProvider:
    def test_register_and_lookup(self):
        provider = _FakeProvider("fake")
        video_gen_registry.register_provider(provider)
        assert video_gen_registry.get_provider("fake") is provider

    def test_rejects_non_provider(self):
        with pytest.raises(TypeError):
            video_gen_registry.register_provider("not a provider")  # type: ignore[arg-type]

    def test_rejects_empty_name(self):
        class Empty(VideoGenProvider):
            @property
            def name(self) -> str:
                return ""

            def generate(self, prompt, **kw):
                return {}

        with pytest.raises(ValueError):
            video_gen_registry.register_provider(Empty())

    def test_reregister_overwrites(self):
        a = _FakeProvider("same")
        b = _FakeProvider("same")
        video_gen_registry.register_provider(a)
        video_gen_registry.register_provider(b)
        assert video_gen_registry.get_provider("same") is b

    def test_list_is_sorted(self):
        video_gen_registry.register_provider(_FakeProvider("zeta"))
        video_gen_registry.register_provider(_FakeProvider("alpha"))
        names = [p.name for p in video_gen_registry.list_providers()]
        assert names == ["alpha", "zeta"]


class TestGetActiveProvider:
    def test_single_provider_autoresolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        video_gen_registry.register_provider(_FakeProvider("solo"))
        active = video_gen_registry.get_active_provider()
        assert active is not None and active.name == "solo"

    def test_no_provider_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert video_gen_registry.get_active_provider() is None

    def test_multi_without_config_returns_none(self, tmp_path, monkeypatch):
        """Unlike image_gen (which falls back to 'fal'), video_gen has no
        legacy default — when there are multiple providers and no config,
        the registry returns None and the tool surfaces a helpful error.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        video_gen_registry.register_provider(_FakeProvider("xai"))
        video_gen_registry.register_provider(_FakeProvider("fal"))
        assert video_gen_registry.get_active_provider() is None

    def test_config_selects_provider(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"video_gen": {"provider": "fal"}})
        )
        video_gen_registry.register_provider(_FakeProvider("xai"))
        video_gen_registry.register_provider(_FakeProvider("fal"))
        active = video_gen_registry.get_active_provider()
        assert active is not None and active.name == "fal"

    def test_unknown_config_falls_back(self, tmp_path, monkeypatch):
        """If video_gen.provider names a provider that isn't registered,
        the single-provider fallback still applies."""
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"video_gen": {"provider": "ghost"}})
        )
        video_gen_registry.register_provider(_FakeProvider("only"))
        active = video_gen_registry.get_active_provider()
        assert active is not None and active.name == "only"
