"""Pet image generation resolves providers from the image-gen registry capabilities."""

from __future__ import annotations

import pytest

from agent.image_gen_provider import ImageGenProvider
from agent.pet.generate import imagegen
import agent.image_gen_registry as registry


class FakeImageProvider(ImageGenProvider):
    def __init__(
        self, name: str, *, refs: bool, available: bool = True, display_name: str = "", raises_available: bool = False,
    ) -> None:
        self._name = name
        self._refs = refs
        self._available = available
        self._display_name = display_name or name.title()
        self._raises_available = raises_available

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    def is_available(self) -> bool:
        if self._raises_available:
            raise RuntimeError("availability probe failed")
        return self._available

    def capabilities(self) -> dict:
        return {
            "modalities": ["text", "image"] if self._refs else ["text"],
            "max_reference_images": 3 if self._refs else 0,
        }

    def generate(self, prompt: str, aspect_ratio: str = "landscape", **kwargs) -> dict:
        return {"success": True, "image": "/tmp/pet.png"}


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    registry._reset_for_tests()
    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    yield
    registry._reset_for_tests()


def test_pet_generator_honors_preferred_plugin_provider_with_reference_capability() -> None:
    registry.register_provider(FakeImageProvider("acme-painter", refs=True, display_name="Acme Painter"))

    chosen = imagegen.resolve_provider(prefer="acme-painter")

    assert chosen.name == "acme-painter"
    assert chosen.supports_references is True


def test_pet_generator_normalizes_forced_provider_name(monkeypatch) -> None:
    registry.register_provider(FakeImageProvider("openai", refs=True))
    monkeypatch.setenv("HERMES_PET_IMAGE_PROVIDER", "OpenAI")

    chosen = imagegen.resolve_provider()

    assert chosen.name == "openai"


def test_pet_provider_fallback_preserves_builtin_preference_order() -> None:
    registry.register_provider(FakeImageProvider("openrouter", refs=True))
    registry.register_provider(FakeImageProvider("openai", refs=True))

    chosen = imagegen.resolve_provider()

    assert chosen.name == "openai"


def test_pet_provider_picker_lists_plugin_reference_providers() -> None:
    registry.register_provider(FakeImageProvider("text-only", refs=False))
    registry.register_provider(FakeImageProvider("acme-painter", refs=True, display_name="Acme Painter"))

    providers = imagegen.list_sprite_providers()

    assert providers == [{"name": "acme-painter", "label": "Acme Painter", "default": True}]


def test_pet_generator_rejects_text_only_provider_when_references_required() -> None:
    registry.register_provider(FakeImageProvider("text-only", refs=False))

    with pytest.raises(imagegen.GenerationError, match="supports reference images"):
        imagegen.resolve_provider(require_references=True)


def test_pet_generator_skips_builtin_provider_that_cannot_consume_local_references() -> None:
    registry.register_provider(FakeImageProvider("fal", refs=True))

    with pytest.raises(imagegen.GenerationError, match="supports reference images"):
        imagegen.resolve_provider(require_references=True)


def test_pet_generator_skips_plugin_with_broken_availability_probe() -> None:
    registry.register_provider(FakeImageProvider("acme-painter", refs=True, raises_available=True))

    with pytest.raises(imagegen.GenerationError, match="supports reference images"):
        imagegen.resolve_provider(prefer="acme-painter")
