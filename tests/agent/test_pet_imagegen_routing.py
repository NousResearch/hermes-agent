"""Pet generation must use the same native Krea route as the managed image tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from hermes_agent.agent import image_gen_registry
from hermes_agent.agent.image_gen_provider import ImageGenProvider
from hermes_agent.agent.pet.generate import imagegen


class RecordingProvider(ImageGenProvider):
    def __init__(self, name: str, image: Path):
        self._name = name
        self.image = image
        self.available = True
        self.calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def generate(self, prompt, aspect_ratio="square", **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "image": str(self.image)}


def _write_config(home: Path, text: str) -> None:
    (home / "config.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def providers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PET_IMAGE_PROVIDER", raising=False)
    _write_config(tmp_path, "image_gen:\n  provider: nous\n  model: krea-3\n")
    output = tmp_path / "generated.png"
    Image.new("RGBA", (8, 8), "blue").save(output)
    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    image_gen_registry._registry.reset_for_tests()
    recorders = {name: RecordingProvider(name, output) for name in ("nous", "openai", "openrouter", "krea")}
    for provider in recorders.values():
        image_gen_registry._registry.register(provider)
    yield SimpleNamespace(home=tmp_path, output=output, **recorders)
    image_gen_registry._registry.reset_for_tests()


@pytest.mark.parametrize("prefer", [None, "nous"])
def test_nous_krea_model_routes_draft_and_row_to_krea(providers, prefer):
    sprite = imagegen.resolve_provider(prefer=prefer)
    assert (sprite.name, sprite.provider) == ("nous", providers.krea)
    imagegen.generate("a pet", provider=sprite)
    imagegen.generate("a pet running", provider=sprite, reference_images=[providers.output])

    assert providers.nous.calls == []
    assert [call["model"] for call in providers.krea.calls] == ["krea-3", "krea-3"]
    assert providers.krea.calls[1]["reference_images"] == [str(providers.output)]


def test_krea_unavailable_skips_nous_and_keeps_scanning(providers):
    providers.krea.available = False
    assert imagegen.resolve_provider().provider is providers.openai
    assert "nous" not in [row["name"] for row in imagegen.list_sprite_providers()]

    providers.openai.available = providers.openrouter.available = False
    for require_references in (True, False):
        with pytest.raises(imagegen.GenerationError, match="Krea gateway"):
            imagegen.resolve_provider(require_references=require_references)
    assert providers.nous.calls == []


@pytest.mark.parametrize(
    ("config", "prefer"),
    [
        # Another backend picked while the stored model is a Krea id.
        ("image_gen:\n  provider: nous\n  model: krea-3\n", "openrouter"),
        ("image_gen:\n  provider: nous\n  model: krea-3\n", "openai"),
        ("image_gen:\n  provider: nous\n  model: krea-3\n", "krea"),
        # Nous without a Krea model keeps the Portal's own chain (env, image_gen.nous.model).
        ("image_gen:\n  provider: nous\n  model: fal-ai/flux-2/klein/9b\n", None),
        ("image_gen:\n  provider: fal\n  model: fal-ai/flux-2/klein/9b\n", None),
        # A key with no value, or a non-mapping value, must not crash the resolver.
        ("image_gen:\n", None),
        ("image_gen: nous\n", None),
    ],
)
def test_other_choices_keep_their_own_model_resolution(providers, config, prefer):
    _write_config(providers.home, config)
    expected = getattr(providers, prefer or "nous")
    sprite = imagegen.resolve_provider(prefer=prefer)
    assert sprite.provider is expected
    imagegen.generate("a pet", provider=sprite)
    assert "model" not in expected.calls[0]
