"""Provider-resolution tests for pet sprite generation.

Reference-capability is derived from ``capabilities()["modalities"]`` (the
``"image"`` modality = image-to-image / editing), so user plugins qualify for
pet generation without being hardcoded in ``_REF_CAPABLE``; the tuple remains a
fallback and preference-order hint. Everything here uses fake providers — no
network, no PIL — so unlike ``test_pet_generate.py`` this module is not gated
behind ``HERMES_RUN_SLOW_PET_TESTS``.
"""

from __future__ import annotations

import pytest

from agent.pet.generate import imagegen


class _Provider:
    """Minimal fake ImageGenProvider.

    ``capabilities()`` only exists when *modalities* is given, mimicking both
    plugins that declare capabilities and bare objects that don't.
    """

    def __init__(self, name, modalities=None, available=True, display_name=None):
        self.name = name
        self._available = available
        if display_name is not None:
            self.display_name = display_name
        if modalities is not None:
            self._modalities = list(modalities)
            self.capabilities = self._capabilities

    def _capabilities(self):
        return {"modalities": list(self._modalities), "max_reference_images": 4}

    def is_available(self):
        return self._available


class _BrokenCapsProvider(_Provider):
    def __init__(self, name, available=True):
        super().__init__(name, available=available)

    def capabilities(self):
        raise RuntimeError("boom")


class _Registry:
    def __init__(self):
        self.providers: dict[str, _Provider] = {}
        self.active: _Provider | None = None

    def add(self, provider: _Provider, active: bool = False) -> _Provider:
        self.providers[provider.name] = provider
        if active:
            self.active = provider
        return provider


@pytest.fixture
def registry(monkeypatch) -> _Registry:
    reg = _Registry()
    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: reg.providers.get(name))
    monkeypatch.setattr("agent.image_gen_registry.get_active_provider", lambda: reg.active)
    monkeypatch.setattr(
        "agent.image_gen_registry.list_providers",
        lambda: [reg.providers[k] for k in sorted(reg.providers)],
    )
    monkeypatch.delenv("HERMES_PET_IMAGE_PROVIDER", raising=False)
    return reg


# ───────────────────────── capability-derived resolution ─────────────────────────


def test_active_plugin_with_image_modality_is_reference_capable(registry):
    """A user plugin declaring img2img resolves as the active pet-gen backend."""
    registry.add(_Provider("comfyui", modalities=["text", "image"]), active=True)

    sprite = imagegen.resolve_provider(require_references=True)
    assert sprite.name == "comfyui"
    assert sprite.supports_references


def test_prefer_accepts_capable_plugin(registry):
    """The desktop picker can choose a capable plugin over the active builtin."""
    registry.add(_Provider("openai", modalities=["text", "image"]), active=True)
    registry.add(_Provider("comfyui", modalities=["text", "image"]))

    assert imagegen.resolve_provider(prefer="comfyui").name == "comfyui"


def test_fallback_discovery_finds_capable_plugin(registry):
    """With no active/preferred provider, a lone capable plugin is discovered."""
    registry.add(_Provider("comfyui", modalities=["text", "image"]))

    sprite = imagegen.resolve_provider(require_references=True)
    assert sprite.name == "comfyui"
    assert sprite.supports_references


def test_builtins_keep_preference_order_over_plugins(registry):
    """Fallback discovery prefers the builtin order; plugins come after."""
    registry.add(_Provider("comfyui", modalities=["text", "image"]))  # sorts before "openai"
    registry.add(_Provider("openai", modalities=["text", "image"]))

    assert imagegen.resolve_provider(require_references=True).name == "openai"


def test_text_only_plugin_is_not_reference_capable(registry):
    """A text-to-image-only plugin never grounds pets, active or preferred."""
    registry.add(_Provider("textgen", modalities=["text"]), active=True)

    with pytest.raises(imagegen.GenerationError):
        imagegen.resolve_provider(require_references=True)
    with pytest.raises(imagegen.GenerationError):
        imagegen.resolve_provider(require_references=True, prefer="textgen")

    # Prompt-only base drafts may still use it, flagged as ungrounded.
    sprite = imagegen.resolve_provider(require_references=False)
    assert sprite.name == "textgen"
    assert not sprite.supports_references


def test_ref_capable_tuple_is_fallback_for_missing_capabilities(registry):
    """Builtins without a capabilities() override keep working (tuple fallback)."""
    registry.add(_Provider("openai"), active=True)  # no capabilities() at all

    sprite = imagegen.resolve_provider(require_references=True)
    assert sprite.name == "openai"
    assert sprite.supports_references


def test_broken_capabilities_falls_back_to_tuple(registry):
    """A crashing capabilities() falls back to the name list instead of raising."""
    registry.add(_BrokenCapsProvider("krea"), active=True)
    assert imagegen.resolve_provider(require_references=True).name == "krea"

    registry.providers.clear()
    registry.add(_BrokenCapsProvider("weird"), active=True)
    with pytest.raises(imagegen.GenerationError):
        imagegen.resolve_provider(require_references=True)


# ───────────────────────── env override ─────────────────────────


def test_env_override_accepts_capable_plugin(registry, monkeypatch):
    """HERMES_PET_IMAGE_PROVIDER honors any registered ref-capable provider."""
    registry.add(_Provider("openai", modalities=["text", "image"]), active=True)
    registry.add(_Provider("comfyui", modalities=["text", "image"]))
    monkeypatch.setenv("HERMES_PET_IMAGE_PROVIDER", "comfyui")

    assert imagegen.resolve_provider(require_references=True).name == "comfyui"


def test_env_override_ignores_text_only_and_unknown(registry, monkeypatch):
    """Text-only or unregistered overrides are ignored (normal resolution wins)."""
    registry.add(_Provider("openai", modalities=["text", "image"]), active=True)
    registry.add(_Provider("textgen", modalities=["text"]))

    monkeypatch.setenv("HERMES_PET_IMAGE_PROVIDER", "textgen")
    assert imagegen.resolve_provider(require_references=True).name == "openai"

    monkeypatch.setenv("HERMES_PET_IMAGE_PROVIDER", "not-registered")
    assert imagegen.resolve_provider(require_references=True).name == "openai"


# ───────────────────────── picker listing ─────────────────────────


def test_list_sprite_providers_includes_capable_plugin(registry):
    """The picker lists capable plugins after the builtins, with a label."""
    registry.add(_Provider("openai", modalities=["text", "image"]), active=True)
    registry.add(
        _Provider("comfyui", modalities=["text", "image"], display_name="ComfyUI")
    )
    registry.add(_Provider("textgen", modalities=["text"]))

    listed = imagegen.list_sprite_providers()
    assert [p["name"] for p in listed] == ["openai", "comfyui"]
    by_name = {p["name"]: p for p in listed}
    # Builtin keeps its friendly label; the plugin falls back to display_name.
    assert by_name["openai"]["label"] == "OpenAI"
    assert by_name["comfyui"]["label"] == "ComfyUI"
    assert by_name["openai"]["default"] is True
    assert by_name["comfyui"]["default"] is False
