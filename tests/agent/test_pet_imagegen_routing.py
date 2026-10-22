"""Pet generation must use the same native Krea route as the managed image tool."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider
from agent.pet.generate import imagegen
from plugins.image_gen import krea


class RecordingProvider(ImageGenProvider):
    def __init__(self, name: str, image: Path):
        self._name = name
        self.image = image
        self.calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def generate(self, prompt, aspect_ratio="square", **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "image": str(self.image)}


@pytest.fixture
def managed_images(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from tools import managed_tool_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_PET_IMAGE_PROVIDER", raising=False)
    monkeypatch.delenv("KREA_API_KEY", raising=False)
    monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
    config = {"image_gen": {"provider": "nous", "model": "krea-3"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "generated.png"
    Image.new("RGBA", (8, 8), "blue").save(output)

    image_gen_registry._registry.reset_for_tests()
    portal = RecordingProvider("nous", output)
    openai = RecordingProvider("openai", output)
    image_gen_registry._registry.register(portal)
    image_gen_registry._registry.register(openai)
    image_gen_registry._registry.register(krea.KreaImageGenProvider())
    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    monkeypatch.setattr(
        managed_tool_gateway,
        "is_managed_tool_gateway_ready",
        lambda name: name == "krea",
    )
    gateway = SimpleNamespace(
        gateway_origin="https://images.example.test",
        nous_user_token="synthetic-nous-token",
    )
    resolver = Mock(return_value=gateway)
    monkeypatch.setattr(managed_tool_gateway, "resolve_managed_tool_gateway", resolver)

    post = Mock(return_value=({"job_id": "synthetic-job"}, None))
    monkeypatch.setattr(krea, "post_json", post)
    poll = Mock(
        return_value={
            "status": "completed",
            "result": {"urls": ["https://images.example.test/result.png"]},
        }
    )
    monkeypatch.setattr(krea, "_poll_krea_job", poll)
    monkeypatch.setattr(krea, "save_url_image", lambda *args, **kwargs: output)
    yield SimpleNamespace(
        home=tmp_path,
        output=output,
        portal=portal,
        openai=openai,
        post=post,
        poll=poll,
        resolver=resolver,
    )
    image_gen_registry._registry.reset_for_tests()


@pytest.mark.parametrize("prefer", [None, "nous", "krea"])
def test_managed_krea_draft_and_reference_row(managed_images, prefer):
    """CLI default and Desktop picks preserve the model, Nous auth, and grounding."""
    env = managed_images
    sprite = imagegen.resolve_provider(prefer=prefer)
    assert sprite.provider.name == "krea"
    assert imagegen.generate("a synthetic pet", provider=sprite) == [env.output]
    assert imagegen.generate(
        "the same pet running",
        provider=sprite,
        reference_images=[env.output],
        aspect_ratio="landscape",
    ) == [env.output]

    assert env.portal.calls == []
    assert env.post.call_count == 2
    for call in env.post.call_args_list:
        assert call.args[0] == "https://images.example.test/generate/image/krea/krea-3"
        assert call.kwargs["headers"]["Authorization"] == "Bearer synthetic-nous-token"
        assert "x-idempotency-key" in call.kwargs["headers"]
    assert "image_urls" not in env.post.call_args_list[0].kwargs["payload"]
    reference = env.post.call_args_list[1].kwargs["payload"]["image_urls"][0]
    assert base64.b64decode(reference.split(",", 1)[1]) == env.output.read_bytes()
    for call in env.poll.call_args_list:
        assert call.args[:2] == ("https://images.example.test", "synthetic-nous-token")
    assert env.resolver.call_args.args == ("krea",)


def test_managed_krea_unavailable_does_not_fall_back_to_portal(
    managed_images, monkeypatch
):
    from tools import managed_tool_gateway

    monkeypatch.setattr(
        managed_tool_gateway, "is_managed_tool_gateway_ready", lambda name: False
    )
    with pytest.raises(imagegen.GenerationError, match="Krea gateway"):
        imagegen.resolve_provider()
    assert managed_images.portal.calls == []
    managed_images.post.assert_not_called()


def test_other_provider_selection_is_preserved(managed_images):
    sprite = imagegen.resolve_provider(prefer="openai")
    assert sprite.provider is managed_images.openai
    assert imagegen.generate("a synthetic pet", provider=sprite) == [
        managed_images.output
    ]
    managed_images.post.assert_not_called()


def test_portal_model_stays_on_portal(managed_images):
    import hermes_yaml as yaml

    config = {
        "image_gen": {"provider": "nous", "model": "synthetic/portal-image-model"}
    }
    (managed_images.home / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    sprite = imagegen.resolve_provider()
    assert sprite.provider is managed_images.portal
    imagegen.generate("a synthetic pet", provider=sprite)
    assert managed_images.portal.calls[0]["model"] == "synthetic/portal-image-model"
    managed_images.post.assert_not_called()


def test_direct_krea_keeps_its_own_credentials(managed_images, monkeypatch):
    import hermes_yaml as yaml

    config = {"image_gen": {"provider": "krea", "model": "krea-3"}}
    (managed_images.home / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    monkeypatch.setenv("KREA_API_KEY", "synthetic-krea-key")
    sprite = imagegen.resolve_provider()
    imagegen.generate("a synthetic pet", provider=sprite)
    call = managed_images.post.call_args
    assert call.args[0] == "https://api.krea.ai/generate/image/krea/krea-3"
    assert call.kwargs["headers"]["Authorization"] == "Bearer synthetic-krea-key"
    assert "x-idempotency-key" not in call.kwargs["headers"]
    managed_images.resolver.assert_not_called()
