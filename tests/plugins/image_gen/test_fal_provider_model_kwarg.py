"""Regression tests for #149095 — a caller-selected FAL model must reach the API.

``FalImageGenProvider.generate()`` forwarded only its fixed kwarg whitelist, so a caller
asking the registered ``fal`` provider for a specific model got whatever
``image_gen.model`` was configured, and the response reported that configured model too.
These tests drive the real pipeline (``tools.image_generation_tool``) with only the FAL
submit/wait layer faked, and assert on the endpoint that was actually submitted.
"""

from __future__ import annotations

import pytest

CONFIGURED = "fal-ai/flux-2-pro"
REQUESTED = "fal-ai/nano-banana-pro"


@pytest.fixture
def fal_submit(monkeypatch, tmp_path):
    """Fake the FAL submit/wait layer; yields the captured ``{"endpoint", "arguments"}``."""
    import tools.image_generation_tool as image_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FAL_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_submit(model, arguments=None, **_kwargs):
        captured["endpoint"] = model
        captured["arguments"] = arguments
        return "handler"

    def fake_wait(_handler, **_kwargs):
        return {"images": [{"url": "https://example.com/out.png", "width": 1024, "height": 768}]}

    monkeypatch.setattr(image_tool, "_submit_fal_request", fake_submit)
    monkeypatch.setattr(image_tool, "_wait_fal_result", fake_wait)
    return captured


def configure_model(monkeypatch, model: str) -> None:
    """Pin ``image_gen.model`` without touching the real user config."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"image_gen": {"model": model}})


def test_caller_model_reaches_the_submitted_endpoint(fal_submit, monkeypatch):
    """The requested catalog model is the one submitted — not the configured default."""
    from plugins.image_gen.fal import FalImageGenProvider

    configure_model(monkeypatch, CONFIGURED)
    result = FalImageGenProvider().generate("a lighthouse at dusk", model=REQUESTED)

    assert result["success"] is True
    assert fal_submit["endpoint"] == REQUESTED
    assert result["model"] == REQUESTED


def test_configured_model_still_wins_without_a_caller_model(fal_submit, monkeypatch):
    """Backward compatibility: no ``model`` kwarg keeps resolving through config."""
    from plugins.image_gen.fal import FalImageGenProvider

    configure_model(monkeypatch, CONFIGURED)
    result = FalImageGenProvider().generate("a lighthouse at dusk")

    assert result["success"] is True
    assert fal_submit["endpoint"] == CONFIGURED
    assert result["model"] == CONFIGURED


def test_unknown_caller_model_falls_back_to_the_configured_one(fal_submit, monkeypatch):
    """An id outside ``FAL_MODELS`` must not break generation — it resolves like config does."""
    from plugins.image_gen.fal import FalImageGenProvider

    configure_model(monkeypatch, CONFIGURED)
    result = FalImageGenProvider().generate("a lighthouse at dusk", model="fal-ai/not-a-model")

    assert result["success"] is True
    assert fal_submit["endpoint"] == CONFIGURED
    assert result["model"] == CONFIGURED
