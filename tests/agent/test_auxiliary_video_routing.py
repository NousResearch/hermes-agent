"""Regression tests for auxiliary.video routing (#132701)."""

from __future__ import annotations

from agent.auxiliary_client import (
    _is_vision_media_task,
    _resolve_task_provider_model,
)
from tools.vision_tools import _aux_call_kwargs


def test_is_vision_media_task_includes_video():
    assert _is_vision_media_task("vision") is True
    assert _is_vision_media_task("video") is True
    assert _is_vision_media_task("compression") is False
    assert _is_vision_media_task(None) is False


def test_aux_call_kwargs_video_task_label():
    kw = _aux_call_kwargs(
        [{"role": "user", "content": "x"}], None, 180.0, min_timeout=180.0, task="video"
    )
    assert kw["task"] == "video"
    assert kw["timeout"] >= 180.0


def test_aux_call_kwargs_vision_default_unchanged():
    kw = _aux_call_kwargs([{"role": "user", "content": "x"}], "m", 120.0)
    assert kw["task"] == "vision"
    assert kw["model"] == "m"


def test_video_resolves_dedicated_provider(monkeypatch):
    def fake_task_config(task: str):
        if task == "video":
            return {"provider": "openrouter", "model": "google/gemini-3.8-flash"}
        if task == "vision":
            return {"provider": "ollama-launch", "model": "deepseek-v4.1-flash:cloud"}
        return {}

    monkeypatch.setattr(
        "agent.auxiliary_client._get_auxiliary_task_config", fake_task_config
    )
    provider, model, *_ = _resolve_task_provider_model("video")
    assert provider == "openrouter"
    assert model == "google/gemini-3.8-flash"


def test_video_inherits_vision_when_unset(monkeypatch):
    def fake_task_config(task: str):
        if task == "video":
            return {"provider": "auto", "model": ""}
        if task == "vision":
            return {"provider": "ollama-launch", "model": "local-vision"}
        return {}

    monkeypatch.setattr(
        "agent.auxiliary_client._get_auxiliary_task_config", fake_task_config
    )
    provider, model, *_ = _resolve_task_provider_model("video")
    assert provider == "ollama-launch"
    assert model == "local-vision"


def test_config_defaults_registers_video():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    video = DEFAULT_CONFIG["auxiliary"]["video"]
    assert video["provider"] == "auto"
    assert video["timeout"] == 180
