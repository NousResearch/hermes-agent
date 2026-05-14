"""Tests for plugin video_gen providers in the tools picker.

Covers the reconfigure path that previously failed to write
``video_gen.provider`` when a user picked an xAI/etc. plugin backend
through Reconfigure tool → Video Generation. The first-time configure
path already handled it; the reconfigure path forgot to mirror it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


class _FakeVideoProvider(VideoGenProvider):
    def __init__(
        self,
        name: str,
        available: bool = True,
        schema: Optional[Dict[str, Any]] = None,
        models: Optional[List[Dict[str, Any]]] = None,
    ):
        self._name = name
        self._available = available
        self._schema = schema or {
            "name": name.title(),
            "badge": "test",
            "tag": f"{name} test tag",
            "env_vars": [{"key": f"{name.upper()}_API_KEY", "prompt": f"{name} key"}],
        }
        self._models = models or [
            {
                "id": f"{name}-video-v1",
                "display": f"{name} v1",
                "speed": "~10s",
                "strengths": "test",
                "price": "$",
            },
        ]

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def list_models(self):
        return list(self._models)

    def default_model(self):
        return self._models[0]["id"] if self._models else None

    def get_setup_schema(self):
        return dict(self._schema)

    def generate(self, prompt, **kw):
        return {"success": True, "video": f"{self._name}://{prompt}"}


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


class TestReconfigureWritesProvider:
    """Regression tests for the video_gen reconfigure path.

    Before the fix, _reconfigure_provider() handled image_gen_plugin_name
    in both the no-env-vars branch and the post-env-vars branch but
    missed video_gen_plugin_name in both. Picking xAI via Reconfigure
    tool → Video Generation silently no-op'd: the env var was already
    set, the env-var loop ran (Enter to keep), and the function fell
    through without ever writing config["video_gen"]["provider"].
    """

    def test_reconfigure_with_env_vars_already_set_writes_provider(
        self, monkeypatch, tmp_path
    ):
        """Env vars present and user accepts current value → still writes
        video_gen.provider via the post-env-vars branch."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        video_gen_registry.register_provider(_FakeVideoProvider("xai_fake"))

        # Picker prompts replaced — no TTY in tests.
        monkeypatch.setattr(tools_config, "_prompt_choice", lambda *a, **kw: 0)
        # User presses Enter to keep the existing key.
        monkeypatch.setattr(tools_config, "_prompt", lambda *a, **kw: "")
        # Pretend the env var is already set so the reconfigure path
        # hits the "Kept current" branch.
        monkeypatch.setattr(
            tools_config,
            "get_env_value",
            lambda key: "sk-fake" if key == "XAI_FAKE_API_KEY" else "",
        )

        config: dict = {}
        provider_row = {
            "name": "xAI",
            "env_vars": [{"key": "XAI_FAKE_API_KEY", "prompt": "xAI key"}],
            "video_gen_plugin_name": "xai_fake",
        }

        tools_config._reconfigure_provider(provider_row, config)

        assert config["video_gen"]["provider"] == "xai_fake"
        assert config["video_gen"]["model"] == "xai_fake-video-v1"
        assert config["video_gen"]["use_gateway"] is False

    def test_reconfigure_with_no_env_vars_writes_provider(
        self, monkeypatch, tmp_path
    ):
        """No env vars at all (managed-style plugin) → writes
        video_gen.provider via the no-env-vars early-return branch."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        video_gen_registry.register_provider(_FakeVideoProvider(
            "noenv_video",
            schema={
                "name": "NoEnvVideo",
                "badge": "free",
                "tag": "",
                "env_vars": [],
            },
        ))
        monkeypatch.setattr(tools_config, "_prompt_choice", lambda *a, **kw: 0)

        config: dict = {}
        provider_row = {
            "name": "NoEnvVideo",
            "env_vars": [],
            "video_gen_plugin_name": "noenv_video",
        }

        tools_config._reconfigure_provider(provider_row, config)

        assert config["video_gen"]["provider"] == "noenv_video"
        assert config["video_gen"]["model"] == "noenv_video-video-v1"
        assert config["video_gen"]["use_gateway"] is False
