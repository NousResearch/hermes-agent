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


class TestPluginVideoProvidersRow:
    """Tests for _plugin_video_gen_providers row contents."""

    def test_post_setup_propagated_when_declared(self, monkeypatch):
        from hermes_cli import tools_config

        video_gen_registry.register_provider(_FakeVideoProvider(
            "xai_video",
            schema={
                "name": "xAI Grok Imagine",
                "badge": "paid",
                "tag": "grok video",
                "env_vars": [],
                "post_setup": "xai_grok",
            },
        ))

        rows = tools_config._plugin_video_gen_providers()
        match = next(r for r in rows if r.get("video_gen_plugin_name") == "xai_video")
        assert match["post_setup"] == "xai_grok"

    def test_post_setup_omitted_when_not_declared(self, monkeypatch):
        from hermes_cli import tools_config

        video_gen_registry.register_provider(_FakeVideoProvider("plain_video"))

        rows = tools_config._plugin_video_gen_providers()
        match = next(r for r in rows if r.get("video_gen_plugin_name") == "plain_video")
        assert "post_setup" not in match


class TestVideoPluginProviderActive:
    """Tests for _is_provider_active recognizing video_gen_plugin_name."""

    def test_active_when_video_gen_provider_matches(self):
        from hermes_cli import tools_config

        config = {"video_gen": {"provider": "xai"}}
        row = {"name": "xAI Grok Imagine", "video_gen_plugin_name": "xai"}

        assert tools_config._is_provider_active(row, config) is True

    def test_inactive_when_video_gen_provider_differs(self):
        from hermes_cli import tools_config

        config = {"video_gen": {"provider": "fal"}}
        row = {"name": "xAI Grok Imagine", "video_gen_plugin_name": "xai"}

        assert tools_config._is_provider_active(row, config) is False

    def test_inactive_when_video_gen_section_missing(self):
        from hermes_cli import tools_config

        row = {"name": "xAI Grok Imagine", "video_gen_plugin_name": "xai"}
        assert tools_config._is_provider_active(row, {}) is False

    def test_detect_active_index_picks_video_plugin_match(self, monkeypatch):
        """When xAI is the configured video_gen provider, the picker should
        default to the xAI row even if FAL_KEY happens to be set in env.

        Regression: previously _detect_active_provider_index() saw
        _is_provider_active(xai) return False (no video_gen branch),
        skipped xAI (empty env_vars), and matched the FAL row via the
        env-var fallback — so the picker visually defaulted to FAL even
        though the user picked xAI. The xAI row uses empty env_vars
        because authentication is handled via xAI Grok OAuth (post_setup
        hook).
        """
        from hermes_cli import tools_config

        monkeypatch.setattr(
            tools_config,
            "get_env_value",
            lambda key: "fal-key" if key == "FAL_KEY" else "",
        )

        config = {"video_gen": {"provider": "xai"}}
        providers = [
            {"name": "xAI Grok Imagine", "env_vars": [], "video_gen_plugin_name": "xai"},
            {
                "name": "FAL.ai",
                "env_vars": [{"key": "FAL_KEY", "prompt": "FAL"}],
                "video_gen_plugin_name": "fal",
            },
        ]

        assert tools_config._detect_active_provider_index(providers, config) == 0
