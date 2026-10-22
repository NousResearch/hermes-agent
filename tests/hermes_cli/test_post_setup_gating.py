"""Tests for the post_setup install-state gate in `_toolset_needs_configuration_prompt`.

Regression coverage for the cua-driver silent-no-op bug (issue #22737).

When a no-key provider's only install side-effect is a `post_setup` hook
(cua-driver, etc.), the gate function used to fall through to the
`_toolset_has_keys` catch-all, which returned True for any provider with
empty `env_vars` — causing `hermes tools` to write the toolset to config
and exit `✓ Saved` without ever invoking the post_setup install. These
tests pin the new predicate-aware behaviour so the regression doesn't
sneak back in.
"""

from __future__ import annotations


class TestPostSetupGate:
    def test_cua_driver_missing_forces_setup(self, monkeypatch, tmp_path):
        """When cua-driver isn't on PATH, the gate must return True so the
        provider-setup flow runs and triggers `_run_post_setup`."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr("shutil.which", lambda name, path=None: None)

        assert tools_config._toolset_needs_configuration_prompt(
            "computer_use", {}
        ) is True

    def test_incompatible_cua_driver_forces_setup(self, monkeypatch):
        from hermes_cli import tools_config, tools_config_post_setup

        monkeypatch.setattr(tools_config_post_setup, "_cua_driver_install_ready", lambda: False)

        assert tools_config._toolset_needs_configuration_prompt(
            "computer_use", {}
        ) is True

    def test_compatible_cua_driver_skips_setup(self, monkeypatch):
        from hermes_cli import tools_config, tools_config_post_setup

        monkeypatch.setattr(tools_config_post_setup, "_cua_driver_install_ready", lambda: True)

        assert tools_config._toolset_needs_configuration_prompt(
            "computer_use", {}
        ) is False


    def test_post_setup_predicate_exception_does_not_block(self, monkeypatch):
        """A predicate that raises must be treated as 'satisfied' so a
        broken check can't strand the user in an infinite setup loop."""
        from hermes_cli import tools_config

        def _boom():
            raise RuntimeError("predicate broken")

        monkeypatch.setitem(tools_config._POST_SETUP_INSTALLED, "cua_driver", _boom)
        assert tools_config._post_setup_already_installed("cua_driver") is True


class TestBrowserBackendPrompt:
    """Regression: `_toolset_needs_configuration_prompt` for the browser toolset
    only checked `browser.cloud_provider` (set by `browser_provider` rows),
    ignoring `browser.backend` (set by the `browser_backend` "Browser Use" row).
    This made the provider picker re-appear every time `hermes tools` was
    opened, even when Browser Use was already configured.
    """


    def test_browser_cloud_provider_set_skips_provider_picker(self, monkeypatch, tmp_path):
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = {"browser": {"cloud_provider": "local"}}
        assert tools_config._toolset_needs_configuration_prompt("browser", config) is False


    def test_browser_empty_still_prompts(self, monkeypatch, tmp_path):
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = {"browser": None}
        assert tools_config._toolset_needs_configuration_prompt("browser", config) is True

    def test_browser_backend_off_still_skips_prompt(self, monkeypatch, tmp_path):
        """YAML 1.1 parses unquoted `off` as boolean False — the helper must
        normalise it, and the gate should still treat it as 'configured'."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = {"browser": {"backend": False}}  # what YAML `off` becomes
        assert tools_config._toolset_needs_configuration_prompt("browser", config) is False


class TestBrowserBackendPromptThroughLoader:
    """The browser gate must hold through the real config loader.

    `load_config()` merges ``DEFAULT_CONFIG``, where ``browser.backend`` is ``""`` — so the key is
    present on every install and a presence test would suppress the picker everywhere. These pin the
    behaviour against the merged dict a real ``hermes tools`` run feeds the gate.
    """

    def _home(self, tmp_path, body: str):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
        return tmp_path

    def test_unset_browser_still_prompts(self, monkeypatch, tmp_path):
        from hermes_cli import tools_config
        from hermes_cli.config import load_config

        monkeypatch.setenv("HERMES_HOME", str(self._home(tmp_path, "cli: {}\n")))
        config = load_config()
        assert config["browser"]["backend"] == ""  # defaults merge fills the key
        assert tools_config._toolset_needs_configuration_prompt("browser", config) is True

    def test_explicit_backend_skips_prompt(self, monkeypatch, tmp_path):
        from hermes_cli import tools_config
        from hermes_cli.config import load_config

        monkeypatch.setenv("HERMES_HOME", str(self._home(tmp_path, "browser:\n  backend: browser-use\n")))
        config = load_config()
        assert tools_config._toolset_needs_configuration_prompt("browser", config) is False
