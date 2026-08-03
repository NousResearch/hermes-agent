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
        monkeypatch.setattr(tools_config.shutil, "which", lambda name: None)

        assert tools_config._toolset_needs_configuration_prompt(
            "computer_use", {}
        ) is True

    def test_cua_driver_installed_skips_setup(self, monkeypatch, tmp_path):
        """When cua-driver is already on PATH, the gate must return False
        so a re-save through `hermes tools` doesn't re-prompt the user."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            tools_config.shutil,
            "which",
            lambda name: "/usr/local/bin/cua-driver" if name == "cua-driver" else None,
        )

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

    def test_unregistered_post_setup_treated_as_satisfied(self):
        """post_setup keys without a registered predicate must default to
        'satisfied' so we don't change behaviour for hooks we haven't
        explicitly opted in (kittentts, piper, agent_browser, etc.)."""
        from hermes_cli import tools_config

        assert tools_config._post_setup_already_installed("does_not_exist") is True

    def test_cua_driver_predicate_registered(self):
        """Keep an explicit pin on the cua_driver entry so accidental
        deletion of the registry row would fail this test rather than
        silently restore the original silent-no-op bug."""
        from hermes_cli import tools_config

        assert "cua_driver" in tools_config._POST_SETUP_INSTALLED
