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


import pytest


@pytest.mark.parametrize("key,extra", [
    ("faster_whisper", "stt-whisper"), ("kittentts", "kittentts"),
    ("piper", "piper"), ("ddgs", "ddgs"), ("langfuse", "langfuse"),
])
@pytest.mark.parametrize("succeeds", [True, False])
def test_python_provider_setup_records_extra_and_preserves_failure(key, extra, succeeds, monkeypatch, capsys):
    import pm
    from hermes_cli import tools_config_post_setup as post, plugins_cmd

    calls = []
    enabled = []
    monkeypatch.setattr(post, "_importable", lambda module: False)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda names: enabled.extend(names))

    def sync(extras, *, explicit):
        calls.append((extras, explicit))
        if not succeeds:
            raise pm.InstallError("venv", "resolution refused")

    monkeypatch.setattr(pm, "sync_venv", sync)
    post._run_post_setup(key)
    assert calls == [([extra], True)]
    output = capsys.readouterr().out
    if succeeds:
        assert "Restart Hermes" in output
        if key == "langfuse":
            assert enabled == ["observability/langfuse"]
    else:
        assert "resolution refused" in output
        assert "Retry with: hermes tools" in output
        assert not enabled
