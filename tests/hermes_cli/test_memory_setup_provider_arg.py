"""Tests for `hermes memory setup [provider]` routing.

The `memory setup` subcommand accepts an optional positional ``provider`` so a
fresh install can configure a specific provider directly (e.g.
``hermes memory setup honcho``) without the interactive picker — which matters
because the per-provider ``hermes <provider>`` subcommand is only registered
once that provider is active.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import memory_setup


class TestMemorySetupProviderRouting:
    def test_setup_with_provider_arg_skips_picker(self):
        """`memory setup honcho` routes straight to cmd_setup_provider."""
        args = SimpleNamespace(memory_command="setup", provider="honcho")
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        direct.assert_called_once_with("honcho")
        picker.assert_not_called()

    def test_setup_without_provider_runs_picker(self):
        """`memory setup` (no provider) runs the interactive picker."""
        args = SimpleNamespace(memory_command="setup", provider=None)
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        picker.assert_called_once_with(args)
        direct.assert_not_called()

    def test_setup_with_missing_provider_attr_runs_picker(self):
        """A SimpleNamespace lacking `provider` must not crash — fall back to picker."""
        args = SimpleNamespace(memory_command="setup")
        with patch.object(memory_setup, "cmd_setup_provider") as direct, \
             patch.object(memory_setup, "cmd_setup") as picker:
            memory_setup.memory_command(args)
        picker.assert_called_once_with(args)
        direct.assert_not_called()

    def test_picker_defaults_to_active_provider(self):
        """The interactive picker should highlight the configured provider,
        not always fall back to the Built-in only entry."""
        providers = [
            ("honcho", "API key / local", object()),
            ("holographic", "local", object()),
        ]
        expected_items = [
            ("honcho", "— API key / local"),
            ("holographic", "— local"),
            ("Built-in only", "— MEMORY.md / USER.md (default)"),
        ]

        with patch.object(memory_setup, "_get_available_providers", return_value=providers), \
             patch.object(memory_setup, "_curses_select", return_value=memory_setup._CANCELLED) as picker, \
             patch.object(memory_setup, "_print_cancelled_setup"), \
             patch("hermes_cli.config.load_config_readonly", return_value={"memory": {"provider": "holographic"}}):
            memory_setup.cmd_setup(SimpleNamespace())

        picker.assert_called_once_with(
            "Memory provider setup",
            expected_items,
            default=1,
            cancel_returns=memory_setup._CANCELLED,
        )

    def test_picker_defaults_to_builtin_when_active_provider_is_unavailable_or_malformed(self):
        """Missing, unavailable, and malformed provider config use the safe fallback."""
        providers = [
            ("honcho", "API key / local", object()),
            ("holographic", "local", object()),
        ]
        expected_items = [
            ("honcho", "— API key / local"),
            ("holographic", "— local"),
            ("Built-in only", "— MEMORY.md / USER.md (default)"),
        ]
        configs = [
            {},
            {"memory": {"provider": "unavailable"}},
            {"memory": "malformed"},
        ]

        for config in configs:
            with patch.object(memory_setup, "_get_available_providers", return_value=providers), \
                 patch.object(memory_setup, "_curses_select", return_value=memory_setup._CANCELLED) as picker, \
                 patch.object(memory_setup, "_print_cancelled_setup"), \
                 patch("hermes_cli.config.load_config_readonly", return_value=config):
                memory_setup.cmd_setup(SimpleNamespace())

            picker.assert_called_once_with(
                "Memory provider setup",
                expected_items,
                default=2,
                cancel_returns=memory_setup._CANCELLED,
            )

    def test_unknown_provider_reports_and_returns_early(self, capsys):
        """An unknown provider name surfaces a helpful message and returns
        before any config load/save (the not-found guard precedes those imports)."""
        memory_setup.cmd_setup_provider("notaprovider")
        out = capsys.readouterr().out
        assert "not found" in out
        assert "hermes memory setup" in out


class TestInstallDependenciesRunner:
    """`_install_dependencies` must route through the canonical
    ``_pip_install`` ladder (uv → pip → ensurepip): uv when present, standard
    pip when uv is unavailable, and an ensurepip bootstrap for pip-less venvs
    instead of dead-ending with "cannot install"."""

    def _run_with_missing_dep(self, tmp_path, which_side_effect, run_behavior=None):
        """Drive _install_dependencies for a plugin that declares one missing
        pip dep, capturing every subprocess.run argv issued by the ladder."""
        import sys

        (tmp_path / "plugin.yaml").write_text(
            "pip_dependencies:\n  - definitely-not-installed-xyz\n", encoding="utf-8"
        )
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if run_behavior:
                return run_behavior(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("plugins.memory.find_provider_dir", return_value=tmp_path), \
             patch("hermes_cli.tools_config.shutil.which", side_effect=which_side_effect), \
             patch("hermes_cli.tools_config.subprocess.run", fake_run):
            memory_setup._install_dependencies("x")
        return calls, sys.executable

    def test_uses_uv_when_available(self, tmp_path):
        calls, _ = self._run_with_missing_dep(
            tmp_path, lambda b: "/usr/bin/uv" if b == "uv" else None
        )
        assert calls
        assert calls[0][:3] == ["/usr/bin/uv", "pip", "install"]

    def test_falls_back_to_pip_when_uv_missing(self, tmp_path):
        """No uv but pip importable -> python -m pip install."""
        calls, py = self._run_with_missing_dep(tmp_path, lambda b: None)
        assert calls
        # Ladder probes pip first, then installs with it.
        assert calls[0][:3] == [py, "-m", "pip"]
        assert calls[-1][:4] == [py, "-m", "pip", "install"]

    def test_bootstraps_pip_via_ensurepip_when_missing(self, tmp_path):
        """Neither uv nor pip -> ensurepip bootstrap, then pip install."""
        def behavior(cmd):
            if cmd[-1] == "--version":
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        calls, py = self._run_with_missing_dep(tmp_path, lambda b: None, behavior)
        assert any("ensurepip" in c for c in calls)
        assert calls[-1][:4] == [py, "-m", "pip", "install"]
