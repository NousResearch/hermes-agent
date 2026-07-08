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

    def test_unknown_provider_reports_and_returns_early(self, capsys):
        """An unknown provider name surfaces a helpful message and returns
        before any config load/save (the not-found guard precedes those imports)."""
        memory_setup.cmd_setup_provider("notaprovider")
        out = capsys.readouterr().out
        assert "not found" in out
        assert "hermes memory setup" in out


class TestInstallDependenciesRunner:
    """`_install_dependencies` must install via `uv` when present and fall back
    to standard `pip` when `uv` is unavailable (e.g. slim containers / CI images
    that don't ship uv) instead of dead-ending with "cannot install"."""

    def _run_with_missing_dep(self, tmp_path, which_side_effect):
        """Drive _install_dependencies for a plugin that declares one missing
        pip dep, capturing the subprocess.run argv (or None if never called)."""
        import sys

        (tmp_path / "plugin.yaml").write_text(
            "pip_dependencies:\n  - definitely-not-installed-xyz\n", encoding="utf-8"
        )
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return SimpleNamespace()

        with patch("plugins.memory.find_provider_dir", return_value=tmp_path), \
             patch("shutil.which", side_effect=which_side_effect), \
             patch("subprocess.run", fake_run):
            memory_setup._install_dependencies("x")
        return captured.get("cmd"), sys.executable

    def test_uses_uv_when_available(self, tmp_path):
        cmd, _ = self._run_with_missing_dep(
            tmp_path, lambda b: "/usr/bin/uv" if b == "uv" else None
        )
        assert cmd is not None
        assert cmd[:3] == ["/usr/bin/uv", "pip", "install"]

    def test_falls_back_to_pip_when_uv_missing(self, tmp_path, capsys):
        """The salvaged behavior (#5954): no uv but pip present -> python -m pip."""
        cmd, py = self._run_with_missing_dep(
            tmp_path, lambda b: "/usr/bin/pip3" if b == "pip3" else None
        )
        assert cmd is not None
        assert cmd[:4] == [py, "-m", "pip", "install"]
        assert "Falling back to standard pip" in capsys.readouterr().out

    def test_aborts_when_neither_uv_nor_pip(self, tmp_path, capsys):
        cmd, _ = self._run_with_missing_dep(tmp_path, lambda b: None)
        assert cmd is None  # no install attempted
        assert "cannot install dependencies" in capsys.readouterr().out
