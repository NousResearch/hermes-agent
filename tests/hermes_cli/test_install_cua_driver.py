"""Tests for ``install_cua_driver`` upgrade semantics.

The cua-driver upstream installer always pulls the latest release tag, so
re-running it is the canonical upgrade path. ``install_cua_driver(upgrade=True)``
must:

* Be macOS-only — no-op silently on Linux/Windows so ``hermes update`` can
  call it unconditionally without warning every non-macOS user.
* Re-run the installer even when the binary is already on PATH (this is the
  fix for the "we only pulled cua-driver once on enable" complaint).
* Preserve original ``upgrade=False`` behaviour for the toolset-enable flow:
  skip if installed, install otherwise, warn on non-macOS.
"""

from __future__ import annotations

from unittest.mock import patch


class TestInstallCuaDriverUpgrade:
    def test_upgrade_on_non_macos_is_silent_noop(self):
        """``hermes update`` calls install_cua_driver(upgrade=True) for every
        user. On Linux/Windows it must return False without printing the
        "macOS-only; skipping" warning that the toolset-enable path emits."""
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="Linux"):
            assert tools_config.install_cua_driver(upgrade=True) is False
            warn.assert_not_called()

    def test_non_upgrade_on_non_macos_warns(self):
        """The toolset-enable path (upgrade=False) should still warn loudly
        when the user tries to enable Computer Use on a non-macOS host."""
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="Linux"):
            assert tools_config.install_cua_driver(upgrade=False) is False
            warn.assert_called()

    def test_upgrade_on_macos_with_binary_runs_installer(self):
        """When cua-driver is already on PATH and upgrade=True, we must
        re-run the upstream installer (this is the fix for the bug report).
        """
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl") else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            # Refresh path uses non-verbose mode so we don't re-print the
            # "grant macOS permissions" block on every `hermes update`.
            kwargs = runner.call_args.kwargs
            assert kwargs.get("verbose") is False

    def test_upgrade_on_macos_without_binary_runs_installer(self):
        """upgrade=True with cua-driver missing must still trigger an
        install — equivalent to a fresh install. (Don't silently no-op.)"""
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()

    def test_non_upgrade_on_macos_with_binary_skips_install(self):
        """Original toolset-enable behaviour: cua-driver already installed
        + upgrade=False → confirm and return without re-running installer.
        This is the behaviour that ``hermes tools`` (re)enable depends on,
        so the new helper must not regress it."""
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl") else None), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_not_called()

    def test_non_upgrade_on_macos_without_binary_runs_installer(self):
        """Original fresh-install path must still work."""
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_called_once()

    def test_upgrade_without_curl_does_not_crash(self):
        """If curl isn't on PATH we can't refresh — must warn and return
        the current install state, not raise."""
        from hermes_cli import tools_config

        # cua-driver present, curl missing.
        def _which(name):
            return "/usr/local/bin/cua-driver" if name == "cua-driver" else None

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which", side_effect=_which), \
             patch.object(tools_config, "_print_warning"):
            assert tools_config.install_cua_driver(upgrade=True) is True
