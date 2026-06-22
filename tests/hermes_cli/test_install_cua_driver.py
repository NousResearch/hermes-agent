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

The pre-install arch probe that used to live alongside this function was
deleted (see top-of-file comment in tools_config.py) — the upstream
installer has CUA_DRIVER_RS_BAKED_VERSION baked in by CD and errors
cleanly on missing-arch assets, and the upgrade path uses
``cua_driver_update_check()`` (which shells `cua-driver check-update
--json` against the already-installed binary).
"""

from __future__ import annotations

from unittest.mock import patch


class TestInstallCuaDriverUpgrade:
    def test_upgrade_on_non_macos_is_silent_noop(self):
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="Linux"):
            assert tools_config.install_cua_driver(upgrade=True) is False
            warn.assert_not_called()

    def test_non_upgrade_on_non_macos_warns(self):
        from hermes_cli import tools_config

        with patch.object(tools_config, "_print_warning") as warn, \
             patch("platform.system", return_value="Linux"):
            assert tools_config.install_cua_driver(upgrade=False) is False
            warn.assert_called()

    def test_upgrade_on_macos_with_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            kwargs = runner.call_args.kwargs
            assert kwargs.get("verbose") is False

    def test_upgrade_on_macos_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()

    def test_non_upgrade_on_macos_with_binary_skips_install(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in {"cua-driver", "curl"} else None), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner, \
             patch("subprocess.run"):
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_not_called()

    def test_non_upgrade_on_macos_without_binary_runs_installer(self):
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_called_once()


class TestArchProbeRemoval:
    """Regression tests for the deletion of `_check_cua_driver_asset_for_arch`.

    The old probe queried ``/releases/latest`` on trycua/cua and inspected
    asset names. That was wrong in two ways:

    1. cua-driver-rs releases are marked **prerelease** on every cut, so
       ``/releases/latest`` returns the Python ``cua-agent`` / ``cua-computer``
       package instead — a release with zero binary assets. The probe then
       reported "no asset for $arch" on Linux x86_64, Windows, macOS Intel,
       Linux arm64 — every non-Apple-Silicon host.
    2. Even with the right endpoint, it duplicated tag-resolution the upstream
       installer already does correctly via ``CUA_DRIVER_RS_BAKED_VERSION``
       (auto-baked by CD on every release).

    The fix: stop probing. Trust the upstream installer for fresh installs
    (it has the baked version + correct API fallback) and the
    ``cua-driver check-update --json`` MCP-binary native command for the
    upgrade path.
    """

    def test_probe_function_is_gone(self):
        from hermes_cli import tools_config
        assert not hasattr(tools_config, "_check_cua_driver_asset_for_arch")
        assert not hasattr(tools_config, "_latest_cua_driver_rs_release")

    def test_fresh_install_does_not_call_github_api(self):
        """Pre-install no longer probes the GitHub API — the upstream
        ``install.sh`` resolves the tag from its baked CUA_DRIVER_RS_BAKED_VERSION
        line. install.sh errors cleanly when the arch has no asset, so the
        probe was duplicate gatekeeping.
        """
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch("urllib.request.urlopen") as urlopen, \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True
            runner.assert_called_once()
            urlopen.assert_not_called()

    def test_upgrade_with_binary_does_not_call_github_api_directly(self):
        """The upgrade path no longer hits GitHub from Python — it delegates
        to the upstream ``install.sh`` (which has the baked release tag and
        the proper API fallback). When cua-driver is already installed,
        ``cua_driver_update_check()`` (added in a separate change) further
        short-circuits the network re-install via the binary's native
        ``check-update --json`` verb.
        """
        from hermes_cli import tools_config

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl") else None), \
             patch("urllib.request.urlopen") as urlopen, \
             patch("subprocess.run"), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_called_once()
            # Probe deleted — no direct GitHub API call from Python.
            urlopen.assert_not_called()
