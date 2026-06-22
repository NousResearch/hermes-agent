"""Tests for ``install_cua_driver`` upgrade semantics and architecture pre-check.

The cua-driver upstream installer always pulls the latest release tag, so
re-running it is the canonical upgrade path. ``install_cua_driver(upgrade=True)``
must:

* Be macOS-only — no-op silently on Linux/Windows so ``hermes update`` can
  call it unconditionally without warning every non-macOS user.
* Re-run the installer even when the binary is already on PATH (this is the
  fix for the "we only pulled cua-driver once on enable" complaint).
* Preserve original ``upgrade=False`` behaviour for the toolset-enable flow:
  skip if installed, install otherwise, warn on non-macOS.
* Pre-check architecture compatibility before downloading to avoid raw 404
  errors on Intel macOS when the upstream release lacks x86_64 assets.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


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
             patch.object(tools_config, "_check_cua_driver_asset_for_arch",
                          return_value=True), \
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
             patch.object(tools_config, "_check_cua_driver_asset_for_arch",
                          return_value=True), \
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
             patch.object(tools_config, "_check_cua_driver_asset_for_arch",
                          return_value=True), \
             patch.object(tools_config, "_run_cua_driver_installer",
                          return_value=True) as runner:
            assert tools_config.install_cua_driver(upgrade=False) is True


class TestCheckCuaDriverAssetForArch:
    def test_arm64_always_returns_true(self):
        from hermes_cli import tools_config

        with patch("platform.machine", return_value="arm64"):
            assert tools_config._check_cua_driver_asset_for_arch() is True

    def test_x86_64_with_asset_returns_true(self):
        from hermes_cli import tools_config

        release = {
            "tag_name": "cua-driver-v0.1.6",
            "assets": [
                {"name": "cua-driver-0.1.6-darwin-arm64.tar.gz"},
                {"name": "cua-driver-0.1.6-darwin-x86_64.tar.gz"},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(release).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            assert tools_config._check_cua_driver_asset_for_arch() is True

    def test_x86_64_without_asset_returns_false(self):
        from hermes_cli import tools_config

        release = {
            "tag_name": "cua-driver-v0.1.6",
            "assets": [
                {"name": "cua-driver-0.1.6-darwin-arm64.tar.gz"},
                {"name": "cua-driver.tar.gz"},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(release).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.object(tools_config, "_print_warning") as warn, \
             patch.object(tools_config, "_print_info"):
            assert tools_config._check_cua_driver_asset_for_arch() is False
            warn.assert_called_once()
            assert "no Intel" in warn.call_args[0][0].lower() or "x86_64" in warn.call_args[0][0]

    def test_x86_64_api_failure_returns_true(self):
        """Network failure should fail open — let the installer handle it."""
        from hermes_cli import tools_config

        with patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            assert tools_config._check_cua_driver_asset_for_arch() is True

    def test_fresh_install_x86_64_no_asset_skips_installer(self):
        """When the latest release has no Intel asset, skip the installer."""
        from hermes_cli import tools_config

        release = {
            "tag_name": "cua-driver-v0.1.6",
            "assets": [{"name": "cua-driver-0.1.6-darwin-arm64.tar.gz"}],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(release).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner:
            assert tools_config.install_cua_driver(upgrade=False) is False
            runner.assert_not_called()

    def test_upgrade_x86_64_no_asset_returns_existing_status(self):
        """On upgrade with no Intel asset, return whether binary existed."""
        from hermes_cli import tools_config

        release = {
            "tag_name": "cua-driver-v0.1.6",
            "assets": [{"name": "cua-driver-0.1.6-darwin-arm64.tar.gz"}],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(release).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        # With binary installed — returns True (binary exists)
        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/local/bin/" + n
                                                 if n in ("cua-driver", "curl") else None), \
             patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner:
            assert tools_config.install_cua_driver(upgrade=True) is True
            runner.assert_not_called()

        # Without binary — returns False
        with patch("platform.system", return_value="Darwin"), \
             patch.object(tools_config.shutil, "which",
                          side_effect=lambda n: "/usr/bin/curl" if n == "curl" else None), \
             patch("platform.machine", return_value="x86_64"), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch.object(tools_config, "_print_warning"), \
             patch.object(tools_config, "_print_info"), \
             patch.object(tools_config, "_run_cua_driver_installer") as runner:
            assert tools_config.install_cua_driver(upgrade=True) is False
            runner.assert_not_called()


class TestComputerUseStatusLines:
    def test_non_macos_status_explains_platform_gate(self, monkeypatch):
        from hermes_cli import main as hermes_main

        monkeypatch.setattr(hermes_main.sys, "platform", "win32")
        lines = hermes_main._computer_use_status_lines()

        joined = "\n".join(lines)
        assert "unavailable on this platform (win32)" in joined
        assert "macOS-only" in joined
        assert "hermes computer-use install" not in joined
        assert "browser toolset" in joined

    def test_non_macos_status_includes_safe_desktop_control_routes(self, monkeypatch):
        from hermes_cli import main as hermes_main

        monkeypatch.setattr(hermes_main.sys, "platform", "win32")
        lines = hermes_main._computer_use_status_lines()
        joined = "\n".join(lines)

        assert "desktop_control safe routing:" in joined
        assert "web UI -> browser" in joined
        assert "local files/logs -> file_terminal" in joined
        assert "screenshots -> vision" in joined
        assert "native Windows GUI mutation -> unsupported" in joined

    def test_status_json_exposes_structured_capabilities(self, monkeypatch):
        from hermes_cli import main as hermes_main

        monkeypatch.setattr(hermes_main.sys, "platform", "win32")
        status = hermes_main._computer_use_status_json(probe_browser=False)

        assert status["platform"] == "win32"
        assert status["computer_use"]["available"] is False
        assert status["computer_use"]["native_gui_mutation_allowed"] is False
        assert "browser" in status
        assert "routes" in status
        assert set(status["risk_policy"]["windows_native_gui_forbidden_actions"]) >= {
            "click",
            "drag",
            "type",
        }

    def test_macos_status_without_driver_keeps_install_hint(self, monkeypatch):
        from hermes_cli import main as hermes_main

        monkeypatch.setattr(hermes_main.sys, "platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert hermes_main._computer_use_status_lines() == [
            "cua-driver: not installed",
            "  Run: hermes computer-use install",
        ]

    def test_macos_status_with_driver_includes_version(self, monkeypatch):
        from hermes_cli import main as hermes_main

        class _Completed:
            stdout = "cua-driver 0.1.6\n"

        monkeypatch.setattr(hermes_main.sys, "platform", "darwin")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/cua-driver")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _Completed())

        lines = hermes_main._computer_use_status_lines()
        assert lines[0] == "cua-driver: installed at /usr/local/bin/cua-driver (cua-driver 0.1.6)"
        assert lines[1] == "  Refresh to latest: hermes computer-use install --upgrade"
