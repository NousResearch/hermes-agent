"""Regression tests: `hermes desktop` names a missing Linux C++ toolchain (#102081).

node-pty ships prebuilds for Windows/macOS only, so on Linux its install
script always compiles locally and needs ``make`` plus a C++ compiler. When
they are missing, the install fails inside a wall of npm log text with exit
127 and the generic advice cannot help — the CLI must print the missing
tools and the distro-specific install command instead.
"""

import argparse
import subprocess
import sys

from hermes_cli.native_build_hint import (
    linux_native_build_toolchain_hint,
    missing_native_build_tools,
)


class TestMissingNativeBuildTools:
    def test_reports_only_absent_tools_in_declaration_order(self):
        def fake_which(tool: str):
            return "/usr/bin/make" if tool == "make" else None

        assert missing_native_build_tools(which=fake_which) == ("g++",)

    def test_empty_when_toolchain_present(self):
        assert missing_native_build_tools(which=lambda tool: f"/usr/bin/{tool}") == ()


class TestLinuxNativeBuildToolchainHint:
    def test_none_off_linux(self):
        assert linux_native_build_toolchain_hint(platform="darwin", which=lambda tool: None) is None
        assert linux_native_build_toolchain_hint(platform="win32", which=lambda tool: None) is None

    def test_hint_when_whole_toolchain_missing(self):
        hint = linux_native_build_toolchain_hint(platform="linux", which=lambda tool: None)
        assert hint is not None
        assert "missing from PATH: make, g++" in hint
        assert "build-essential" in hint  # Debian/Ubuntu
        assert "gcc-c++" in hint           # Fedora/RHEL
        assert "base-devel" in hint        # Arch
        assert "run this command again" in hint

    def test_hint_lists_only_the_missing_tools(self):
        def fake_which(tool: str):
            return "/usr/bin/make" if tool == "make" else None

        hint = linux_native_build_toolchain_hint(platform="linux", which=fake_which)
        assert hint is not None
        assert "missing from PATH: g++" in hint
        assert "make, g++" not in hint

    def test_none_when_toolchain_present(self):
        hint = linux_native_build_toolchain_hint(
            platform="linux", which=lambda tool: f"/usr/bin/{tool}"
        )
        assert hint is None


class TestCmdGuiFailurePath:
    """The `hermes desktop` failure exit must carry the hint end-to-end."""

    @staticmethod
    def _force_toolchainless_build_failure(root, monkeypatch, *, platform: str):
        """A checkout whose forced build fails in dependency preparation, on a
        simulated host whose PATH has no native-build tools. Returns nothing."""
        from hermes_cli import main as cli_main
        from hermes_cli import main_desktop
        from hermes_cli import source_build as source_build_mod

        desktop_dir = root / "apps" / "desktop"
        desktop_dir.mkdir(parents=True)
        (desktop_dir / "package.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
        monkeypatch.setattr(source_build_mod, "source_build_env",
                            lambda env=None, **kw: {**(env or {}), "PATH": "/usr/bin"})
        monkeypatch.setattr(main_desktop, "_desktop_launch_env", lambda args: ({}, []))
        monkeypatch.setattr(main_desktop, "_register_linux_desktop_entry", lambda **kw: None)
        monkeypatch.setattr(source_build_mod, "prepare_source_dependencies",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                subprocess.CalledProcessError(127, ["npm", "ci"])))
        # The hint probe must resolve against the simulated host, not this
        # test machine's real PATH or OS. `npm` lookup happens first with a
        # `path=` kwarg, so the stub must accept it. The hint module's own
        # `shutil.which` is patched too — the host running these tests has a
        # toolchain, so the real probe would (correctly) find nothing missing.
        monkeypatch.setattr(sys, "platform", platform, raising=False)
        # Both modules share the one global shutil/sys; patching through
        # either patches both. One stub serves the `path=`-kwarg npm lookup
        # and the hint's bare probe.
        monkeypatch.setattr(main_desktop.shutil, "which",
                            lambda tool, path=None: None)
        import hermes_cli.native_build_hint as native_build_hint
        monkeypatch.setattr(native_build_hint.sys, "platform", platform, raising=False)
        return main_desktop

    @staticmethod
    def _gui_args():
        return argparse.Namespace(skip_build=False, build_only=False, force_build=True,
                                  source=False, fake_boot=False, ignore_existing=False,
                                  hermes_root=None, cwd=None, setup_tcc_identity=False,
                                  identity=None)

    def test_build_failure_prints_toolchain_hint_on_linux(self, tmp_path, monkeypatch, capsys):
        main_desktop = self._force_toolchainless_build_failure(
            tmp_path / "hermes-agent", monkeypatch, platform="linux")

        try:
            main_desktop.cmd_gui(self._gui_args())
            raise AssertionError("cmd_gui should have exited nonzero")
        except SystemExit as exc:
            assert exc.code == 1

        out = capsys.readouterr().out
        assert "✗ Desktop GUI build failed" in out
        assert "missing from PATH: make, g++" in out
        assert "build-essential" in out

    def test_build_failure_stays_silent_off_linux(self, tmp_path, monkeypatch, capsys):
        main_desktop = self._force_toolchainless_build_failure(
            tmp_path / "hermes-agent", monkeypatch, platform="darwin")

        try:
            main_desktop.cmd_gui(self._gui_args())
            raise AssertionError("cmd_gui should have exited nonzero")
        except SystemExit as exc:
            assert exc.code == 1

        out = capsys.readouterr().out
        assert "✗ Desktop GUI build failed" in out
        assert "missing from PATH" not in out
