"""CuaDriver's staged-verification repair installs the host X11 client libraries.

On headless Linux hosts the cua-driver --version probe dies on the FIRST
missing X11 library (libX11, then libXi, then libXtst…), one package-manager
round per message. The repair enumerates the full set via ldd (pm.x11libs),
installs the distro packages non-interactively (pm.libatomic machinery), and
re-probes. The fake staged binary mirrors that failure signature exactly.
"""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("pytest")

from pm.packages import CuaDriver
from pm.store import current_target


def _fake_cua_binary(staged_dir, messages: list[str]):
    """A staged 'binary' whose --version probe fails with the shared-library
    signature (exit 127), emitting each `messages` line on stderr."""
    staged_dir.mkdir(parents=True, exist_ok=True)
    binary = staged_dir / "cua-driver"
    body = "\n".join(f"echo {msg!r} >&2" for msg in messages)
    binary.write_text(f"#!/bin/sh\n{body}\nexit 127\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


@pytest.fixture
def no_host_install(monkeypatch):
    """Stub the real package-manager attempt out; record the soname set."""
    calls: list[list[str]] = []

    def record(sonames: list[str]) -> tuple[bool, str]:
        calls.append(list(sonames))
        return True, "install hint"  # pretend the manager succeeded

    monkeypatch.setattr("pm.x11libs.try_install_x11_client_libs", record)
    return calls


class TestCuaDriverX11Repair:
    def test_repair_enumerates_the_full_missing_set(self, tmp_path, monkeypatch, no_host_install):
        """ldd lists ALL missing libraries in one pass; the probe only shows one."""
        _fake_cua_binary(tmp_path / "tree", [
            "cua-driver: error while loading shared libraries: libX11.so.6: "
            "cannot open shared object file: No such file or directory"])
        pkg = CuaDriver()
        # ldd on our shell script: fake the enumeration the repair relies on
        monkeypatch.setattr("pm.x11libs.missing_shared_libraries", lambda b: [
            "libX11.so.6", "libXi.so.6", "libXtst.so.6", "libxkbcommon.so.0"])

        reason = pkg.verify(tmp_path / "tree", current_target())
        assert "loading shared libraries" in reason
        reason2, _remedy = pkg.repair_staged_verification(
            tmp_path / "tree", current_target(), reason)
        assert no_host_install == [["libX11.so.6", "libXi.so.6",
                                    "libXtst.so.6", "libxkbcommon.so.0"]]
        # repair re-probes: still failing (fake binary), so the reason returns
        assert "loading shared libraries" in reason2

    def test_repair_reports_success_when_reprobe_passes(self, tmp_path, monkeypatch, no_host_install):
        _fake_cua_binary(tmp_path / "tree", [
            "cua-driver: error while loading shared libraries: libX11.so.6: "
            "cannot open shared object file: No such file or directory"])
        pkg = CuaDriver()
        reason = pkg.verify(tmp_path / "tree", current_target())
        monkeypatch.setattr("pm.x11libs.missing_shared_libraries", lambda b: ["libX11.so.6"])
        real_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if cmd and str(cmd[0]).endswith("cua-driver"):
                return subprocess.CompletedProcess(cmd, 0, b"", b"")
            return real_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_run)
        reason2, _remedy = pkg.repair_staged_verification(
            tmp_path / "tree", current_target(), reason)
        assert reason2 == ""

    def test_unmapped_soname_still_names_the_libraries(
            self, tmp_path, monkeypatch):
        """A soname the family table does not map stays in the remedy verbatim."""
        _fake_cua_binary(tmp_path / "tree", [
            "cua-driver: error while loading shared libraries: libWeird.so.9: "
            "cannot open shared object file: No such file or directory"])
        pkg = CuaDriver()
        monkeypatch.setattr("pm.x11libs.missing_shared_libraries", lambda b: ["libWeird.so.9"])
        monkeypatch.setattr("pm.x11libs.try_install_x11_client_libs",
                            lambda sonames: (False, "no packages matched"))
        reason = pkg.verify(tmp_path / "tree", current_target())
        _reason2, remedy = pkg.repair_staged_verification(
            tmp_path / "tree", current_target(), reason)
        assert "libWeird.so.9" in remedy

    def test_repair_is_a_noop_outside_shared_library_failures(self, tmp_path):
        pkg = CuaDriver()
        reason2, remedy = pkg.repair_staged_verification(
            tmp_path, current_target(), "arch mismatch for this target")
        assert reason2 == "arch mismatch for this target"
        assert remedy == ""


class TestFamilyTables:
    """The dnf/yum tables must spell Fedora names — Debian names make dnf
    abort the whole transaction ("Unable to find a match"), so a repair that
    mixes families installs nothing on Fedora/RHEL."""

    def test_dnf_yum_share_an_explicit_fedora_table(self):
        from pm import x11libs
        assert x11libs._FAMILIES["dnf"] is x11libs._FEDORA
        assert x11libs._FAMILIES["yum"] is x11libs._FEDORA

    def test_fedora_table_covers_every_debian_soname(self):
        from pm import x11libs
        assert set(x11libs._DEBIAN) <= set(x11libs._FEDORA)

    def test_no_debian_name_leaks_into_dnf_or_yum(self):
        from pm import x11libs
        debian_names = set(x11libs._DEBIAN.values())
        for family in ("dnf", "yum"):
            names = set(x11libs._FAMILIES[family].values())
            assert names.isdisjoint(debian_names), \
                f"{family} reuses Debian names: {sorted(names & debian_names)}"

    def test_dnf_command_uses_fedora_names(self, monkeypatch):
        from pm import x11libs
        monkeypatch.setattr("pm.libatomic._manager_order", lambda tokens: ["dnf", "apt-get"])
        monkeypatch.setattr("pm.libatomic._release_tokens", lambda: ())
        monkeypatch.setattr(x11libs.shutil, "which",
                            lambda name: "/usr/bin/dnf" if name == "dnf" else None)
        manager, pkgs = x11libs._packages_for(
            ["libXrandr.so.2", "libXext.so.6", "libXfixes.so.3", "libXrender.so.1",
             "libXinerama.so.1"])
        assert manager == "dnf"
        assert pkgs == ["libXrandr", "libXext", "libXfixes", "libXrender", "libXinerama"]
        assert x11libs._install_command(manager, pkgs) == (
            "dnf", "install", "-y", "libXrandr", "libXext", "libXfixes",
            "libXrender", "libXinerama")

    def test_libxcursor_maps_on_every_family(self):
        """Xcursor was once a dead selector (looked handled, was not)."""
        from pm import x11libs
        for family, table in x11libs._FAMILIES.items():
            assert "libXcursor.so.1" in table, f"{family} does not map libXcursor.so.1"
            assert table["libXcursor.so.1"]