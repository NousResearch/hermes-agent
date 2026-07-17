"""Host-specific gating in ``hermes_cli.gateway._all_platforms()`` plus
service-manager platform detection.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same filtered
list.

Currently:
- Matrix is hidden on Windows. The ``[matrix]`` extra pulls
  ``mautrix[encryption]`` -> ``python-olm``, which has no Windows wheel
  and needs ``make`` + libolm to build from sdist. There's no native
  Windows path that works.

Service-manager detectors (``supports_systemd_services``,
``supports_freebsd_rc``, ``is_macos``, ``is_windows``) are also gated on
``sys.platform`` — mutually exclusive per host.
"""



class TestMatrixHiddenOnWindows:
    def test_matrix_present_on_linux(self, monkeypatch):
        """Sanity: matrix is still in the picker on Linux/macOS."""
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "linux")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be available on Linux"

    def test_matrix_present_on_macos(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "darwin")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be available on macOS"

    def test_matrix_hidden_on_windows(self, monkeypatch):
        """The actual gate: matrix must NOT appear on Windows."""
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "win32")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" not in keys, (
            "matrix must be hidden on Windows — python-olm has no "
            "Windows wheel and no native build path"
        )

    def test_other_platforms_unaffected_on_windows(self, monkeypatch):
        """Gating must only drop matrix, not collateral damage."""
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "win32")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        # A representative sample of platforms that have no Windows
        # blockers — picker should still surface them.
        for must_have in ("telegram", "discord", "slack", "mattermost"):
            assert must_have in keys, (
                f"{must_have} disappeared from Windows picker — gate is "
                "over-filtering"
            )


class TestFreebsdRcPresentOnFreebsd:
    """The service-manager detectors are mutually exclusive on each host.
    On FreeBSD, ``supports_freebsd_rc`` picks up the port-installed rc.d
    script; no other manager reports available."""

    def _stub_env(self, monkeypatch, platform):
        """Neutralize the file-existence + PATH sniffs so the detector's
        platform check is the only variable."""
        import hermes_cli.gateway as gateway_mod
        from pathlib import Path

        monkeypatch.setattr(gateway_mod.sys, "platform", platform)
        # Every escalator + rc.d + service(8) sniff returns "present"
        monkeypatch.setattr(gateway_mod.shutil, "which", lambda name: f"/x/{name}")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        # Systemd operational probe would otherwise shell out; short-circuit.
        monkeypatch.setattr(gateway_mod, "_wsl_systemd_operational", lambda: True)
        monkeypatch.setattr(gateway_mod, "_container_systemd_operational", lambda: True)
        # is_freebsd, is_termux, is_wsl, is_container all delegate to
        # sys.platform / env; freeze them on the FreeBSD detector's
        # dependencies.
        monkeypatch.setattr(gateway_mod, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_mod, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway_mod, "is_container", lambda: False)
        monkeypatch.setattr(gateway_mod, "_freebsd_is_root", lambda: True)

    def test_freebsd_rc_detected_on_freebsd(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod
        self._stub_env(monkeypatch, "freebsd16")
        assert gateway_mod.supports_freebsd_rc() is True

    def test_freebsd_rc_absent_on_linux(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod
        self._stub_env(monkeypatch, "linux")
        assert gateway_mod.supports_freebsd_rc() is False

    def test_freebsd_rc_absent_on_macos(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod
        self._stub_env(monkeypatch, "darwin")
        assert gateway_mod.supports_freebsd_rc() is False

    def test_freebsd_rc_absent_on_windows(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod
        self._stub_env(monkeypatch, "win32")
        assert gateway_mod.supports_freebsd_rc() is False

    def test_service_managers_mutually_exclusive_on_freebsd(self, monkeypatch):
        # On FreeBSD, freebsd_rc claims the host and neither systemd nor
        # macOS/Windows detectors fire.
        import hermes_cli.gateway as gateway_mod
        self._stub_env(monkeypatch, "freebsd16")
        assert gateway_mod.supports_freebsd_rc() is True
        assert gateway_mod.supports_systemd_services() is False
        assert gateway_mod.is_macos() is False
        assert gateway_mod.is_windows() is False
