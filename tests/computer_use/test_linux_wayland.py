"""Hermes-side native Wayland policy and Arch diagnosis tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch


def _wayland_env(**extra: str) -> dict[str, str]:
    env = {
        "XDG_SESSION_TYPE": "wayland",
        "WAYLAND_DISPLAY": "wayland-1",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "XDG_CURRENT_DESKTOP": "GNOME",
        "PATH": "/usr/bin:/bin",
    }
    env.update(extra)
    return env


def test_detects_native_wayland_without_display(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    session = detect_linux_session(_wayland_env())
    assert session.kind == "wayland"
    assert session.desktop == "gnome"


def test_detects_wayland_with_xwayland(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    session = detect_linux_session(_wayland_env(DISPLAY=":0"))
    assert session.kind == "wayland-xwayland"


def test_detects_stale_wayland_socket(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
    session = detect_linux_session(_wayland_env())
    assert session.kind == "wayland-stale"
    assert "runtime socket" in " ".join(session.reasons)


def test_parses_dbus_path_before_semicolon_options():
    from tools.computer_use.linux_wayland import _dbus_socket_path

    assert _dbus_socket_path("unix:path=/run/user/1000/bus,guid=abc") == Path("/run/user/1000/bus")


def test_detects_x11_and_headless(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    assert detect_linux_session({"DISPLAY": ":1", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"}).kind == "x11"
    assert detect_linux_session({}).kind == "headless"


def test_auto_mode_requires_machine_readable_driver_feature(monkeypatch):
    from tools.computer_use.linux_wayland import native_wayland_enabled

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    with patch("tools.computer_use.linux_wayland.subprocess.run") as run:
        run.return_value = Mock(returncode=0, stdout='{"binary_version":"99.0.0"}')
        assert not native_wayland_enabled("/driver", {}, _wayland_env())
        run.return_value = Mock(returncode=0, stdout='{"features":{"wayland_native":true}}')
        assert native_wayland_enabled("/driver", {}, _wayland_env())


def test_explicit_disable_wins_over_inherited_export(monkeypatch):
    from tools.computer_use.linux_wayland import WAYLAND_ENABLE_ENV, native_wayland_child_env

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    env = _wayland_env(**{WAYLAND_ENABLE_ENV: "1"})
    got = native_wayland_child_env("/driver", {"computer_use": {"linux": {"wayland": {"enabled": "disabled"}}}}, env)
    assert got[WAYLAND_ENABLE_ENV] == "0"


def test_explicit_enable_is_forwarded(monkeypatch):
    from tools.computer_use.linux_wayland import WAYLAND_ENABLE_ENV, native_wayland_child_env

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    got = native_wayland_child_env("/old-driver", {"computer_use": {"linux": {"wayland": {"enabled": "enabled"}}}}, _wayland_env())
    assert got[WAYLAND_ENABLE_ENV] == "1"


def test_arch_diagnosis_selects_one_desktop_portal(monkeypatch):
    from tools.computer_use import linux_wayland

    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)
    monkeypatch.setattr(linux_wayland, "_package_installed", lambda package, env: package != "xdg-desktop-portal-gnome")
    monkeypatch.setattr(linux_wayland, "driver_supports_native_wayland", lambda *_: True)
    monkeypatch.setattr(linux_wayland, "_run", lambda args, env: "org.freedesktop.portal.Desktop\norg.a11y.Bus" if args[0] == "busctl" else "active")
    report = linux_wayland.diagnose_arch_wayland("/driver", {}, _wayland_env())
    assert report["selected_portal_package"] == "xdg-desktop-portal-gnome"
    assert "xdg-desktop-portal-gnome" in report["missing_packages"]
    assert "xdg-desktop-portal-kde" not in report["packages"]


def test_arch_install_hint_is_copyable():
    from tools.computer_use.linux_wayland import arch_install_hint

    assert arch_install_hint({"missing_packages": ["pipewire", "xdg-desktop-portal-gnome"]}) == (
        "Install only the missing packages for this desktop: sudo pacman -S pipewire xdg-desktop-portal-gnome"
    )
