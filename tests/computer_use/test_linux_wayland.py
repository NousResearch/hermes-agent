"""Hermes-side native Wayland policy and Linux desktop diagnosis tests."""

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


def _socket_ready(monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    monkeypatch.setattr("pathlib.Path.is_socket", lambda _: True)


def test_detects_native_wayland_without_display(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    _socket_ready(monkeypatch)
    session = detect_linux_session(_wayland_env())
    assert session.kind == "wayland"
    assert session.desktop == "gnome"


def test_detects_wayland_with_xwayland(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    _socket_ready(monkeypatch)
    assert detect_linux_session(_wayland_env(DISPLAY=":0")).kind == "wayland-xwayland"


def test_detects_stale_wayland_socket(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
    session = detect_linux_session(_wayland_env())
    assert session.kind == "wayland-stale"
    assert "runtime socket" in " ".join(session.reasons)


def test_stale_wayland_with_display_keeps_x11_route_visible(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
    session = detect_linux_session(_wayland_env(DISPLAY=":0"))
    assert session.kind == "wayland-stale-xwayland"
    assert session.display == ":0"
    assert "runtime socket" in " ".join(session.reasons)


def test_parses_dbus_path_before_semicolon_options():
    from tools.computer_use.linux_wayland import _dbus_socket_path

    assert _dbus_socket_path("unix:path=/run/user/1000/bus,guid=abc") == Path("/run/user/1000/bus")


def test_detects_x11_and_headless(monkeypatch):
    from tools.computer_use.linux_wayland import detect_linux_session

    _socket_ready(monkeypatch)
    assert detect_linux_session({"DISPLAY": ":1", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1/bus"}).kind == "x11"
    assert detect_linux_session({}).kind == "headless"


def test_probe_features_parses_all_feature_bits_and_invokes_manifest_once(monkeypatch):
    from tools.computer_use.linux_wayland import probe_driver_features

    run = Mock(return_value=Mock(returncode=0, stdout='{"features":{"wayland_native":true,"portal_input":false,"portal_capture":true}}'))
    monkeypatch.setattr("tools.computer_use.linux_wayland.subprocess.run", run)

    features = probe_driver_features("/driver", _wayland_env())

    assert features.wayland_native is True
    assert features.portal_input is False
    assert features.portal_capture is True
    assert features.manifest_supported is True
    run.assert_called_once()
    assert run.call_args.args[0] == ["/driver", "manifest"]


def test_probe_features_fails_closed_for_unknown_or_malformed_values(monkeypatch):
    from tools.computer_use.linux_wayland import probe_driver_features

    with patch("tools.computer_use.linux_wayland.subprocess.run") as run:
        run.return_value = Mock(returncode=0, stdout='{"features":{"wayland_native":"true","portal_input":1}}')
        features = probe_driver_features("/driver", _wayland_env())
    assert features.manifest_supported is True
    assert features.wayland_native is features.portal_input is features.portal_capture is False


def test_auto_mode_requires_native_feature_but_explicit_override_is_preserved(monkeypatch):
    from tools.computer_use.linux_wayland import CuaDriverFeatures, native_wayland_enabled

    _socket_ready(monkeypatch)
    no_features = CuaDriverFeatures(manifest_supported=True)
    native = CuaDriverFeatures(wayland_native=True, manifest_supported=True)
    assert not native_wayland_enabled("/driver", {}, _wayland_env(), features=no_features)
    assert native_wayland_enabled("/driver", {}, _wayland_env(), features=native)
    assert native_wayland_enabled("/driver", {"computer_use": {"linux": {"wayland": {"enabled": "enabled"}}}}, _wayland_env(), features=no_features)
    assert not native_wayland_enabled("/driver", {"computer_use": {"linux": {"wayland": {"enabled": "disabled"}}}}, _wayland_env(), features=native)


def test_stale_wayland_socket_refuses_auto_and_explicit_enabled(monkeypatch):
    from tools.computer_use.linux_wayland import CuaDriverFeatures, native_wayland_enabled

    monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
    native = CuaDriverFeatures(wayland_native=True, manifest_supported=True)
    stale_xwayland = _wayland_env(DISPLAY=":0")
    assert not native_wayland_enabled("/driver", {}, stale_xwayland, features=native)
    assert not native_wayland_enabled(
        "/driver",
        {"computer_use": {"linux": {"wayland": {"enabled": "enabled"}}}},
        stale_xwayland,
        features=native,
    )


def test_explicit_disabled_refuses_valid_wayland_socket(monkeypatch):
    from tools.computer_use.linux_wayland import CuaDriverFeatures, native_wayland_enabled

    _socket_ready(monkeypatch)
    native = CuaDriverFeatures(wayland_native=True, manifest_supported=True)
    assert not native_wayland_enabled(
        "/driver",
        {"computer_use": {"linux": {"wayland": {"enabled": "disabled"}}}},
        _wayland_env(DISPLAY=":0"),
        features=native,
    )


def test_explicit_disable_wins_over_inherited_export(monkeypatch):
    from tools.computer_use.linux_wayland import WAYLAND_ENABLE_ENV, native_wayland_child_env

    _socket_ready(monkeypatch)
    env = _wayland_env(**{WAYLAND_ENABLE_ENV: "1"})
    got = native_wayland_child_env("/driver", {"computer_use": {"linux": {"wayland": {"enabled": "disabled"}}}}, env)
    assert got[WAYLAND_ENABLE_ENV] == "0"


def test_explicit_enable_is_forwarded(monkeypatch):
    from tools.computer_use.linux_wayland import WAYLAND_ENABLE_ENV, native_wayland_child_env

    _socket_ready(monkeypatch)
    got = native_wayland_child_env("/old-driver", {"computer_use": {"linux": {"wayland": {"enabled": "enabled"}}}}, _wayland_env())
    assert got[WAYLAND_ENABLE_ENV] == "1"


def test_parse_os_release_identifies_arch_and_derivative():
    from tools.computer_use.linux_wayland import parse_os_release

    arch = parse_os_release('ID=arch\nNAME="Arch Linux"\nPRETTY_NAME="Arch Linux"\n')
    manjaro = parse_os_release('ID=manjaro\nID_LIKE="arch"\nNAME=Manjaro\nPRETTY_NAME="Manjaro Linux"\n')
    assert arch.is_arch_like is True
    assert manjaro.is_arch_like is True
    assert manjaro.id_like == ("arch",)


def test_parse_os_release_rejects_non_arch_families_and_missing_file(tmp_path):
    from tools.computer_use.linux_wayland import detect_linux_distribution, parse_os_release

    for content in (
        'ID=ubuntu\nID_LIKE=debian\nNAME="Ubuntu"\n',
        'ID=fedora\nID_LIKE="fedora rhel"\nNAME=Fedora\n',
        'ID=nixos\nNAME=NixOS\n',
    ):
        assert parse_os_release(content).is_arch_like is False
    assert detect_linux_distribution(tmp_path / "missing-os-release").id is None


def test_arch_package_diagnosis_requires_arch_identity_and_pacman(monkeypatch):
    from tools.computer_use import linux_wayland

    non_arch = {"distribution": {"arch_like": False}, "session": {"compositor": "gnome"}}
    monkeypatch.setattr(linux_wayland.shutil, "which", lambda *args, **kwargs: "/usr/bin/pacman")
    report = linux_wayland.diagnose_arch_packages(non_arch, _wayland_env())
    assert report["applicable"] is False
    assert report["missing_packages"] == []
    assert linux_wayland.arch_install_hint({"arch_packages": report}) is None

    arch = {"distribution": {"arch_like": True}, "session": {"compositor": "gnome"}}
    monkeypatch.setattr(linux_wayland.shutil, "which", lambda *args, **kwargs: None)
    report = linux_wayland.diagnose_arch_packages(arch, _wayland_env())
    assert report["applicable"] is False
    assert report["reason"] == "pacman executable is unavailable"


def test_arch_diagnosis_selects_one_desktop_portal(monkeypatch):
    from tools.computer_use import linux_wayland

    _socket_ready(monkeypatch)
    monkeypatch.setattr(linux_wayland, "_package_installed", lambda package, pacman, env: package != "xdg-desktop-portal-gnome")
    monkeypatch.setattr(linux_wayland.shutil, "which", lambda *args, **kwargs: "/usr/bin/pacman")
    arch = {"distribution": {"arch_like": True}, "session": {"compositor": "gnome"}}
    report = linux_wayland.diagnose_arch_packages(arch, _wayland_env())
    assert report["applicable"] is True
    assert report["selected_portal_package"] == "xdg-desktop-portal-gnome"
    assert "xdg-desktop-portal-gnome" in report["missing_packages"]
    assert "xdg-desktop-portal-kde" not in report["packages"]


def test_linux_desktop_diagnosis_preserves_wlroots_candidates_without_portal(monkeypatch, tmp_path):
    from tools.computer_use import linux_wayland

    _socket_ready(monkeypatch)
    monkeypatch.setattr(linux_wayland, "probe_driver_features", lambda *_: linux_wayland.CuaDriverFeatures(wayland_native=True, manifest_supported=True))
    monkeypatch.setattr(linux_wayland, "_bus_name_owned", lambda *_: False)
    monkeypatch.setattr(linux_wayland, "_service_active", lambda *_: False)
    env = _wayland_env(XDG_CURRENT_DESKTOP="sway")
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=ubuntu\n")
    report = linux_wayland.diagnose_linux_desktop("/driver", {}, env, os_release_path=os_release)
    caps = report["capabilities"]
    assert report["driver_features"] == {"wayland_native": True, "portal_input": False, "portal_capture": False, "manifest_supported": True}
    assert caps["capture_path"] == "wlroots_native_capture_candidate"
    assert caps["input_path"] == "wlroots_virtual_pointer_candidate"


def test_gnome_explains_missing_portal_input(monkeypatch, tmp_path):
    from tools.computer_use import linux_wayland

    _socket_ready(monkeypatch)
    monkeypatch.setattr(linux_wayland, "probe_driver_features", lambda *_: linux_wayland.CuaDriverFeatures(wayland_native=True, manifest_supported=True))
    monkeypatch.setattr(linux_wayland, "_bus_name_owned", lambda *_: True)
    monkeypatch.setattr(linux_wayland, "_service_active", lambda *_: True)
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=fedora\n")
    report = linux_wayland.diagnose_linux_desktop("/driver", {}, _wayland_env(), os_release_path=os_release)
    assert any("without portal_input" in reason for reason in report["capabilities"]["degraded_reasons"])


def test_portal_input_is_a_candidate_until_driver_verifies_delivery(monkeypatch, tmp_path):
    from tools.computer_use import linux_wayland

    _socket_ready(monkeypatch)
    features = linux_wayland.CuaDriverFeatures(
        wayland_native=True,
        portal_input=True,
        manifest_supported=True,
    )
    monkeypatch.setattr(linux_wayland, "probe_driver_features", lambda *_: features)
    monkeypatch.setattr(linux_wayland, "_bus_name_owned", lambda *_: True)
    monkeypatch.setattr(linux_wayland, "_service_active", lambda *_: True)
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=fedora\n")

    caps = linux_wayland.diagnose_linux_desktop(
        "/driver", {}, _wayland_env(), os_release_path=os_release
    )["capabilities"]
    assert caps["input_path"] == "portal_remote_desktop_input_candidate"
    assert caps["consent_expected"] is True
    assert caps["foreground_pointer_input"] is False
    assert caps["foreground_keyboard_input"] is False


def test_stale_xwayland_diagnosis_uses_x11_without_enabling_native_wayland(monkeypatch, tmp_path):
    from tools.computer_use import linux_wayland

    monkeypatch.setattr("pathlib.Path.exists", lambda _: False)
    monkeypatch.setattr(linux_wayland, "probe_driver_features", lambda *_: linux_wayland.CuaDriverFeatures(wayland_native=True, manifest_supported=True))
    monkeypatch.setattr(linux_wayland, "_bus_name_owned", lambda *_: False)
    monkeypatch.setattr(linux_wayland, "_service_active", lambda *_: False)
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=ubuntu\n")

    report = linux_wayland.diagnose_linux_desktop(
        "/driver", {}, _wayland_env(DISPLAY=":0"), os_release_path=os_release
    )
    assert report["native_wayland_enabled"] is False
    assert report["session"]["kind"] == "wayland-stale-xwayland"
    assert report["capabilities"]["input_path"] == "x11"
    assert any("WAYLAND_DISPLAY is set" in reason for reason in report["capabilities"]["degraded_reasons"])


def test_arch_install_hint_is_copyable_only_when_arch_packages_apply():
    from tools.computer_use.linux_wayland import arch_install_hint

    assert arch_install_hint({"arch_packages": {"applicable": True, "missing_packages": ["pipewire", "xdg-desktop-portal-gnome"]}}) == (
        "Install only the missing packages for this desktop: sudo pacman -S pipewire xdg-desktop-portal-gnome"
    )
    assert arch_install_hint({"arch_packages": {"applicable": False, "missing_packages": ["pipewire"]}}) is None
