from hermes_cli.web_server import _display_system_platform


def test_windows_11_build_displays_as_windows_11():
    info = _display_system_platform(
        system="Windows",
        release="10",
        version="10.0.26200",
        platform_label="Windows-10-10.0.26200-SP0",
    )

    assert info["os"] == "Windows"
    assert info["os_release"] == "11"
    assert info["os_version"] == "10.0.26200"
    assert info["platform"] == "Windows-11-10.0.26200-SP0"


def test_windows_10_build_keeps_windows_10_label():
    info = _display_system_platform(
        system="Windows",
        release="10",
        version="10.0.19045",
        platform_label="Windows-10-10.0.19045-SP0",
    )

    assert info["os"] == "Windows"
    assert info["os_release"] == "10"
    assert info["platform"] == "Windows-10-10.0.19045-SP0"


def test_non_windows_platform_unchanged():
    info = _display_system_platform(
        system="Linux",
        release="6.8.0",
        version="#1 SMP",
        platform_label="Linux-6.8.0-x86_64-with-glibc2.39",
    )

    assert info == {
        "os": "Linux",
        "os_release": "6.8.0",
        "os_version": "#1 SMP",
        "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
    }
