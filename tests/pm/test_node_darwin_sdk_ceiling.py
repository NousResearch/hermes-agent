"""A darwin host older than the SDK a Node release was built against must not
be offered that release: dyld aborts it on every launch and each startup
"repair" re-downloads the same unusable bytes.
"""
import sys

from pm import update


def _index(monkeypatch, versions):
    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": f"v{v}"} for v in versions
    ])


def _host(monkeypatch, sdk):
    monkeypatch.setattr(update.sys, "platform", "darwin")
    monkeypatch.setattr(update, "_installed_macos_sdk", lambda: sdk)


def test_macos_12_host_is_capped_below_the_unsatisfiable_majors(monkeypatch):
    from pm.registry import get_package

    _index(monkeypatch, ["27.0.0", "26.12.1", "24.9.0", "22.23.4", "22.22.3"])
    _host(monkeypatch, (12, 3))
    versions = get_package("node").latest_versions("darwin-x64")
    assert versions == ["22.23.4", "22.22.3"]


def test_modern_darwin_host_keeps_upstream_newest(monkeypatch):
    from pm.registry import get_package

    _index(monkeypatch, ["27.0.0", "26.12.1", "22.23.4"])
    _host(monkeypatch, (15, 2))
    assert get_package("node").latest_versions("darwin-x64") == ["27.0.0", "26.12.1", "22.23.4"]


def test_unknown_sdk_never_caps(monkeypatch):
    from pm.registry import get_package

    _index(monkeypatch, ["26.12.1", "22.23.4"])
    _host(monkeypatch, None)
    assert get_package("node").latest_versions("darwin-x64") == ["26.12.1", "22.23.4"]


def test_linux_targets_are_never_capped(monkeypatch):
    from pm.registry import get_package

    _index(monkeypatch, ["26.12.1", "22.23.4"])
    monkeypatch.setattr(update.sys, "platform", "linux")
    assert get_package("node").latest_versions("linux-x64") == ["26.12.1", "22.23.4"]
