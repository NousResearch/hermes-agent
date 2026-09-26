"""The package lock must never choose an unstable Node release."""
from pm import update


def test_node_candidates_exclude_prereleases(monkeypatch):
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.8.0-alpha.0"}, {"version": "v26.7.0"},
        {"version": "v24.0.0-rc.1"}, {"version": "v24.20.0"},
    ])
    versions = get_package("node").latest_versions("linux-x64")
    assert versions == ["26.7.0", "24.20.0"]
    assert get_package("termux-docker").latest_versions("linux-arm64-bionic") == []


def test_node_candidates_cap_legacy_macos_at_node22(monkeypatch):
    from pm import packages
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.7.0"}, {"version": "v24.20.0"},
        {"version": "v23.11.1"}, {"version": "v22.22.2"},
    ])
    monkeypatch.setattr(update.platform, "mac_ver", lambda: ("12.7.6", ("", "", ""), "x86_64"))
    monkeypatch.setattr(packages, "current_target", lambda: "darwin-x64")

    assert get_package("node").latest_versions("darwin-x64") == ["22.22.2"]


def test_node_candidates_keep_latest_on_supported_macos(monkeypatch):
    from pm import packages
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.7.0"}, {"version": "v24.20.0"},
    ])
    monkeypatch.setattr(update.platform, "mac_ver", lambda: ("13.5", ("", "", ""), "x86_64"))
    monkeypatch.setattr(packages, "current_target", lambda: "darwin-x64")

    assert get_package("node").latest_versions("darwin-x64") == ["26.7.0", "24.20.0"]
