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


def test_node_candidates_keep_catalogue_for_shared_resolution(monkeypatch):
    from pm import packages
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.7.0"}, {"version": "v24.20.0"},
        {"version": "v23.11.1"}, {"version": "v22.22.2"},
    ])
    monkeypatch.setattr(update.platform, "mac_ver", lambda: ("12.7.6", ("", "", ""), "x86_64"))
    monkeypatch.setattr(packages, "current_target", lambda: "darwin-x64")

    assert get_package("node").latest_versions("darwin-x64") == [
        "26.7.0", "24.20.0", "23.11.1", "22.22.2"
    ]

    assert update.legacy_macos_node_version(
        get_package("node").latest_versions("darwin-x64"),
        "darwin-x64",
        host_target="darwin-x64",
    ) == "22.22.2"


def test_legacy_macos_uses_fallback_only_for_native_artifact(monkeypatch):
    from pm import packages
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.7.0"}, {"version": "v22.23.3"},
    ])
    monkeypatch.setattr(update.platform, "mac_ver", lambda: ("12.7.6", ("", "", ""), "x86_64"))
    monkeypatch.setattr(packages, "current_target", lambda: "darwin-x64")

    node = get_package("node")
    artifacts = {
        target: [{"url": node.fetch_urls("26.7.0", target)[0], "sha256": "a" * 64}]
        for target in ("darwin-x64", "linux-x64")
    }
    decision = update.resolve_package(
        node, ["darwin-x64", "linux-x64"], "26.7.0", artifacts=artifacts
    )

    assert decision.version == "26.7.0"
    assert decision.per_target == {"darwin-x64": "22.23.3", "linux-x64": "26.7.0"}
    assert set(decision.artifact_updates) == {"darwin-x64"}
    assert decision.changed


def test_node_candidates_keep_latest_on_supported_macos(monkeypatch):
    from pm import packages
    from pm.registry import get_package

    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"version": "v26.7.0"}, {"version": "v24.20.0"},
    ])
    monkeypatch.setattr(update.platform, "mac_ver", lambda: ("13.5", ("", "", ""), "x86_64"))
    monkeypatch.setattr(packages, "current_target", lambda: "darwin-x64")

    assert update.legacy_macos_node_version(
        ["26.7.0", "24.20.0"], "darwin-x64", host_target="darwin-x64"
    ) is None