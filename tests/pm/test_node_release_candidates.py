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
