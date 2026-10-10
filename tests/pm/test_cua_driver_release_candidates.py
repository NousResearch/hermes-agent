"""trycua flags every cua-driver-rs release as a GitHub prerelease, so the
stable channel only resolves when the prerelease flag is opted in."""
from pm import update
from pm.registry import get_package

# trycua/cua releases, as the API returns them: stable-shaped cua-driver-rs
# tags carry prerelease=true, nightlies and sandbox builds are separate shapes.
TRYCUA_RELEASES = [
    {"tag_name": "cua-driver-rs-v0.30.2", "prerelease": True, "draft": False},
    {"tag_name": "nightly-cua-driver-rs-v0.30.2-nightly.20260928", "prerelease": True, "draft": False},
    {"tag_name": "cua-driver-rs-vsandbox-v0.4.3", "prerelease": True, "draft": False},
    {"tag_name": "sandbox-v0.8.0", "prerelease": False, "draft": False},
    {"tag_name": "cua-driver-rs-v0.21.0", "prerelease": True, "draft": False},
    {"tag_name": "cua-driver-rs-v0.29.0-draft", "prerelease": False, "draft": True},
    {"tag_name": "latest", "prerelease": False, "draft": False},
]


def test_cua_driver_candidates_resolve_prerelease_flagged_stable_tags(monkeypatch):
    monkeypatch.setattr(update, "_get_json", lambda url: TRYCUA_RELEASES)
    assert get_package("cua-driver").latest_versions("darwin-arm64") == ["0.30.2", "0.21.0"]


def test_default_filter_still_drops_prereleases(monkeypatch):
    monkeypatch.setattr(update, "_get_json", lambda url: TRYCUA_RELEASES)
    assert update.github_release_tags("trycua/cua", strip_prefix="cua-driver-rs-v") == []


def test_include_prereleases_never_admits_drafts_or_wrong_shapes(monkeypatch):
    monkeypatch.setattr(update, "_get_json", lambda url: TRYCUA_RELEASES)
    tags = update.github_release_tags(
        "trycua/cua", strip_prefix="cua-driver-rs-v", include_prereleases=True
    )
    assert tags == ["0.30.2", "0.21.0"]
