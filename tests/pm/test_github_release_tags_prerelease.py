"""trycua/cua flags every cua-driver-rs release as prerelease; the resolver must still see them."""

from pm import update
from pm.packages import CuaDriver

RELEASES = [
    {"tag_name": "cua-driver-rs-v0.33.1", "prerelease": True, "draft": False},
    {"tag_name": "nightly-cua-driver-rs-v0.30.5-nightly.20260929", "prerelease": True, "draft": False},
    {"tag_name": "cua-driver-rs-vsandbox-v0.4.3", "prerelease": True, "draft": False},
    {"tag_name": "cua-driver-rs-v0.34.0", "prerelease": True, "draft": True},
    {"tag_name": "cua-driver-rs-v0.21.0", "prerelease": False, "draft": False},
]


def _fake_index(monkeypatch):
    monkeypatch.setattr(update, "_get_json", lambda url: RELEASES if "page=1" in url else [])


def test_prereleases_skipped_by_default(monkeypatch):
    _fake_index(monkeypatch)
    assert update.github_release_tags("trycua/cua", strip_prefix="cua-driver-rs-v") == ["0.21.0"]


def test_prereleases_kept_on_request_but_drafts_and_non_versions_dropped(monkeypatch):
    _fake_index(monkeypatch)
    tags = update.github_release_tags("trycua/cua", strip_prefix="cua-driver-rs-v", include_prerelease=True)
    assert tags == ["0.33.1", "0.21.0"]


def test_cua_driver_latest_versions_see_prerelease_builds(monkeypatch):
    _fake_index(monkeypatch)
    assert CuaDriver().latest_versions("win32-x64")[0] == "0.33.1"
