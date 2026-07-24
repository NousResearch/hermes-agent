"""Release-tag policy: new releases use semver, old CalVer tags remain readable."""

import importlib.util
from pathlib import Path


_RELEASE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("hermes_release", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def test_release_tag_uses_the_semver_version():
    assert release.release_tag_for_version("0.20.0") == "v0.20.0"


def test_last_tag_prefers_semver_over_newer_looking_legacy_calver(monkeypatch):
    monkeypatch.setattr(
        release,
        "git",
        lambda *_args: "v2026.7.20\nv0.20.0\nv0.19.0",
    )

    assert release.get_last_tag() == "v0.20.0"


def test_last_tag_falls_back_to_legacy_calver_history(monkeypatch):
    monkeypatch.setattr(release, "git", lambda *_args: "v2026.7.20\nv2026.7.7")

    assert release.get_last_tag() == "v2026.7.20"
