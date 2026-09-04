"""Release-tag policy: new releases use semver, old CalVer tags remain readable."""

import importlib.util
from pathlib import Path

import pytest


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


class _FakeResult:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def _remotes(monkeypatch, names: list[str]):
    monkeypatch.setattr(
        release, "git_result", lambda *_args, **_kw: _FakeResult("\n".join(names))
    )


def test_single_remote_is_used_without_a_flag(monkeypatch):
    _remotes(monkeypatch, ["origin"])

    assert release.resolve_push_remote(None) == "origin"


def test_multiple_remotes_require_an_explicit_flag(monkeypatch):
    _remotes(monkeypatch, ["fork", "origin"])

    with pytest.raises(SystemExit, match="pass --remote"):
        release.resolve_push_remote(None)


def test_explicit_remote_is_honored_among_many(monkeypatch):
    _remotes(monkeypatch, ["fork", "origin"])

    assert release.resolve_push_remote("fork") == "fork"


def test_unknown_remote_is_rejected(monkeypatch):
    _remotes(monkeypatch, ["fork", "origin"])

    with pytest.raises(SystemExit, match="not configured"):
        release.resolve_push_remote("upstream")


def test_no_remotes_is_rejected(monkeypatch):
    _remotes(monkeypatch, [])

    with pytest.raises(SystemExit, match="no git remotes"):
        release.resolve_push_remote(None)


def test_github_repo_parsed_from_ssh_and_https_urls(monkeypatch):
    urls = {
        "fork": "git@github.com:ethernet8023/hermes-agent.git",
        "origin": "https://github.com/NousResearch/hermes-agent",
        "gitlab": "git@gitlab.com:someone/elsewhere.git",
    }
    monkeypatch.setattr(
        release,
        "git_result",
        lambda *args, **_kw: _FakeResult(urls[args[-1]]),
    )

    assert release.remote_github_repo("fork") == "ethernet8023/hermes-agent"
    assert release.remote_github_repo("origin") == "NousResearch/hermes-agent"
    assert release.remote_github_repo("gitlab") is None
