"""A stable update names the checkout by the release it moved to (#143656).

The checkout's version is the newest release tag merged into HEAD. A stable update
fetched the release commit with ``--no-tags``, so HEAD sat on vX.Y.Z while /health,
the install stamp and the receipt still named the previous release.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd_check
from hermes_cli.source_releases import SourceTarget
from hermes_cli.version_info import _git_version_info


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_stable_update_fetch_brings_the_release_tag(tmp_path, monkeypatch):
    origin, checkout = tmp_path / "origin", tmp_path / "checkout"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "f").write_text("1")
    _git(origin, "add", "f")
    _git(origin, "commit", "-qm", "one")
    _git(origin, "tag", "v0.22.2")
    _git(tmp_path, "clone", "-q", str(origin), str(checkout))
    (origin / "f").write_text("2")
    _git(origin, "commit", "-qam", "two")
    _git(origin, "tag", "v0.22.3")
    release = _git(origin, "rev-parse", "HEAD")

    monkeypatch.setattr(update_cmd_check, "_uc", lambda: SimpleNamespace(
        _m=lambda: SimpleNamespace(PROJECT_ROOT=checkout), _update_run_channel=lambda _a: "stable"))
    monkeypatch.setattr("hermes_cli.source_releases.resolve_source_target", lambda *a, **k: SourceTarget(
        "stable", "stable", "NousResearch/hermes-agent", commit=release, version="0.22.3"))
    monkeypatch.setattr("hermes_cli.config.require_readable_config_before_write", lambda _p: {})
    request = {"home": str(tmp_path)}
    target_ref, _sha, _is_head, _repo = update_cmd_check.select_apply_target(
        SimpleNamespace(branch=None), "main", request, git_cmd=["git"], stop=lambda: None)

    _git(checkout, "fetch", "--no-tags", "origin", target_ref)
    update_cmd_check.fetch_release_tag(["git"], checkout, request)
    _git(checkout, "merge", "-q", "--ff-only", release)

    info = _git_version_info(checkout)
    assert (info.commit, info.base_version, info.display_version) == (release, "0.22.3", "0.22.3")


@pytest.mark.parametrize("git_cmd", [["git"], [sys.executable + "-missing-git"]])
def test_an_unfetchable_release_tag_is_reported_never_fatal(tmp_path, capsys, git_cmd):
    """Best effort means the update goes on, but not silently: without the tag every version
    reader keeps naming the previous release. A git that cannot even run (missing binary, a
    custody refusal: both OSErrors) is the same non-fatal miss, not a crash of the update."""
    _git(tmp_path, "init", "-q", "-b", "main")  # no origin: the fetch itself fails

    update_cmd_check.fetch_release_tag(git_cmd, tmp_path, {"release_tag": "v9.9.9"})

    assert "Could not fetch the v9.9.9 tag" in capsys.readouterr().out
