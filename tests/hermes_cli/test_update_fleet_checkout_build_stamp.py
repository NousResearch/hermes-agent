"""``checkout_contains`` on a build-stamped image (no ``.git``): ancestry collapses to stamp equality.

On a Docker/Cloud image ``git merge-base`` has nothing to walk, so the probe was always False and the
pending-restart catch-up printed "every gateway already serves the checkout" and "still off the checkout
code" in the same run. The stamp IS the checkout there.
"""

from hermes_cli import update_cmd_fleet_checkout as chk


def _identity(monkeypatch, sha, source):
    import hermes_cli.build_info as bi
    monkeypatch.setattr(bi, "get_code_identity", lambda refresh=False: {"sha": sha, "short_sha": sha[:8], "source": source})


def test_build_stamped_image_contains_exactly_its_stamp(monkeypatch):
    stamped = "b936546561aa0d2e6d0f7c3d1a9c5e8f2b4d6a70"
    _identity(monkeypatch, stamped, "build-file")
    # no git call may be attempted on an image: make one blow up if it is
    monkeypatch.setattr(chk.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("git probe on a build-stamped image")))
    assert chk.checkout_contains(stamped) is True
    assert chk.checkout_contains(stamped[:12]) is True  # short form recorded by an older writer
    assert chk.checkout_contains("0000000000aa0d2e6d0f7c3d1a9c5e8f2b4d6a70") is False


def test_git_checkout_still_walks_ancestry(monkeypatch):
    """Control: a source install keeps asking git, so a carried hotfix past the pulled SHA still counts."""
    _identity(monkeypatch, "deadbeef" * 5, "git")
    calls = []

    class _R:
        returncode = 0

    monkeypatch.setattr(chk.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _R())
    assert chk.checkout_contains("cafebabe" * 5) is True
    from hermes_cli.update_custody import git_subcommand

    assert calls and calls[0][0] == "git" and git_subcommand(calls[0][1:]) == "merge-base"
    assert calls[0][-3:] == ["--is-ancestor", "cafebabe" * 5, "HEAD"]


def test_partial_clone_never_downloads_history_for_a_sha_it_lacks(monkeypatch, tmp_path):
    """A recorded SHA missing from a partial clone is "not contained", answered offline: an
    on-demand fetch of a commit pulls its whole history and kept growing ``.git`` (#129712)."""
    import subprocess
    from types import SimpleNamespace

    from hermes_cli import update_cmd

    def git(cwd, *args):
        return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                              capture_output=True, text=True, check=True).stdout.strip()

    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "config", "uploadpack.allowFilter", "true")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(origin, "commit", "-q", "--allow-empty", "-m", "base")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", "--filter=blob:none", f"file://{origin}", str(clone))
    git(origin, "commit", "-q", "--allow-empty", "-m", "newer")
    unseen = git(origin, "rev-parse", "HEAD")
    packs = sorted((clone / ".git" / "objects" / "pack").glob("*.pack"))
    _identity(monkeypatch, git(clone, "rev-parse", "HEAD"), "git")
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=str(clone)))

    assert chk.checkout_contains(unseen) is False

    assert sorted((clone / ".git" / "objects" / "pack").glob("*.pack")) == packs
