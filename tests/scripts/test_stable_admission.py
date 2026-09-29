"""Stable admission reads the version and the attempt from the attempt ref, not from the checkout.

``main`` carries ``0.0.0`` by design, so comparing the tag against
``pyproject.toml`` would refuse every release. The attempt ref
``rc.<N>-vX.Y.Z`` names the version and the attempt, and the only question
about the commit is whether it is on ``main``.
"""
import json
import subprocess

import pytest


SSH_SIGNATURE = """-----BEGIN SSH SIGNATURE-----
U1NIU0lHAAAAAQAAADMAAAALc3NoLWVkMjU1MTkAAAAggv7kgN4gq9ilCWVWxT+yhG7bsr
I9MDkVe9kzlnz6azsAAAADZ2l0AAAAAAAAAAZzaGE1MTIAAABTAAAAC3NzaC1lZDI1NTE5
AAAAQEpfHRbnoIdqp6f7VhHqY66Im8aoqVRR8z+VchVeNSWOVEOlM7jM7QzMcvNLeE+B3V
yR4zgvmhXTUJgyPe+zMgM=
-----END SSH SIGNATURE-----"""


def test_admission_reads_the_version_and_attempt_from_the_attempt_ref():
    from scripts.releases.stable import admit_claim

    commit = "a" * 40
    admitted = admit_claim("rc.2-v0.21.5", commit, on_main=lambda sha: sha == commit)

    assert admitted == {
        "claim_tag": "rc.2-v0.21.5", "tag": "v0.21.5",
        "version": "0.21.5", "attempt": 2, "commit": commit,
    }


def test_admission_refuses_a_claim_for_a_commit_off_main():
    from scripts.releases.stable import admit_claim

    with pytest.raises(ValueError, match="not on main"):
        admit_claim("rc.1-v0.21.5", "b" * 40, on_main=lambda _sha: False)


@pytest.mark.parametrize("tag", ["v0.21.5", "v0.21.5-rc", "abandoned-rc.1-v0.21.5"])
def test_admission_refuses_a_tag_that_is_not_an_attempt_ref(tag):
    from scripts.releases.stable import admit_claim

    with pytest.raises(ValueError, match="not a claim tag"):
        admit_claim(tag, "a" * 40, on_main=lambda _sha: True)


def test_a_signed_claim_tag_is_read_through_the_jobs_own_git_read(tmp_path, monkeypatch):
    """The job reads a claim with ``git tag -l --format=%(contents)``, and a
    signed tag appends its armor to that message. The reader has to parse the
    record before the armor, or every signed claim fails admission."""
    from scripts.releases.stable import _claim_metadata

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    record = {"schema": 1, "version": "0.21.5", "attempt": 19, "commit": "3" * 40,
              "autopublish": False, "skipBundles": False, "skipTests": True,
              "claimEpoch": 1790701941}
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("commit", "--allow-empty", "-qm", "fixture")
    git("tag", "-a", "rc.19-v0.21.5", "-m",
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" + SSH_SIGNATURE)

    raw = subprocess.check_output(
        ["git", "tag", "-l", "rc.19-v0.21.5", "--format=%(contents)"], cwd=repo, text=True).strip()
    assert "-----BEGIN SSH SIGNATURE-----" in raw
    with pytest.raises(json.JSONDecodeError):  # the shape that failed admission
        json.loads(raw)
    assert _claim_metadata(raw, version="0.21.5", attempt=19, commit="3" * 40) == record
