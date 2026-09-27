"""Stable admission reads the version and the attempt from the attempt ref, not from the checkout.

``main`` carries ``0.0.0`` by design, so comparing the tag against
``pyproject.toml`` would refuse every release. The attempt ref
``rc.<N>-vX.Y.Z`` names the version and the attempt, and the only question
about the commit is whether it is on ``main``.
"""
import pytest


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
